"""无密钥价格快照、usage 校验与 credits 换算。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from lunagentic_research_swarm.errors import LRSError

PriceSource = Literal[
    "plugin_override",
    "host_config",
    "host_unavailable_free",
    "host_unpriced_free",
    "actual_model_unknown_free",
]


def _nonnegative_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须为非负有限数")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} 必须为非负有限数")
    return normalized


@dataclass(frozen=True, slots=True)
class PriceProfile:
    price_in: float = 0.0
    cache: bool = False
    cache_price_in: float = 0.0
    price_out: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_in", _nonnegative_finite(self.price_in, "price_in"))
        if not isinstance(self.cache, bool):
            raise ValueError("cache 必须为布尔值")
        object.__setattr__(self, "cache_price_in", _nonnegative_finite(self.cache_price_in, "cache_price_in"))
        object.__setattr__(self, "price_out", _nonnegative_finite(self.price_out, "price_out"))

    @property
    def total_is_free(self) -> bool:
        return self.price_in == 0.0 and self.cache_price_in == 0.0 and self.price_out == 0.0


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    source: Literal["actual", "estimated"] = "estimated"


@dataclass(frozen=True, slots=True)
class ResolvedPrice:
    model_name: str
    profile: PriceProfile
    source: PriceSource


@dataclass(frozen=True, slots=True)
class ChargedUsage:
    price: ResolvedPrice
    usage: TokenUsage
    credits: float


@dataclass(frozen=True, slots=True)
class CallReconciliation:
    status: Literal["actual", "estimated", "estimated_unreconciled"]
    actual_credits: float
    adjustment_credits: float
    charged: ChargedUsage | None = None


def _usage_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LRSError("invalid_usage", f"无效 LLM usage：{field_name} 必须为非负整数")
    return value


def normalize_actual_usage(
    *,
    prompt_tokens: Any,
    completion_tokens: Any,
    cache_hit_tokens: Any,
    cache_miss_tokens: Any,
) -> TokenUsage:
    """校验 Host usage，并把未分类的 prompt token 保守记为 cache miss。"""

    prompt = _usage_integer(prompt_tokens, "prompt_tokens")
    completion = _usage_integer(completion_tokens, "completion_tokens")
    hit = _usage_integer(cache_hit_tokens, "cache_hit_tokens")
    miss = _usage_integer(cache_miss_tokens, "cache_miss_tokens")
    if hit + miss > prompt:
        raise LRSError("invalid_usage", "无效 LLM usage：cache hit 与 miss 之和超过 prompt tokens")
    miss += prompt - hit - miss
    return TokenUsage(prompt, completion, hit, miss, source="actual")


def price_units_to_credits(price_units: float) -> float:
    """Host 配置价格数值 1.0 固定换算为 100 credits。"""

    return float(price_units) * 100.0


def charge(profile: PriceProfile, usage: TokenUsage) -> float:
    """按完整价格 profile 计算一次调用的 credits。"""

    normalized = normalize_actual_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_hit_tokens=usage.cache_hit_tokens,
        cache_miss_tokens=usage.cache_miss_tokens,
    )
    if profile.cache:
        input_units = (
            normalized.cache_miss_tokens * profile.price_in
            + normalized.cache_hit_tokens * profile.cache_price_in
        ) / 1_000_000
    else:
        input_units = normalized.prompt_tokens * profile.price_in / 1_000_000
    output_units = normalized.completion_tokens * profile.price_out / 1_000_000
    return price_units_to_credits(input_units + output_units)


def _field(source: Any, name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _profile_from_source(source: Any) -> PriceProfile:
    return PriceProfile(
        price_in=_field(source, "price_in", 0.0),
        cache=_field(source, "cache", False),
        cache_price_in=_field(source, "cache_price_in", 0.0),
        price_out=_field(source, "price_out", 0.0),
    )


def sanitize_host_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """只复制价格与 task model-list 白名单，不保留原 snapshot 引用。"""

    models_raw = snapshot.get("models", [])
    tasks_raw = snapshot.get("model_task_config", {})
    if not isinstance(models_raw, list) or not isinstance(tasks_raw, Mapping):
        raise ValueError("Host 模型快照结构无效")

    models: list[dict[str, Any]] = []
    for raw in models_raw:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str) or not raw["name"].strip():
            raise ValueError("Host 模型快照包含无效模型条目")
        item: dict[str, Any] = {"name": raw["name"].strip()}
        for field_name in ("price_in", "cache", "cache_price_in", "price_out"):
            if field_name in raw:
                item[field_name] = raw[field_name]
        if len(item) > 1:
            _profile_from_source(item)
        models.append(item)

    tasks: dict[str, dict[str, list[str]]] = {}
    for task_name, raw_task in tasks_raw.items():
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("Host 模型快照包含无效 task 条目")
        # Host model_dump 会把 PluginConfigBase 元数据（field_docs / suppress_any_warning）
        # 混进 model_task_config；这些不是 task 条目，直接跳过。
        if not isinstance(raw_task, Mapping) or "model_list" not in raw_task:
            continue
        model_list = raw_task.get("model_list")
        if not isinstance(model_list, list) or not all(isinstance(name, str) for name in model_list):
            raise ValueError("Host task model_list 必须为字符串列表")
        tasks[task_name] = {"model_list": [name for name in model_list if name.strip()]}
    return {"models": models, "model_task_config": tasks}


class PriceCatalog:
    """只持有可公开价格字段的可热更新目录。"""

    def __init__(
        self,
        *,
        plugin_overrides: Mapping[str, Any],
        host_models: Mapping[str, PriceProfile],
        task_models: Mapping[str, list[str]],
        host_available: bool,
        host_priced_models: set[str] | None = None,
    ) -> None:
        self._plugin_overrides = {str(name): _profile_from_source(value) for name, value in plugin_overrides.items()}
        self._host_models = {str(name): profile for name, profile in host_models.items()}
        self._task_models = {str(task): list(models) for task, models in task_models.items()}
        self._host_available = bool(host_available)
        self._host_priced_models = set(host_priced_models if host_priced_models is not None else host_models)
        self._refresh_fingerprint()

    @classmethod
    def from_sources(
        cls,
        plugin_overrides: Mapping[str, Any],
        host_models: Mapping[str, PriceProfile | Mapping[str, Any]],
        task_models: Mapping[str, list[str]],
    ) -> PriceCatalog:
        profiles = {
            str(name): value if isinstance(value, PriceProfile) else _profile_from_source(value)
            for name, value in host_models.items()
        }
        return cls(
            plugin_overrides=plugin_overrides,
            host_models=profiles,
            task_models=task_models,
            host_available=True,
            host_priced_models=set(profiles),
        )

    @classmethod
    def from_host_snapshot(
        cls, snapshot: Mapping[str, Any] | None, plugin_overrides: Mapping[str, Any]
    ) -> PriceCatalog:
        if snapshot is None:
            return cls(
                plugin_overrides=plugin_overrides,
                host_models={},
                task_models={},
                host_available=False,
            )
        safe = sanitize_host_snapshot(snapshot)
        models: dict[str, PriceProfile] = {}
        priced: set[str] = set()
        for raw in safe["models"]:
            name = raw["name"]
            if len(raw) > 1:
                models[name] = _profile_from_source(raw)
                priced.add(name)
            else:
                models[name] = PriceProfile()
        tasks = {name: list(raw["model_list"]) for name, raw in safe["model_task_config"].items()}
        return cls(
            plugin_overrides=plugin_overrides,
            host_models=models,
            task_models=tasks,
            host_available=True,
            host_priced_models=priced,
        )

    def replace_host_snapshot(self, snapshot: Mapping[str, Any] | None) -> None:
        replacement = type(self).from_host_snapshot(snapshot, self._plugin_overrides)
        self._host_models = replacement._host_models
        self._task_models = replacement._task_models
        self._host_available = replacement._host_available
        self._host_priced_models = replacement._host_priced_models
        self._refresh_fingerprint()

    def with_plugin_overrides(self, plugin_overrides: Mapping[str, Any]) -> PriceCatalog:
        """复用已脱敏 Host 状态构造新目录，不原地污染已冻结 round。"""

        return type(self)(
            plugin_overrides=plugin_overrides,
            host_models=self._host_models,
            task_models=self._task_models,
            host_available=self._host_available,
            host_priced_models=self._host_priced_models,
        )

    def _refresh_fingerprint(self) -> None:
        payload = {
            "host_available": self._host_available,
            "plugin_overrides": {
                name: self._profile_dict(profile) for name, profile in sorted(self._plugin_overrides.items())
            },
            "host_models": {name: self._profile_dict(profile) for name, profile in sorted(self._host_models.items())},
            "host_priced_models": sorted(self._host_priced_models),
            "task_models": {name: list(models) for name, models in sorted(self._task_models.items())},
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _profile_dict(profile: PriceProfile) -> dict[str, Any]:
        return {
            "price_in": profile.price_in,
            "cache": profile.cache,
            "cache_price_in": profile.cache_price_in,
            "price_out": profile.price_out,
        }

    def __repr__(self) -> str:
        return (
            f"PriceCatalog(fingerprint={self.fingerprint!r}, models={len(self._host_models)}, "
            f"tasks={len(self._task_models)}, overrides={len(self._plugin_overrides)})"
        )

    def debug_snapshot(self) -> dict[str, Any]:
        """导出不含 provider/认证信息的可诊断价格与 task 快照。"""

        model_names = set(self._host_models) | set(self._plugin_overrides)
        return {
            "models": {
                name: self._profile_dict(self.resolve_model(name).profile)
                for name in sorted(model_names)
            },
            "tasks": {name: list(models) for name, models in sorted(self._task_models.items())},
        }

    def resolve_model(self, model_name: str, *, actual: bool = False) -> ResolvedPrice:
        if model_name in self._plugin_overrides:
            return ResolvedPrice(model_name, self._plugin_overrides[model_name], "plugin_override")
        if not self._host_available:
            return ResolvedPrice(model_name, PriceProfile(), "host_unavailable_free")
        if model_name in self._host_priced_models:
            return ResolvedPrice(model_name, self._host_models[model_name], "host_config")
        source: PriceSource = (
            "actual_model_unknown_free" if actual and model_name not in self._host_models else "host_unpriced_free"
        )
        return ResolvedPrice(model_name, PriceProfile(), source)

    def estimate_model_for_selector(self, selector: str | Any) -> ResolvedPrice:
        from .gateway import ModelSelector

        parsed = selector if isinstance(selector, ModelSelector) else ModelSelector.parse(selector)
        if parsed.scheme == "model":
            return self.resolve_model(parsed.name)
        models = self._task_models.get(parsed.name, [])
        return self.resolve_model(models[0] if models else "")

    def charge_actual(
        self,
        *,
        actual_model_name: str,
        prompt_tokens: Any,
        completion_tokens: Any,
        cache_hit_tokens: Any,
        cache_miss_tokens: Any,
    ) -> ChargedUsage:
        usage = normalize_actual_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
        )
        resolved = self.resolve_model(actual_model_name, actual=True)
        return ChargedUsage(resolved, usage, charge(resolved.profile, usage))

    def estimate_root_minimum(
        self,
        selector: str | Any,
        *,
        miss_input_tokens: int = 500_000,
        output_tokens: int = 50_000,
    ) -> ChargedUsage:
        """Design §11.3 low-budget probe; thresholds come from ``[budget]`` config."""

        resolved = self.estimate_model_for_selector(selector)
        miss = max(0, int(miss_input_tokens))
        completion = max(0, int(output_tokens))
        usage = TokenUsage(miss, completion, 0, miss, source="estimated")
        return ChargedUsage(resolved, usage, charge(resolved.profile, usage))

    def low_budget_warning(
        self,
        selector: str | Any,
        effective_budget: float,
        *,
        miss_input_tokens: int = 500_000,
        output_tokens: int = 50_000,
    ) -> str | None:
        estimate = self.estimate_root_minimum(
            selector, miss_input_tokens=miss_input_tokens, output_tokens=output_tokens
        )
        if estimate.credits == 0.0 or effective_budget >= estimate.credits:
            return None
        return f"有效预算 {effective_budget:g} credits 低于根调用保守估算 {estimate.credits:g} credits；任务仍会启动"

    def reconcile_call(
        self,
        *,
        estimated_credits: float,
        success: bool,
        actual_model_name: str,
        usage: TokenUsage | None,
    ) -> CallReconciliation:
        estimated = _nonnegative_finite(estimated_credits, "estimated_credits")
        if usage is None:
            status: Literal["estimated", "estimated_unreconciled"] = (
                "estimated" if success else "estimated_unreconciled"
            )
            return CallReconciliation(status, estimated, 0.0)
        charged = self.charge_actual(
            actual_model_name=actual_model_name,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_tokens=usage.cache_hit_tokens,
            cache_miss_tokens=usage.cache_miss_tokens,
        )
        return CallReconciliation("actual", charged.credits, estimated - charged.credits, charged)
