"""统一生成请求、公开 task 路由与物理 pinning 路由。"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from lunagentic_research_swarm.errors import LRSError

from .physical_pinning import PhysicalPinningAdapter, read_initial_host_model_snapshot
from .pricing import TokenUsage, normalize_actual_usage, sanitize_host_snapshot


@dataclass(frozen=True, slots=True)
class ModelSelector:
    scheme: Literal["task", "model"]
    name: str
    raw: str

    @classmethod
    def parse(cls, raw_selector: str) -> ModelSelector:
        if not isinstance(raw_selector, str) or raw_selector != raw_selector.strip() or ":" not in raw_selector:
            raise LRSError("invalid_selector", f"无效模型 selector：{raw_selector}")
        scheme, name = raw_selector.split(":", 1)
        if scheme not in {"task", "model"} or not name or name != name.strip():
            raise LRSError("invalid_selector", f"无效模型 selector：{raw_selector}")
        return cls(scheme=scheme, name=name, raw=raw_selector)  # type: ignore[arg-type]


def resolve_generation_selector(configured_selector: str, force_selector: str = "") -> ModelSelector:
    """非空 force selector 统一覆盖生成式 LLM；embedding 调用方不使用本函数。"""

    return ModelSelector.parse(force_selector if force_selector else configured_selector)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    selector: ModelSelector | str
    messages: str | Sequence[Mapping[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        parsed = self.selector if isinstance(self.selector, ModelSelector) else ModelSelector.parse(self.selector)
        object.__setattr__(self, "selector", parsed)
        value = self.max_tokens
        if isinstance(value, bool) or (
            value is not None and (not isinstance(value, int) or value < 0 or value > 65536)
        ):
            raise ValueError("max_tokens 必须为 0..65536 的整数或 None")
        if value == 0:
            object.__setattr__(self, "max_tokens", None)


@dataclass(frozen=True, slots=True)
class GenerationError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    response: str
    tool_calls: list[dict[str, Any]] | None
    model_name: str
    usage: TokenUsage | None
    success: bool
    error: GenerationError | None
    duration: float


class HostModelSnapshotReader:
    """一次性读取初始 Host snapshot，并只缓存白名单副本。"""

    def __init__(self, *, loader: Callable[[], dict[str, Any] | None] = read_initial_host_model_snapshot) -> None:
        self._loader = loader
        self._attempted = False
        self._snapshot: dict[str, Any] | None = None
        self.health_issue: str | None = None

    def read(self) -> dict[str, Any] | None:
        if self._attempted:
            return self._snapshot
        self._attempted = True
        try:
            raw = self._loader()
            if raw is None:
                self.health_issue = "host_model_snapshot_unavailable"
                return None
            self._snapshot = sanitize_host_snapshot(raw)
        except Exception:
            self.health_issue = "host_model_snapshot_unavailable"
            self._snapshot = None
        return self._snapshot

    def __repr__(self) -> str:
        return f"HostModelSnapshotReader(attempted={self._attempted}, health_issue={self.health_issue!r})"


class LLMGateway:
    """对上层冻结单一、无 reasoning 的生成结果形状。"""

    def __init__(self, ctx: Any, *, physical_pinning: PhysicalPinningAdapter | Any | None = None) -> None:
        self._ctx = ctx
        self._physical_pinning = physical_pinning or PhysicalPinningAdapter()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        selector = request.selector
        assert isinstance(selector, ModelSelector)
        try:
            if selector.scheme == "task":
                kwargs = {
                    "prompt": request.messages,
                    "model": selector.name,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                }
                if request.tools is None:
                    raw = await self._ctx.llm.generate(**kwargs)
                else:
                    raw = await self._ctx.llm.generate_with_tools(tools=request.tools, **kwargs)
            else:
                raw = await self._physical_pinning.generate(
                    physical_name=selector.name,
                    prompt=request.messages,
                    tools=request.tools,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            return self._normalize_result(raw, time.perf_counter() - started)
        except LRSError as exc:
            return self._failure(exc.code, exc.message, time.perf_counter() - started)
        except Exception as exc:
            return self._failure("llm_generation_failed", f"LLM 调用失败：{exc}", time.perf_counter() - started)

    @staticmethod
    def _failure(
        code: str,
        message: str,
        duration: float,
        *,
        usage: TokenUsage | None = None,
        model_name: str = "",
        response: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            response=response,
            tool_calls=None,
            model_name=model_name,
            usage=usage,
            success=False,
            error=GenerationError(code, message),
            duration=duration,
        )

    def _normalize_result(self, raw: Any, duration: float) -> GenerationResult:
        if not isinstance(raw, Mapping):
            return self._failure("llm_invalid_result", "Host 返回了无效 LLM 结果", duration)
        if not bool(raw.get("success")):
            raw_error = raw.get("error")
            if isinstance(raw_error, Mapping):
                code = str(raw_error.get("code") or "llm_generation_failed")
                message = str(raw_error.get("message") or "LLM 调用失败")
            else:
                code = "llm_generation_failed"
                message = str(raw_error or "LLM 调用失败")
            usage_fields = (
                raw.get("prompt_tokens", 0),
                raw.get("completion_tokens", 0),
                raw.get("prompt_cache_hit_tokens", 0),
                raw.get("prompt_cache_miss_tokens", 0),
            )
            usage: TokenUsage | None = None
            if any(value not in (None, 0) for value in usage_fields):
                try:
                    usage = normalize_actual_usage(
                        prompt_tokens=usage_fields[0],
                        completion_tokens=usage_fields[1],
                        cache_hit_tokens=usage_fields[2],
                        cache_miss_tokens=usage_fields[3],
                    )
                except LRSError as exc:
                    return self._failure(exc.code, exc.message, duration)
            return self._failure(
                code,
                message,
                duration,
                usage=usage,
                model_name=str(raw.get("model") or raw.get("model_name") or ""),
                response=str(raw.get("response", "") or ""),
            )
        try:
            usage = normalize_actual_usage(
                prompt_tokens=raw.get("prompt_tokens", 0),
                completion_tokens=raw.get("completion_tokens", 0),
                cache_hit_tokens=raw.get("prompt_cache_hit_tokens", 0),
                cache_miss_tokens=raw.get("prompt_cache_miss_tokens", 0),
            )
        except LRSError as exc:
            return self._failure(exc.code, exc.message, duration)
        raw_calls = raw.get("tool_calls")
        tool_calls = [dict(item) for item in raw_calls] if isinstance(raw_calls, list) else None
        return GenerationResult(
            response=str(raw.get("response", "") or ""),
            tool_calls=tool_calls,
            model_name=str(raw.get("model") or raw.get("model_name") or ""),
            usage=usage,
            success=True,
            error=None,
            duration=duration,
        )
