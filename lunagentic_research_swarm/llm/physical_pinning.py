"""隔离的 Host 内部物理模型固定适配器。

此模块是 LRS 唯一接触 ``src.*`` 的位置；接口不兼容时必须显式失败。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def read_initial_host_model_snapshot() -> dict[str, Any] | None:
    """读取 Host 已验证的内存模型配置；不猜测 TOML 路径。"""

    try:
        from src.config.config import model_config
    except (ImportError, ModuleNotFoundError):
        return None
    return model_config.model_dump(mode="json")


def _supports_parameters(callable_object: Any, required: set[str]) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()) or required <= set(
        parameters
    )


def _error(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "response": "", "error": {"code": code, "message": message}}


def _tool_calls_to_payload(raw_calls: Any) -> list[dict[str, Any]] | None:
    if raw_calls is None:
        return None
    result: list[dict[str, Any]] = []
    for raw in raw_calls:
        if isinstance(raw, Mapping):
            result.append(dict(raw))
            continue
        call_id = str(getattr(raw, "call_id", "") or getattr(raw, "id", ""))
        name = str(getattr(raw, "func_name", "") or getattr(raw, "name", ""))
        arguments = getattr(raw, "args", {}) or {}
        item: dict[str, Any] = {"id": call_id, "function": {"name": name, "arguments": arguments}}
        extra_content = getattr(raw, "extra_content", None)
        if extra_content:
            item["extra_content"] = extra_content
        result.append(item)
    return result


@dataclass(frozen=True, slots=True)
class PhysicalPinningStatus:
    available: bool
    error_code: str | None = None


_TASK_CONFIG_FIELDS = {"model_list", "temperature", "max_tokens", "selection_strategy", "slow_threshold"}
_GENERATION_FIELDS = {"temperature", "max_tokens", "tools"}


class PhysicalPinningAdapter:
    """把一个 synthetic task 固定为唯一物理模型。"""

    def __init__(self, *, message_factory_builder: Callable[[Any], Callable[..., Any]] | None = None) -> None:
        self._message_factory_builder = message_factory_builder

    @staticmethod
    def _load_host_contracts() -> tuple[type[Any], type[Any]]:
        from src.config.model_configs import TaskConfig
        from src.llm_models.utils_model import LLMOrchestrator

        return TaskConfig, LLMOrchestrator

    def _resolve_message_factory_builder(self) -> Callable[[Any], Callable[..., Any]]:
        if self._message_factory_builder is not None:
            return self._message_factory_builder
        from src.services.llm_service import _build_prompt_message_factory

        return _build_prompt_message_factory

    def check_compatibility(self) -> PhysicalPinningStatus:
        """在加载期检查当前 Host 是否同时支持文本与消息列表 pinning。"""

        try:
            TaskConfig, LLMOrchestrator = self._load_host_contracts()
        except Exception:
            return PhysicalPinningStatus(False, "physical_pinning_unsupported")
        plain = getattr(LLMOrchestrator, "generate_response_async", None)
        messages = getattr(LLMOrchestrator, "generate_response_with_message_async", None)
        compatible = (
            _supports_parameters(TaskConfig, _TASK_CONFIG_FIELDS)
            and _supports_parameters(LLMOrchestrator.__init__, {"task_name", "request_type", "session_id"})
            and inspect.iscoroutinefunction(plain)
            and _supports_parameters(plain, _GENERATION_FIELDS | {"prompt"})
            and inspect.iscoroutinefunction(messages)
            and _supports_parameters(messages, _GENERATION_FIELDS | {"message_factory"})
        )
        return PhysicalPinningStatus(compatible, None if compatible else "physical_pinning_unsupported")

    async def generate(
        self,
        *,
        physical_name: str,
        prompt: str | Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        try:
            TaskConfig, LLMOrchestrator = self._load_host_contracts()
        except Exception:
            return _error("physical_pinning_unsupported", "当前 Host 不支持物理模型固定")

        init_required = {"task_name", "request_type", "session_id"}
        method_name = "generate_response_async" if isinstance(prompt, str) else "generate_response_with_message_async"
        method = getattr(LLMOrchestrator, method_name, None)
        method_required = set(_GENERATION_FIELDS)
        method_required.add("prompt" if isinstance(prompt, str) else "message_factory")
        if not _supports_parameters(LLMOrchestrator.__init__, init_required) or not _supports_parameters(
            method, method_required
        ):
            return _error("physical_pinning_unsupported", "当前 Host 的物理模型固定接口签名不兼容")

        fallback_temperature = temperature if temperature is not None else 0.3
        try:
            task_config = TaskConfig(
                model_list=[physical_name],
                temperature=fallback_temperature,
                max_tokens=65536,
                selection_strategy="random",
                slow_threshold=30.0,
            )

            class _PinnedOrchestrator(LLMOrchestrator):  # type: ignore[misc, valid-type]
                def __init__(self, pinned_config: Any) -> None:
                    self._pinned_config = pinned_config
                    super().__init__(
                        task_name="replyer",
                        request_type="plugin.lunagentic_research_swarm",
                        session_id="",
                    )

                def _get_task_config_or_raise(self) -> Any:
                    return self._pinned_config

                def _refresh_task_config(self) -> Any:
                    return self._pinned_config

            orchestrator = _PinnedOrchestrator(task_config)
            common = {
                "temperature": temperature,
                # None 让 ModelInfo / extra_params 优先，synthetic TaskConfig 仅作最终 fallback。
                "max_tokens": None,
                "tools": tools,
            }
            if isinstance(prompt, str):
                result = await orchestrator.generate_response_async(prompt=prompt, **common)
            else:
                raw_factory = self._resolve_message_factory_builder()(list(prompt))

                async def message_factory(*args: Any) -> Any:
                    built = raw_factory(*args)
                    return await built if inspect.isawaitable(built) else built

                result = await orchestrator.generate_response_with_message_async(
                    message_factory=message_factory,
                    **common,
                )
        except TypeError:
            return _error("physical_pinning_unsupported", "当前 Host 的物理模型固定接口签名不兼容")
        except Exception as exc:
            return _error("physical_pinning_failed", f"物理模型 `{physical_name}` 调用失败：{exc}")

        return {
            "success": True,
            "response": str(getattr(result, "response", "") or ""),
            "model": str(getattr(result, "model_name", physical_name) or physical_name),
            "model_name": str(getattr(result, "model_name", physical_name) or physical_name),
            "tool_calls": _tool_calls_to_payload(getattr(result, "tool_calls", None)),
            "prompt_tokens": getattr(result, "prompt_tokens", 0),
            "completion_tokens": getattr(result, "completion_tokens", 0),
            "total_tokens": getattr(result, "total_tokens", 0),
            "prompt_cache_hit_tokens": getattr(result, "prompt_cache_hit_tokens", 0),
            "prompt_cache_miss_tokens": getattr(result, "prompt_cache_miss_tokens", 0),
        }
