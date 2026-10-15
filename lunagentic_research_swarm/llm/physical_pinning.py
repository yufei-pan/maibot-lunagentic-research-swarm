"""``model:`` selector 的物理模型固定适配器。

Host ≥ 1.2.5 的公开 ``llm.generate`` / ``llm.generate_with_tools`` capability 支持 ``model_name``
直选具体模型（SDK ≥ 2.8.1 暴露 ``task_name`` / ``model_name`` 参数）：指定模型失败时 Host 直接报错，
不会换用任务内其他模型，正好满足 pinning 语义。因此本模块不再 import Host 内部的 LLM 实现。

唯一保留的 ``src.*`` 读取是初次加载时的 ``model_config`` 价格快照——公开 capability 只给任务名，
不给价格；读取失败只会让价格按 0 计，不影响加载。
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


# pinning 请求借用的 Host 任务：只提供 hard_timeout 与 temperature/max_tokens 的最终 fallback，
# 实际模型由 model_name 固定。与 Host 对未指定任务的插件调用所用默认任务一致。
PINNED_MODEL_HOST_TASK = "utils"


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


@dataclass(frozen=True, slots=True)
class PhysicalPinningStatus:
    available: bool
    error_code: str | None = None


_ROUTE_FIELDS = {"task_name", "model_name"}


class PhysicalPinningAdapter:
    """通过公开 LLM capability 把一次调用固定到唯一物理模型。"""

    def __init__(self, ctx: Any | None = None) -> None:
        self._ctx = ctx

    def _llm_methods(self) -> tuple[Any, Any]:
        llm = getattr(self._ctx, "llm", None)
        return getattr(llm, "generate", None), getattr(llm, "generate_with_tools", None)

    def check_compatibility(self) -> PhysicalPinningStatus:
        """在加载期检查 SDK 的 ``ctx.llm`` 是否同时支持文本与工具调用的 model_name 直选。"""

        plain, with_tools = self._llm_methods()
        compatible = (
            inspect.iscoroutinefunction(plain)
            and _supports_parameters(plain, _ROUTE_FIELDS)
            and inspect.iscoroutinefunction(with_tools)
            and _supports_parameters(with_tools, _ROUTE_FIELDS)
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
        rpc_timeout_ms: int | None = None,
    ) -> Any:
        plain, with_tools = self._llm_methods()
        method = plain if tools is None else with_tools
        if method is None or not _supports_parameters(method, _ROUTE_FIELDS):
            return _error("physical_pinning_unsupported", "当前 SDK 的 ctx.llm 不支持 model_name 直选模型")

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "task_name": PINNED_MODEL_HOST_TASK,
            "model_name": physical_name,
            "temperature": temperature,
            # None 时 Host 依次取 ModelInfo.max_tokens / extra_params.max_tokens / 任务 max_tokens。
            "max_tokens": max_tokens,
        }
        if rpc_timeout_ms is not None:
            kwargs["rpc_timeout_ms"] = rpc_timeout_ms
        if tools is None:
            return await method(**kwargs)
        return await method(tools=tools, **kwargs)
