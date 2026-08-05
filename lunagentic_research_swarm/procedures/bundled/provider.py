"""本地 builtin Procedure provider：describe + invoke 统一入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lunagentic_research_swarm.extensions.contracts import ProcedureResult
from lunagentic_research_swarm.procedures.bundled.memory import (
    MEMORY_HANDLERS,
    memory_procedure_definitions,
)


class BundledProcedureProvider:
    """通过与第三方相同的 registry/invoker 路径暴露内置 Procedures。"""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._handlers = dict(MEMORY_HANDLERS)

    def describe(self) -> list[dict[str, Any]]:
        """返回可交给 ProcedureRegistry.replace_provider 的 model_dump payload。"""

        return [item.model_dump(mode="json") for item in memory_procedure_definitions()]

    async def invoke(self, procedure_id: str, arguments: Mapping[str, Any] | None = None) -> ProcedureResult:
        """按 procedure_id 分发到 handler map；未知 ID 返回 invalid_arguments。"""

        handler = self._handlers.get(procedure_id)
        if handler is None:
            return ProcedureResult(
                success=False,
                data=None,
                error={"code": "invalid_arguments", "message": f"未知 Procedure：{procedure_id}"},
                metadata={},
            )
        return await handler(self.ctx, dict(arguments or {}))
