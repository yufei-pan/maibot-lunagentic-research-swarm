"""Production scheduler worker for committed runtime effects."""

from __future__ import annotations

from typing import Any

from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    OpenReportEpoch,
    PerformAgentCall,
    PerformBranchSummary,
    PerformFormalization,
    PerformProcedureBatch,
)


class RuntimeEffectRunner:
    """Execute effects without owning runtime state.

    The manager is bound after scheduler construction to break the
    manager -> scheduler -> worker -> manager construction cycle.
    """

    def __init__(self, turn_worker: Any) -> None:
        self._turn_worker = turn_worker
        self._manager: Any | None = None

    def bind_manager(self, manager: Any) -> None:
        if manager is None:
            raise TypeError("manager 不能为空")
        if self._manager is not None and self._manager is not manager:
            raise RuntimeError("runtime effect runner 已绑定 manager")
        self._manager = manager

    def _require_manager(self) -> Any:
        if self._manager is None:
            raise RuntimeError("runtime effect runner 尚未绑定 manager")
        return self._manager

    async def run(self, effect: Any, _token: Any = None) -> Any:
        if isinstance(effect, PerformFormalization):
            # ResearchManager owns the short-lived raw formalization coroutine.
            return None

        manager = self._require_manager()
        if isinstance(effect, PerformAgentCall):
            prepared = await manager.prepare_agent_effect(effect)
            completed = await self._turn_worker.perform_agent_call(prepared)
            await manager.handle_runtime_event(completed)
            return completed
        if isinstance(effect, PerformProcedureBatch):
            prepare = getattr(manager, "prepare_procedure_effect", None)
            prepared = await prepare(effect) if callable(prepare) else effect
            completed = await self._turn_worker.perform_procedure_batch(prepared)
            await manager.handle_runtime_event(completed)
            return completed
        if isinstance(effect, PerformBranchSummary):
            await manager.handle_branch_summary_effect(effect)
            return None
        if isinstance(effect, OpenReportEpoch):
            await manager.handle_runtime_effect(effect)
            return None
        if isinstance(effect, NotifyToolWaiter) and effect.payload.get("action") == "materialize_child":
            await manager.materialize_child_effect(effect)
        return None

    __call__ = run


__all__ = ["RuntimeEffectRunner"]
