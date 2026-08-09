"""Production scheduler worker for committed runtime effects."""

from __future__ import annotations

import logging
from typing import Any

from lunagentic_research_swarm.errors import LRSError
from lunagentic_research_swarm.runtime.reducer import (
    ArmDeadline,
    ArmPauseExpiry,
    DeliverOutbox,
    NotifyToolWaiter,
    OpenReportEpoch,
    PerformAgentCall,
    PerformBranchSummary,
    PerformFormalization,
    PerformProcedureBatch,
    ReleaseRawContext,
)


_LOG = logging.getLogger(__name__)


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

    @staticmethod
    async def _fail_branch(manager: Any, effect: Any, exc: Exception, default_code: str) -> None:
        """Convert a crashed agent/procedure effect into a durable branch failure.

        The scheduler only records the exception type on an un-awaited future, so
        an unhandled worker error would otherwise strand the branch forever.
        """

        _LOG.exception("LRS runtime effect 执行失败：kind=%s", getattr(effect, "kind", "?"))
        code = exc.code if isinstance(exc, LRSError) else default_code
        message = exc.message if isinstance(exc, LRSError) else f"{type(exc).__name__}: {exc}"
        fail = getattr(manager, "fail_agent_effect", None)
        if not callable(fail):
            return
        try:
            await fail(effect, error_code=str(code), error_message=str(message))
        except Exception:
            _LOG.exception("LRS 无法终结失败分支：kind=%s", getattr(effect, "kind", "?"))

    async def run(self, effect: Any, _token: Any = None) -> Any:
        if isinstance(effect, PerformFormalization):
            # ResearchManager owns the short-lived raw formalization coroutine.
            return None

        manager = self._require_manager()
        if isinstance(effect, PerformAgentCall):
            try:
                payload = dict(getattr(effect, "payload", None) or {})
                # Reducer-built protocol corrections already reserved credits and
                # attached the correction user message + model: pin. Re-running
                # prepare_agent_effect would wipe those and double-reserve.
                if int(payload.get("correction_count") or 0) >= 1 and payload.get("messages") is not None:
                    prepared = effect
                else:
                    prepared = await manager.prepare_agent_effect(effect)
                completed = await self._turn_worker.perform_agent_call(prepared)
            except Exception as exc:
                await self._fail_branch(manager, effect, exc, "agent_effect_failed")
                return None
            await manager.handle_runtime_event(completed)
            return completed
        if isinstance(effect, PerformProcedureBatch):
            try:
                prepare = getattr(manager, "prepare_procedure_effect", None)
                prepared = await prepare(effect) if callable(prepare) else effect
                completed = await self._turn_worker.perform_procedure_batch(prepared)
            except Exception as exc:
                await self._fail_branch(manager, effect, exc, "procedure_effect_failed")
                return None
            await manager.handle_runtime_event(completed)
            return completed
        if isinstance(effect, PerformBranchSummary):
            await manager.handle_branch_summary_effect(effect)
            return None
        if isinstance(effect, OpenReportEpoch):
            await manager.handle_runtime_effect(effect)
            return None
        if isinstance(effect, ArmDeadline):
            await manager.arm_deadline_effect(effect)
            return None
        if isinstance(effect, ArmPauseExpiry):
            await manager.arm_pause_expiry_effect(effect)
            return None
        if isinstance(effect, ReleaseRawContext):
            await manager.release_raw_context_effect(effect)
            return None
        if isinstance(effect, DeliverOutbox):
            await manager.deliver_outbox_effect(effect)
            return None
        if isinstance(effect, NotifyToolWaiter):
            await manager.notify_tool_waiter_effect(effect)
            return None
        return None

    __call__ = run


__all__ = ["RuntimeEffectRunner"]
