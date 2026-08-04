"""Serial, transaction-before-effect driver for one research task."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

from lunagentic_research_swarm.errors import STORAGE_COMMIT_FAILED
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.events import RuntimeEvent
from lunagentic_research_swarm.runtime.reducer import _replace_state, _state_round_id, reduce_event
from lunagentic_research_swarm.storage.sqlite import StoreCommand


class TaskController:
    """Owns the authoritative in-memory state and serializes reducer transitions."""

    def __init__(
        self,
        state: Any,
        *,
        store: Any,
        scheduler: Any | None = None,
        executor: Any | None = None,
        health: Any = None,
    ) -> None:
        self.state = state
        self.store = store
        self.scheduler = scheduler if scheduler is not None else executor
        if self.scheduler is None:
            raise TypeError("TaskController 需要 scheduler/executor")
        self.health = health
        self._inbox: deque[RuntimeEvent] = deque()
        self.stopped = False

    async def submit(self, event: RuntimeEvent) -> bool:
        if self.stopped:
            return False
        self._inbox.append(event)
        return True

    async def drain_once(self) -> bool:
        if self.stopped:
            self._inbox.clear()
            return False
        if not self._inbox:
            return False
        await self._apply(self._inbox.popleft())
        return True

    async def drain(self) -> None:
        while self._inbox and not self.stopped:
            await self.drain_once()

    async def _apply(self, event: RuntimeEvent) -> None:
        transition = reduce_event(self.state, event)
        if transition.ignored:
            self.state = transition.next_state
            return
        try:
            await self.store.transact(transition.commands)
        except Exception as exc:
            await self._fail_after_storage_error(event, exc)
            return
        self.state = transition.next_state
        for effect in transition.effects:
            await self.scheduler.enqueue(effect)

    async def _fail_after_storage_error(self, event: RuntimeEvent, exc: Exception) -> None:
        self.state = _replace_state(
            self.state,
            status=TaskStatus.FAILED,
            failure_code=STORAGE_COMMIT_FAILED,
        )
        fallback = (
            StoreCommand(
                "update_round_status",
                {
                    "round_id": _state_round_id(self.state) or event.round_id,
                    "status": TaskStatus.FAILED.value,
                    "report_deadline_at": None,
                    "ended_at": event.occurred_at.timestamp(),
                },
            ),
        )
        try:
            await self.store.transact(fallback)
        except Exception as fallback_exc:
            self.stopped = True
            self._inbox.clear()
            self._record_health(
                {
                    "status": "degraded",
                    "code": STORAGE_COMMIT_FAILED,
                    "message": str(fallback_exc),
                    "original": str(exc),
                }
            )

    def _record_health(self, payload: Mapping[str, Any]) -> None:
        if callable(self.health):
            self.health(payload)
        elif isinstance(self.health, dict):
            self.health["runtime"] = dict(payload)
        elif self.health is not None:
            try:
                setattr(self.health, "runtime", dict(payload))
            except Exception:
                pass
