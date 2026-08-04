"""Serial, transaction-before-effect driver for one research task."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

from lunagentic_research_swarm.errors import STORAGE_COMMIT_FAILED
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.events import RuntimeEvent
from lunagentic_research_swarm.runtime.reducer import RuntimeState, _replace_state, reduce_event
from lunagentic_research_swarm.storage.sqlite import StoreCommand


class TaskController:
    """Owns the authoritative in-memory state and serializes reducer transitions."""

    def __init__(self, state: RuntimeState, *, store: Any, scheduler: Any, health: Any = None) -> None:
        self.state = state
        self.store = store
        self.scheduler = scheduler
        self.health = health
        self._inbox: deque[RuntimeEvent] = deque()
        self.stopped = False

    async def submit(self, event: RuntimeEvent) -> None:
        self._inbox.append(event)

    async def drain_once(self) -> bool:
        if not self._inbox:
            return False
        await self.apply(self._inbox.popleft())
        return True

    async def drain(self) -> None:
        while await self.drain_once():
            pass

    async def apply(self, event: RuntimeEvent) -> None:
        transition = reduce_event(self.state, event)
        if transition.ignored:
            return
        try:
            await self.store.transact(transition.commands)
        except Exception as exc:
            self.state = _replace_state(self.state, status=TaskStatus.FAILED, failure_code=STORAGE_COMMIT_FAILED)
            self._record_health({"status": "degraded", "code": STORAGE_COMMIT_FAILED, "message": str(exc)})
            return
        self.state = transition.next_state
        for effect in transition.effects:
            await self.scheduler.enqueue(effect)

    def _record_health(self, payload: Mapping[str, Any]) -> None:
        if callable(self.health):
            self.health(payload)
        elif isinstance(self.health, dict):
            self.health["runtime"] = dict(payload)
