"""Serial, transaction-before-effect driver for one research task."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from lunagentic_research_swarm.errors import STORAGE_COMMIT_FAILED
from lunagentic_research_swarm.feedback import REMINDER_TERMINAL_STATUSES
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.events import ContinueRequested, RuntimeEvent
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
        feedback: Any | None = None,
    ) -> None:
        self.state = state
        self.store = store
        self.scheduler = scheduler if scheduler is not None else executor
        if self.scheduler is None:
            raise TypeError("TaskController 需要 scheduler/executor")
        self.health = health
        self.feedback = feedback
        self._inbox: deque[RuntimeEvent] = deque()
        self._lock = asyncio.Lock()
        self.stopped = False

    async def submit(self, event: RuntimeEvent) -> bool:
        async with self._lock:
            if self.stopped:
                return False
            self._inbox.append(event)
            return True

    async def drain_once(self) -> bool:
        async with self._lock:
            return await self._drain_once_locked()

    async def drain(self) -> None:
        async with self._lock:
            while self._inbox and not self.stopped:
                await self._drain_once_locked()

    async def submit_and_drain(self, event: RuntimeEvent) -> bool:
        """Append and drain through the task's sole state/effect boundary."""

        async with self._lock:
            if self.stopped:
                return False
            self._inbox.append(event)
            while self._inbox and not self.stopped:
                await self._drain_once_locked()
            return not self.stopped

    async def apply(
        self,
        event: RuntimeEvent,
        *,
        extra_commands: Sequence[StoreCommand] = (),
        effects: Sequence[Any] | None = None,
        state_changes: Mapping[str, Any] | None = None,
    ) -> bool:
        """Atomically apply an event with manager-supplied durable details."""

        async with self._lock:
            if self.stopped:
                return False
            return await self._apply(
                event,
                extra_commands=extra_commands,
                effects=effects,
                state_changes=state_changes,
            )

    async def _drain_once_locked(self) -> bool:
        if self.stopped:
            self._inbox.clear()
            return False
        if not self._inbox:
            return False
        await self._apply(self._inbox.popleft())
        return True

    async def _apply(
        self,
        event: RuntimeEvent,
        *,
        extra_commands: Sequence[StoreCommand] = (),
        effects: Sequence[Any] | None = None,
        state_changes: Mapping[str, Any] | None = None,
    ) -> bool:
        transition = reduce_event(self.state, event)
        if transition.ignored:
            self.state = transition.next_state
            return False
        try:
            commands = transition.commands if transition.error is not None else (*transition.commands, *extra_commands)
            if transition.error is None:
                commands = (*commands, *self._feedback_commands(event, transition.next_state))
            await self.store.transact(commands)
        except Exception as exc:
            await self._fail_after_storage_error(event, exc)
            return False
        if transition.error is not None:
            # A reducer error may carry its own durable compensation (for
            # example a rejected terminal continue persists its credit pool),
            # but it must never accept manager-supplied state, commands, or
            # effects intended for a successful transition.
            self.state = transition.next_state
            for effect in transition.effects:
                await self.scheduler.enqueue(effect)
            return False
        self.state = _replace_state(transition.next_state, **dict(state_changes or {}))
        selected_effects = transition.effects if effects is None else effects
        for effect in selected_effects:
            await self.scheduler.enqueue(effect)
        return True

    def _feedback_commands(self, event: RuntimeEvent, next_state: Any) -> tuple[StoreCommand, ...]:
        """终态同事务插入 reminder；continue/new round 取消 pending。"""

        feedback = self.feedback
        if feedback is None:
            return ()
        builder = getattr(feedback, "commands_for_status_transition", None)
        if not callable(builder):
            return ()
        new_status = getattr(getattr(next_state, "status", None), "value", None) or str(
            getattr(next_state, "status", "")
        )
        cancel_round_id: str | None = None
        if isinstance(event, ContinueRequested):
            cancel_round_id = str(event.round_id)
        elif new_status not in REMINDER_TERMINAL_STATUSES:
            return ()
        round_id = _state_round_id(next_state) or event.round_id
        ended_at = float(event.occurred_at.timestamp())
        return tuple(
            builder(
                task_id=event.task_id,
                round_id=str(round_id),
                new_status=str(new_status),
                ended_at=ended_at,
                cancel_round_id=cancel_round_id,
            )
        )

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
