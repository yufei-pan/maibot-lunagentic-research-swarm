"""终态反馈提醒：600 秒一次、continue/feedback 取消、EXPIRED 等不提醒。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.feedback import FeedbackService, REMINDER_TERMINAL_STATUSES
from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


class FakeClock:
    def __init__(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += float(seconds)
        return self.value


class FakeMaisaka:
    def __init__(self) -> None:
        self.trigger_calls = 0
        self.trigger_kwargs: list[dict[str, Any]] = []

    async def trigger(self, stream_id: str, intent: str, **kwargs: Any) -> None:
        self.trigger_calls += 1
        self.trigger_kwargs.append({"stream_id": stream_id, "intent": intent, **kwargs})

    async def append(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass
class ReminderHarness:
    store: SQLiteStateStore
    service: FeedbackService
    outbox: MaisakaOutbox
    clock: FakeClock
    maisaka: FakeMaisaka
    task_id: str = "lrs_remind"
    round_id: str = "rnd_remind"
    _seq: int = field(default=0, init=False)

    @classmethod
    async def create(cls, tmp_path: Path) -> ReminderHarness:
        store = SQLiteStateStore(tmp_path / "reminders.sqlite3")
        await store.open()
        clock = FakeClock(1000.0)
        maisaka = FakeMaisaka()
        outbox = MaisakaOutbox(store, maisaka, clock=clock, poll_interval_seconds=0.01)
        service = FeedbackService(
            store,
            outbox=outbox,
            clock=clock,
            feedback_wait_seconds=600,
            reminders_enabled=True,
            index_lessons=False,
        )
        await store.transact(
            [
                StoreCommand(
                    "insert_task",
                    {
                        "task_id": "lrs_remind",
                        "stream_id": "stream-remind",
                        "current_round_number": 1,
                        "created_at": 1.0,
                        "updated_at": 1.0,
                    },
                ),
                StoreCommand(
                    "insert_round",
                    {
                        "round_id": "rnd_remind",
                        "task_id": "lrs_remind",
                        "round_number": 1,
                        "generation": 0,
                        "status": "RUNNING",
                        "time_budget_seconds": 60,
                        "credit_pool": 10.0,
                        "started_at": 1.0,
                    },
                ),
            ]
        )
        return cls(store=store, service=service, outbox=outbox, clock=clock, maisaka=maisaka)

    async def close(self) -> None:
        await self.outbox.close()
        await self.store.close()

    async def finish(self, status: str) -> None:
        ended_at = float(self.clock())
        await self.store.transact(
            [
                StoreCommand(
                    "update_round_status",
                    {
                        "round_id": self.round_id,
                        "status": status,
                        "report_deadline_at": None,
                        "ended_at": ended_at,
                    },
                ),
            ]
        )
        if status in REMINDER_TERMINAL_STATUSES:
            await self.service.schedule(task_id=self.task_id, round_id=self.round_id, ended_at=ended_at)

    async def run_due(self) -> None:
        await self.service.process_due()
        await self.outbox.deliver_once()

    async def submit_feedback(self) -> None:
        await self.service.submit(task_id=self.task_id, disposition="accepted", notes="已审阅")

    async def continue_round(self) -> None:
        await self.service.cancel_due_to_continue(task_id=self.task_id, round_id=self.round_id)
        self._seq += 1
        new_round = f"rnd_remind_{self._seq}"
        await self.store.transact(
            [
                StoreCommand(
                    "insert_round",
                    {
                        "round_id": new_round,
                        "task_id": self.task_id,
                        "round_number": 1 + self._seq,
                        "generation": self._seq,
                        "status": "RUNNING",
                        "time_budget_seconds": 60,
                        "credit_pool": 10.0,
                        "started_at": float(self.clock()),
                    },
                ),
                StoreCommand(
                    "set_task_current_round",
                    {
                        "task_id": self.task_id,
                        "current_round_number": 1 + self._seq,
                        "updated_at": float(self.clock()),
                    },
                ),
            ]
        )
        self.round_id = new_round


@pytest.fixture
async def reminder_harness(tmp_path: Path) -> AsyncIterator[ReminderHarness]:
    harness = await ReminderHarness.create(tmp_path)
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.parametrize("status", ["COMPLETED", "COMPLETED_WITH_ERRORS", "STOPPED"])
@pytest.mark.asyncio
async def test_terminal_status_schedules_one_reminder(status: str, reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish(status)
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1
    intent = reminder_harness.maisaka.trigger_kwargs[0]["intent"]
    assert "submit_research_feedback" in intent or "feedback" in intent.lower()
    assert reminder_harness.task_id in str(reminder_harness.maisaka.trigger_kwargs[0])


@pytest.mark.parametrize("status", ["EXPIRED", "INTERRUPTED", "FAILED"])
@pytest.mark.asyncio
async def test_non_reminder_terminals_do_not_schedule(status: str, reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish(status)
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_feedback_cancels_pending_reminder(reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish("COMPLETED")
    await reminder_harness.submit_feedback()
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_continue_cancels_pending_reminder(reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish("STOPPED")
    await reminder_harness.continue_round()
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0
