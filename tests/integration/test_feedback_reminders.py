"""终态反馈提醒：经 TaskController._feedback_commands 调度/取消，非 harness 本地 if。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.feedback import FeedbackService
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.controller import TaskController
from lunagentic_research_swarm.runtime.events import (
    ContinueRequested,
    FinalReportCompleted,
    FinalReportFailed,
    PauseExpired,
    StopRequested,
)
from lunagentic_research_swarm.runtime.reducer import RuntimeState
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


class FakeScheduler:
    def __init__(self) -> None:
        self.enqueued: list[Any] = []

    async def enqueue(self, effect: Any) -> bool:
        self.enqueued.append(effect)
        return True


class RecordingStore:
    """记录每次 transact 批次，同时委托真实 SQLiteStateStore。"""

    def __init__(self, inner: SQLiteStateStore) -> None:
        self.inner = inner
        self.batches: list[tuple[StoreCommand, ...]] = []

    async def open(self) -> None:
        await self.inner.open()

    async def close(self) -> None:
        await self.inner.close()

    async def transact(self, commands: Sequence[StoreCommand]) -> None:
        batch = tuple(commands)
        self.batches.append(batch)
        await self.inner.transact(batch)

    async def run_locked(self, fn: Any) -> Any:
        return await self.inner.run_locked(fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class ReminderHarness:
    store: RecordingStore
    service: FeedbackService
    outbox: MaisakaOutbox
    clock: FakeClock
    maisaka: FakeMaisaka
    scheduler: FakeScheduler
    controller: TaskController
    task_id: str = "lrs_remind"
    stream_id: str = "stream-remind"
    round_id: str = "rnd_remind"
    generation: int = 0
    _seq: int = field(default=0, init=False)
    _event_seq: int = field(default=0, init=False)

    @classmethod
    async def create(cls, tmp_path: Path) -> ReminderHarness:
        raw = SQLiteStateStore(tmp_path / "reminders.sqlite3")
        store = RecordingStore(raw)
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
        store.batches.clear()
        scheduler = FakeScheduler()
        controller = TaskController(
            RuntimeState("lrs_remind", TaskStatus.RUNNING, generation=0, active_round_id="rnd_remind"),
            store=store,
            scheduler=scheduler,
            feedback=service,
        )
        return cls(
            store=store,
            service=service,
            outbox=outbox,
            clock=clock,
            maisaka=maisaka,
            scheduler=scheduler,
            controller=controller,
        )

    async def close(self) -> None:
        await self.outbox.close()
        await self.store.close()

    def _event_id(self, prefix: str) -> str:
        self._event_seq += 1
        return f"{prefix}_{self._event_seq}"

    def _occurred_at(self) -> datetime:
        return datetime.fromtimestamp(float(self.clock()), tz=timezone.utc)

    async def finish(self, status: str) -> None:
        """经真实 TaskController._feedback_commands 进入终态（非 harness 本地 schedule）。"""

        if status == "COMPLETED":
            self.controller.state = RuntimeState(
                self.task_id,
                TaskStatus.FINALIZING,
                generation=self.generation,
                active_round_id=self.round_id,
            )
            accepted = await self.controller.apply(
                FinalReportCompleted(
                    self._event_id("final_ok"),
                    self.task_id,
                    self.round_id,
                    self.generation,
                    occurred_at=self._occurred_at(),
                    report_id="rpt_ok",
                )
            )
        elif status == "COMPLETED_WITH_ERRORS":
            self.controller.state = RuntimeState(
                self.task_id,
                TaskStatus.FINALIZING,
                generation=self.generation,
                active_round_id=self.round_id,
            )
            accepted = await self.controller.apply(
                FinalReportFailed(
                    self._event_id("final_err"),
                    self.task_id,
                    self.round_id,
                    self.generation,
                    occurred_at=self._occurred_at(),
                    error_code="report_failed",
                    error_message="合成失败",
                )
            )
        elif status == "STOPPED":
            self.controller.state = RuntimeState(
                self.task_id,
                TaskStatus.RUNNING,
                generation=self.generation,
                active_round_id=self.round_id,
            )
            accepted = await self.controller.apply(
                StopRequested(
                    self._event_id("stop"),
                    self.task_id,
                    self.round_id,
                    self.generation,
                    occurred_at=self._occurred_at(),
                    reason="测试停止",
                )
            )
            self.generation = int(self.controller.state.generation)
        elif status == "EXPIRED":
            self.controller.state = RuntimeState(
                self.task_id,
                TaskStatus.PAUSED,
                generation=self.generation,
                active_round_id=self.round_id,
            )
            accepted = await self.controller.apply(
                PauseExpired(
                    self._event_id("expire"),
                    self.task_id,
                    self.round_id,
                    self.generation,
                    occurred_at=self._occurred_at(),
                )
            )
        else:
            raise AssertionError(f"unsupported terminal via controller: {status}")
        assert accepted, f"controller rejected transition to {status}"
        assert self.controller.state.status.value == status

    async def reminder_rows(self) -> list[dict[str, Any]]:
        def _read(connection: Any) -> list[dict[str, Any]]:
            rows = connection.execute(
                "SELECT reminder_id, round_id, status, due_at FROM feedback_reminders ORDER BY reminder_id"
            ).fetchall()
            return [dict(row) for row in rows]

        return await self.store.run_locked(_read)

    async def run_due(self) -> None:
        await self.service.process_due()
        await self.outbox.deliver_once()

    async def submit_feedback(self) -> None:
        await self.service.submit(
            task_id=self.task_id,
            disposition="accepted",
            notes="已审阅",
            stream_id=self.stream_id,
        )

    async def continue_round(self) -> None:
        """经 ContinueRequested → _feedback_commands 取消 pending，而非直接 cancel_due_to_continue。"""

        self._seq += 1
        new_round = f"rnd_remind_{self._seq}"
        next_generation = int(self.controller.state.generation) + 1
        accepted = await self.controller.apply(
            ContinueRequested(
                self._event_id("continue"),
                self.task_id,
                self.round_id,
                self.generation,
                occurred_at=self._occurred_at(),
                active_leaves={},
                next_round_id=new_round,
                next_generation=next_generation,
                round_number=1 + self._seq,
                time_budget_seconds=60,
            )
        )
        assert accepted, "ContinueRequested must be accepted"
        assert self.controller.state.status is TaskStatus.RUNNING
        self.round_id = new_round
        self.generation = next_generation


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
    reminders = await reminder_harness.reminder_rows()
    assert len(reminders) == 1
    assert reminders[0]["status"] == "pending"
    assert reminders[0]["round_id"] == reminder_harness.round_id

    # reminder INSERT 与 update_round_status 必须在同一 transact 批次。
    terminal_batches = [
        batch
        for batch in reminder_harness.store.batches
        if any(cmd.kind == "insert_feedback_reminder" for cmd in batch)
    ]
    assert terminal_batches, "controller must emit insert_feedback_reminder"
    batch = terminal_batches[-1]
    kinds = {cmd.kind for cmd in batch}
    assert "insert_feedback_reminder" in kinds
    assert "update_round_status" in kinds

    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1
    intent = reminder_harness.maisaka.trigger_kwargs[0]["intent"]
    assert "submit_research_feedback" in intent or "feedback" in intent.lower()
    assert reminder_harness.task_id in str(reminder_harness.maisaka.trigger_kwargs[0])


@pytest.mark.asyncio
async def test_expired_via_controller_does_not_schedule(reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish("EXPIRED")
    assert await reminder_harness.reminder_rows() == []
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0
    assert not any(
        cmd.kind == "insert_feedback_reminder"
        for batch in reminder_harness.store.batches
        for cmd in batch
    )


@pytest.mark.asyncio
async def test_feedback_cancels_pending_reminder(reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish("COMPLETED")
    await reminder_harness.submit_feedback()
    reminders = await reminder_harness.reminder_rows()
    assert reminders and all(row["status"] == "cancelled" for row in reminders)
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_submit_suppresses_already_triggered_reminder_outbox(
    reminder_harness: ReminderHarness,
) -> None:
    """C-I1: process_due 已插入 outbox 后提交反馈，不得再向 Maisaka nudge。"""

    await reminder_harness.finish("COMPLETED")
    reminder_harness.clock.advance(600)
    await reminder_harness.service.process_due()

    def _outbox(connection: Any) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT outbox_id, status, idempotency_key FROM maisaka_outbox "
            "WHERE idempotency_key = ?",
            (f"lrs:feedback-reminder:{reminder_harness.round_id}",),
        ).fetchall()
        return [dict(row) for row in rows]

    pending = await reminder_harness.store.run_locked(_outbox)
    assert pending and pending[0]["status"].upper() == "PENDING"

    await reminder_harness.submit_feedback()
    after = await reminder_harness.store.run_locked(_outbox)
    assert after and after[0]["status"].lower() == "cancelled"

    await reminder_harness.outbox.deliver_once()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_deliver_skips_feedback_reminder_when_feedback_already_submitted(
    reminder_harness: ReminderHarness,
) -> None:
    """C-I1 defense: even if outbox stays PENDING, deliver re-checks feedback_events."""

    await reminder_harness.finish("COMPLETED")
    reminder_harness.clock.advance(600)
    await reminder_harness.service.process_due()
    await reminder_harness.submit_feedback()

    # Force outbox back to PENDING to simulate cancel race / missed cancel.
    def _reopen(connection: Any) -> None:
        connection.execute(
            """
            UPDATE maisaka_outbox
            SET status = 'PENDING', last_error = NULL, next_attempt_at = 0
            WHERE idempotency_key = ?
            """,
            (f"lrs:feedback-reminder:{reminder_harness.round_id}",),
        )

    await reminder_harness.store.run_locked(_reopen)
    await reminder_harness.outbox.deliver_once()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_continue_cancels_pending_reminder(reminder_harness: ReminderHarness) -> None:
    await reminder_harness.finish("STOPPED")
    await reminder_harness.continue_round()
    reminders = await reminder_harness.reminder_rows()
    assert reminders and all(row["status"] == "cancelled" for row in reminders)
    cancel_batches = [
        batch
        for batch in reminder_harness.store.batches
        if any(cmd.kind == "cancel_pending_feedback_reminders" for cmd in batch)
    ]
    assert cancel_batches, "ContinueRequested must emit cancel via _feedback_commands"
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0
