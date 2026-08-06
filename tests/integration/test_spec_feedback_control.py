"""Design §7.4 / §20.3 / §18.4 — STOPPED schedules reminder; INTERRUPTED crash/continue does not."""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.reducer import RuntimeState

from .test_feedback_reminders import ReminderHarness, reminder_harness  # noqa: F401


@pytest.mark.asyncio
async def test_spec_7_4_20_3_stopped_schedules_feedback_reminder(
    reminder_harness: ReminderHarness,
) -> None:
    """§7.4 / §20.3: Maisaka/user STOPPED starts feedback wait (unlike EXPIRED/INTERRUPTED)."""

    await reminder_harness.finish("STOPPED")
    reminders = await reminder_harness.reminder_rows()
    assert len(reminders) == 1
    assert reminders[0]["status"] == "pending"
    assert reminders[0]["round_id"] == reminder_harness.round_id

    terminal_batches = [
        batch
        for batch in reminder_harness.store.batches
        if any(cmd.kind == "insert_feedback_reminder" for cmd in batch)
    ]
    assert terminal_batches, "StopRequested must emit insert_feedback_reminder via _feedback_commands"
    kinds = {cmd.kind for cmd in terminal_batches[-1]}
    assert "insert_feedback_reminder" in kinds
    assert "update_round_status" in kinds

    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1


@pytest.mark.asyncio
async def test_spec_20_3_expired_unlike_stopped_does_not_schedule(
    reminder_harness: ReminderHarness,
) -> None:
    """§20.3: EXPIRED does not trigger reminder (contrast with STOPPED)."""

    await reminder_harness.finish("EXPIRED")
    assert await reminder_harness.reminder_rows() == []
    assert not any(
        cmd.kind == "insert_feedback_reminder"
        for batch in reminder_harness.store.batches
        for cmd in batch
    )
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_spec_18_4_20_3_crash_mark_interrupted_does_not_insert_reminder(
    reminder_harness: ReminderHarness,
) -> None:
    """§18.4 / §20.3: crash recovery mark INTERRUPTED never inserts a feedback reminder."""

    interrupted_round = reminder_harness.round_id
    count = await reminder_harness.store.mark_active_rounds_interrupted(float(reminder_harness.clock()))
    assert count == 1

    task = await reminder_harness.store.load_task(reminder_harness.task_id)
    assert task is not None and task.current_round is not None
    assert task.current_round.status is TaskStatus.INTERRUPTED
    assert task.current_round.round_id == interrupted_round

    assert await reminder_harness.reminder_rows() == []
    assert not any(
        cmd.kind == "insert_feedback_reminder"
        for batch in reminder_harness.store.batches
        for cmd in batch
    )

    # FeedbackService contract: INTERRUPTED is a no-reminder terminal even if asked explicitly.
    assert (
        reminder_harness.service.commands_for_status_transition(
            task_id=reminder_harness.task_id,
            round_id=interrupted_round,
            new_status=TaskStatus.INTERRUPTED.value,
            ended_at=float(reminder_harness.clock()),
        )
        == ()
    )

    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0


@pytest.mark.asyncio
async def test_spec_18_4_20_3_continue_after_interrupted_creates_no_reminder_for_interrupted_round(
    reminder_harness: ReminderHarness,
) -> None:
    """§18.4 / §20.3: continue new round after INTERRUPTED must not reminder the interrupted round."""

    interrupted_round = reminder_harness.round_id
    count = await reminder_harness.store.mark_active_rounds_interrupted(float(reminder_harness.clock()))
    assert count == 1
    reminder_harness.controller.state = RuntimeState(
        reminder_harness.task_id,
        TaskStatus.INTERRUPTED,
        generation=reminder_harness.generation,
        active_round_id=interrupted_round,
    )

    await reminder_harness.continue_round()
    assert reminder_harness.controller.state.status is TaskStatus.RUNNING
    assert reminder_harness.round_id != interrupted_round

    reminders = await reminder_harness.reminder_rows()
    assert not any(row["round_id"] == interrupted_round for row in reminders), (
        "interrupted round must not receive a feedback reminder"
    )
    assert not any(row["status"] == "pending" for row in reminders)

    assert not any(
        cmd.kind == "insert_feedback_reminder"
        for batch in reminder_harness.store.batches
        for cmd in batch
    ), "ContinueRequested after INTERRUPTED must not schedule a reminder"

    # Continue may cancel pending for the old round (none exist); that must not create one either.
    cancel_batches = [
        batch
        for batch in reminder_harness.store.batches
        if any(cmd.kind == "cancel_pending_feedback_reminders" for cmd in batch)
    ]
    assert cancel_batches, "ContinueRequested should still emit cancel via _feedback_commands"

    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 0
