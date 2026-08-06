"""Feedback reminder rules for INTERRUPTED (§20.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lunagentic_research_swarm.feedback import FeedbackService, REMINDER_TERMINAL_STATUSES
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore


def test_spec_20_3_interrupted_is_not_a_reminder_terminal() -> None:
    assert TaskStatus.INTERRUPTED.value not in REMINDER_TERMINAL_STATUSES
    assert TaskStatus.EXPIRED.value not in REMINDER_TERMINAL_STATUSES
    assert TaskStatus.FAILED.value not in REMINDER_TERMINAL_STATUSES
    assert TaskStatus.COMPLETED.value in REMINDER_TERMINAL_STATUSES
    assert TaskStatus.STOPPED.value in REMINDER_TERMINAL_STATUSES


@pytest.mark.asyncio
async def test_spec_20_3_commands_for_interrupted_do_not_insert_reminder(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "feedback.sqlite3")
    await store.open()
    try:
        service = FeedbackService(
            store=store,
            reminders_enabled=True,
            feedback_wait_seconds=600,
            clock=lambda: 1000.0,
        )
        interrupted = service.commands_for_status_transition(
            task_id="task-1",
            round_id="round-1",
            new_status=TaskStatus.INTERRUPTED.value,
            ended_at=1000.0,
        )
        assert interrupted == ()
        completed = [
            cmd.kind
            for cmd in service.commands_for_status_transition(
                task_id="task-1",
                round_id="round-1",
                new_status=TaskStatus.COMPLETED.value,
                ended_at=1000.0,
            )
        ]
        assert "insert_feedback_reminder" in completed
    finally:
        await store.close()
