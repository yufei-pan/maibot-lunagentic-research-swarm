"""Module-level coverage for lifecycle + outbox + continue deadline semantics (§7, §17.2)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.events import (
    ContinueRequested,
    FormalizationSucceeded,
    PauseExpired,
    PauseRequested,
    ReportCompleted,
    ReportDeadlineReached,
    StopRequested,
)
from lunagentic_research_swarm.runtime.reducer import RuntimeState, reduce_event


NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_spec_7_lifecycle_pause_expire_stop_continue_matrix() -> None:
    running = RuntimeState("t", TaskStatus.RUNNING, generation=0, active_round_id="r1", active_leaves={"a": 1.0})
    pausing = reduce_event(running, PauseRequested("e1", "t", "r1", 0, occurred_at=NOW))
    assert pausing.next_state.status is TaskStatus.PAUSING

    paused = RuntimeState("t", TaskStatus.PAUSED, generation=0, active_round_id="r1", active_leaves={"a": 1.0})
    expired = reduce_event(paused, PauseExpired("e2", "t", "r1", 0, occurred_at=NOW))
    assert expired.next_state.status is TaskStatus.EXPIRED
    assert expired.next_state.raw_context_released is True

    stopped = reduce_event(running, StopRequested("e3", "t", "r1", 0, occurred_at=NOW, reason="user"))
    assert stopped.next_state.status is TaskStatus.STOPPED
    assert stopped.next_state.generation == 1


def test_spec_13_1_intermediate_report_completed_with_leaves_returns_running() -> None:
    state = RuntimeState(
        "t",
        TaskStatus.REPORTING,
        generation=0,
        active_round_id="r1",
        report_epoch=1,
        active_leaves={"a": 2.0},
    )
    transition = reduce_event(state, ReportCompleted("e", "t", "r1", 0, occurred_at=NOW, report_id="rpt"))
    assert transition.next_state.status is TaskStatus.RUNNING


def test_spec_13_1_empty_leaf_deadline_opens_final_epoch() -> None:
    state = RuntimeState("t", TaskStatus.RUNNING, generation=0, active_round_id="r1", active_leaves={})
    transition = reduce_event(
        state,
        ReportDeadlineReached("e", "t", "r1", 0, occurred_at=NOW, epoch=1),
    )
    assert transition.next_state.status is TaskStatus.FINALIZING
    assert [type(effect).__name__ for effect in transition.effects] == ["OpenReportEpoch"]
    assert transition.effects[0].payload["kind"] == "FINAL"


def test_spec_7_2_formalization_success_starts_running_with_root_leaf() -> None:
    state = RuntimeState("t", TaskStatus.FORMALIZING, generation=0, active_round_id="r1")
    # FormalizationSucceeded requires sha — use FormalizedTask.create path via event fields.
    from lunagentic_research_swarm.models import FormalizedTask

    formalized = FormalizedTask.create("正式目标")
    transition = reduce_event(
        state,
        FormalizationSucceeded(
            "e",
            "t",
            "r1",
            0,
            occurred_at=NOW,
            formalized_text=formalized.text,
            formalized_sha256=formalized.sha256,
        ),
    )
    assert transition.next_state.status is TaskStatus.RUNNING
    assert transition.next_state.formalized_task is not None


def test_spec_7_5_continue_from_expired_requires_funds_or_leaves() -> None:
    expired = RuntimeState(
        "t",
        TaskStatus.EXPIRED,
        generation=1,
        active_round_id="r1",
        credit_pool=-5.0,
        active_leaves={},
    )
    transition = reduce_event(
        expired,
        ContinueRequested(
            "e",
            "t",
            "r1",
            1,
            occurred_at=NOW,
            adjustment=0.0,
            next_round_id="r2",
        ),
    )
    # Insufficient funds path surfaces as error/notify rather than silent RUNNING.
    assert transition.next_state.status is TaskStatus.EXPIRED or transition.error is not None or transition.ignored
