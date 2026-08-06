from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.models import TaskSnapshot, TaskStatus
from lunagentic_research_swarm.runtime.events import (
    AllInflightSettled,
    FinalEpochCommitted,
    FinalReportCompleted,
    FinalReportFailed,
    FormalizationFailed,
    FormalizationSucceeded,
    PauseRequested,
    ReportCompleted,
    StopRequested,
    ContinueRequested,
    PauseExpired,
    event_from_json,
    event_to_json,
    AgentCallRequested,
    OutboxDelivered,
    GraceExpired,
    ReportDeadlineReached,
)
from lunagentic_research_swarm.runtime.reducer import RuntimeState, reduce_event


def state_factory(status: str, *, generation: int = 0) -> TaskSnapshot:
    return TaskSnapshot(
        task_id="task-1",
        status=TaskStatus(status),
        generation=generation,
        active_round_id="round-1",
    )


def event_factory(event_type: str, *, generation: int = 0):
    event_class = {
        "FormalizationSucceeded": FormalizationSucceeded,
        "FormalizationFailed": FormalizationFailed,
        "PauseRequested": PauseRequested,
        "AllInflightSettled": AllInflightSettled,
        "StopRequested": StopRequested,
        "ReportCompleted": ReportCompleted,
        "FinalReportCompleted": FinalReportCompleted,
        "FinalReportFailed": FinalReportFailed,
    }[event_type]
    return event_class(
        event_id=f"evt-{event_type}",
        task_id="task-1",
        round_id="round-1",
        generation=generation,
        occurred_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("status", "event_type", "expected"),
    [
        ("FORMALIZING", "FormalizationSucceeded", "RUNNING"),
        ("FORMALIZING", "FormalizationFailed", "FAILED"),
        ("RUNNING", "PauseRequested", "PAUSING"),
        ("PAUSING", "AllInflightSettled", "PAUSED"),
        ("RUNNING", "StopRequested", "STOPPED"),
        ("REPORTING", "ReportCompleted", "FINALIZING"),  # empty leaves → §13.4
        ("FINALIZING", "FinalReportCompleted", "COMPLETED"),
        ("FINALIZING", "FinalReportFailed", "COMPLETED_WITH_ERRORS"),
    ],
)
def test_lifecycle_transitions(status: str, event_type: str, expected: str) -> None:
    transition = reduce_event(state_factory(status), event_factory(event_type))
    assert transition.next_state.status.value == expected


def test_report_completed_with_active_leaves_returns_to_running() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.REPORTING,
        generation=0,
        active_round_id="round-1",
        report_epoch=1,
        active_leaves={"still-running": 2.0},
    )
    transition = reduce_event(state, event_factory("ReportCompleted"))
    assert transition.next_state.status is TaskStatus.RUNNING


def test_late_generation_is_ignored_without_commands_or_effects() -> None:
    transition = reduce_event(
        state_factory("RUNNING", generation=2),
        event_factory("StopRequested", generation=1),
    )

    assert transition.ignored
    assert transition.reason == "late_generation"
    assert transition.commands == ()
    assert transition.effects == ()


@pytest.mark.parametrize(
    ("event", "reason"),
    [
        (ReportDeadlineReached("old-deadline", "task-1", "round-1", 0, epoch=2), "stale_report_epoch"),
        (ReportDeadlineReached("future-deadline", "task-1", "round-1", 0, epoch=10), "future_report_epoch"),
        (GraceExpired("old-grace", "task-1", "round-1", 0, epoch=7), "stale_report_epoch"),
    ],
)
def test_stale_or_future_report_epoch_event_cannot_advance_current_epoch(event, reason: str) -> None:
    status = TaskStatus.REPORTING if isinstance(event, GraceExpired) else TaskStatus.RUNNING
    state = RuntimeState("task-1", status, generation=0, active_round_id="round-1", report_epoch=8)

    transition = reduce_event(state, event)

    assert transition.ignored
    assert transition.reason == reason
    assert transition.next_state == state
    assert transition.commands == ()
    assert transition.effects == ()


def test_invalid_state_is_explicit_error_effect() -> None:
    transition = reduce_event(state_factory("PAUSED"), event_factory("PauseRequested"))

    assert transition.error is not None
    assert transition.error.code == "invalid_state"
    assert transition.next_state == state_factory("PAUSED")
    assert transition.commands == ()


def test_pause_expiry_only_expires_and_releases_raw_context() -> None:
    state = RuntimeState("task-1", TaskStatus.PAUSED, generation=0, active_round_id="round-1")
    event = PauseExpired("evt-expire", "task-1", "round-1", 0)

    transition = reduce_event(state, event)

    assert transition.next_state.status is TaskStatus.EXPIRED
    assert [type(effect).__name__ for effect in transition.effects] == ["ReleaseRawContext"]
    assert not any(type(effect).__name__ in {"PerformTaskSummary", "NotifyToolWaiter"} for effect in transition.effects)


def test_terminal_continue_uses_event_supplied_round_identity() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.COMPLETED,
        generation=2,
        active_round_id="round-1",
        credit_pool=2.0,
    )
    event = ContinueRequested(
        "evt-continue",
        "task-1",
        "round-1",
        2,
        next_round_id="round-2",
        next_generation=3,
        round_number=2,
    )

    transition = reduce_event(state, event)

    assert transition.next_state.active_round_id == "round-2"
    assert transition.next_state.generation == 3
    assert transition.next_state.status is TaskStatus.RUNNING
    assert transition.effects[0].round_id == "round-2"


def test_extended_events_round_trip_without_reducer_clock() -> None:
    event = ContinueRequested(
        "evt-continue",
        "task-1",
        "round-1",
        1,
        adjustment=-1.5,
        active_leaves={"a": 2.0},
        next_round_id="round-2",
    )

    decoded = event_from_json(event_to_json(event))

    assert decoded == event
    assert decoded.active_leaves == {"a": 2.0}


def test_terminal_continue_never_reopens_old_round() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.COMPLETED,
        generation=2,
        active_round_id="round-old",
        credit_pool=2.0,
    )
    event = ContinueRequested(
        "evt-new-round",
        "task-1",
        "round-old",
        2,
        next_round_id="round-new",
        next_generation=3,
        round_number=2,
    )

    transition = reduce_event(state, event)

    updates = [command for command in transition.commands if command.kind == "update_round_status"]
    assert updates == []
    assert all(
        command.values.get("round_id") != "round-old"
        for command in transition.commands
        if command.kind in {"update_round_status", "insert_lifecycle_event"}
    )
    lifecycle = [command for command in transition.commands if command.kind == "insert_lifecycle_event"]
    assert lifecycle and lifecycle[0].values["round_id"] == "round-new"


def test_continue_empty_event_leaves_override_state_leaves() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.COMPLETED,
        generation=2,
        active_round_id="round-old",
        credit_pool=2.0,
        active_leaves={"still-running": 1.0},
    )
    event = ContinueRequested(
        "evt-empty-leaves",
        "task-1",
        "round-old",
        2,
        active_leaves={},
        next_round_id="round-new",
        next_generation=3,
        round_number=2,
    )

    transition = reduce_event(state, event)

    assert transition.next_state.active_round_id == "round-new"
    assert transition.next_state.status is TaskStatus.RUNNING
    assert any(command.kind == "insert_round" for command in transition.commands)


def test_pausing_rejects_new_agent_call_without_effect() -> None:
    event = AgentCallRequested(
        "evt-agent",
        "task-1",
        "round-1",
        0,
        branch_id="branch-1",
        call_id="call-1",
        agent_id="agent-1",
    )

    transition = reduce_event(RuntimeState("task-1", TaskStatus.PAUSING, active_round_id="round-1"), event)

    assert transition.error is not None
    assert transition.error.code == "invalid_state"
    assert not any(type(effect).__name__ == "PerformAgentCall" for effect in transition.effects)


def test_outbox_delivery_is_invalid_before_completed_terminal_status() -> None:
    event = OutboxDelivered("evt-outbox", "task-1", "round-1", 0, outbox_id="out-1")

    transition = reduce_event(RuntimeState("task-1", TaskStatus.RUNNING, active_round_id="round-1"), event)

    assert transition.error is not None
    assert transition.error.code == "invalid_state"
    assert any(type(effect).__name__ == "NotifyToolWaiter" for effect in transition.effects)


def test_last_branch_finalized_enters_finalizing() -> None:
    from lunagentic_research_swarm.runtime.events import BranchFinalized

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"branch-1": 3.0},
        credit_pool=1.0,
    )
    event = BranchFinalized(
        "evt-final",
        "task-1",
        "round-1",
        0,
        branch_id="branch-1",
        summary_id="sum-1",
        reason="no_further_work",
    )

    transition = reduce_event(state, event)

    assert transition.next_state.status is TaskStatus.FINALIZING
    assert transition.next_state.active_leaves == {}
    assert transition.next_state.credit_pool == pytest.approx(4.0)
    assert any(command.kind == "settle_branch" for command in transition.commands)
    assert any(command.kind == "update_round_status" for command in transition.commands)


def test_branch_finalized_during_reporting_keeps_reporting_when_leaves_empty() -> None:
    from lunagentic_research_swarm.runtime.events import BranchFinalized

    state = RuntimeState(
        "task-1",
        TaskStatus.REPORTING,
        generation=0,
        active_round_id="round-1",
        report_epoch=1,
        active_leaves={"branch-1": 1.0},
    )
    event = BranchFinalized(
        "evt-final",
        "task-1",
        "round-1",
        0,
        branch_id="branch-1",
        summary_id="sum-1",
    )

    transition = reduce_event(state, event)

    assert transition.next_state.status is TaskStatus.REPORTING
    assert transition.next_state.active_leaves == {}


def test_report_completed_with_empty_leaves_enters_finalizing() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.REPORTING,
        generation=0,
        active_round_id="round-1",
        report_epoch=1,
        active_leaves={},
    )
    event = ReportCompleted("evt-report", "task-1", "round-1", 0, report_id="rpt-1")

    transition = reduce_event(state, event)

    assert transition.next_state.status is TaskStatus.FINALIZING


def test_report_deadline_with_empty_leaves_enters_finalizing() -> None:
    """Wall-clock deadline with no live leaves opens a FINAL epoch (not INTERMEDIATE)."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={},
    )
    event = ReportDeadlineReached(
        "evt-deadline",
        "task-1",
        "round-1",
        0,
        epoch=1,
    )

    transition = reduce_event(state, event)

    assert transition.next_state.status is TaskStatus.FINALIZING
    assert transition.next_state.report_epoch == 1
    assert [type(effect).__name__ for effect in transition.effects] == ["OpenReportEpoch"]
    assert transition.effects[0].payload["kind"] == "FINAL"


def test_final_report_completed_requires_finalizing() -> None:
    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.REPORTING, generation=0, active_round_id="round-1", report_epoch=1),
        FinalReportCompleted("evt-final", "task-1", "round-1", 0, report_id="rpt-1"),
    )
    assert transition.error is not None
    assert transition.error.code == "invalid_state"


def test_final_epoch_committed_same_epoch_from_reporting() -> None:
    """Post-synthesis same-epoch FINAL freeze reuses FINALIZING entry (no OpenReportEpoch)."""

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.REPORTING, generation=0, active_round_id="round-1", report_epoch=1),
        FinalEpochCommitted("evt-commit", "task-1", "round-1", 0, epoch=1),
    )
    assert transition.next_state.status is TaskStatus.FINALIZING
    assert transition.next_state.report_epoch == 1
    assert transition.effects == ()


def test_stop_accepted_from_finalizing() -> None:
    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.FINALIZING, generation=0, active_round_id="round-1"),
        StopRequested("evt-stop", "task-1", "round-1", 0),
    )
    assert transition.next_state.status is TaskStatus.STOPPED
    assert transition.next_state.generation == 1


def test_final_epoch_committed_bumps_epoch_while_already_finalizing() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.FINALIZING,
        generation=0,
        active_round_id="round-1",
        report_epoch=0,
        active_leaves={},
    )
    event = FinalEpochCommitted(
        "evt-commit",
        "task-1",
        "round-1",
        0,
        epoch=1,
    )

    transition = reduce_event(state, event)

    assert transition.next_state.status is TaskStatus.FINALIZING
    assert transition.next_state.report_epoch == 1
    assert transition.effects == ()
