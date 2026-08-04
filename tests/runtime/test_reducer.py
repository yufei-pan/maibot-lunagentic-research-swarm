from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.models import TaskSnapshot, TaskStatus
from lunagentic_research_swarm.runtime.events import (
    AllInflightSettled,
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
        ("REPORTING", "ReportCompleted", "RUNNING"),
        ("FINALIZING", "FinalReportCompleted", "COMPLETED"),
        ("FINALIZING", "FinalReportFailed", "COMPLETED_WITH_ERRORS"),
    ],
)
def test_lifecycle_transitions(status: str, event_type: str, expected: str) -> None:
    transition = reduce_event(state_factory(status), event_factory(event_type))
    assert transition.next_state.status.value == expected


def test_late_generation_is_ignored_without_commands_or_effects() -> None:
    transition = reduce_event(
        state_factory("RUNNING", generation=2),
        event_factory("StopRequested", generation=1),
    )

    assert transition.ignored
    assert transition.reason == "late_generation"
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
