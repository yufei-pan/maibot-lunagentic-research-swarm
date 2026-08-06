"""Module-level credit-conservation + ChildMaterialized debit pins (§11.4 / A3 credit hold)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.credits import allocate_children
from lunagentic_research_swarm.runtime.events import ChildMaterialized
from lunagentic_research_swarm.runtime.reducer import RuntimeState, reduce_event


NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_spec_11_4_child_materialize_debits_parent_to_parent_credits_after() -> None:
    """Materialization transfers credits; parent ends at parent_credits_after."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 10.0},
        credit_pool=0.0,
    )
    allocation = allocate_children(10.0, [("child", 4.0)])
    assert allocation.allocations["child"] == 4.0
    parent_after = 10.0 - 4.0

    transition = reduce_event(
        state,
        ChildMaterialized(
            "evt-1",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="child",
            parent_branch_id="parent",
            agent_id="agent.child",
            credits=4.0,
            depth=1,
            retire_parent=False,
            parent_credits_after=parent_after,
        ),
    )
    assert transition.next_state.active_leaves["parent"] == pytest.approx(parent_after)
    assert transition.next_state.active_leaves["child"] == pytest.approx(4.0)
    # Conservation across live leaves (no invent/destroy at materialize).
    assert sum(transition.next_state.active_leaves.values()) == pytest.approx(10.0)


def test_spec_11_4_retire_parent_settles_remainder_into_pool() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 10.0},
        credit_pool=1.0,
    )
    transition = reduce_event(
        state,
        ChildMaterialized(
            "evt-2",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="child",
            parent_branch_id="parent",
            agent_id="agent.child",
            credits=7.0,
            depth=1,
            retire_parent=True,
            pool_return=3.0,
        ),
    )
    assert "parent" not in transition.next_state.active_leaves
    assert transition.next_state.active_leaves["child"] == pytest.approx(7.0)
    # Remaining parent credits fold into pool when parent retires.
    assert transition.next_state.credit_pool == pytest.approx(1.0 + 3.0)
