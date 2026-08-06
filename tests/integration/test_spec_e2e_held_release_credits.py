"""E2E held-release credit conservation via real ``_release_held_delegations`` (§11 / §13.1).

Unlike ``test_spec_e2e_control_and_grace`` (launch stub that bypasses credit debit),
these cases keep the manager-wired ``release_held_delegations`` callback and drain
``materialize_child`` effects through ``ResearchManager.materialize_child_effect``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from lunagentic_research_swarm.models import BranchLifecycle
from lunagentic_research_swarm.runtime.reducer import NotifyToolWaiter


def _materialize_effects(enqueued: list[Any]) -> list[NotifyToolWaiter]:
    return [
        item
        for item in enqueued
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]


def _ledger_total(controller: Any) -> float:
    leaves = dict(controller.state.active_leaves)
    return math.fsum(leaves.values()) + float(controller.state.credit_pool)


async def _drain_materialize(harness, *, expect: int | None = None) -> list[NotifyToolWaiter]:
    effects = _materialize_effects(harness.scheduler.enqueued)
    if expect is not None:
        assert len(effects) == expect, (
            f"expected {expect} materialize_child effects, got {len(effects)}; "
            f"actions={[getattr(i, 'payload', {}).get('action') for i in harness.scheduler.enqueued]}"
        )
    for effect in effects:
        await harness.manager.materialize_child_effect(effect)
    return effects


@pytest.mark.asyncio
async def test_e2e_held_release_open_epoch_conserves_credits_exact_split(
    runtime_harness,
) -> None:
    """§11.4 / §11.7 / §13.1 — manual checkpoint hold → open_epoch release → no invent.

    Root alone checkpoints with two held children (60+40). All-active-checkpointed
    opens an epoch, driving real ``_release_held_delegations`` (not a launch stub).
    """

    harness = runtime_harness
    await harness.start("held-release exact split", credits=100.0, time_budget=120)
    await harness.formalize("正式 held-release exact split")

    harness.wire_live_agents("agent.child-a", "agent.child-b")
    assert harness.coordinator is not None
    assert harness.coordinator.release_held_delegations is not None
    # Guard: grace E2E stubs must not leak into this pin.
    assert harness.coordinator.launch_delegation is not None

    status = await harness.manager.status(harness.task_id)
    assert len(status["active_leaves"]) == 1
    root_id = str(status["active_leaves"][0]["branch_id"])
    controller = harness.manager._controllers[harness.task_id]
    initial_total = _ledger_total(controller)
    assert initial_total == pytest.approx(100.0)

    harness.scheduler.enqueued.clear()
    await harness.coordinator.on_branch_safe_point(
        root_id,
        checkpoint=True,
        delegations=(
            {
                "branch_id": f"{root_id}:1",
                "agent_id": "agent.child-a",
                "task": "held-a",
                "credits": 60.0,
            },
            {
                "branch_id": f"{root_id}:2",
                "agent_id": "agent.child-b",
                "task": "held-b",
                "credits": 40.0,
            },
        ),
    )
    # Single active leaf checkpointed → open_epoch → real release before return.
    assert harness.coordinator._held == {}
    assert harness.coordinator.current_epoch is not None

    materialize = await _drain_materialize(harness, expect=2)
    assert materialize[0].payload["retire_parent"] is True
    assert materialize[0].payload["pool_return"] == pytest.approx(0.0)
    assert materialize[0].payload["credits"] == pytest.approx(60.0)
    assert materialize[1].payload["retire_parent"] is False
    assert materialize[1].payload["credits"] == pytest.approx(40.0)
    assert all(item.payload.get("parent_credits_after") is None for item in materialize)

    leaves = dict(controller.state.active_leaves)
    assert root_id not in leaves
    assert leaves[f"{root_id}:1"] == pytest.approx(60.0)
    assert leaves[f"{root_id}:2"] == pytest.approx(40.0)
    assert controller.state.credit_pool == pytest.approx(0.0)
    assert _ledger_total(controller) == pytest.approx(initial_total)
    # Coordinator graph receives real materialize inserts (not stub BranchRuntime).
    assert f"{root_id}:1" in harness.coordinator.branches
    assert f"{root_id}:2" in harness.coordinator.branches
    assert harness.coordinator.branches[root_id].lifecycle is BranchLifecycle.FINALIZED


@pytest.mark.asyncio
async def test_e2e_held_release_open_epoch_returns_unallocated_remainder_to_pool(
    runtime_harness,
) -> None:
    """§11.7 / §13.1 — held child requests less than parent; remainder → dormant pool."""

    harness = runtime_harness
    await harness.start("held-release pool remainder", credits=100.0, time_budget=120)
    await harness.formalize("正式 held-release pool remainder")

    harness.wire_live_agents("agent.child-partial")
    assert harness.coordinator.release_held_delegations is not None

    status = await harness.manager.status(harness.task_id)
    root_id = str(status["active_leaves"][0]["branch_id"])
    controller = harness.manager._controllers[harness.task_id]
    initial_total = _ledger_total(controller)

    harness.scheduler.enqueued.clear()
    await harness.coordinator.on_branch_safe_point(
        root_id,
        checkpoint=True,
        delegations=(
            {
                "branch_id": f"{root_id}:1",
                "agent_id": "agent.child-partial",
                "task": "partial",
                "credits": 30.0,
            },
        ),
    )

    materialize = await _drain_materialize(harness, expect=1)
    assert materialize[0].payload["retire_parent"] is True
    assert materialize[0].payload["pool_return"] == pytest.approx(70.0)
    assert materialize[0].payload["credits"] == pytest.approx(30.0)

    leaves = dict(controller.state.active_leaves)
    assert root_id not in leaves
    assert leaves[f"{root_id}:1"] == pytest.approx(30.0)
    assert controller.state.credit_pool == pytest.approx(70.0)
    assert _ledger_total(controller) == pytest.approx(initial_total)


@pytest.mark.asyncio
async def test_e2e_held_release_in_grace_conserves_parent_leaf_credits(
    runtime_harness,
) -> None:
    """§11.4 / §13.1 — in-grace checkpoint release debits the holding parent only.

    Real root→A/B allocate first; open frontier; A checkpoints with a held child
    during grace → ``_release_held_branch`` → real ``_release_held_delegations``.
    Sibling B credits must stay put; no invent across A-child + pool + B.
    """

    harness = runtime_harness
    await harness.start("held-release in-grace", credits=100.0, time_budget=120)
    await harness.formalize("正式 held-release in-grace")

    leaves_before = await harness.root_allocates_real({"A": 50.0, "B": 50.0})
    assert leaves_before["A"] == pytest.approx(50.0)
    assert leaves_before["B"] == pytest.approx(50.0)

    # Grandchild agent must be in catalog.entries for held-release planning.
    harness.wire_live_agents("agent.A-child")
    assert harness.coordinator.release_held_delegations is not None

    controller = harness.manager._controllers[harness.task_id]
    initial_total = _ledger_total(controller)
    assert initial_total == pytest.approx(100.0)

    epoch = await harness.coordinator.open_epoch()
    assert list(epoch.frontier) == ["A", "B"]
    assert "A" in epoch.frontier

    harness.scheduler.enqueued.clear()
    await harness.coordinator.on_branch_safe_point(
        "A",
        checkpoint=True,
        delegations=(
            {
                "branch_id": "A:1",
                "agent_id": "agent.A-child",
                "task": "grace-held",
                "credits": 30.0,
            },
        ),
    )
    # In-frontier checkpoint releases immediately after branch summary.
    assert "A" not in harness.coordinator._held

    materialize = await _drain_materialize(harness, expect=1)
    assert materialize[0].payload["parent_branch_id"] == "A"
    assert materialize[0].payload["credits"] == pytest.approx(30.0)
    assert materialize[0].payload["retire_parent"] is True
    assert materialize[0].payload["pool_return"] == pytest.approx(20.0)

    leaves = dict(controller.state.active_leaves)
    assert "A" not in leaves
    assert leaves["A:1"] == pytest.approx(30.0)
    assert leaves["B"] == pytest.approx(50.0)
    assert controller.state.credit_pool == pytest.approx(20.0)
    assert _ledger_total(controller) == pytest.approx(initial_total)
    assert math.fsum(leaves.values()) + controller.state.credit_pool == pytest.approx(100.0)
