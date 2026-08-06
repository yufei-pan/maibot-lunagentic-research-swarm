"""E2E live-tree credit reallocation via manager continue barrier (§11.6).

Distinct from ``test_spec_coverage_credits`` (pure ``redistribute_pool``) and E8
(synthetic all-zero ``root_delegates`` seed): these cases materialize live
children through ``root_allocates_real``, leave dormant pool remainder, then
drive ``ResearchManager.continue_task`` — the same control path
``continue_deep_research`` uses — so live balances update through the reducer
barrier (no invent).
"""

from __future__ import annotations

import math
from typing import Any

import pytest


def _ledger_total(controller: Any) -> float:
    leaves = dict(controller.state.active_leaves)
    return math.fsum(leaves.values()) + float(controller.state.credit_pool)


async def _durable_branch_balances(store: Any, round_id: str) -> dict[str, float]:
    def _query(connection: Any) -> dict[str, float]:
        rows = connection.execute(
            "SELECT branch_id, credit_balance FROM branches WHERE round_id = ?",
            (round_id,),
        ).fetchall()
        return {str(row["branch_id"]): float(row["credit_balance"]) for row in rows}

    return await store.run_locked(_query)


@pytest.mark.asyncio
async def test_e2e_live_tree_continue_redistributes_pool_and_adjustment_proportionally(
    runtime_harness,
) -> None:
    """§11.6 — live A/B + dormant pool; continue activates pool+adjustment by positive weight."""

    harness = runtime_harness
    await harness.start("live reallocation proportional", credits=100.0, time_budget=120)
    await harness.formalize("正式 live reallocation proportional")

    # Under-subscribe so unused parent credits settle into dormant pool on retire.
    leaves = await harness.root_allocates_real({"A": 60.0, "B": 20.0})
    assert set(leaves) == {"A", "B"}
    assert leaves["A"] == pytest.approx(60.0)
    assert leaves["B"] == pytest.approx(20.0)

    controller = harness.manager._controllers[harness.task_id]
    assert controller.state.credit_pool == pytest.approx(20.0)
    initial_total = _ledger_total(controller)
    assert initial_total == pytest.approx(100.0)
    # Root retired; only live children remain.
    assert harness._root_branch_id not in leaves
    assert set(controller.state.active_leaves) == {"A", "B"}

    await harness.manager.pause(harness.task_id)
    assert (await harness.manager.status(harness.task_id))["status"] == "PAUSED"

    adjustment = 10.0
    result = await harness.manager.continue_task(
        harness.task_id,
        credit_adjustment=adjustment,
        time_budget_seconds=90,
    )

    assert result["status"] == "RUNNING"
    assert result["effective_time_budget_seconds"] == 90
    by_id = {item["branch_id"]: float(item["credits"]) for item in result["active_leaves"]}
    # available = pool(20) + adj(10) = 30; weights 60:20 → +22.5 / +7.5
    assert set(by_id) == {"A", "B"}
    assert by_id["A"] == pytest.approx(82.5)
    assert by_id["B"] == pytest.approx(27.5)
    assert controller.state.credit_pool == pytest.approx(0.0)
    assert _ledger_total(controller) == pytest.approx(initial_total + adjustment)

    # Manager branch cache and durable round/branch rows track the barrier.
    mgr_branches = harness.manager._branches[harness.task_id]
    assert set(mgr_branches) == {"A", "B"}
    assert float(mgr_branches["A"]["credits"]) == pytest.approx(82.5)
    assert float(mgr_branches["B"]["credits"]) == pytest.approx(27.5)
    task = await harness.store.load_task(harness.task_id)
    assert task is not None and task.current_round is not None
    assert float(task.current_round.credit_pool) == pytest.approx(0.0)
    durable = await _durable_branch_balances(harness.store, str(task.current_round.round_id))
    assert durable["A"] == pytest.approx(82.5)
    assert durable["B"] == pytest.approx(27.5)


@pytest.mark.asyncio
async def test_e2e_live_tree_zero_leaf_gets_no_share_of_pool_redistribution(
    runtime_harness,
) -> None:
    """§11.6 — among live leaves, only positive balances weigh; zero leaf stays put."""

    harness = runtime_harness
    await harness.start("live reallocation zero weight", credits=100.0, time_budget=120)
    await harness.formalize("正式 live reallocation zero weight")

    leaves = await harness.root_allocates_real({"pos": 40.0, "zero": 0.0})
    assert leaves["pos"] == pytest.approx(40.0)
    assert leaves["zero"] == pytest.approx(0.0)

    controller = harness.manager._controllers[harness.task_id]
    assert controller.state.credit_pool == pytest.approx(60.0)
    before = _ledger_total(controller)
    assert before == pytest.approx(100.0)

    await harness.manager.pause(harness.task_id)
    result = await harness.manager.continue_task(harness.task_id, credit_adjustment=0.0)

    assert result["status"] == "RUNNING"
    by_id = {item["branch_id"]: float(item["credits"]) for item in result["active_leaves"]}
    # Only ``pos`` weighs → entire dormant pool (60) lands on it; zero stays 0.
    assert by_id["pos"] == pytest.approx(100.0)
    assert by_id["zero"] == pytest.approx(0.0)
    assert controller.state.credit_pool == pytest.approx(0.0)
    assert _ledger_total(controller) == pytest.approx(before)

    task = await harness.store.load_task(harness.task_id)
    assert task is not None and task.current_round is not None
    assert float(task.current_round.credit_pool) == pytest.approx(0.0)
    durable = await _durable_branch_balances(harness.store, str(task.current_round.round_id))
    assert durable["pos"] == pytest.approx(100.0)
    assert durable["zero"] == pytest.approx(0.0)
