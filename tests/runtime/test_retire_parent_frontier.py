"""Regression: retire_parent must not wedge an open report frontier.

Production bug (lrs_e6a1b1c1…): a frontier parent was FINALIZED via retire_parent
without becoming frontier-ready, grace was never armed (epoch already open at
deadline), and REPORTING stuck with branch summaries but no task report.

The production path is ProcedureBatchCompleted → materialize_child (not the
coordinator in-frontier checkpoint path).
"""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.models import BranchLifecycle, TaskStatus
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted, ReportDeadlineReached
from lunagentic_research_swarm.runtime.reducer import NotifyToolWaiter


@pytest.mark.asyncio
async def test_retire_parent_marks_frontier_entry_ready_and_allows_synthesis(
    runtime_harness,
) -> None:
    """retire_parent while parent is in a frozen frontier must make that entry ready."""

    harness = runtime_harness
    await harness.start("retire-parent frontier", credits=100.0, time_budget=120)
    await harness.formalize("正式：frontier 内父节点退休不得卡住报告")

    leaves = await harness.root_allocates_real({"A": 50.0, "B": 50.0})
    assert set(leaves) == {"A", "B"}
    harness.wire_live_agents("agent.A-child")

    epoch = await harness.coordinator.open_epoch()
    assert set(epoch.frontier) == {"A", "B"}
    assert epoch.frontier["A"].ready is False
    assert epoch.synthesis_started is False

    controller = harness.manager._controllers[harness.task_id]
    status = await harness.manager.status(harness.task_id)
    parent_messages = tuple(
        dict(item)
        for item in (harness.manager._branches[harness.task_id].get("A") or {}).get("messages", ())
    ) or ({"role": "assistant", "content": "A pre-retire evidence"},)

    harness.scheduler.enqueued.clear()
    # Real turn completion with delegation — bypasses coordinator in-frontier
    # checkpoint; this is how retire_parent left frontier slots not-ready.
    await harness.manager.handle_runtime_event(
        ProcedureBatchCompleted(
            event_id=f"{harness.task_id}:A-delegate",
            task_id=harness.task_id,
            round_id=str(status["round_id"]),
            generation=int(status["generation"]),
            branch_id="A",
            call_id="A:delegate",
            result_id="A:delegate-result",
            credits_after=50.0,
            report="A hands off",
            delegations=(
                {
                    "branch_id": "A:1",
                    "agent_id": "agent.A-child",
                    "task": "continue-A",
                    "credits": 30.0,
                },
            ),
            parent_messages=parent_messages,
            parent_depth=1,
            live_agent_ids=("agent.A-child",),
            agent_calls_started=int(controller.state.agent_calls_started),
        )
    )
    materialize = [
        item
        for item in harness.scheduler.enqueued
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]
    assert len(materialize) == 1
    assert materialize[0].payload["retire_parent"] is True

    await harness.manager.materialize_child_effect(materialize[0])

    assert harness.coordinator.branches["A"].lifecycle is BranchLifecycle.FINALIZED
    assert epoch.frontier["A"].ready is True, (
        "retire_parent must mark the frozen frontier slot ready "
        f"(got checkpoint={epoch.frontier['A'].checkpoint_summary_id!r} "
        f"terminal={epoch.frontier['A'].terminal_summary_id!r} "
        f"failed={epoch.frontier['A'].failed})"
    )
    assert "A:1" not in epoch.frontier

    # Sibling finishes → frontier fully ready → synthesis must run (not wedge).
    await harness.coordinator.on_branch_safe_point("B", terminal=True)
    await harness.coordinator.wait_for_synthesis()
    assert epoch.synthesis_finished is True
    assert harness.coordinator.reports, "expected a report after frontier became ready"


@pytest.mark.asyncio
async def test_deadline_arms_grace_when_epoch_already_open(runtime_harness) -> None:
    """OpenReportEpoch for an already-open epoch must still arm grace."""

    harness = runtime_harness
    await harness.start("deadline grace rearm", credits=80.0, time_budget=30)
    await harness.formalize("正式：已有 epoch 时 deadline 仍须武装 grace")
    await harness.root_allocates_real({"A": 40.0, "B": 40.0})

    # Early epoch (as first-terminal / early open would create) — no grace job yet.
    epoch = await harness.coordinator.open_epoch(epoch=1)
    assert epoch.epoch == 1
    assert harness.manager._grace_jobs.get(harness.task_id) is None

    status = await harness.manager.status(harness.task_id)
    await harness.manager.handle_runtime_event(
        ReportDeadlineReached(
            f"{harness.task_id}:deadline",
            harness.task_id,
            str(status["round_id"]),
            int(status["generation"]),
            epoch=1,
        )
    )
    assert (await harness.manager.status(harness.task_id))["status"] == TaskStatus.REPORTING.value
    grace_job = harness.manager._grace_jobs.get(harness.task_id)
    assert grace_job is not None and not grace_job.done(), (
        "deadline must arm grace even when coordinator epoch 1 already exists"
    )
