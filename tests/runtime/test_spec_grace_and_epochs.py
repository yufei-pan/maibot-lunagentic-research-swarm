"""Design §13.1 / §7.5 / §13.2 — grace manual checkpoint, deadline re-arm, multi-epoch coverage."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import BranchLifecycle, BranchRuntime, FormalizedTask, ReportKind, SummaryKind
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator
from lunagentic_research_swarm.runtime.events import GraceExpired, ReportDeadlineReached
from lunagentic_research_swarm.runtime.reducer import ArmDeadline

from .test_controller_controls import _running_manager
from .test_controller_start import harness  # noqa: F401


class FakeStore:
    def __init__(self) -> None:
        self.commands = []

    async def transact(self, commands) -> None:
        self.commands.extend(commands)


class FakeSummarizer:
    def __init__(self) -> None:
        self.branch_requests = []
        self.task_requests = []

    async def finalize_branch(self, request):
        self.branch_requests.append(request)
        return SummaryResult(True, f"branch:{request.branch_history[-1]['content']}", "fake", None, None)

    async def finalize_task(self, request):
        self.task_requests.append(request)
        return SummaryResult(True, "task synthesis", "fake", None, None)


@dataclass
class EpochHarness:
    store: FakeStore
    summarizer: FakeSummarizer
    coordinator: ReportCoordinator
    launched: list[dict[str, object]]
    clock_value: list[float]

    def advance(self, seconds: float) -> None:
        self.clock_value[0] += float(seconds)


@pytest.fixture
def epoch_harness() -> EpochHarness:
    task = FormalizedTask.create("正式任务：grace/epoch 规格")
    clock_value = [10.0]
    launched: list[dict[str, object]] = []

    async def launch(_parent: str, child: dict[str, object]) -> None:
        launched.append(dict(child))

    branch_a = BranchRuntime(
        branch_id="A",
        task=task,
        catalog_fingerprint="catalog",
        generation=0,
        messages=[{"role": "assistant", "content": "A evidence"}],
        credits=10.0,
        depth=0,
    )
    branch_b = BranchRuntime(
        branch_id="B",
        task=task,
        catalog_fingerprint="catalog",
        generation=0,
        messages=[{"role": "assistant", "content": "B evidence"}],
        credits=5.0,
        depth=1,
    )
    coordinator = ReportCoordinator(
        task_id="task",
        round_id="round",
        formalized_task=task,
        branches={"A": branch_a, "B": branch_b},
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        clock=lambda: clock_value[0],
        launch_delegation=launch,
        time_budget_seconds=120,
        grace_period_seconds=60,
    )
    return EpochHarness(coordinator.store, coordinator.summarizer, coordinator, launched, clock_value)


@pytest.mark.asyncio
async def test_spec_13_1_manual_checkpoint_during_grace_covers_current_frontier(
    epoch_harness: EpochHarness,
) -> None:
    """§13.1 — manual checkpoint inside grace belongs to the open epoch (not a later hold)."""

    epoch = await epoch_harness.coordinator.open_epoch()
    await epoch_harness.coordinator.on_branch_safe_point(
        "A",
        checkpoint=True,
        delegations=({"branch_id": "A-child", "task": "held-until-release"},),
    )

    assert epoch.frontier["A"].checkpoint_requested is True
    assert epoch.frontier["A"].checkpoint_summary_id is not None
    assert epoch.frontier["A"].ready is True
    # Sibling still in-flight: synthesis must not start until frontier is covered.
    assert epoch.synthesis_started is False

    await epoch_harness.coordinator.on_branch_safe_point("B", checkpoint=True)
    await epoch_harness.coordinator.wait_for_synthesis()

    assert epoch_harness.coordinator.reports[0].kind is ReportKind.INTERMEDIATE
    assert epoch_harness.coordinator.reports[0].coverage.has_checkpoint is True


@pytest.mark.asyncio
async def test_spec_13_1_manual_checkpoint_during_grace_continues_after_summary(
    epoch_harness: EpochHarness,
) -> None:
    """§13.1 — grace-window manual checkpoint continues after summary; not stuck waiting."""

    await epoch_harness.coordinator.open_epoch()
    await epoch_harness.coordinator.on_branch_safe_point(
        "A",
        checkpoint=True,
        delegations=({"branch_id": "A-child", "task": "next"},),
    )
    # Keep B covered so task synthesis can finish; continue must not depend on next epoch.
    await epoch_harness.coordinator.on_branch_safe_point("B", checkpoint=True)
    await epoch_harness.coordinator.wait_for_synthesis()

    branch = epoch_harness.coordinator.branches["A"]
    assert branch.lifecycle is not BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT
    assert branch.lifecycle is BranchLifecycle.READY
    assert [item["branch_id"] for item in epoch_harness.launched] == ["A-child"]
    assert epoch_harness.coordinator._held == {}


@pytest.mark.asyncio
async def test_spec_13_1_intermediate_synthesis_rearms_coordinator_deadline(
    epoch_harness: EpochHarness,
) -> None:
    """§13.1 — each intermediate report resets the wall-clock deadline on the coordinator."""

    prior_deadline = epoch_harness.coordinator.deadline_at
    await epoch_harness.coordinator.on_branch_safe_point("A", checkpoint=True)
    # Move the clock before the all-checkpointed early epoch opens synthesis.
    epoch_harness.advance(50.0)
    await epoch_harness.coordinator.on_branch_safe_point("B", checkpoint=True)
    await epoch_harness.coordinator.wait_for_synthesis()

    record = epoch_harness.coordinator.reports[0]
    assert record.kind is ReportKind.INTERMEDIATE
    assert epoch_harness.coordinator.deadline_at == pytest.approx(record.created_at + 120)
    assert epoch_harness.coordinator.deadline_at > prior_deadline


@pytest.mark.asyncio
async def test_spec_13_1_intermediate_return_to_running_rearms_manager_deadline_timer(harness) -> None:
    """§13.1 — manager re-arms `_deadline_jobs` after intermediate ReportCompleted → RUNNING."""

    manager, _, _, _, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    coordinator = manager.report_coordinators[task_id]
    root_id = before["active_leaves"][0]["branch_id"]
    root = coordinator.branches[root_id]
    # Keep a live sibling so the epoch stays INTERMEDIATE (not FINAL).
    coordinator.branches["B"] = BranchRuntime(
        branch_id="B",
        task=root.task,
        catalog_fingerprint=root.catalog_fingerprint,
        generation=0,
        messages=[{"role": "assistant", "content": "B still running"}],
        credits=1.0,
        depth=1,
    )
    controller = manager._controllers[task_id]
    controller.state = replace(
        controller.state,
        active_leaves={**dict(controller.state.active_leaves), "B": 1.0},
    )

    old_job = manager._deadline_jobs.get(task_id)
    prior_deadline = float(coordinator.deadline_at)

    await manager.handle_runtime_event(
        ReportDeadlineReached("deadline", task_id, before["round_id"], before["generation"], epoch=1)
    )
    await manager.handle_runtime_event(
        GraceExpired("grace", task_id, before["round_id"], before["generation"], epoch=1)
    )
    await coordinator.wait_for_synthesis()

    status = await manager.status(task_id)
    assert status["status"] == "RUNNING"
    assert coordinator.reports[-1].kind is ReportKind.INTERMEDIATE
    assert coordinator.deadline_at >= prior_deadline
    new_job = manager._deadline_jobs.get(task_id)
    assert new_job is not None and not new_job.done()
    assert new_job is not old_job


@pytest.mark.asyncio
async def test_spec_7_5_continue_restart_arms_deadline_effect(harness) -> None:
    """§7.5 / §13.1 — continue into a new round arms ArmDeadline (and manager timer)."""

    manager, _, _, scheduler, task_id = await _running_manager(harness)
    await manager.stop(task_id)
    # Clear formalization-time deadline arming noise.
    before_effects = len(scheduler.enqueued)
    prior_job = manager._deadline_jobs.get(task_id)

    continued = await manager.continue_task(task_id, credit_adjustment=5.0, time_budget_seconds=90)

    assert continued["status"] == "RUNNING"
    new_effects = scheduler.enqueued[before_effects:]
    assert any(isinstance(effect, ArmDeadline) for effect in new_effects)
    deadline = next(effect for effect in new_effects if isinstance(effect, ArmDeadline))
    assert deadline.payload.get("kind") == "report_deadline"
    assert float(deadline.payload["due_at"]) > 0
    job = manager._deadline_jobs.get(task_id)
    assert job is not None and not job.done()
    assert job is not prior_job


@pytest.mark.asyncio
async def test_spec_7_5_continue_with_live_leaves_rearms_deadline(harness) -> None:
    """§7.5 — continue from PAUSED with active leaves resets the report clock."""

    manager, _, _, scheduler, task_id = await _running_manager(harness)
    await manager.pause(task_id)
    status = await manager.status(task_id)
    assert status["status"] == "PAUSED"
    assert status["active_leaves"]

    before_effects = len(scheduler.enqueued)
    prior_job = manager._deadline_jobs.get(task_id)

    continued = await manager.continue_task(task_id, credit_adjustment=5.0, time_budget_seconds=90)

    assert continued["status"] == "RUNNING"
    assert continued["active_leaves"]
    new_effects = scheduler.enqueued[before_effects:]
    assert any(isinstance(effect, ArmDeadline) for effect in new_effects)
    deadline = next(effect for effect in new_effects if isinstance(effect, ArmDeadline))
    assert deadline.payload.get("kind") == "report_deadline"
    assert float(deadline.payload["due_at"]) > 0
    job = manager._deadline_jobs.get(task_id)
    assert job is not None and not job.done()
    assert job is not prior_job


@pytest.mark.asyncio
async def test_spec_13_2_later_epoch_includes_early_terminal_and_stays_intermediate(
    epoch_harness: EpochHarness,
) -> None:
    """§13.2 — early-terminal A remains in later coverage with B checkpoint; kind stays INTERMEDIATE."""

    epoch1 = await epoch_harness.coordinator.open_epoch()
    await epoch_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await epoch_harness.coordinator.on_grace_expired(epoch1.epoch)
    await epoch_harness.coordinator.wait_for_synthesis()

    first = epoch_harness.coordinator.reports[0]
    assert first.kind is ReportKind.INTERMEDIATE
    assert {item.branch_id for item in first.coverage.items} == {"A", "B"}
    assert any(item.kind is SummaryKind.BRANCH_FINAL and item.branch_id == "A" for item in first.coverage.items)
    assert any(item.kind is SummaryKind.CHECKPOINT and item.branch_id == "B" for item in first.coverage.items)

    # Later epoch: only B remains active; A's terminal must still enter coverage.
    epoch_harness.advance(200.0)
    epoch_harness.coordinator.branches["B"].messages.append(
        {"role": "assistant", "content": "B later checkpoint evidence"}
    )
    epoch_harness.coordinator.branches["B"].latest_checkpoint_id = None
    epoch2 = await epoch_harness.coordinator.open_epoch()
    assert list(epoch2.frontier) == ["B"]
    assert "A" not in epoch2.frontier

    await epoch_harness.coordinator.on_branch_safe_point("B", checkpoint=True)
    await epoch_harness.coordinator.wait_for_synthesis()

    later = epoch_harness.coordinator.reports[-1]
    assert later.epoch == epoch2.epoch
    assert later.kind is ReportKind.INTERMEDIATE
    assert later.kind is not ReportKind.FINAL
    assert later.coverage.has_checkpoint is True
    by_branch = {item.branch_id: item for item in later.coverage.items}
    assert by_branch["A"].kind is SummaryKind.BRANCH_FINAL
    assert by_branch["A"].text == "branch:A evidence"
    assert by_branch["B"].kind is SummaryKind.CHECKPOINT
    assert by_branch["B"].text == "branch:B later checkpoint evidence"
