from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import BranchLifecycle, BranchRuntime, FormalizedTask, ReportKind
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator


class FakeStore:
    def __init__(self) -> None:
        self.commands = []

    async def transact(self, commands) -> None:
        self.commands.extend(commands)


class FakeSummarizer:
    def __init__(self) -> None:
        self.task_failure: SimpleNamespace | None = None
        self.branch_failure: SimpleNamespace | None = None
        self.gate = asyncio.Event()
        self.gate.set()
        self.task_requests = []
        self.branch_requests = []

    def block(self) -> None:
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()

    async def finalize_branch(self, request):
        self.branch_requests.append(request)
        await self.gate.wait()
        if self.branch_failure is not None:
            return SummaryResult(False, "", "fake", None, self.branch_failure)
        return SummaryResult(True, f"branch:{request.branch_history[-1]['content']}", "fake", None, None)

    async def finalize_task(self, request):
        self.task_requests.append(request)
        await self.gate.wait()
        if self.task_failure is not None:
            return SummaryResult(False, "", "fake", None, self.task_failure)
        return SummaryResult(True, "task synthesis", "fake", None, None)


@dataclass
class ReportHarness:
    store: FakeStore
    summarizer: FakeSummarizer
    coordinator: ReportCoordinator
    launched: list[str]

    def branch(self, branch_id: str) -> BranchRuntime:
        return self.coordinator.branches[branch_id]


@pytest.fixture
def report_harness() -> ReportHarness:
    task = FormalizedTask.create("任务文本必须逐字保持\r\n  空格")
    branch = BranchRuntime(
        branch_id="A", task=task, catalog_fingerprint="catalog", generation=0,
        messages=[{"role": "assistant", "content": "A stable"}], credits=10.0, depth=0,
    )
    launched: list[str] = []
    coordinator = ReportCoordinator(
        task_id="task", round_id="round", formalized_task=task, branches={"A": branch},
        store=FakeStore(), summarizer=FakeSummarizer(), clock=lambda: 10.0,
        launch_delegation=lambda _parent, child: launched.append(child["branch_id"]),
        time_budget_seconds=120, grace_period_seconds=60,
    )
    return ReportHarness(coordinator.store, coordinator.summarizer, coordinator, launched)


@pytest.mark.asyncio
async def test_manual_checkpoint_holds_children_until_report(report_harness: ReportHarness) -> None:
    branch = report_harness.branch("A")
    # Keep another active leaf so this is a manual due epoch rather than the
    # separate all-active-checkpointed early-report rule.
    report_harness.coordinator.branches["C"] = BranchRuntime(
        branch_id="C", task=branch.task, catalog_fingerprint="catalog", generation=0,
        messages=[{"role": "assistant", "content": "C stable"}], credits=1.0, depth=1,
    )
    await report_harness.coordinator.on_branch_safe_point(
        "A", checkpoint=True, delegations=({"branch_id": "B", "task": "child"},)
    )

    assert branch.lifecycle is BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT
    assert report_harness.launched == []

    report_harness.summarizer.block()
    epoch = await report_harness.coordinator.open_epoch()
    assert epoch.kind is ReportKind.INTERMEDIATE
    assert report_harness.launched == ["B"]
    report_harness.summarizer.release()
    await report_harness.coordinator.wait_for_synthesis()


@pytest.mark.asyncio
async def test_all_active_branches_checkpointed_reports_early(report_harness: ReportHarness) -> None:
    epoch = await report_harness.coordinator.on_branch_safe_point("A", checkpoint=True)

    assert epoch is not None
    assert epoch.kind is ReportKind.INTERMEDIATE
    assert epoch.frozen_at < report_harness.coordinator.deadline_at


@pytest.mark.asyncio
async def test_intermediate_does_not_upgrade_if_branches_finish_during_synthesis(report_harness: ReportHarness) -> None:
    await report_harness.coordinator.on_branch_safe_point("A", checkpoint=True)
    epoch = await report_harness.coordinator.open_epoch()
    await report_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await report_harness.coordinator.wait_for_synthesis()

    assert [record.kind for record in report_harness.coordinator.reports] == [
        ReportKind.INTERMEDIATE,
        ReportKind.FINAL,
    ]
    assert report_harness.coordinator.epochs[-1].kind is ReportKind.FINAL


@pytest.mark.asyncio
async def test_terminal_last_branch_opens_and_delivers_final_epoch_without_prior_deadline(
    report_harness: ReportHarness,
) -> None:
    epoch = await report_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await report_harness.coordinator.wait_for_synthesis()

    assert epoch is not None
    assert epoch.kind is ReportKind.FINAL
    assert report_harness.coordinator.reports[-1].kind is ReportKind.FINAL


@pytest.mark.asyncio
async def test_synthesis_uses_coverage_snapshot_taken_before_later_terminal_summary(
    report_harness: ReportHarness,
) -> None:
    epoch = await report_harness.coordinator.open_epoch()

    await report_harness.coordinator.on_branch_safe_point("A", checkpoint=True)
    checkpoint_id = epoch.frontier["A"].checkpoint_summary_id
    # The synthesis task has been scheduled but has not yet run.  A terminal
    # summary arriving in this gap must belong to the next epoch, not rewrite
    # this intermediate report's frozen input set.
    await report_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await report_harness.coordinator.wait_for_synthesis()

    report = report_harness.coordinator.reports[0]
    assert report.kind is ReportKind.INTERMEDIATE
    assert [item.summary_id for item in report.coverage.items] == [checkpoint_id]
    assert report_harness.coordinator.reports[-1].kind is ReportKind.FINAL


@pytest.mark.asyncio
async def test_final_synthesis_failure_record_preserves_safe_error_fields(report_harness: ReportHarness) -> None:
    branch = report_harness.branch("A")
    report_harness.coordinator.branches["B"] = BranchRuntime(
        branch_id="B", task=branch.task, catalog_fingerprint="catalog", generation=0,
        messages=[{"role": "assistant", "content": "B stable"}], credits=1.0, depth=1,
    )
    report_harness.summarizer.task_failure = SimpleNamespace(
        code="provider_unavailable", message="summary provider unavailable"
    )

    await report_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await report_harness.coordinator.on_branch_safe_point("B", terminal=True)
    await report_harness.coordinator.wait_for_synthesis()

    record = report_harness.coordinator.reports[-1]
    assert record.kind is ReportKind.FINAL
    assert record.status == "FAILED"
    assert record.error_code == "provider_unavailable"
    assert record.error_message == "summary provider unavailable"
    assert "任务文本必须逐字保持" not in record.error_message
    persisted = [dict(command.values) for command in report_harness.store.commands if command.kind == "insert_report"]
    assert persisted[-1]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_failed_branch_summary_is_durable_and_final_coverage_degrades(report_harness: ReportHarness) -> None:
    report_harness.summarizer.branch_failure = SimpleNamespace(
        code="provider_unavailable", message="branch summary unavailable"
    )

    await report_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await report_harness.coordinator.wait_for_synthesis()

    summaries = [dict(command.values) for command in report_harness.store.commands if command.kind == "insert_summary"]
    assert summaries[-1]["status"] == "FAILED"
    assert summaries[-1]["error_code"] == "provider_unavailable"
    assert report_harness.coordinator.reports[-1].status == "DEGRADED"
