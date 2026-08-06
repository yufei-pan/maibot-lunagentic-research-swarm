"""Design §13 report-epoch behaviors that need explicit pin tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask, ReportKind
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator


class FakeStore:
    def __init__(self) -> None:
        self.commands = []

    async def transact(self, commands) -> None:
        self.commands.extend(commands)


class FakeSummarizer:
    def __init__(self) -> None:
        self.task_requests = []
        self.branch_requests = []

    async def finalize_branch(self, request):
        self.branch_requests.append(request)
        return SummaryResult(True, f"branch:{request.branch_history[-1]['content']}", "fake", None, None)

    async def finalize_task(self, request):
        self.task_requests.append(request)
        return SummaryResult(True, "should-not-be-needed", "fake", None, None)


@dataclass
class ReportHarness:
    store: FakeStore
    summarizer: FakeSummarizer
    coordinator: ReportCoordinator


@pytest.fixture
def report_harness() -> ReportHarness:
    task = FormalizedTask.create("任务文本必须逐字保持")
    branch = BranchRuntime(
        branch_id="A",
        task=task,
        catalog_fingerprint="catalog",
        generation=0,
        messages=[{"role": "assistant", "content": "A terminal evidence"}],
        credits=10.0,
        depth=0,
    )
    summarizer = FakeSummarizer()
    store = FakeStore()
    coordinator = ReportCoordinator(
        task_id="task",
        round_id="round",
        formalized_task=task,
        branches={"A": branch},
        store=store,
        summarizer=summarizer,
        clock=lambda: 10.0,
        time_budget_seconds=120,
        grace_period_seconds=60,
    )
    return ReportHarness(store, summarizer, coordinator)


@pytest.mark.asyncio
async def test_spec_13_4_single_terminal_summary_skips_finalize_task(report_harness: ReportHarness) -> None:
    """§13.4 — one terminal summary becomes the final body without calling finalize_task."""

    await report_harness.coordinator.on_branch_safe_point("A", terminal=True)
    await report_harness.coordinator.wait_for_synthesis()

    assert report_harness.summarizer.task_requests == []
    assert report_harness.coordinator.reports[-1].kind is ReportKind.FINAL
    assert "A terminal evidence" in report_harness.coordinator.reports[-1].text


@pytest.mark.asyncio
async def test_spec_13_2_checkpoint_coverage_freezes_as_intermediate(report_harness: ReportHarness) -> None:
    """§13.2 — any checkpoint in coverage forces INTERMEDIATE, even if siblings later finalize."""

    sibling = BranchRuntime(
        branch_id="B",
        task=report_harness.coordinator.formalized_task,
        catalog_fingerprint="catalog",
        generation=0,
        messages=[{"role": "assistant", "content": "B running"}],
        credits=1.0,
        depth=1,
    )
    report_harness.coordinator.branches["B"] = sibling
    epoch = await report_harness.coordinator.open_epoch()
    await report_harness.coordinator.on_branch_safe_point("A", checkpoint=True)
    # B is still in-flight until grace clones it into coverage as a checkpoint.
    await report_harness.coordinator.on_grace_expired(epoch=epoch.epoch)
    await report_harness.coordinator.wait_for_synthesis()

    assert report_harness.coordinator.reports[0].kind is ReportKind.INTERMEDIATE
    assert report_harness.coordinator.reports[0].coverage.has_checkpoint is True
    # Coverage has >1 items so task finalizer may run — kind must stay INTERMEDIATE.
    assert report_harness.coordinator.reports[0].kind is not ReportKind.FINAL
