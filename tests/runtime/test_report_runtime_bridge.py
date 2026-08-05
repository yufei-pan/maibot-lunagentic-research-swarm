from __future__ import annotations

import pytest

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import BranchRuntime, TaskStatus
from lunagentic_research_swarm.runtime.events import GraceExpired, ReportDeadlineReached
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator

from .test_controller_controls import _running_manager
from .test_controller_start import harness


class FakeReportCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def open_epoch(self, *, epoch: int | None = None) -> None:
        self.calls.append(("deadline", epoch))

    async def on_grace_expired(self, epoch: int | None = None) -> None:
        self.calls.append(("grace", epoch))

    async def on_branch_safe_point(self, branch_id: str, **kwargs: object) -> None:
        self.calls.append(("safe_point", (branch_id, kwargs)))


@pytest.mark.asyncio
async def test_manager_routes_durable_report_lifecycle_events_to_injected_coordinator(harness) -> None:
    registered: list[FakeReportCoordinator] = []

    def factory(**_kwargs: object) -> FakeReportCoordinator:
        coordinator = FakeReportCoordinator()
        registered.append(coordinator)
        return coordinator

    manager, _, _, _, task_id = await _running_manager(harness, report_coordinator_factory=factory)
    before = await manager.status(task_id)
    coordinator = manager.report_coordinators[task_id]

    await manager.handle_runtime_event(
        ReportDeadlineReached("deadline", task_id, before["round_id"], before["generation"], epoch=1)
    )
    await manager.handle_runtime_event(
        GraceExpired("grace", task_id, before["round_id"], before["generation"])
    )
    await manager.handle_branch_safe_point(task_id, before["active_leaves"][0]["branch_id"], terminal=True)

    assert coordinator.calls == [
        ("deadline", 1),
        ("grace", 1),
        ("safe_point", (before["active_leaves"][0]["branch_id"], {"checkpoint": False, "terminal": True, "delegations": ()})),
    ]
    assert registered == [coordinator]


@pytest.mark.asyncio
async def test_default_coordinator_completes_intermediate_then_final_report_through_controller(harness) -> None:
    manager, _, _, _, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    coordinator = manager.report_coordinators[task_id]

    assert isinstance(coordinator, ReportCoordinator)

    await manager.handle_runtime_event(
        ReportDeadlineReached("deadline", task_id, before["round_id"], before["generation"], epoch=1)
    )
    await manager.handle_runtime_event(
        GraceExpired("grace", task_id, before["round_id"], before["generation"], epoch=1)
    )
    await coordinator.wait_for_synthesis()
    assert (await manager.status(task_id))["status"] == TaskStatus.RUNNING.value

    await manager.handle_branch_safe_point(task_id, before["active_leaves"][0]["branch_id"], terminal=True)
    await coordinator.wait_for_synthesis()

    assert (await manager.status(task_id))["status"] == TaskStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_failed_final_synthesis_completes_with_errors_without_prompt_leak(harness) -> None:
    manager, _, summarizer, _, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    coordinator = manager.report_coordinators[task_id]
    root_id = before["active_leaves"][0]["branch_id"]
    root = coordinator.branches[root_id]
    coordinator.branches["B"] = BranchRuntime(
        branch_id="B", task=root.task, catalog_fingerprint=root.catalog_fingerprint, generation=0,
        messages=[{"role": "assistant", "content": "B evidence"}], credits=1.0, depth=1,
    )

    async def fail_finalizer(_request):
        return SummaryResult(
            False, "", "fake-model", None,
            type("Error", (), {"code": "provider_unavailable", "message": "summary provider unavailable"})(),
        )

    summarizer.finalize_task = fail_finalizer
    await manager.handle_branch_safe_point(task_id, root_id, terminal=True)
    await manager.handle_branch_safe_point(task_id, "B", terminal=True)
    await coordinator.wait_for_synthesis()

    assert (await manager.status(task_id))["status"] == TaskStatus.COMPLETED_WITH_ERRORS.value
    assert coordinator.reports[-1].status == "FAILED"
    assert "形式化后的调查任务" not in coordinator.reports[-1].error_message


@pytest.mark.asyncio
async def test_failed_intermediate_synthesis_returns_to_running(harness) -> None:
    manager, _, summarizer, _, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    coordinator = manager.report_coordinators[task_id]
    root_id = before["active_leaves"][0]["branch_id"]
    root = coordinator.branches[root_id]
    coordinator.branches["B"] = BranchRuntime(
        branch_id="B", task=root.task, catalog_fingerprint=root.catalog_fingerprint, generation=0,
        messages=[{"role": "assistant", "content": "B evidence"}], credits=1.0, depth=1,
    )

    async def fail_finalizer(_request):
        error = type("Error", (), {"code": "provider_unavailable", "message": "unavailable"})()
        return SummaryResult(False, "", "fake-model", None, error)

    summarizer.finalize_task = fail_finalizer
    await manager.handle_runtime_event(
        ReportDeadlineReached("deadline", task_id, before["round_id"], before["generation"], epoch=1)
    )
    await manager.handle_branch_safe_point(task_id, root_id, checkpoint=True)
    await manager.handle_branch_safe_point(task_id, "B", checkpoint=True)
    await coordinator.wait_for_synthesis()

    assert coordinator.reports[-1].kind.value == "INTERMEDIATE"
    assert coordinator.reports[-1].status == "FAILED"
    assert (await manager.status(task_id))["status"] == TaskStatus.RUNNING.value
