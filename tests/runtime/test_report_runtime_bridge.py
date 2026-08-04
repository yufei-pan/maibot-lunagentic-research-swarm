from __future__ import annotations

import pytest

from lunagentic_research_swarm.runtime.events import GraceExpired, ReportDeadlineReached

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
    manager, _, _, _, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    coordinator = FakeReportCoordinator()
    manager.report_coordinators[task_id] = coordinator

    await manager.handle_runtime_event(
        ReportDeadlineReached("deadline", task_id, before["round_id"], before["generation"], epoch=7)
    )
    await manager.handle_runtime_event(
        GraceExpired("grace", task_id, before["round_id"], before["generation"])
    )
    await manager.handle_branch_safe_point(task_id, before["active_leaves"][0]["branch_id"], terminal=True)

    assert coordinator.calls == [
        ("deadline", 7),
        ("grace", 7),
        ("safe_point", (before["active_leaves"][0]["branch_id"], {"checkpoint": False, "terminal": True, "delegations": ()})),
    ]
