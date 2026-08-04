from __future__ import annotations

import pytest

from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator


class Store:
    async def transact(self, commands) -> None:
        pass


class Summarizer:
    def __init__(self) -> None:
        self.branch_requests = []

    async def finalize_branch(self, request):
        self.branch_requests.append(request)
        from lunagentic_research_swarm.llm.summarizer import SummaryResult

        return SummaryResult(True, "checkpoint", "fake", None, None)

    async def finalize_task(self, request):
        from lunagentic_research_swarm.llm.summarizer import SummaryResult

        return SummaryResult(True, "report", "fake", None, None)


@pytest.mark.asyncio
async def test_agent_return_during_grace_is_checkpointed_but_original_continues() -> None:
    task = FormalizedTask.create("formal task")
    branch = BranchRuntime("A", task, "catalog", 0, [{"role": "assistant", "content": "stable"}], 5.0, 0)
    launched: list[str] = []
    coordinator = ReportCoordinator(
        task_id="task", round_id="round", formalized_task=task, branches={"A": branch},
        store=Store(), summarizer=Summarizer(), clock=lambda: 20.0,
        launch_delegation=lambda _parent, child: launched.append(child["branch_id"]),
    )
    epoch = await coordinator.open_epoch()

    await coordinator.on_branch_safe_point(
        "A", checkpoint=False, delegations=({"branch_id": "B", "task": "next"},)
    )

    assert epoch.frontier["A"].checkpoint_requested is True
    assert launched == ["B"]
    assert branch.latest_checkpoint_id is not None


@pytest.mark.asyncio
async def test_grace_expiry_clones_last_stable_context_without_stopping_original() -> None:
    task = FormalizedTask.create("formal task")
    branch = BranchRuntime("A", task, "catalog", 0, [{"role": "assistant", "content": "before call"}], 5.0, 0)
    coordinator = ReportCoordinator(
        task_id="task", round_id="round", formalized_task=task, branches={"A": branch},
        store=Store(), summarizer=Summarizer(), clock=lambda: 80.0,
    )
    epoch = await coordinator.open_epoch()
    branch.messages.append({"role": "assistant", "content": "late result"})

    await coordinator.on_grace_expired(epoch.epoch)

    assert epoch.frontier["A"].checkpoint_requested is True
    assert epoch.frontier["A"].stable_history[-1]["content"] == "before call"
    assert branch.messages[-1]["content"] == "late result"


@pytest.mark.asyncio
async def test_grace_return_snapshots_history_before_child_launch_and_marks_child_next_epoch() -> None:
    task = FormalizedTask.create("formal task")
    branch = BranchRuntime("A", task, "catalog", 0, [{"role": "assistant", "content": "pre-return"}], 5.0, 0)
    launched: list[dict[str, object]] = []

    async def launch(_parent: str, child: dict[str, object]) -> None:
        # A real launch can update the parent branch's mutable bookkeeping.
        # The current epoch checkpoint must not see this post-return mutation.
        branch.messages.append({"role": "assistant", "content": "child-launch mutation"})
        launched.append(child)

    summarizer = Summarizer()
    coordinator = ReportCoordinator(
        task_id="task", round_id="round", formalized_task=task, branches={"A": branch},
        store=Store(), summarizer=summarizer, clock=lambda: 20.0, launch_delegation=launch,
    )
    epoch = await coordinator.open_epoch()

    await coordinator.on_branch_safe_point(
        "A", delegations=({"branch_id": "B", "task": "next"},)
    )

    assert summarizer.branch_requests[-1].branch_history[-1]["content"] == "pre-return"
    assert launched == [{"branch_id": "B", "task": "next", "report_epoch": epoch.epoch + 1}]
