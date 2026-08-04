from __future__ import annotations

import pytest

from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator


class Store:
    async def transact(self, commands) -> None:
        pass


class Summarizer:
    async def finalize_branch(self, request):
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
