from __future__ import annotations

import asyncio

import pytest

from lunagentic_research_swarm.models import BranchLifecycle, ReportKind, TaskStatus
from lunagentic_research_swarm.runtime.controller import TaskController
from lunagentic_research_swarm.runtime.events import (
    AgentCallCompleted,
    ContinueRequested,
    StopRequested,
)
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall, PerformProcedureBatch, RuntimeState
from lunagentic_research_swarm.runtime.scheduler import FairScheduler


class _MemoryStore:
    def __init__(self) -> None:
        self.commands = []

    async def transact(self, commands) -> None:
        self.commands.extend(commands)


class _QueueScheduler:
    def __init__(self) -> None:
        self.enqueued = []

    async def enqueue(self, effect) -> bool:
        self.enqueued.append(effect)
        return True


@pytest.mark.asyncio
async def test_checkpoint_holds_children_until_report_epoch(runtime_harness) -> None:
    harness = runtime_harness
    await harness.start("调查", credits=20.0, time_budget=120)
    await harness.formalize("正式调查")
    await harness.root_delegates({"A": 10.0, "B": 10.0})
    assert harness.coordinator is not None

    launched: list[dict[str, object]] = []

    async def launch(_parent: str, child: dict[str, object]) -> None:
        launched.append(dict(child))

    harness.coordinator.launch_delegation = launch
    await harness.coordinator.on_branch_safe_point(
        "A", checkpoint=True, delegations=({"branch_id": "A-child", "agent_id": "sibling"},)
    )

    assert launched == []
    assert harness.coordinator.branches["A"].lifecycle is BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT

    await harness.coordinator.open_epoch()
    assert [item["branch_id"] for item in launched] == ["A-child"]


@pytest.mark.asyncio
async def test_stop_generation_discards_late_agent_completion() -> None:
    store = _MemoryStore()
    scheduler = _QueueScheduler()
    controller = TaskController(
        RuntimeState("task", TaskStatus.RUNNING, generation=0, active_round_id="round"),
        store=store,
        scheduler=scheduler,
    )

    await controller.apply(StopRequested("stop", "task", "round", 0, reason="planner"))
    command_count = len(store.commands)
    await controller.apply(
        AgentCallCompleted(
            "late", "task", "round", 0, branch_id="branch", call_id="call", result_id="result",
            actual_charge=0.0, balance_before_reconciliation=1.0,
            protocol_result={"report": "late", "procedures": (), "delegations": ()},
        )
    )

    assert controller.state.status is TaskStatus.STOPPED
    assert controller.state.generation == 1
    assert len(store.commands) == command_count
    assert not any(isinstance(effect, PerformProcedureBatch) for effect in scheduler.enqueued)


@pytest.mark.asyncio
async def test_negative_continue_barrier_finalizes_negative_leaves() -> None:
    store = _MemoryStore()
    scheduler = _QueueScheduler()
    controller = TaskController(
        RuntimeState("task", TaskStatus.PAUSED, generation=0, active_round_id="round", active_leaves={"A": 0.0, "B": 0.0}),
        store=store,
        scheduler=scheduler,
    )

    await controller.apply(
        ContinueRequested(
            "continue", "task", "round", 0,
            adjustment=-2.0,
            active_leaves={"A": 0.0, "B": 0.0},
            time_budget_seconds=120,
        )
    )

    assert controller.state.active_leaves == {}
    balances = [command.values for command in store.commands if command.kind == "update_branch_balance"]
    assert {item["branch_id"] for item in balances} == {"A", "B"}
    assert all(item["lifecycle"] == "FINALIZED" and item["credit_balance"] == -1.0 for item in balances)


@pytest.mark.asyncio
async def test_pause_keeps_queued_child_from_starting_until_resume() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    executed: list[str] = []

    async def worker(effect, _token):
        name = str(effect.payload["name"])
        executed.append(name)
        if name == "parent":
            started.set()
            await release.wait()

    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=worker)
    task = "task"
    await scheduler.start()
    await scheduler.enqueue(PerformAgentCall(task, "round", 0, payload={"name": "parent"}))
    await asyncio.wait_for(started.wait(), 1)
    scheduler.pause_task(task)
    await scheduler.enqueue(PerformAgentCall(task, "round", 0, payload={"name": "child"}))
    release.set()
    await asyncio.sleep(0.05)
    assert executed == ["parent"]

    scheduler.resume_task(task)
    for _ in range(20):
        if executed == ["parent", "child"]:
            break
        await asyncio.sleep(0.01)
    await scheduler.close()
    assert executed == ["parent", "child"]


@pytest.mark.asyncio
async def test_wide_fanout_does_not_starve_second_task() -> None:
    executed: list[str] = []

    async def worker(effect, _token):
        executed.append(str(effect.payload["task_marker"]))

    scheduler = FairScheduler(global_llm=1, per_task_llm=8, per_task_procedure=1, worker=worker)
    await scheduler.start()
    for index in range(8):
        await scheduler.enqueue(PerformAgentCall("wide", "round", 0, payload={"task_marker": f"A{index}"}))
    await scheduler.enqueue(PerformAgentCall("other", "round", 0, payload={"task_marker": "B"}))
    for _ in range(50):
        if len(executed) == 9:
            break
        await asyncio.sleep(0.01)
    await scheduler.close()

    assert len(executed) == 9
    assert executed.index("B") < 8


@pytest.mark.asyncio
async def test_all_branches_finishing_during_synthesis_keeps_intermediate_then_final(runtime_harness) -> None:
    harness = runtime_harness
    await harness.start("调查", credits=20.0, time_budget=120)
    await harness.formalize("正式调查")
    await harness.root_delegates({"A": 10.0, "B": 10.0})
    assert harness.coordinator is not None
    harness.summarizer.task_gate.clear()

    await harness.coordinator.on_branch_safe_point("A", checkpoint=True)
    await harness.coordinator.on_branch_safe_point("B", checkpoint=True)
    assert harness.coordinator.current_epoch is not None
    assert harness.coordinator.current_epoch.kind is ReportKind.INTERMEDIATE

    await harness.coordinator.on_branch_safe_point("A", terminal=True)
    await harness.coordinator.on_branch_safe_point("B", terminal=True)
    assert len(harness.coordinator.reports) == 0

    harness.summarizer.task_gate.set()
    await harness.coordinator.wait_for_synthesis()

    assert [record.kind for record in harness.coordinator.reports] == [ReportKind.INTERMEDIATE, ReportKind.FINAL]
