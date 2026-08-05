from __future__ import annotations

from dataclasses import dataclass

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.events import ChildMaterialized, ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    OpenReportEpoch,
    PerformAgentCall,
    PerformBranchSummary,
    PerformFormalization,
    PerformProcedureBatch,
    RuntimeState,
    reduce_event,
)


@dataclass(frozen=True)
class _Completed:
    kind: str


class _TurnWorker:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def perform_agent_call(self, effect):
        self.calls.append(effect)
        return _Completed("agent")

    async def perform_procedure_batch(self, effect):
        self.calls.append(effect)
        return _Completed("procedure")


class _Manager:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.effects: list[object] = []
        self.summaries: list[object] = []
        self.children: list[object] = []
        self.notifications: list[object] = []

    async def prepare_agent_effect(self, effect):
        return effect

    async def handle_runtime_event(self, event):
        self.events.append(event)

    async def handle_runtime_effect(self, effect):
        self.effects.append(effect)

    async def handle_branch_summary_effect(self, effect):
        self.summaries.append(effect)

    async def materialize_child_effect(self, effect):
        self.children.append(effect)

    async def notify_tool_waiter_effect(self, effect):
        if effect.payload.get("action") == "materialize_child":
            await self.materialize_child_effect(effect)
            return
        self.notifications.append(effect)


@pytest.mark.asyncio
async def test_runner_dispatches_effects_to_production_boundaries() -> None:
    manager = _Manager()
    worker = _TurnWorker()
    runner = RuntimeEffectRunner(worker)
    runner.bind_manager(manager)

    formalization = PerformFormalization("task", "round", 0)
    agent = PerformAgentCall("task", "round", 0)
    procedure = PerformProcedureBatch("task", "round", 0)
    summary = PerformBranchSummary("task", "round", 0)
    report = OpenReportEpoch("task", "round", 0)
    child = NotifyToolWaiter(
        "task", "round", 0, payload={"action": "materialize_child", "branch_id": "child"}
    )

    assert await runner.run(formalization) is None
    assert await runner.run(agent) == _Completed("agent")
    assert await runner.run(procedure) == _Completed("procedure")
    assert await runner.run(summary) is None
    assert await runner.run(report) is None
    assert await runner.run(child) is None

    assert [event.kind for event in manager.events] == ["agent", "procedure"]
    assert manager.summaries == [summary]
    assert manager.effects == [report]
    assert manager.children == [child]


@pytest.mark.asyncio
async def test_runner_requires_manager_before_stateful_effect() -> None:
    runner = RuntimeEffectRunner(_TurnWorker())

    with pytest.raises(RuntimeError, match="manager"):
        await runner.run(PerformAgentCall("task", "round", 0))


@pytest.mark.asyncio
async def test_non_materialization_notification_is_delivered() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.notifications: list[object] = []
            self.children: list[object] = []

        async def notify_tool_waiter_effect(self, effect):
            self.notifications.append(effect)

        async def materialize_child_effect(self, effect):
            self.children.append(effect)

    manager = _Manager()
    runner = RuntimeEffectRunner(_TurnWorker())
    runner.bind_manager(manager)

    effect = NotifyToolWaiter("task", "round", 0, payload={"action": "wake_tool", "error_code": "x"})
    result = await runner.run(effect)

    assert result is None
    assert manager.notifications == [effect]
    assert manager.children == []


def test_child_materialization_retires_parent_and_returns_unallocated_credit() -> None:
    delegated = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 10.0}),
        ProcedureBatchCompleted(
            "procedure",
            "task",
            "round",
            0,
            branch_id="parent",
            credits_after=10.0,
            delegations=({"agent_id": "agent.child", "task": "work", "credits": 4.0},),
            live_agent_ids=("agent.child",),
        ),
    )
    child_effect = next(effect for effect in delegated.effects if isinstance(effect, NotifyToolWaiter))
    assert child_effect.payload["pool_return"] == 6.0

    materialized = reduce_event(
        delegated.next_state,
        ChildMaterialized(
            "child",
            "task",
            "round",
            0,
            branch_id="parent:1",
            parent_branch_id="parent",
            agent_id="agent.child",
            credits=4.0,
            depth=1,
            retire_parent=True,
            pool_return=6.0,
        ),
    )

    assert materialized.next_state.active_leaves == {"parent:1": 4.0}
    assert materialized.next_state.credit_pool == 6.0
    assert [command.kind for command in materialized.commands] == ["insert_branch", "settle_delegating_parent"]
