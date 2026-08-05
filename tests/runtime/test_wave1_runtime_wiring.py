"""Wave-1 runtime production wiring: deadlines, reservation, compact, effects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CoreProcedureDecision
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    ArmDeadline,
    ArmPauseExpiry,
    DeliverOutbox,
    NotifyToolWaiter,
    PerformAgentCall,
    PerformProcedureBatch,
    ReleaseRawContext,
    RuntimeState,
    reduce_event,
)
from .test_controller_start import harness  # noqa: F401


@pytest.mark.asyncio
async def test_formalization_arms_production_deadline_timer(harness) -> None:
    manager, _, _, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    task_id = result["task_id"]

    assert any(isinstance(effect, ArmDeadline) for effect in scheduler.enqueued)
    assert task_id in manager._deadline_jobs
    assert not manager._deadline_jobs[task_id].done()
    coordinator = manager.report_coordinators[task_id]
    assert coordinator.deadline_at > coordinator.started_at


@pytest.mark.asyncio
async def test_prepare_agent_effect_reserves_research_credits_via_store(harness) -> None:
    manager, store, _, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    agent_effect = next(effect for effect in reversed(scheduler.enqueued) if isinstance(effect, PerformAgentCall))
    before = len(store.commands)

    prepared = await manager.prepare_agent_effect(agent_effect)

    assert prepared.payload["estimated_charge"] > 0.0
    assert prepared.payload["credits_after_reservation"] < 100.0
    new_commands = store.commands[before:]
    assert any(command.kind == "insert_llm_usage" for command in new_commands)
    assert any(
        command.kind == "insert_credit_ledger" and command.values.get("entry_kind") == "input_reservation"
        for command in new_commands
    )
    usage = next(command for command in new_commands if command.kind == "insert_llm_usage")
    assert usage.values["reconciliation_status"] == "reserved"
    assert usage.values["estimated_charge"] == pytest.approx(prepared.payload["estimated_charge"])


@pytest.mark.asyncio
async def test_runner_handles_deadline_pause_outbox_release_and_error_notify() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        async def arm_deadline_effect(self, effect):
            self.calls.append(("deadline", effect))

        async def arm_pause_expiry_effect(self, effect):
            self.calls.append(("pause", effect))

        async def release_raw_context_effect(self, effect):
            self.calls.append(("release", effect))

        async def deliver_outbox_effect(self, effect):
            self.calls.append(("outbox", effect))

        async def notify_tool_waiter_effect(self, effect):
            self.calls.append(("notify", effect))

    manager = _Manager()
    runner = RuntimeEffectRunner(SimpleNamespace())
    runner.bind_manager(manager)

    deadline = ArmDeadline("task", "round", 0, payload={"due_at": 1.0, "kind": "report_deadline"})
    pause = ArmPauseExpiry("task", "round", 0, payload={"due_at": 2.0})
    release = ReleaseRawContext("task", "round", 0)
    outbox = DeliverOutbox("task", "round", 0, payload={"report_id": "r1"})
    notify = NotifyToolWaiter("task", "round", 0, payload={"error_code": "final_report_failed"})

    assert await runner.run(deadline) is None
    assert await runner.run(pause) is None
    assert await runner.run(release) is None
    assert await runner.run(outbox) is None
    assert await runner.run(notify) is None
    assert [name for name, _ in manager.calls] == ["deadline", "pause", "release", "outbox", "notify"]


@pytest.mark.asyncio
async def test_notify_tool_waiter_delivers_error_payload(harness) -> None:
    manager, _, _, _, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    waiter = manager.register_tool_waiter(result["task_id"])
    effect = NotifyToolWaiter(
        result["task_id"],
        "round",
        0,
        payload={"error_code": "final_report_failed", "error_message": "最终报告生成失败。"},
    )

    await manager.notify_tool_waiter_effect(effect)

    assert waiter.is_set()
    assert manager._tool_notifications[result["task_id"]][-1]["error_code"] == "final_report_failed"


@pytest.mark.asyncio
async def test_continue_task_allows_completed_with_errors(harness) -> None:
    manager, _, _, _, task_id = await _running(harness)
    controller = manager._controllers[task_id]
    controller.state = RuntimeState(
        task_id,
        TaskStatus.COMPLETED_WITH_ERRORS,
        generation=controller.state.generation,
        active_round_id=controller.state.active_round_id,
        formalized_task=controller.state.formalized_task,
        credit_pool=2.0,
    )
    manager._branches[task_id].clear()

    continued = await manager.continue_task(task_id, credit_adjustment=1.0, stream_id="s")

    assert continued["status"] == "RUNNING"


@pytest.mark.asyncio
async def test_compact_rewrites_messages_and_launches_children_without_checkpoint_hold() -> None:
    class _Summarizer:
        async def compact_branch(self, request):
            return SimpleNamespace(success=True, text="压缩后的摘要", model_name="fake", error=None)

    executor = ProcedureExecutor(catalog=SimpleNamespace(get=lambda _pid: None), summarizer=_Summarizer())
    effect = PerformProcedureBatch(
        "task-1",
        "round-1",
        0,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "formalized_task": "正式任务",
            "messages": (
                {"role": "system", "content": "swarm"},
                {"role": "user", "content": "正式任务"},
                {"role": "assistant", "content": "很长的可变历史"},
                {"role": "user", "content": "runtime header"},
            ),
            "requests": ({"procedure_id": "core.compact", "arguments": {}},),
            "delegations": ({"agent_id": "agent.child", "task": "child work", "credits": 1.0},),
            "credits_after": 3.0,
        },
    )

    completed = await executor.invoke_many(effect)

    assert completed.controls.compact is True
    assert completed.results[-1].procedure_id == "core.compact"
    assert completed.parent_messages[-1]["content"] == "分支压缩摘要：压缩后的摘要"
    assert all("很长的可变历史" not in str(item) for item in completed.parent_messages)

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, active_round_id="round-1", active_leaves={"branch-1": 3.0}),
        ProcedureBatchCompleted(
            "evt",
            "task-1",
            "round-1",
            0,
            branch_id="branch-1",
            credits_after=3.0,
            controls=CoreProcedureDecision(compact=True),
            delegations=({"agent_id": "agent.child", "task": "child work", "credits": 1.0},),
            parent_messages=completed.parent_messages,
            live_agent_ids=("agent.child",),
        ),
    )
    assert not any(
        getattr(item, "payload", {}).get("reason") in {"compact", "checkpoint"}
        for item in transition.effects
    )
    assert any(
        isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
        for item in transition.effects
    )


async def _running(harness):
    manager, store, summarizer, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    return manager, store, summarizer, scheduler, result["task_id"]
