"""E2E protocol correction / native / self-loop / interrupted reopen (E9, E10, E13, E14).

Builds on Task 4 unit pins and Task 3 feedback interrupted wiring. Uses FakeLLMGateway
enqueue, TurnWorker, manager handle_runtime_event, ProcedureBatchCompleted materialize,
and mark_active_rounds_interrupted + continue. No Host LLM.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeLLMResponse

from lunagentic_research_swarm.feedback import FeedbackService
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformAgentCall,
    PerformBranchSummary,
    PerformProcedureBatch,
)
from lunagentic_research_swarm.runtime.turns import TurnWorker

from .test_feedback_reminders import RecordingStore

ROOT_AGENT = "builtin.quick_thinker"


class _ChargePricing:
    def __init__(self, credits: float = 0.25) -> None:
        self.credits = float(credits)

    def charge_actual(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(credits=self.credits)


def _materialize_effects(enqueued: list[Any]) -> list[NotifyToolWaiter]:
    return [
        item
        for item in enqueued
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]


def _agent_call_effects(enqueued: list[Any]) -> list[PerformAgentCall]:
    return [item for item in enqueued if isinstance(item, PerformAgentCall)]


def _procedure_effects(enqueued: list[Any]) -> list[PerformProcedureBatch]:
    return [item for item in enqueued if isinstance(item, PerformProcedureBatch)]


def _summary_effects(enqueued: list[Any]) -> list[PerformBranchSummary]:
    return [item for item in enqueued if isinstance(item, PerformBranchSummary)]


def _native_tool_call(*, report: str = "", procedures: list[dict[str, Any]] | None = None, delegations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": "call_native_e2e",
        "function": {
            "name": "submit_swarm_turn",
            "arguments": {
                "report": report,
                "procedures": list(procedures or []),
                "delegations": list(delegations or []),
            },
        },
    }


async def _run_procedure_batch(harness, effect: PerformProcedureBatch):
    """Drive PerformProcedureBatch through real ProcedureExecutor (core.terminate etc.)."""

    executor = ProcedureExecutor(SimpleNamespace(get=lambda _pid: None), harness.procedures)
    batch_raw = await executor.invoke_many(effect)
    return replace(
        batch_raw,
        report=str(effect.payload.get("report", "")),
        delegations=tuple(effect.payload.get("delegations", ())),
        credits_after=float(effect.payload.get("credits_after", 0.0)),
        parent_messages=batch_raw.parent_messages or tuple(effect.payload.get("messages", ())),
        parent_depth=int(effect.payload.get("branch_depth", 0)),
        live_agent_ids=effect.payload.get("live_agent_ids"),
        max_delegations_per_turn=int(effect.payload.get("max_delegations_per_turn", 8)),
        max_branch_depth=int(effect.payload.get("max_branch_depth", 32)),
        max_agent_calls_per_task=int(effect.payload.get("max_agent_calls_per_task", 256)),
        agent_calls_started=int(effect.payload.get("agent_calls_started", 0)),
    )


async def _reminder_rows(store, *, task_id: str | None = None) -> list[dict[str, Any]]:
    def _read(connection: Any) -> list[dict[str, Any]]:
        if task_id is None:
            rows = connection.execute(
                "SELECT reminder_id, round_id, status, due_at FROM feedback_reminders ORDER BY reminder_id"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT reminder_id, round_id, status, due_at
                FROM feedback_reminders
                WHERE task_id = ?
                ORDER BY reminder_id
                """,
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    return await store.run_locked(_read)


# ---------------------------------------------------------------------------
# E9 — invalid envelope → one correction → terminate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e9_protocol_invalid_one_correction_then_terminate(runtime_harness) -> None:
    """§9.2 / §23.1 / E9 — FakeLLM invalid then valid+terminate; exactly one correction."""

    harness = runtime_harness
    await harness.start("协议纠正", credits=40.0, time_budget=90)
    await harness.formalize("正式协议纠正")
    status = await harness.manager.status(harness.task_id)
    root_id = str(status["active_leaves"][0]["branch_id"])
    controller = harness.manager._controllers[harness.task_id]
    credits = float(status["active_leaves"][0]["credits"])

    harness.llm.enqueue(
        FakeLLMResponse(
            text='{"report": 7, "procedures": [], "delegations": []}',
            model="physical-v1",
        ),
        FakeLLMResponse(
            payload={
                "report": "corrected",
                "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}}],
                "delegations": [],
            },
            model="physical-v1",
        ),
    )
    worker = TurnWorker(harness.llm, harness.procedures, pricing=_ChargePricing(0.25))

    first_effect = PerformAgentCall(
        harness.task_id,
        harness.round_id,
        int(status["generation"]),
        event_id=f"{harness.task_id}:e9-call",
        payload={
            "branch_id": root_id,
            "call_id": "call-e9",
            "selector": "task:reasoning",
            "protocol": "json_envelope",
            "messages": (
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "formalized"},
            ),
            "estimated_charge": 0.5,
            "credits_after_reservation": credits - 0.5,
            "correction_estimated_charge": 0.1,
            "max_correction_turns": 1,
            "pinning_supported": True,
            "branch_depth": 0,
            "live_agent_ids": (ROOT_AGENT,),
            "agent_calls_started": int(controller.state.agent_calls_started),
        },
    )

    first = await worker.perform_agent_call(first_effect)
    assert first.protocol_error is not None
    assert first.protocol_result is None
    assert first.actual_model_name == "physical-v1"

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(first)

    corrections = _agent_call_effects(harness.scheduler.enqueued)
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction.payload["correction_count"] == 1
    assert correction.payload["call_id"] == "call-e9:correction"
    assert correction.payload["selector"] == "model:physical-v1"
    assert correction.payload["messages"][-1]["role"] == "user"
    assert (
        "协议" in correction.payload["messages"][-1]["content"]
        or "report" in correction.payload["messages"][-1]["content"]
    )

    harness.scheduler.enqueued.clear()
    second = await worker.perform_agent_call(correction)
    assert second.protocol_error is None
    assert second.protocol_result is not None
    assert second.correction_count == 1
    assert any(
        item.get("procedure_id") == CORE_TERMINATE_ID for item in second.protocol_result["procedures"]
    )

    await harness.manager.handle_runtime_event(second)
    batches = _procedure_effects(harness.scheduler.enqueued)
    assert len(batches) == 1
    assert any(item.get("procedure_id") == CORE_TERMINATE_ID for item in batches[0].payload["requests"])
    assert not _agent_call_effects(harness.scheduler.enqueued), "valid envelope must not schedule another correction"

    batch = await _run_procedure_batch(harness, batches[0])
    assert batch.controls is not None and batch.controls.terminate is True

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(batch)
    summaries = _summary_effects(harness.scheduler.enqueued)
    assert len(summaries) == 1
    assert summaries[0].payload["reason"] == "terminate"
    assert summaries[0].payload["branch_id"] == root_id
    assert len(harness.llm.calls) == 2


# ---------------------------------------------------------------------------
# E10 — native tool-only turn (empty assistant prose)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e10_native_tool_only_empty_assistant_text_completes(runtime_harness) -> None:
    """§9.3 / E10 — native submit_swarm_turn with empty prose completes via FakeLLMGateway."""

    harness = runtime_harness
    await harness.start("native 无正文", credits=30.0, time_budget=60)
    await harness.formalize("正式 native 无正文")
    status = await harness.manager.status(harness.task_id)
    root_id = str(status["active_leaves"][0]["branch_id"])
    credits = float(status["active_leaves"][0]["credits"])
    controller = harness.manager._controllers[harness.task_id]

    harness.llm.enqueue(
        FakeLLMResponse(
            text="",
            tool_calls=[
                _native_tool_call(
                    report="",
                    procedures=[{"procedure_id": CORE_TERMINATE_ID, "arguments": {}}],
                )
            ],
            model="native-model",
            usage={"prompt_tokens": 8, "completion_tokens": 1, "cache_hit_tokens": 0, "cache_miss_tokens": 8},
        )
    )
    worker = TurnWorker(harness.llm, harness.procedures, pricing=_ChargePricing(0.1))
    effect = PerformAgentCall(
        harness.task_id,
        harness.round_id,
        int(status["generation"]),
        event_id=f"{harness.task_id}:e10-native",
        payload={
            "branch_id": root_id,
            "call_id": "call-native",
            "selector": "task:reasoning",
            "protocol": "native_tools",
            "messages": ({"role": "user", "content": "task"},),
            "estimated_charge": 0.5,
            "credits_after_reservation": credits - 0.5,
            "branch_depth": 0,
            "live_agent_ids": (ROOT_AGENT,),
            "agent_calls_started": int(controller.state.agent_calls_started),
        },
    )

    completed = await worker.perform_agent_call(effect)
    assert completed.protocol_error is None
    assert completed.protocol == "native_tools"
    assert completed.protocol_result is not None
    assert completed.protocol_result["report"] == ""
    assert any(item.get("procedure_id") == CORE_TERMINATE_ID for item in completed.protocol_result["procedures"])
    assert harness.llm.calls[0]["tools"] is not None
    assert harness.llm.calls[0]["tools"][0]["function"]["name"] == "submit_swarm_turn"

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(completed)
    batches = _procedure_effects(harness.scheduler.enqueued)
    assert len(batches) == 1
    assert batches[0].payload["report"] == ""

    batch = await _run_procedure_batch(harness, batches[0])
    assert batch.controls is not None and batch.controls.terminate is True
    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(batch)
    summaries = _summary_effects(harness.scheduler.enqueued)
    assert len(summaries) == 1
    assert summaries[0].payload["reason"] == "terminate"


# ---------------------------------------------------------------------------
# E13 — self-delegation tool loop then terminate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e13_self_delegation_materialize_then_terminate(runtime_harness) -> None:
    """§9.1 / E13 — parent delegates to itself → materialize child → child terminates."""

    harness = runtime_harness
    await harness.start("自调用循环", credits=50.0, time_budget=90)
    await harness.formalize("正式自调用循环")
    status = await harness.manager.status(harness.task_id)
    root_id = str(status["active_leaves"][0]["branch_id"])
    credits = float(status["active_leaves"][0]["credits"])
    controller = harness.manager._controllers[harness.task_id]
    generation = int(status["generation"])

    # Parent turn: self-delegate under credit/depth limits.
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "parent self-loop",
                "procedures": [],
                "delegations": [
                    {"agent_id": ROOT_AGENT, "task": "再读一遍再决策", "credits": 20.0},
                ],
            },
            model="physical-v1",
        )
    )
    worker = TurnWorker(harness.llm, harness.procedures, pricing=_ChargePricing(0.2))
    parent_effect = PerformAgentCall(
        harness.task_id,
        harness.round_id,
        generation,
        event_id=f"{harness.task_id}:e13-parent",
        payload={
            "branch_id": root_id,
            "call_id": "call-parent",
            "selector": "model:physical-v1",
            "protocol": "json_envelope",
            "messages": (
                {"role": "user", "content": "formalized"},
                {"role": "assistant", "content": "prior"},
            ),
            "estimated_charge": 0.5,
            "credits_after_reservation": credits - 0.5,
            "branch_depth": 0,
            "live_agent_ids": (ROOT_AGENT,),
            "agent_calls_started": int(controller.state.agent_calls_started),
            "max_delegations_per_turn": 8,
            "max_branch_depth": 32,
            "max_agent_calls_per_task": 256,
        },
    )

    parent_completed = await worker.perform_agent_call(parent_effect)
    assert parent_completed.protocol_error is None
    assert parent_completed.protocol_result is not None
    assert parent_completed.protocol_result["delegations"][0]["agent_id"] == ROOT_AGENT

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(parent_completed)
    parent_batches = _procedure_effects(harness.scheduler.enqueued)
    assert len(parent_batches) == 1

    parent_batch = await _run_procedure_batch(harness, parent_batches[0])
    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(parent_batch)

    children = _materialize_effects(harness.scheduler.enqueued)
    assert len(children) == 1
    assert children[0].payload["agent_id"] == ROOT_AGENT
    assert children[0].payload["parent_branch_id"] == root_id
    assert children[0].payload["depth"] == 1
    assert children[0].payload["credits"] == pytest.approx(20.0)
    assert "再读一遍再决策" in children[0].payload["messages"][-1]["content"]

    await harness.manager.materialize_child_effect(children[0])
    child_id = str(children[0].payload["branch_id"])
    assert child_id in controller.state.active_leaves
    child_launches = [
        item
        for item in harness.scheduler.enqueued
        if isinstance(item, PerformAgentCall) and item.payload.get("branch_id") == child_id
    ]
    assert child_launches, "materialize must enqueue child's first agent call"

    # Child turn: terminate the self-loop.
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "child done",
                "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}}],
                "delegations": [],
            },
            model="physical-v1",
        )
    )
    child_credits = float(controller.state.active_leaves[child_id])
    child_effect = PerformAgentCall(
        harness.task_id,
        harness.round_id,
        controller.state.generation,
        event_id=f"{harness.task_id}:e13-child",
        payload={
            "branch_id": child_id,
            "call_id": "call-child",
            "selector": "model:physical-v1",
            "protocol": "json_envelope",
            "messages": tuple(children[0].payload["messages"]),
            "estimated_charge": 0.5,
            "credits_after_reservation": child_credits - 0.5,
            "branch_depth": 1,
            "live_agent_ids": (ROOT_AGENT,),
            "agent_calls_started": int(controller.state.agent_calls_started),
        },
    )
    child_completed = await worker.perform_agent_call(child_effect)
    assert child_completed.protocol_error is None

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(child_completed)
    child_batches = _procedure_effects(harness.scheduler.enqueued)
    assert len(child_batches) == 1
    child_batch = await _run_procedure_batch(harness, child_batches[0])
    assert child_batch.controls is not None and child_batch.controls.terminate is True

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(child_batch)
    summaries = _summary_effects(harness.scheduler.enqueued)
    assert len(summaries) == 1
    assert summaries[0].payload["branch_id"] == child_id
    assert summaries[0].payload["reason"] == "terminate"
    assert len(harness.llm.calls) == 2


# ---------------------------------------------------------------------------
# E14 — stranded RUNNING → mark interrupted → continue round 2; no reminder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e14_interrupted_reopen_continue_no_reminder_for_interrupted_round(
    runtime_harness,
) -> None:
    """§18.4 / §20.3 / E14 — crash mark INTERRUPTED → restore_tasks → continue; no reminder."""

    harness = runtime_harness
    feedback = FeedbackService(
        store=harness.store,
        reminders_enabled=True,
        feedback_wait_seconds=600,
        clock=harness.clock,
    )
    # Wire FeedbackService so restore/continue controllers use _feedback_commands.
    harness.manager._feedback_service = feedback

    await harness.start("中断再开", credits=40.0, time_budget=120)
    harness.manager._controllers[harness.task_id].feedback = feedback
    await harness.formalize("正式中断再开")

    first = await harness.manager.status(harness.task_id)
    interrupted_round = str(first["round_id"])
    assert first["status"] == "RUNNING"
    assert first["round_number"] == 1
    # Pre-crash: credits live on the root leaf; durable round pool is already 0.
    assert first["active_leaves"] and float(first["active_leaves"][0]["credits"]) == pytest.approx(40.0)

    # Production startup: mark stranded RUNNING → INTERRUPTED (no feedback insert).
    count = await harness.store.mark_active_rounds_interrupted(float(harness.clock()))
    assert count == 1
    task = await harness.store.load_task(harness.task_id)
    assert task is not None and task.current_round is not None
    assert task.current_round.status is TaskStatus.INTERRUPTED
    assert task.current_round.round_id == interrupted_round
    durable_pool = float(task.current_round.credit_pool)
    assert durable_pool == pytest.approx(0.0)
    assert await _reminder_rows(harness.store, task_id=harness.task_id) == []
    assert (
        feedback.commands_for_status_transition(
            task_id=harness.task_id,
            round_id=interrupted_round,
            new_status=TaskStatus.INTERRUPTED.value,
            ended_at=float(harness.clock()),
        )
        == ()
    )

    # Simulate process restart: drop ephemerals, then rehydrate via restore_tasks (§18.3/§18.4).
    # Leaf credits are not persisted — restore must NOT invent a leaf→pool fold.
    task_id = harness.task_id
    harness.manager._controllers.pop(task_id, None)
    harness.manager._branches.pop(task_id, None)
    harness.manager.report_coordinators.pop(task_id, None)
    harness.manager._round_snapshots.pop(task_id, None)
    harness.manager._prompt_builders.pop(task_id, None)
    restored = await harness.manager.restore_tasks()
    assert restored == 1
    controller = harness.manager._controllers[task_id]
    assert controller.feedback is feedback
    assert controller.state.status is TaskStatus.INTERRUPTED
    assert controller.state.active_round_id == interrupted_round
    assert controller.state.credit_pool == pytest.approx(durable_pool)
    assert controller.state.active_leaves == {}
    assert harness.manager._branches[task_id] == {}
    assert controller.state.raw_context_released is True
    assert controller.state.formalized_task is not None

    # Record store batches so continue feedback wiring is proven (Task 3 cancel pattern).
    recording = RecordingStore(harness.store)
    harness.manager.store = recording
    controller.store = recording
    recording.batches.clear()

    # Real restore leaves 0 pool funds; continue with adj=0 is the honest outcome (0-credit root).
    continued = await harness.manager.continue_task(
        task_id, credit_adjustment=0.0, time_budget_seconds=90
    )
    assert continued["status"] == "RUNNING"
    assert continued["round_number"] == 2
    assert continued["generation"] == first["generation"] + 1
    assert continued["round_id"] != interrupted_round
    assert continued["active_leaves"]
    assert float(continued["active_leaves"][0]["credits"]) == pytest.approx(0.0)
    harness.round_id = str(continued["round_id"])

    assert not any(
        cmd.kind == "insert_feedback_reminder"
        for batch in recording.batches
        for cmd in batch
    ), "ContinueRequested after INTERRUPTED must not schedule a reminder"
    cancel_batches = [
        batch
        for batch in recording.batches
        if any(cmd.kind == "cancel_pending_feedback_reminders" for cmd in batch)
    ]
    assert cancel_batches, "ContinueRequested should emit cancel via _feedback_commands"
    cancel_cmds = [
        cmd
        for batch in cancel_batches
        for cmd in batch
        if cmd.kind == "cancel_pending_feedback_reminders"
    ]
    assert any(str(cmd.values.get("round_id")) == interrupted_round for cmd in cancel_cmds)

    reminders = await _reminder_rows(harness.store, task_id=task_id)
    assert not any(row["round_id"] == interrupted_round for row in reminders), (
        "interrupted round must not receive a feedback reminder"
    )
    assert not any(row["status"] == "pending" for row in reminders)

    # Restart enqueues a fresh root PerformAgentCall from summary layer.
    root_calls = [
        item
        for item in harness.scheduler.enqueued
        if isinstance(item, PerformAgentCall) and item.payload.get("root") is True
    ]
    assert root_calls, "continue after INTERRUPTED must enqueue root restart call"
    assert root_calls[-1].payload.get("formalized_text") == "正式中断再开"
