from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.llm.gateway import GenerationError, GenerationResult
from lunagentic_research_swarm.llm.pricing import TokenUsage
from lunagentic_research_swarm.llm.protocol import DelegationRequest, ProcedureRequest, SwarmTurnEnvelope
from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask, TaskStatus
from lunagentic_research_swarm.procedures.core import CoreProcedureDecision
from lunagentic_research_swarm.runtime.events import (
    AgentCallCompleted,
    AgentCallFailed,
    AgentCallRequested,
    ProcedureBatchCompleted,
)
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformAgentCall,
    PerformBranchSummary,
    PerformProcedureBatch,
    RuntimeState,
    reduce_event,
)
from lunagentic_research_swarm.runtime.turns import TurnLimits, TurnWorker, resolve_completed_turn

NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


class RecordingProcedures:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, requests: tuple[ProcedureRequest, ...]) -> None:
        self.calls.extend(request.procedure_id for request in requests)


class FakeLLM:
    def __init__(self, result: GenerationResult) -> None:
        self.result = result
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return self.result


class FakePricing:
    def charge_actual(self, **kwargs):
        return SimpleNamespace(credits=1.5)


def branch(*, credits: float = 10.0, depth: int = 0) -> BranchRuntime:
    return BranchRuntime(
        branch_id="branch-root",
        task=FormalizedTask.create("逐字保持：调查 α 与 β。\n第二行。"),
        catalog_fingerprint="catalog-1",
        generation=0,
        messages=[{"role": "assistant", "content": "已有工作"}],
        credits=credits,
        depth=depth,
    )


def envelope(*, procedures=(), delegations=()) -> SwarmTurnEnvelope:
    return SwarmTurnEnvelope(report="本轮结果", procedures=list(procedures), delegations=list(delegations))


def delegation(agent_id: str, credits: float) -> DelegationRequest:
    return DelegationRequest(agent_id=agent_id, task=f"交给 {agent_id}", credits=credits)


@pytest.mark.asyncio
async def test_turn_worker_returns_normalized_protocol_event_without_mutating_effect() -> None:
    llm = FakeLLM(
        GenerationResult(
            response='{"report":"ok","procedures":[],"delegations":[]}',
            tool_calls=None,
            model_name="physical-v1",
            usage=TokenUsage(10, 2, 0, 10, source="actual"),
            success=True,
            error=None,
            duration=0.25,
        )
    )
    effect = PerformAgentCall(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        event_id="evt-call",
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "selector": "task:reasoning",
            "protocol": "json_envelope",
            "messages": ({"role": "user", "content": "任务"},),
            "estimated_charge": 2.0,
            "credits_after_reservation": 8.0,
        },
    )
    original_payload = dict(effect.payload)

    event = await TurnWorker(llm, RecordingProcedures(), pricing=FakePricing()).perform_agent_call(effect)

    assert event.actual_model_name == "physical-v1"
    assert event.actual_charge == 1.5
    assert event.protocol_result == {"report": "ok", "procedures": (), "delegations": ()}
    assert event.usage == {"prompt_tokens": 10, "completion_tokens": 2, "cache_hit_tokens": 0, "cache_miss_tokens": 10}
    assert dict(effect.payload) == original_payload


@pytest.mark.asyncio
async def test_zero_credit_agent_can_delegate_to_zero_credit_children() -> None:
    result = await resolve_completed_turn(
        branch=branch(credits=0.0),
        envelope=envelope(delegations=(delegation("agent.a", 10), delegation("agent.b", 5))),
        actual_charge=0.0,
        procedures=RecordingProcedures(),
        live_agents={"agent.a", "agent.b"},
    )

    assert [child.credits for child in result.children] == [0.0, 0.0]
    assert result.finalize_reason is None


@pytest.mark.asyncio
async def test_procedures_finish_before_negative_credit_blocks_children() -> None:
    procedures = RecordingProcedures()
    result = await resolve_completed_turn(
        branch=branch(credits=1.0),
        envelope=envelope(
            procedures=(ProcedureRequest(procedure_id="builtin.echo"),),
            delegations=(delegation("agent.a", 1),),
        ),
        actual_charge=5.0,
        procedures=procedures,
        live_agents={"agent.a"},
    )

    assert procedures.calls == ["builtin.echo"]
    assert result.children == ()
    assert result.finalize_reason == "negative_credit"


@pytest.mark.asyncio
async def test_removed_agent_edge_is_finalized_without_blocking_sibling() -> None:
    result = await resolve_completed_turn(
        branch=branch(),
        envelope=envelope(
            delegations=(delegation("agent.removed", 1), delegation("agent.valid", 1)),
        ),
        actual_charge=0.0,
        procedures=RecordingProcedures(),
        live_agents={"agent.valid"},
        limits=TurnLimits(max_agent_calls_per_task=256),
        agent_calls_started=255,
    )

    assert [child.agent_id for child in result.children] == ["agent.valid"]
    assert result.edge_finalizations[0].reason == "agent_unavailable"
    assert result.edge_finalizations[0].messages[-1]["content"].endswith("agent_unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "depth", "calls", "delegations", "reason", "child_ids"),
    [
        (TurnLimits(max_delegations_per_turn=1), 0, 0, 2, "delegation_limit_exceeded", ["agent.a"]),
        (TurnLimits(max_branch_depth=1), 1, 0, 1, "branch_depth_exceeded", []),
        (TurnLimits(max_agent_calls_per_task=2), 0, 2, 1, "agent_call_limit_exceeded", []),
    ],
)
async def test_structural_limit_finalizes_each_rejected_edge_without_hiding_valid_edges(
    limits: TurnLimits,
    depth: int,
    calls: int,
    delegations: int,
    reason: str,
    child_ids: list[str],
) -> None:
    requests = tuple(delegation(f"agent.{chr(97 + index)}", 1) for index in range(delegations))
    result = await resolve_completed_turn(
        branch=branch(depth=depth),
        envelope=envelope(delegations=requests),
        actual_charge=0.0,
        procedures=RecordingProcedures(),
        live_agents={request.agent_id for request in requests},
        limits=limits,
        agent_calls_started=calls,
    )

    assert [child.agent_id for child in result.children] == child_ids
    assert [edge.reason for edge in result.edge_finalizations] == [reason] * (delegations - len(child_ids))


def test_reducer_reserves_before_scheduling_agent_call() -> None:
    state = RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1")
    event = AgentCallRequested(
        "evt-request",
        "task-1",
        "round-1",
        0,
        occurred_at=NOW,
        branch_id="branch-1",
        call_id="call-1",
        agent_id="agent.a",
        selector="task:reasoning",
        prompt_tokens=100,
        estimated_charge=1.25,
        balance_before=4.0,
        usage_id="usage-1",
        ledger_id="ledger-1",
    )

    transition = reduce_event(state, event)

    assert [command.kind for command in transition.commands] == [
        "insert_llm_usage",
        "insert_credit_ledger",
        "insert_lifecycle_event",
    ]
    assert isinstance(transition.effects[0], PerformAgentCall)
    assert transition.effects[0].payload["credits_after_reservation"] == pytest.approx(2.75)


def test_agent_reservation_lifecycle_metadata_omits_raw_messages() -> None:
    state = RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1")
    event = AgentCallRequested(
        "evt-request-private",
        "task-1",
        "round-1",
        0,
        occurred_at=NOW,
        branch_id="branch-1",
        call_id="call-1",
        agent_id="agent.a",
        selector="task:reasoning",
        messages=({"role": "user", "content": "SECRET_RAW_PROMPT"},),
        prompt_tokens=100,
        estimated_charge=1.25,
        balance_before=4.0,
    )

    transition = reduce_event(state, event)

    lifecycle = next(command for command in transition.commands if command.kind == "insert_lifecycle_event")
    metadata = json.loads(lifecycle.values["metadata_json"])
    assert "messages" not in metadata
    assert "SECRET_RAW_PROMPT" not in lifecycle.values["metadata_json"]
    assert metadata["selector"] == "task:reasoning"
    assert metadata["estimated_charge"] == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_protocol_correction_appends_to_original_request_messages() -> None:
    original_messages = (
        {"role": "system", "content": "stable prefix"},
        {"role": "user", "content": "formalized task"},
        {"role": "user", "content": "runtime header"},
    )
    llm = FakeLLM(
        GenerationResult(
            response='{"report": 7}',
            tool_calls=None,
            model_name="physical-v1",
            usage=TokenUsage(10, 2, 0, 10, source="actual"),
            success=True,
            error=None,
            duration=0.25,
        )
    )
    effect = PerformAgentCall(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        event_id="evt-call",
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "selector": "task:reasoning",
            "protocol": "json_envelope",
            "messages": original_messages,
            "estimated_charge": 2.0,
            "credits_after_reservation": 8.0,
            "correction_call_id": "call-1:correction",
            "correction_usage_id": "usage-correction",
            "correction_ledger_id": "ledger-correction",
            "correction_estimated_charge": 0.1,
        },
    )

    completed = await TurnWorker(llm, RecordingProcedures(), pricing=FakePricing()).perform_agent_call(effect)
    assert isinstance(completed, AgentCallCompleted)
    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1"),
        completed,
    )

    correction = next(item for item in transition.effects if isinstance(item, PerformAgentCall))
    assert correction.payload["messages"][:-1] == original_messages
    assert correction.payload["messages"][-1]["role"] == "user"


@pytest.mark.asyncio
async def test_failed_agent_call_with_usage_reconciles_reserved_estimate() -> None:
    llm = FakeLLM(
        GenerationResult(
            response="",
            tool_calls=None,
            model_name="physical-v1",
            usage=TokenUsage(10, 2, 0, 10, source="actual"),
            success=False,
            error=GenerationError("provider_error", "provider failed after usage"),
            duration=0.25,
        )
    )
    effect = PerformAgentCall(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        event_id="evt-call",
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "selector": "task:reasoning",
            "messages": ({"role": "user", "content": "task"},),
            "estimated_charge": 2.0,
            "credits_after_reservation": 8.0,
        },
    )

    failed = await TurnWorker(llm, RecordingProcedures(), pricing=FakePricing()).perform_agent_call(effect)
    assert isinstance(failed, AgentCallFailed)
    assert failed.usage == {"prompt_tokens": 10, "completion_tokens": 2, "cache_hit_tokens": 0, "cache_miss_tokens": 10}
    assert failed.actual_model_name == "physical-v1"
    assert failed.actual_charge == pytest.approx(1.5)

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1"),
        failed,
    )
    usage = next(command for command in transition.commands if command.kind == "insert_llm_usage")
    ledger = next(command for command in transition.commands if command.kind == "insert_credit_ledger")
    assert usage.values["reconciliation_status"] == "actual"
    assert usage.values["actual_charge"] == pytest.approx(1.5)
    assert ledger.values["amount"] == pytest.approx(0.5)
    assert any(isinstance(item, PerformBranchSummary) for item in transition.effects)


def test_failed_agent_call_without_usage_stays_estimated_unreconciled() -> None:
    failed = AgentCallFailed(
        "evt-failed",
        "task-1",
        "round-1",
        0,
        occurred_at=NOW,
        branch_id="branch-1",
        call_id="call-1",
        error_code="provider_error",
        error_message="failed before usage",
        estimated_charge=2.0,
        balance_before_reconciliation=8.0,
    )

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1"),
        failed,
    )

    usage = next(command for command in transition.commands if command.kind == "insert_llm_usage")
    assert usage.values["reconciliation_status"] == "estimated_unreconciled"
    assert usage.values["actual_charge"] is None
    assert not any(command.kind == "insert_credit_ledger" for command in transition.commands)


def test_protocol_correction_is_pinned_to_first_actual_model_and_only_once() -> None:
    state = RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1")
    event = AgentCallCompleted(
        "evt-completed",
        "task-1",
        "round-1",
        0,
        occurred_at=NOW,
        branch_id="branch-1",
        call_id="call-1",
        result_id="result-1",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "cache_hit_tokens": 0, "cache_miss_tokens": 10},
        actual_model_name="physical-model-v1",
        actual_charge=0.25,
        estimated_charge=0.5,
        balance_before_reconciliation=1.0,
        protocol_error={"message": "invalid", "errors": [{"pointer": "/report", "message": "required"}]},
        correction_count=0,
        correction_call_id="call-correction",
        correction_usage_id="usage-correction",
        correction_ledger_id="ledger-correction",
        correction_estimated_charge=0.1,
        pinning_supported=True,
    )

    transition = reduce_event(state, event)

    correction = next(effect for effect in transition.effects if isinstance(effect, PerformAgentCall))
    assert correction.payload["selector"] == "model:physical-model-v1"
    assert correction.payload["correction_count"] == 1
    assert correction.payload["messages"][-1]["role"] == "user"
    assert "协议无效" in correction.payload["messages"][-1]["content"]

    second = reduce_event(
        state,
        AgentCallCompleted(
            "evt-second-invalid",
            "task-1",
            "round-1",
            0,
            branch_id="branch-1",
            call_id="call-correction",
            result_id="result-2",
            actual_model_name="physical-model-v1",
            actual_charge=0.0,
            estimated_charge=0.1,
            balance_before_reconciliation=0.9,
            protocol_error={"message": "still invalid", "errors": []},
            correction_count=1,
        ),
    )
    finalizer = next(effect for effect in second.effects if isinstance(effect, PerformBranchSummary))
    assert finalizer.payload["reason"] == "protocol_invalid"


def test_procedure_completion_applies_control_then_negative_then_checkpoint_order() -> None:
    state = RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1")
    base = dict(
        event_id="evt-procedures",
        task_id="task-1",
        round_id="round-1",
        generation=0,
        branch_id="branch-1",
        call_id="call-1",
        result_id="result-1",
        delegations=({"agent_id": "agent.a", "task": "child", "credits": 1.0},),
    )

    terminated = reduce_event(
        state,
        ProcedureBatchCompleted(**base, credits_after=-2.0, controls=CoreProcedureDecision(terminate=True)),
    )
    negative = reduce_event(
        state,
        ProcedureBatchCompleted(**base, credits_after=-2.0, controls=CoreProcedureDecision()),
    )
    checkpoint = reduce_event(
        state,
        ProcedureBatchCompleted(**base, credits_after=2.0, controls=CoreProcedureDecision(checkpoint=True)),
    )

    assert terminated.effects[0].payload["reason"] == "terminate"
    assert negative.effects[0].payload["reason"] == "negative_credit"
    assert checkpoint.effects[0].payload["reason"] == "checkpoint"
    assert checkpoint.effects[0].payload["held_delegations"] == base["delegations"]


def test_procedure_completion_materializes_valid_sibling_and_finalizes_rejected_edges() -> None:
    event = ProcedureBatchCompleted(
        event_id="evt-procedures-materialize",
        task_id="task-1",
        round_id="round-1",
        generation=0,
        branch_id="branch-1",
        call_id="call-1",
        result_id="result-1",
        credits_after=3.0,
        delegations=(
            {"agent_id": "agent.removed", "task": "removed assignment", "credits": 1.0},
            {"agent_id": "agent.valid", "task": "valid assignment", "credits": 1.0},
            {"agent_id": "agent.extra", "task": "extra assignment", "credits": 1.0},
        ),
        parent_messages=({"role": "user", "content": "formalized task"},),
        parent_depth=0,
        live_agent_ids=("agent.valid", "agent.extra"),
        max_delegations_per_turn=2,
        max_branch_depth=32,
        max_agent_calls_per_task=256,
        agent_calls_started=0,
    )

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1"),
        event,
    )

    children = [
        item for item in transition.effects
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]
    edges = [item for item in transition.effects if isinstance(item, PerformBranchSummary)]
    assert [item.payload["agent_id"] for item in children] == ["agent.valid"]
    assert children[0].payload["messages"][-1]["content"] == "assignment: valid assignment"
    assert [(item.payload["agent_id"], item.payload["reason"]) for item in edges] == [
        ("agent.removed", "agent_unavailable"),
        ("agent.extra", "delegation_limit_exceeded"),
    ]
    assert edges[0].payload["assignment"] == "removed assignment"
    assert edges[0].payload["messages"][-1]["content"].endswith("agent_unavailable")
    with pytest.raises(TypeError):
        edges[0].payload["messages"][0]["content"] = "mutated"  # type: ignore[index]


def test_missing_agent_does_not_consume_last_task_call_slot_before_valid_sibling() -> None:
    event = ProcedureBatchCompleted(
        event_id="evt-missing-before-valid",
        task_id="task-1",
        round_id="round-1",
        generation=0,
        branch_id="branch-1",
        call_id="call-1",
        credits_after=2.0,
        delegations=(
            {"agent_id": "agent.removed", "task": "removed assignment", "credits": 1.0},
            {"agent_id": "agent.valid", "task": "valid assignment", "credits": 1.0},
        ),
        live_agent_ids=("agent.valid",),
        max_agent_calls_per_task=256,
        agent_calls_started=255,
    )

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1"),
        event,
    )

    children = [
        item
        for item in transition.effects
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]
    edges = [item for item in transition.effects if isinstance(item, PerformBranchSummary)]
    assert [item.payload["agent_id"] for item in children] == ["agent.valid"]
    assert [(item.payload["agent_id"], item.payload["reason"]) for item in edges] == [
        ("agent.removed", "agent_unavailable")
    ]


def test_reducer_atomically_reserves_task_call_slot_across_stale_procedure_events() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        agent_calls_started=255,
    )

    def completion(event_id: str, branch_id: str, agent_id: str) -> ProcedureBatchCompleted:
        return ProcedureBatchCompleted(
            event_id=event_id,
            task_id="task-1",
            round_id="round-1",
            generation=0,
            branch_id=branch_id,
            call_id=f"{branch_id}:call",
            credits_after=1.0,
            delegations=({"agent_id": agent_id, "task": "child", "credits": 1.0},),
            live_agent_ids=(agent_id,),
            max_agent_calls_per_task=256,
            # 两个并发 worker 都观察到同一旧快照；reducer state 必须具有权威性。
            agent_calls_started=255,
        )

    first = reduce_event(state, completion("evt-first", "branch-a", "agent.a"))
    second = reduce_event(first.next_state, completion("evt-second", "branch-b", "agent.b"))

    first_child = next(
        item
        for item in first.effects
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    )
    assert first.next_state.agent_calls_started == 256
    assert first_child.payload["agent_calls_started"] == 256
    assert first_child.payload["max_agent_calls_per_task"] == 256
    assert second.next_state.agent_calls_started == 256
    assert not any(
        isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
        for item in second.effects
    )
    second_edge = next(item for item in second.effects if isinstance(item, PerformBranchSummary))
    assert second_edge.payload["reason"] == "agent_call_limit_exceeded"


def test_procedure_completion_preserves_monotonic_call_counter_through_continue_barrier() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"branch-1": 1.0},
        continue_barrier=True,
        agent_calls_started=254,
    )
    event = ProcedureBatchCompleted(
        event_id="evt-continue-counter",
        task_id="task-1",
        round_id="round-1",
        generation=0,
        branch_id="branch-1",
        call_id="call-1",
        credits_after=1.0,
        agent_calls_started=255,
    )

    transition = reduce_event(state, event)

    assert transition.next_state.agent_calls_started == 255


@pytest.mark.parametrize(
    ("parent_depth", "agent_calls_started", "reason"),
    [
        (32, 0, "branch_depth_exceeded"),
        (0, 256, "agent_call_limit_exceeded"),
    ],
)
def test_procedure_completion_reaches_each_structural_edge_limit(
    parent_depth: int,
    agent_calls_started: int,
    reason: str,
) -> None:
    event = ProcedureBatchCompleted(
        event_id=f"evt-{reason}",
        task_id="task-1",
        round_id="round-1",
        generation=0,
        branch_id="branch-1",
        call_id="call-1",
        credits_after=1.0,
        delegations=({"agent_id": "agent.valid", "task": "child", "credits": 1.0},),
        parent_depth=parent_depth,
        live_agent_ids=("agent.valid",),
        max_delegations_per_turn=8,
        max_branch_depth=32,
        max_agent_calls_per_task=256,
        agent_calls_started=agent_calls_started,
    )

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, generation=0, active_round_id="round-1"),
        event,
    )

    edge = next(item for item in transition.effects if isinstance(item, PerformBranchSummary))
    assert edge.payload["reason"] == reason
    assert not any(
        isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
        for item in transition.effects
    )


class CompletedProcedures:
    async def invoke_many(self, effect: PerformProcedureBatch) -> ProcedureBatchCompleted:
        return ProcedureBatchCompleted(
            event_id="evt-procedures",
            task_id=effect.task_id,
            round_id=str(effect.round_id),
            generation=effect.generation,
            branch_id=str(effect.payload["branch_id"]),
            call_id=str(effect.payload["call_id"]),
        )


@pytest.mark.asyncio
async def test_procedure_worker_preserves_materialization_snapshot_for_reducer_boundary() -> None:
    effect = PerformProcedureBatch(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "delegations": ({"agent_id": "agent.valid", "task": "child", "credits": 1.0},),
            "credits_after": 1.0,
            "messages": ({"role": "user", "content": "formalized task"},),
            "branch_depth": 7,
            "live_agent_ids": ("agent.valid",),
            "max_delegations_per_turn": 4,
            "max_branch_depth": 9,
            "max_agent_calls_per_task": 11,
            "agent_calls_started": 3,
        },
    )

    event = await TurnWorker(FakeLLM, CompletedProcedures()).perform_procedure_batch(effect)

    assert event.parent_messages == effect.payload["messages"]
    assert event.parent_depth == 7
    assert event.live_agent_ids == ("agent.valid",)
    assert event.max_delegations_per_turn == 4
    assert event.max_branch_depth == 9
    assert event.max_agent_calls_per_task == 11
    assert event.agent_calls_started == 3


@pytest.mark.asyncio
async def test_procedure_worker_uses_effect_round_catalog_factory() -> None:
    catalog = object()
    seen: list[object] = []

    def factory(value: object) -> CompletedProcedures:
        seen.append(value)
        return CompletedProcedures()

    effect = PerformProcedureBatch(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        payload={"branch_id": "branch-1", "call_id": "call-1", "procedure_catalog": catalog},
    )

    await TurnWorker(object(), CompletedProcedures(), procedure_factory=factory).perform_procedure_batch(effect)

    assert seen == [catalog]
