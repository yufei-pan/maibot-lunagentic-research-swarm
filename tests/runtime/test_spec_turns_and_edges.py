"""Design §9.1 / §9.3 / §10 / §23.1 — call_id, self-delegation, mixed rejects, correction, native."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.llm.gateway import GenerationRequest, GenerationResult
from lunagentic_research_swarm.llm.pricing import TokenUsage
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID, CoreProcedureDecision
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor, procedure_result_summary
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.events import AgentCallCompleted, ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformAgentCall,
    PerformBranchSummary,
    PerformProcedureBatch,
    RuntimeState,
    reduce_event,
)
from lunagentic_research_swarm.runtime.turns import TurnWorker

from fakes import FakeLLMGateway, FakeLLMResponse

NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


class QueuedFakeLLM:
    """TurnWorker-facing LLM that pops GenerationResult / Exception from a queue."""

    def __init__(self, *results: GenerationResult | Exception) -> None:
        self.results = list(results)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("QueuedFakeLLM 队列已空")
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakePricing:
    def charge_actual(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(credits=0.25)


def _definition(procedure_id: str) -> ProcedureDefinition:
    return ProcedureDefinition.model_validate(
        {
            "procedure_id": procedure_id,
            "version": "1",
            "display_name": procedure_id,
            "description": "测试 Procedure",
            "arguments_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "idempotent": False,
            "timeout_seconds": 30.0,
        }
    )


def _catalog(*procedure_ids: str) -> ProcedureCatalogSnapshot:
    return ProcedureCatalogSnapshot(
        [
            ProcedureCatalogEntry(
                definition=_definition(procedure_id),
                provider_plugin_id="provider.tools",
                api_name="provider.tools.invoke_procedure",
                api_version="1",
                fingerprint=f"fingerprint:{procedure_id}",
            )
            for procedure_id in procedure_ids
        ]
    )


class FakeAPI:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def call(self, name: str, *, version: str = "", **kwargs: Any) -> Any:
        self.calls.append({"name": name, "version": version, **kwargs})
        procedure_id = str(kwargs["procedure_id"])
        response = self.responses[procedure_id]
        if callable(response):
            value = response()
            if asyncio.iscoroutine(value):
                return await value
            return value
        return response


def _usage(**overrides: int) -> dict[str, int]:
    base = {"prompt_tokens": 10, "completion_tokens": 2, "cache_hit_tokens": 0, "cache_miss_tokens": 10}
    base.update(overrides)
    return base


def _generation(
    *,
    response: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "physical-v1",
) -> GenerationResult:
    return GenerationResult(
        response=response,
        tool_calls=tool_calls,
        model_name=model,
        usage=TokenUsage(10, 2, 0, 10, source="actual"),
        success=True,
        error=None,
        duration=0.1,
    )


def _materialize_children(effects: Any) -> list[NotifyToolWaiter]:
    return [
        item
        for item in effects
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]


def _edge_summaries(effects: Any) -> list[PerformBranchSummary]:
    return [
        item
        for item in effects
        if isinstance(item, PerformBranchSummary) and item.payload.get("edge_finalization")
    ]


# ---------------------------------------------------------------------------
# §9.1 — procedure call_id optional; echo; request order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spec_9_1_procedure_call_id_optional_echoed_and_request_order_preserved() -> None:
    """§9.1 — call_id optional; when present echoed in result metadata; order = request order."""

    async def slow_first() -> dict[str, Any]:
        await asyncio.sleep(0.02)
        return {"success": True, "data": {"index": 0}, "error": None, "metadata": {}}

    async def fast_second() -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"success": True, "data": {"index": 1}, "error": None, "metadata": {}}

    async def mid_third() -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"success": True, "data": {"index": 2}, "error": None, "metadata": {}}

    api = FakeAPI(
        {
            "builtin.alpha": slow_first,
            "builtin.beta": fast_second,
            "builtin.gamma": mid_third,
        }
    )
    executor = ProcedureExecutor(
        _catalog("builtin.alpha", "builtin.beta", "builtin.gamma"),
        api=api,
    )
    # call_ids intentionally unsorted (z / missing / a) so order-by-call_id would fail.
    requests = [
        ProcedureRequest(procedure_id="builtin.alpha", call_id="z-last", arguments={}),
        ProcedureRequest(procedure_id="builtin.beta", arguments={}),
        ProcedureRequest(procedure_id="builtin.gamma", call_id="a-first", arguments={}),
    ]
    effect = PerformProcedureBatch(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        payload={
            "branch_id": "branch-1",
            "call_id": "turn-1",
            "turn_id": "turn-1",
            "agent_id": "agent.reader",
            "requests": requests,
        },
    )

    event = await executor.invoke_many(effect)

    assert [item.procedure_id for item in event.results] == [
        "builtin.alpha",
        "builtin.beta",
        "builtin.gamma",
    ]
    assert [item.result.data["index"] for item in event.results] == [0, 1, 2]
    assert event.results[0].call_id == "z-last"
    assert event.results[1].call_id == ""
    assert event.results[2].call_id == "a-first"

    summaries = [procedure_result_summary(item) for item in event.results]
    assert summaries[0]["call_id"] == "z-last"
    assert "call_id" not in summaries[1]
    assert summaries[2]["call_id"] == "a-first"
    # as_dict also surfaces call_id for event persistence / audit.
    assert event.results[0].as_dict()["call_id"] == "z-last"
    assert event.results[1].as_dict()["call_id"] == ""


# ---------------------------------------------------------------------------
# §9.1 — successful self-delegation materializes child
# ---------------------------------------------------------------------------


def test_spec_9_1_self_delegation_materializes_child_under_limits() -> None:
    """§9.1 — agent may list itself as next agent; child materializes under depth/call limits."""

    parent_agent = "agent.reader"
    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 5.0},
        agent_calls_started=3,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-self",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=5.0,
            delegations=(
                {"agent_id": parent_agent, "task": "读完结果后再决策", "credits": 4.0},
            ),
            parent_messages=(
                {"role": "user", "content": "formalized"},
                {"role": "assistant", "content": "procedure done"},
            ),
            parent_depth=1,
            live_agent_ids=(parent_agent,),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=3,
        ),
    )

    children = _materialize_children(transition.effects)
    assert len(children) == 1
    assert children[0].payload["agent_id"] == parent_agent
    assert children[0].payload["branch_id"] == "parent:1"
    assert children[0].payload["depth"] == 2
    assert children[0].payload["credits"] == pytest.approx(4.0)
    assert children[0].payload["messages"][-1]["content"] == "assignment: 读完结果后再决策"
    assert transition.next_state.agent_calls_started == 4
    assert not _edge_summaries(transition.effects)


def test_spec_9_1_empty_delegations_with_non_control_procedure_auto_self_delegates() -> None:
    """§9.1 — empty delegations + ordinary procedure → implicit self-continue delegation."""

    parent_agent = "builtin.quick_thinker"
    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 8.0},
        agent_calls_started=1,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-auto-self",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=7.5,
            results=(SimpleNamespace(procedure_id="builtin.web_search"),),
            delegations=(),
            agent_id=parent_agent,
            parent_messages=({"role": "user", "content": "formalized"},),
            parent_depth=0,
            live_agent_ids=(parent_agent,),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=1,
        ),
    )

    children = _materialize_children(transition.effects)
    assert len(children) == 1
    assert children[0].payload["agent_id"] == parent_agent
    assert children[0].payload["credits"] == pytest.approx(7.5)
    assignment = str(children[0].payload.get("assignment", ""))
    assert "上一 turn 的 Procedure 结果" in assignment
    # 兜底自委派也要说明下一步能做什么，否则这一 turn 只会原地复述。
    assert "core.terminate" in assignment
    assert not any(
        isinstance(item, PerformBranchSummary) and item.payload.get("reason") == "no_further_work"
        for item in transition.effects
    )


def test_spec_9_1_empty_delegations_without_ordinary_procedure_is_natural_end() -> None:
    """§9.1 — empty delegations and no ordinary procedure → no_further_work."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 8.0},
        agent_calls_started=1,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-natural",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=8.0,
            results=(),
            delegations=(),
            agent_id="builtin.quick_thinker",
            parent_depth=0,
            live_agent_ids=("builtin.quick_thinker",),
            agent_calls_started=1,
        ),
    )
    assert _materialize_children(transition.effects) == []
    summaries = [
        item
        for item in transition.effects
        if isinstance(item, PerformBranchSummary) and item.payload.get("reason") == "no_further_work"
    ]
    assert len(summaries) == 1


def test_spec_9_1_terminate_wins_over_auto_self_delegation() -> None:
    """§9.1 — terminate ignores empty-delegation auto self-continue."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 8.0},
        agent_calls_started=1,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-term",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=8.0,
            results=(SimpleNamespace(procedure_id="builtin.web_search"),),
            controls=CoreProcedureDecision(terminate=True),
            delegations=(),
            agent_id="builtin.quick_thinker",
            parent_depth=0,
            live_agent_ids=("builtin.quick_thinker",),
            agent_calls_started=1,
        ),
    )
    assert _materialize_children(transition.effects) == []
    assert any(
        isinstance(item, PerformBranchSummary) and item.payload.get("reason") == "terminate"
        for item in transition.effects
    )


# ---------------------------------------------------------------------------
# §10 — mixed rejected edges / credits conserved
# ---------------------------------------------------------------------------


def test_spec_10_all_agent_unavailable_retries_parent_once_credits_conserved() -> None:
    """§10 — all edges agent_unavailable → parent retry once; credits stay on parent."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 10.0},
        agent_calls_started=2,
        credit_pool=0.0,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-unavailable",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=10.0,
            delegations=(
                {"agent_id": "agent.gone_a", "task": "a", "credits": 6.0},
                {"agent_id": "agent.gone_b", "task": "b", "credits": 4.0},
            ),
            parent_messages=({"role": "assistant", "content": "parent"},),
            parent_depth=0,
            live_agent_ids=(),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=2,
        ),
    )

    assert _materialize_children(transition.effects) == []
    retries = [item for item in transition.effects if isinstance(item, PerformAgentCall)]
    assert len(retries) == 1
    assert retries[0].payload["branch_id"] == "parent"
    notice = retries[0].payload["appended_messages"][0]["content"]
    assert "agent_unavailable" in notice
    assert "agent.gone_a" in notice and "agent.gone_b" in notice
    # Credits never left the parent for rejected edges.
    assert transition.next_state.active_leaves == {"parent": 10.0}
    assert transition.next_state.credit_pool == 0.0
    # Parent retry consumes one call slot.
    assert transition.next_state.agent_calls_started == 3


def test_spec_10_mixed_unavailable_and_valid_finalizes_sibling_edge_credits_conserved() -> None:
    """§10 — mixed: unavailable edge finalized; valid sibling materializes; rejected share stays out."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 10.0},
        agent_calls_started=1,
        credit_pool=0.0,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-mixed",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=10.0,
            delegations=(
                {"agent_id": "agent.removed", "task": "removed work", "credits": 5.0},
                {"agent_id": "agent.valid", "task": "valid work", "credits": 5.0},
            ),
            parent_messages=({"role": "assistant", "content": "parent"},),
            parent_depth=0,
            live_agent_ids=("agent.valid",),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=1,
        ),
    )

    children = _materialize_children(transition.effects)
    edges = _edge_summaries(transition.effects)
    assert [item.payload["agent_id"] for item in children] == ["agent.valid"]
    assert children[0].payload["credits"] == pytest.approx(5.0)
    # Rejected edge's 5.0 never allocated to a branch (credits=0 on edge summary).
    assert [(item.payload["agent_id"], item.payload["reason"], item.payload["credits"]) for item in edges] == [
        ("agent.removed", "agent_unavailable", 0.0),
    ]
    # Unallocated remainder returns with the retiring parent (retire_parent on first live child).
    assert children[0].payload["retire_parent"] is True
    assert children[0].payload["pool_return"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("parent_depth", "agent_calls_started", "reason"),
    [
        (32, 0, "branch_depth_exceeded"),
        (0, 256, "agent_call_limit_exceeded"),
    ],
)
def test_spec_10_all_deterministic_limits_finalize_parent_credits_conserved(
    parent_depth: int,
    agent_calls_started: int,
    reason: str,
) -> None:
    """§10 — deterministic depth/call-limit on all edges → finalize parent; no children; credits stay."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 8.0},
        agent_calls_started=agent_calls_started,
        credit_pool=1.0,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            f"evt-{reason}",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=8.0,
            delegations=(
                {"agent_id": "agent.valid", "task": "child a", "credits": 4.0},
                {"agent_id": "agent.valid", "task": "child b", "credits": 4.0},
            ),
            parent_messages=({"role": "assistant", "content": "parent"},),
            parent_depth=parent_depth,
            live_agent_ids=("agent.valid",),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=agent_calls_started,
        ),
    )

    assert _materialize_children(transition.effects) == []
    assert not any(isinstance(item, PerformAgentCall) for item in transition.effects)
    summaries = [item for item in transition.effects if isinstance(item, PerformBranchSummary)]
    assert len(summaries) == 1
    assert summaries[0].payload["branch_id"] == "parent"
    assert summaries[0].payload["reason"] == reason
    assert reason in summaries[0].payload["appended_messages"][0]["content"]
    # Credits conserved on parent until finalization consumes the leaf.
    assert transition.next_state.active_leaves["parent"] == pytest.approx(8.0)
    assert transition.next_state.credit_pool == pytest.approx(1.0)


def test_spec_10_mixed_call_limit_and_valid_finalizes_sibling_edge_credits_conserved() -> None:
    """§10 — mixed: one edge hits agent_call_limit; valid sibling materializes; no invent.

    Distinct from all-deterministic→parent finalize and unavailable+valid. Depth cannot
    reject only one sibling in a batch (same parent_depth); call-limit reservation can:
    started=255 / max=256 accepts the first live edge then rejects the next.
    """

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 10.0},
        agent_calls_started=255,
        credit_pool=2.0,
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "evt-mixed-call-limit",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent",
            call_id="call-1",
            result_id="result-1",
            credits_after=10.0,
            delegations=(
                {"agent_id": "agent.valid", "task": "valid work", "credits": 6.0},
                {"agent_id": "agent.sibling", "task": "over budget work", "credits": 4.0},
            ),
            parent_messages=({"role": "assistant", "content": "parent"},),
            parent_depth=0,
            live_agent_ids=("agent.valid", "agent.sibling"),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=255,
        ),
    )

    children = _materialize_children(transition.effects)
    edges = _edge_summaries(transition.effects)
    assert [item.payload["agent_id"] for item in children] == ["agent.valid"]
    assert children[0].payload["credits"] == pytest.approx(6.0)
    assert [(item.payload["agent_id"], item.payload["reason"], item.payload["credits"]) for item in edges] == [
        ("agent.sibling", "agent_call_limit_exceeded", 0.0),
    ]
    # Rejected share never allocated; first live child retires parent + returns remainder.
    assert children[0].payload["retire_parent"] is True
    assert children[0].payload["pool_return"] == pytest.approx(4.0)
    # Call reservation consumed exactly one slot for the live child (255 → 256).
    assert transition.next_state.agent_calls_started == 256
    # Pool untouched by reducer until materialize applies; no invent on leaves.
    assert transition.next_state.credit_pool == pytest.approx(2.0)
    assert not any(isinstance(item, PerformAgentCall) for item in transition.effects)


# ---------------------------------------------------------------------------
# §9.2 / §23.1 — protocol invalid → one correction → terminate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spec_9_2_23_1_protocol_invalid_one_correction_then_terminate() -> None:
    """§9.2 / §23.1 — FakeLLM enqueue(invalid, valid+terminate); exactly one correction when credits allow."""

    invalid = _generation(response='{"report": 7, "procedures": [], "delegations": []}')
    valid = _generation(
        response=json.dumps(
            {
                "report": "corrected",
                "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}}],
                "delegations": [],
            },
            ensure_ascii=False,
        )
    )
    llm = QueuedFakeLLM(invalid, valid)
    worker = TurnWorker(llm, SimpleNamespace(), pricing=FakePricing())
    messages = (
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "formalized task"},
    )
    first_effect = PerformAgentCall(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        event_id="evt-call",
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "selector": "task:reasoning",
            "protocol": "json_envelope",
            "messages": messages,
            "estimated_charge": 0.5,
            "credits_after_reservation": 5.0,
            "correction_estimated_charge": 0.1,
            "max_correction_turns": 1,
            "pinning_supported": True,
            "branch_depth": 0,
            "live_agent_ids": (),
            "agent_calls_started": 1,
        },
    )

    first = await worker.perform_agent_call(first_effect)
    assert isinstance(first, AgentCallCompleted)
    assert first.protocol_error is not None
    assert first.protocol_result is None

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"branch-1": 5.5},
        agent_calls_started=1,
    )
    correction_transition = reduce_event(state, first)
    corrections = [item for item in correction_transition.effects if isinstance(item, PerformAgentCall)]
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction.payload["correction_count"] == 1
    assert correction.payload["call_id"] == "call-1:correction"
    assert correction.payload["selector"] == "model:physical-v1"
    assert correction.payload["messages"][:-1] == messages
    assert correction.payload["messages"][-1]["role"] == "user"
    assert "协议" in correction.payload["messages"][-1]["content"] or "report" in correction.payload[
        "messages"
    ][-1]["content"]
    assert correction_transition.next_state.agent_calls_started == 2

    second = await worker.perform_agent_call(correction)
    assert isinstance(second, AgentCallCompleted)
    assert second.protocol_error is None
    assert second.protocol_result is not None
    assert second.correction_count == 1

    second_transition = reduce_event(correction_transition.next_state, second)
    batches = [item for item in second_transition.effects if isinstance(item, PerformProcedureBatch)]
    assert len(batches) == 1
    assert any(item.get("procedure_id") == CORE_TERMINATE_ID for item in batches[0].payload["requests"])
    # No second correction scheduled after a valid envelope.
    assert not any(isinstance(item, PerformAgentCall) for item in second_transition.effects)

    # Procedure path with terminate control finalizes the branch.
    finalize = reduce_event(
        second_transition.next_state,
        ProcedureBatchCompleted(
            "evt-proc",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="branch-1",
            call_id="call-1:correction",
            result_id="result-final",
            credits_after=float(batches[0].payload["credits_after"]),
            controls=CoreProcedureDecision(terminate=True),
            delegations=(),
            parent_messages=messages,
            parent_depth=0,
            live_agent_ids=(),
            agent_calls_started=2,
        ),
    )
    summaries = [item for item in finalize.effects if isinstance(item, PerformBranchSummary)]
    assert len(summaries) == 1
    assert summaries[0].payload["reason"] == "terminate"
    assert len(llm.requests) == 2


# ---------------------------------------------------------------------------
# §9.3 — native submit_swarm_turn with empty assistant text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spec_9_3_native_submit_swarm_turn_empty_assistant_text_completes_turn() -> None:
    """§9.3 — native_tools path with empty assistant text completes via TurnWorker + FakeLLMGateway."""

    gateway = FakeLLMGateway()
    gateway.enqueue(
        FakeLLMResponse(
            text="",
            tool_calls=[
                {
                    "id": "call_native_1",
                    "function": {
                        "name": "submit_swarm_turn",
                        "arguments": {
                            "report": "",
                            "procedures": [],
                            "delegations": [],
                        },
                    },
                }
            ],
            usage={"prompt_tokens": 8, "completion_tokens": 1, "cache_hit_tokens": 0, "cache_miss_tokens": 8},
        )
    )
    worker = TurnWorker(gateway, SimpleNamespace(), pricing=FakePricing())
    effect = PerformAgentCall(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        event_id="evt-native",
        payload={
            "branch_id": "branch-1",
            "call_id": "call-native",
            "selector": "task:reasoning",
            "protocol": "native_tools",
            "messages": ({"role": "user", "content": "task"},),
            "estimated_charge": 0.5,
            "credits_after_reservation": 4.0,
        },
    )

    completed = await worker.perform_agent_call(effect)

    assert isinstance(completed, AgentCallCompleted)
    assert completed.protocol_error is None
    assert completed.protocol == "native_tools"
    assert completed.protocol_result is not None
    assert completed.protocol_result["report"] == ""
    assert list(completed.protocol_result["procedures"]) == []
    assert list(completed.protocol_result["delegations"]) == []
    assert gateway.calls[0]["tools"] is not None
    assert gateway.calls[0]["tools"][0]["function"]["name"] == "submit_swarm_turn"

    transition = reduce_event(
        RuntimeState(
            "task-1",
            TaskStatus.RUNNING,
            generation=0,
            active_round_id="round-1",
            active_leaves={"branch-1": 4.5},
        ),
        completed,
    )
    batches = [item for item in transition.effects if isinstance(item, PerformProcedureBatch)]
    assert len(batches) == 1
    assert batches[0].payload["report"] == ""
    assert batches[0].payload["delegations"] == ()
