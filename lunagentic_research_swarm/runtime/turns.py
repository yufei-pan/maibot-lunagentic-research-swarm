"""普通智能体 worker 与完成 turn 的确定性分支解析。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import Any

from lunagentic_research_swarm.llm.gateway import GenerationRequest
from lunagentic_research_swarm.llm.protocol import (
    SWARM_TURN_TOOLS,
    DelegationRequest,
    ProtocolError,
    SwarmTurnEnvelope,
    parse_json_envelope,
    parse_native_tool_result,
)
from lunagentic_research_swarm.models import BranchRuntime
from lunagentic_research_swarm.procedures.core import split_procedure_requests
from lunagentic_research_swarm.runtime.credits import allocate_children
from lunagentic_research_swarm.runtime.events import AgentCallCompleted, AgentCallFailed, ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall, PerformProcedureBatch


@dataclass(frozen=True, slots=True)
class TurnLimits:
    max_delegations_per_turn: int = 8
    max_branch_depth: int = 32
    max_agent_calls_per_task: int = 256

    def __post_init__(self) -> None:
        for name in ("max_delegations_per_turn", "max_branch_depth", "max_agent_calls_per_task"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须为正整数")


@dataclass(frozen=True, slots=True)
class ChildLaunch:
    branch_id: str
    agent_id: str
    assignment: str
    credits: float
    depth: int
    messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class EdgeFinalization:
    agent_id: str
    assignment: str
    reason: str
    credits: float
    messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CompletedTurnResult:
    children: tuple[ChildLaunch, ...] = ()
    edge_finalizations: tuple[EdgeFinalization, ...] = ()
    finalize_reason: str | None = None
    credits_after_call: float = 0.0
    held_for_checkpoint: bool = False


def _edge_messages(branch: BranchRuntime, request: DelegationRequest, reason: str) -> tuple[Mapping[str, Any], ...]:
    messages = [dict(message) for message in branch.build_prompt_messages()]
    messages.append(
        {
            "role": "user",
            "content": f"assignment: {request.task}\nedge finalized: {reason}",
        }
    )
    return tuple(messages)


def _limit_reason(
    *,
    index: int,
    branch: BranchRuntime,
    limits: TurnLimits,
    agent_calls_started: int,
) -> str | None:
    if index >= limits.max_delegations_per_turn:
        return "delegation_limit_exceeded"
    if branch.depth >= limits.max_branch_depth:
        return "branch_depth_exceeded"
    available_calls = max(0, limits.max_agent_calls_per_task - agent_calls_started)
    if index >= available_calls:
        return "agent_call_limit_exceeded"
    return None


async def _invoke_ordinary(procedures: Any, requests: Sequence[Any]) -> None:
    if not requests:
        return
    invoke = getattr(procedures, "invoke", None)
    if callable(invoke):
        await invoke(tuple(requests))
        return
    invoke_many = getattr(procedures, "invoke_many", None)
    if callable(invoke_many):
        await invoke_many(tuple(requests))
        return
    raise TypeError("Procedure executor 必须实现 invoke 或 invoke_many")


async def resolve_completed_turn(
    *,
    branch: BranchRuntime,
    envelope: SwarmTurnEnvelope,
    actual_charge: float,
    procedures: Any,
    live_agents: Set[str] | set[str] | frozenset[str],
    limits: TurnLimits | None = None,
    agent_calls_started: int = 0,
) -> CompletedTurnResult:
    """按固定顺序处理 Procedure、余额、控制和 children。

    这个函数不写 store；controller/reducer 可在 transaction 成功后发布其结果。
    """

    active_limits = limits or TurnLimits()
    branch.credits -= float(actual_charge)
    ordinary, controls = split_procedure_requests(envelope.procedures)
    await _invoke_ordinary(procedures, ordinary)

    if controls.terminate:
        return CompletedTurnResult(finalize_reason="terminate", credits_after_call=branch.credits)
    if branch.credits < 0:
        return CompletedTurnResult(finalize_reason="negative_credit", credits_after_call=branch.credits)
    if not envelope.delegations:
        if controls.checkpoint:
            branch.pending_delegations.clear()
            return CompletedTurnResult(credits_after_call=branch.credits, held_for_checkpoint=True)
        return CompletedTurnResult(finalize_reason="no_further_work", credits_after_call=branch.credits)

    structurally_valid: list[tuple[int, DelegationRequest]] = []
    rejected: list[tuple[DelegationRequest, str]] = []
    for index, request in enumerate(envelope.delegations):
        reason = _limit_reason(
            index=index,
            branch=branch,
            limits=active_limits,
            agent_calls_started=agent_calls_started,
        )
        if reason is None:
            structurally_valid.append((index, request))
        else:
            rejected.append((request, reason))

    allocations = allocate_children(
        branch.credits,
        [(str(index), request.credits) for index, request in structurally_valid],
    )
    children: list[ChildLaunch] = []
    edge_finalizations: list[EdgeFinalization] = [
        EdgeFinalization(
            request.agent_id,
            request.task,
            reason,
            0.0,
            _edge_messages(branch, request, reason),
        )
        for request, reason in rejected
    ]
    for index, request in structurally_valid:
        credits = allocations.allocations[str(index)]
        child_messages = [dict(message) for message in branch.build_prompt_messages()]
        child_messages.append({"role": "user", "content": f"assignment: {request.task}"})
        if request.agent_id not in live_agents:
            reason = "agent_unavailable"
            edge_finalizations.append(
                EdgeFinalization(
                    request.agent_id,
                    request.task,
                    reason,
                    credits,
                    _edge_messages(branch, request, reason),
                )
            )
            continue
        children.append(
            ChildLaunch(
                branch_id=f"{branch.branch_id}:{index + 1}",
                agent_id=request.agent_id,
                assignment=request.task,
                credits=credits,
                depth=branch.depth + 1,
                messages=tuple(child_messages),
            )
        )

    if controls.checkpoint:
        branch.pending_delegations[:] = [request.model_dump(mode="python") for _, request in structurally_valid]
        return CompletedTurnResult(
            edge_finalizations=tuple(edge_finalizations),
            credits_after_call=branch.credits,
            held_for_checkpoint=True,
        )
    return CompletedTurnResult(
        children=tuple(children),
        edge_finalizations=tuple(edge_finalizations),
        credits_after_call=branch.credits,
    )


class TurnWorker:
    """只执行外部调用与协议解析；不修改 branch 或 store。"""

    def __init__(self, llm: Any, procedures: Any, *, pricing: Any | None = None) -> None:
        self.llm = llm
        self.procedures = procedures
        self.pricing = pricing

    async def perform_agent_call(self, effect: PerformAgentCall) -> AgentCallCompleted | AgentCallFailed:
        payload = effect.payload
        protocol = str(payload.get("protocol", "json_envelope"))
        tools = SWARM_TURN_TOOLS if protocol == "native_tools" else None
        request = GenerationRequest(
            selector=str(payload.get("selector", "")),
            messages=payload.get("messages", ()),
            tools=tools,
            temperature=payload.get("temperature"),
            max_tokens=payload.get("max_tokens"),
        )
        result = await self.llm.generate(request)
        common = {
            "event_id": str(payload.get("completed_event_id") or f"{effect.event_id}:completed"),
            "task_id": effect.task_id,
            "round_id": str(effect.round_id or ""),
            "generation": effect.generation,
            "branch_id": str(payload.get("branch_id", "")),
            "call_id": str(payload.get("call_id", "")),
        }
        usage = None
        if result.usage is not None:
            usage = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cache_hit_tokens": result.usage.cache_hit_tokens,
                "cache_miss_tokens": result.usage.cache_miss_tokens,
            }
        actual_charge: float | None = None
        if self.pricing is not None and result.usage is not None:
            charged = self.pricing.charge_actual(
                actual_model_name=result.model_name,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cache_hit_tokens=result.usage.cache_hit_tokens,
                cache_miss_tokens=result.usage.cache_miss_tokens,
            )
            actual_charge = float(charged.credits)
        if not result.success:
            error = result.error
            return AgentCallFailed(
                **common,
                error_code=error.code if error is not None else "llm_generation_failed",
                error_message=error.message if error is not None else "LLM 调用失败",
                usage=usage,
                actual_model_name=result.model_name,
                actual_charge=actual_charge,
                estimated_charge=float(payload.get("estimated_charge", 0.0)),
                balance_before_reconciliation=float(payload.get("credits_after_reservation", 0.0)),
                selector=str(payload.get("selector", "")),
            )

        envelope: SwarmTurnEnvelope | None = None
        protocol_error: Mapping[str, Any] | None = None
        try:
            if protocol == "native_tools":
                envelope = parse_native_tool_result(result.response, result.tool_calls)
            else:
                envelope = parse_json_envelope(result.response)
        except ProtocolError as exc:
            protocol_error = {"message": exc.message, "errors": list(exc.errors)}
        return AgentCallCompleted(
            **common,
            result_id=str(payload.get("result_id") or f"{common['call_id']}:result"),
            usage=usage,
            actual_model_name=result.model_name,
            actual_charge=actual_charge,
            protocol_result=envelope.model_dump(mode="python") if envelope is not None else None,
            protocol_error=protocol_error,
            correction_count=int(payload.get("correction_count", 0)),
            estimated_charge=float(payload.get("estimated_charge", 0.0)),
            balance_before_reconciliation=float(payload.get("credits_after_reservation", 0.0)),
            pinning_supported=bool(payload.get("pinning_supported", True)),
            messages=tuple(payload.get("messages", ())),
            branch_depth=int(payload.get("branch_depth", 0)),
            live_agent_ids=payload.get("live_agent_ids"),
            max_delegations_per_turn=int(payload.get("max_delegations_per_turn", 8)),
            max_branch_depth=int(payload.get("max_branch_depth", 32)),
            max_agent_calls_per_task=int(payload.get("max_agent_calls_per_task", 256)),
            agent_calls_started=int(payload.get("agent_calls_started", 0)),
        )

    async def perform_procedure_batch(self, effect: PerformProcedureBatch) -> ProcedureBatchCompleted:
        completed = await self.procedures.invoke_many(effect)
        if not isinstance(completed, ProcedureBatchCompleted):
            raise TypeError("Procedure executor 必须返回 ProcedureBatchCompleted")
        payload = effect.payload
        return replace(
            completed,
            report=str(payload.get("report", "")),
            delegations=tuple(payload.get("delegations", ())),
            credits_after=float(payload.get("credits_after", 0.0)),
            parent_messages=tuple(payload.get("messages", ())),
            parent_depth=int(payload.get("branch_depth", 0)),
            live_agent_ids=payload.get("live_agent_ids"),
            max_delegations_per_turn=int(payload.get("max_delegations_per_turn", 8)),
            max_branch_depth=int(payload.get("max_branch_depth", 32)),
            max_agent_calls_per_task=int(payload.get("max_agent_calls_per_task", 256)),
            agent_calls_started=int(payload.get("agent_calls_started", 0)),
        )


__all__ = [
    "ChildLaunch",
    "CompletedTurnResult",
    "EdgeFinalization",
    "TurnLimits",
    "TurnWorker",
    "resolve_completed_turn",
]
