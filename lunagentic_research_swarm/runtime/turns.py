"""普通智能体 worker：只执行外部调用与协议解析。

turn 完成后的分支解析（Procedure、credits 终止、委派与 children）由
``runtime.reducer._delegation_effects`` 独占，符合 spec §5.2 不变量 9：
所有状态转移必须由单一 reducer 完成，worker 不得修改权威任务状态。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from lunagentic_research_swarm.llm.gateway import GenerationRequest
from lunagentic_research_swarm.llm.protocol import (
    SWARM_TURN_TOOLS,
    ProtocolError,
    SwarmTurnEnvelope,
    parse_json_envelope_with_repairs,
    parse_native_tool_result_with_repairs,
)
from lunagentic_research_swarm.procedures.billing import extract_research_credits_charged
from lunagentic_research_swarm.runtime.events import AgentCallCompleted, AgentCallFailed, ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall, PerformProcedureBatch

_LOG = logging.getLogger(__name__)


class TurnWorker:
    """只执行外部调用与协议解析；不修改 branch 或 store。"""

    def __init__(
        self,
        llm: Any,
        procedures: Any,
        *,
        pricing: Any | None = None,
        procedure_factory: Callable[[Any], Any] | None = None,
        debug_store: Any | None = None,
    ) -> None:
        self.llm = llm
        self.procedures = procedures
        self.pricing = pricing
        self.procedure_factory = procedure_factory
        self.debug_store = debug_store

    async def _maybe_save_transcript(
        self,
        *,
        task_id: str,
        round_id: str,
        branch_id: str,
        turn_id: str,
        messages: Any,
        envelope: Mapping[str, Any] | None,
    ) -> None:
        store = self.debug_store
        if store is None or not getattr(store, "store_agent_transcripts", False):
            return
        save = getattr(store, "save_transcript", None)
        if not callable(save):
            return
        try:
            await save(
                task_id=task_id,
                round_id=round_id,
                branch_id=branch_id,
                turn_id=turn_id,
                messages=messages or (),
                envelope=envelope,
            )
        except Exception:
            # DebugStore 通常自行吞掉异常；若调用方覆写抛出，仍不得影响权威 turn。
            record = getattr(store, "_record_failure", None)
            if callable(record):
                try:
                    await record(
                        kind="transcript",
                        error=RuntimeError("debug transcript 写入失败"),
                        task_id=task_id,
                        round_id=round_id,
                    )
                except Exception:
                    pass

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
            await self._maybe_save_transcript(
                task_id=common["task_id"],
                round_id=common["round_id"],
                branch_id=common["branch_id"],
                turn_id=common["call_id"],
                messages=payload.get("messages", ()),
                envelope={"error_code": error.code if error is not None else "llm_generation_failed"},
            )
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
        protocol_repairs: tuple[str, ...] = ()
        try:
            if protocol == "native_tools":
                envelope, protocol_repairs = parse_native_tool_result_with_repairs(
                    result.response, result.tool_calls
                )
            else:
                envelope, protocol_repairs = parse_json_envelope_with_repairs(result.response)
        except ProtocolError as exc:
            protocol_error = {"message": exc.message, "errors": list(exc.errors)}
        if protocol_repairs:
            # spec §9.2：成功的有限修复必须留下 protocol_repaired 记录。
            _LOG.info(
                "protocol_repaired=true task_id=%s branch_id=%s call_id=%s protocol=%s rules=%s",
                common["task_id"],
                common["branch_id"],
                common["call_id"],
                protocol,
                ",".join(protocol_repairs),
            )
        envelope_payload = envelope.model_dump(mode="python") if envelope is not None else protocol_error
        await self._maybe_save_transcript(
            task_id=common["task_id"],
            round_id=common["round_id"],
            branch_id=common["branch_id"],
            turn_id=common["call_id"],
            messages=payload.get("messages", ()),
            envelope=envelope_payload if isinstance(envelope_payload, Mapping) else None,
        )
        return AgentCallCompleted(
            **common,
            result_id=str(payload.get("result_id") or f"{common['call_id']}:result"),
            usage=usage,
            actual_model_name=result.model_name,
            actual_charge=actual_charge,
            protocol=protocol,
            protocol_result=envelope.model_dump(mode="python") if envelope is not None else None,
            protocol_error=protocol_error,
            protocol_repairs=protocol_repairs,
            correction_count=int(payload.get("correction_count", 0)),
            max_correction_turns=int(payload.get("max_correction_turns", 1)),
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
        procedures = self.procedures
        catalog = effect.payload.get("procedure_catalog")
        if catalog is not None and self.procedure_factory is not None:
            procedures = self.procedure_factory(catalog)
        completed = await procedures.invoke_many(effect)
        if not isinstance(completed, ProcedureBatchCompleted):
            raise TypeError("Procedure executor 必须返回 ProcedureBatchCompleted")
        payload = effect.payload
        parent_messages = completed.parent_messages or tuple(payload.get("messages", ()))
        prior = float(payload.get("credits_after", 0.0))
        charged = sum(extract_research_credits_charged(getattr(item, "result", None)) for item in completed.results)
        return replace(
            completed,
            report=str(payload.get("report", "")),
            delegations=tuple(payload.get("delegations", ())),
            credits_after=prior - charged,
            parent_messages=parent_messages,
            parent_depth=int(payload.get("branch_depth", 0)),
            agent_id=str(payload.get("agent_id", "") or ""),
            live_agent_ids=payload.get("live_agent_ids"),
            max_delegations_per_turn=int(payload.get("max_delegations_per_turn", 8)),
            max_branch_depth=int(payload.get("max_branch_depth", 32)),
            max_agent_calls_per_task=int(payload.get("max_agent_calls_per_task", 256)),
            agent_calls_started=int(payload.get("agent_calls_started", 0)),
        )


__all__ = ["TurnWorker"]
