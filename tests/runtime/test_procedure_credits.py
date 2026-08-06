"""Procedure batch research-credit debit + ledger persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CoreProcedureDecision
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.credits import charge_procedure_usage
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch, RuntimeState, reduce_event
from lunagentic_research_swarm.runtime.turns import TurnWorker

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def _definition(procedure_id: str) -> ProcedureDefinition:
    return ProcedureDefinition.model_validate(
        {
            "procedure_id": procedure_id,
            "version": "7",
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


class _FakeAPI:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def call(self, name: str, *, version: str = "", **kwargs: Any) -> Any:
        del name, version
        self.calls.append(kwargs)
        return self.response


def _effect(*, credits_after: float, request: ProcedureRequest) -> PerformProcedureBatch:
    return PerformProcedureBatch(
        task_id="task-1",
        round_id="round-1",
        generation=0,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "agent_id": "agent.reader",
            "credits_after": credits_after,
            "requests": [request],
        },
    )


@pytest.mark.asyncio
async def test_procedure_batch_debits_research_credits_charged() -> None:
    """Fake handler returns research_credits_charged=3.5; payload credits_after=10."""

    api = _FakeAPI(
        {
            "success": True,
            "data": {},
            "error": None,
            "metadata": {},
            "research_credits_charged": 3.5,
        }
    )
    executor = ProcedureExecutor(_catalog("builtin.search"), api=api)
    effect = _effect(
        credits_after=10.0,
        request=ProcedureRequest(procedure_id="builtin.search", credits=0.0),
    )

    completed = await TurnWorker(object(), executor).perform_procedure_batch(effect)

    assert completed.credits_after == pytest.approx(6.5)


@pytest.mark.asyncio
async def test_procedure_batch_missing_charge_bills_zero() -> None:
    """Legacy-shaped result with default charge 0 leaves credits_after unchanged."""

    api = _FakeAPI({"success": True, "data": {}, "error": None, "metadata": {}})
    executor = ProcedureExecutor(_catalog("builtin.search"), api=api)
    effect = _effect(
        credits_after=7.0,
        request=ProcedureRequest(procedure_id="builtin.search"),
    )

    completed = await TurnWorker(object(), executor).perform_procedure_batch(effect)

    assert completed.credits_after == pytest.approx(7.0)


def test_charge_procedure_usage_debits_with_negative_amount() -> None:
    """Mirror agent reservation sign: spend is a negative ledger amount."""

    result = charge_procedure_usage(
        3.5,
        task_id="task-1",
        round_id="round-1",
        branch_id="branch-1",
        call_id="call-1",
        procedure_id="builtin.search",
        budget_hint=4.0,
        balance_after=6.5,
        created_at=NOW,
    )

    assert result.ledger_command is not None
    assert result.ledger_command.kind == "insert_credit_ledger"
    assert result.ledger_command.values["entry_kind"] == "procedure_charge"
    assert result.ledger_command.values["amount"] == pytest.approx(-3.5)
    assert result.ledger_command.values["balance_after"] == pytest.approx(6.5)


def test_procedure_batch_completed_persists_procedure_charge_ledger() -> None:
    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"branch-1": 10.0},
    )
    event = ProcedureBatchCompleted(
        "evt-batch",
        "task-1",
        "round-1",
        0,
        occurred_at=NOW,
        branch_id="branch-1",
        call_id="call-1",
        result_id="result-1",
        results=(
            {
                "procedure_id": "builtin.search",
                "request_id": "req-1",
                "result": ProcedureResult(
                    success=True,
                    data={},
                    error=None,
                    metadata={"agent_id": "agent.reader"},
                    research_credits_charged=3.5,
                ).model_dump(mode="json"),
                "provider_plugin_id": "provider.tools",
                "api_name": "provider.tools.invoke_procedure",
                "api_version": "1",
                "attempts": 1,
                "duration_ms": 1,
            },
        ),
        controls=CoreProcedureDecision(),
        credits_after=6.5,
        parent_messages=({"role": "user", "content": "task"},),
    )

    transition = reduce_event(state, event)

    ledger = [cmd for cmd in transition.commands if cmd.kind == "insert_credit_ledger"]
    assert len(ledger) == 1
    assert ledger[0].values["entry_kind"] == "procedure_charge"
    assert ledger[0].values["amount"] == pytest.approx(-3.5)
    assert ledger[0].values["balance_after"] == pytest.approx(6.5)
    assert transition.next_state.active_leaves["branch-1"] == pytest.approx(6.5)
