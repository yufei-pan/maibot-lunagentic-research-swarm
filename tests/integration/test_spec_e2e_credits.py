"""E2E credits path via real allocate_children + ChildMaterialized (design §25.3 / §11 E1/E2).

``RuntimeHarness.root_delegates`` is synthetic — these cases drive the controller
materialize / TurnWorker / reducer path instead.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeLLMResponse

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformAgentCall,
    PerformBranchSummary,
    PerformProcedureBatch,
)
from lunagentic_research_swarm.runtime.turns import TurnWorker


class _ChargePricing:
    """Deterministic nonzero actual charge for zero-credit child scenarios."""

    def __init__(self, credits: float = 1.5) -> None:
        self.credits = float(credits)

    def charge_actual(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(credits=self.credits)


def _procedure_catalog(*procedure_ids: str) -> ProcedureCatalogSnapshot:
    entries = []
    for procedure_id in procedure_ids:
        definition = ProcedureDefinition.model_validate(
            {
                "procedure_id": procedure_id,
                "version": "1",
                "display_name": procedure_id,
                "description": "e2e credits procedure",
                "arguments_schema": {"type": "object"},
                "result_schema": {"type": "object"},
                "idempotent": False,
                "timeout_seconds": 30.0,
            }
        )
        entries.append(
            ProcedureCatalogEntry(definition, "fake", "fake.invoke", "1", "e2e-proc-fingerprint")
        )
    return ProcedureCatalogSnapshot(entries)


def _materialize_effects(enqueued: list[Any]) -> list[NotifyToolWaiter]:
    return [
        item
        for item in enqueued
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]


def _summary_effects(enqueued: list[Any]) -> list[PerformBranchSummary]:
    return [item for item in enqueued if isinstance(item, PerformBranchSummary)]


@pytest.mark.asyncio
async def test_e2e_e1_root_allocates_abc_50_25_25_via_real_materialize(runtime_harness) -> None:
    """§25.3#1 / E1 — root 100 → A/B/C = 50/25/25 through allocate + ChildMaterialized."""

    harness = runtime_harness
    await harness.start("三路 credits 守恒", credits=100.0, time_budget=120)
    await harness.formalize("正式三路 credits 守恒")

    before = await harness.manager.status(harness.task_id)
    assert len(before["active_leaves"]) == 1
    root_credits = float(before["active_leaves"][0]["credits"])
    assert root_credits == pytest.approx(100.0)
    initial_total = root_credits + float(
        harness.manager._controllers[harness.task_id].state.credit_pool
    )

    leaves = await harness.root_allocates_real({"A": 50.0, "B": 25.0, "C": 25.0})

    assert set(leaves) == {"A", "B", "C"}
    assert leaves["A"] == pytest.approx(50.0)
    assert leaves["B"] == pytest.approx(25.0)
    assert leaves["C"] == pytest.approx(25.0)
    # Exact request sum == parent balance → no pool remainder; leaf sum conserved.
    controller = harness.manager._controllers[harness.task_id]
    assert controller.state.credit_pool == pytest.approx(0.0)
    assert math.fsum(leaves.values()) + controller.state.credit_pool == pytest.approx(initial_total)
    # Parent retired by first live materialize (real ChildMaterialized path).
    assert harness._root_branch_id not in leaves

    status = await harness.manager.status(harness.task_id)
    by_id = {item["branch_id"]: float(item["credits"]) for item in status["active_leaves"]}
    assert set(by_id) == {"A", "B", "C"}
    assert by_id["A"] == pytest.approx(50.0)
    assert by_id["B"] == pytest.approx(25.0)
    assert by_id["C"] == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_e2e_e2_zero_credit_child_runs_procedures_then_blocks_grandchildren(
    runtime_harness,
) -> None:
    """§25.3#4 / E2 — zero child + FakeLLM cost + procedures → finalize, no grandchild."""

    harness = runtime_harness
    await harness.start("零余额后代", credits=100.0, time_budget=120)
    await harness.formalize("正式零余额后代")

    # Requested credits 0 → allocate_children launches child at 0; remainder → pool.
    leaves = await harness.root_allocates_real({"zero": 0.0})
    assert leaves["zero"] == pytest.approx(0.0)
    controller = harness.manager._controllers[harness.task_id]
    assert controller.state.credit_pool == pytest.approx(100.0)

    catalog = _procedure_catalog("fake.search")
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "zero child worked",
                "procedures": [{"procedure_id": "fake.search", "arguments": {"query": "资料"}}],
                "delegations": [
                    {"agent_id": "agent.grandchild", "task": "must not launch", "credits": 5.0},
                ],
            },
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 100,
            },
        )
    )
    pricing = _ChargePricing(1.5)
    worker = TurnWorker(harness.llm, harness.procedures, pricing=pricing)

    agent_completed = await worker.perform_agent_call(
        PerformAgentCall(
            harness.task_id,
            harness.round_id,
            controller.state.generation,
            event_id=f"{harness.task_id}:zero-call",
            payload={
                "selector": "model:gpt-5.6-luna-max",
                "messages": ({"role": "user", "content": "零余额调查"},),
                "branch_id": "zero",
                "call_id": "call-zero",
                "estimated_charge": 0.0,
                "credits_after_reservation": 0.0,
                "branch_depth": 1,
                "live_agent_ids": ("agent.grandchild",),
                "agent_calls_started": int(controller.state.agent_calls_started),
            },
        )
    )
    assert agent_completed.actual_charge == pytest.approx(1.5)
    assert agent_completed.protocol_result is not None
    assert agent_completed.protocol_result["procedures"]
    assert agent_completed.protocol_result["delegations"]

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(agent_completed)

    assert controller.state.active_leaves["zero"] == pytest.approx(-1.5)
    procedure_effects = [
        item for item in harness.scheduler.enqueued if isinstance(item, PerformProcedureBatch)
    ]
    assert len(procedure_effects) == 1
    # Successful protocol still schedules procedures even after balance goes negative.
    assert not _materialize_effects(harness.scheduler.enqueued)
    assert not _summary_effects(harness.scheduler.enqueued)

    procedure_effect = procedure_effects[0]
    assert procedure_effect.payload["credits_after"] == pytest.approx(-1.5)
    assert procedure_effect.payload["delegations"]
    assert procedure_effect.payload["requests"]

    executor = ProcedureExecutor(catalog, harness.procedures)
    batch_raw = await executor.invoke_many(procedure_effect)
    # Mirror TurnWorker.perform_procedure_batch merge so reducer sees credits/delegations.
    batch = replace(
        batch_raw,
        report=str(procedure_effect.payload.get("report", "")),
        delegations=tuple(procedure_effect.payload.get("delegations", ())),
        credits_after=float(procedure_effect.payload.get("credits_after", 0.0)),
        parent_messages=batch_raw.parent_messages
        or tuple(procedure_effect.payload.get("messages", ())),
        parent_depth=int(procedure_effect.payload.get("branch_depth", 0)),
        live_agent_ids=procedure_effect.payload.get("live_agent_ids"),
        agent_calls_started=int(procedure_effect.payload.get("agent_calls_started", 0)),
    )
    assert batch.results
    assert batch.results[0].success is True
    assert any(
        call.get("operation") == "call" and call.get("procedure_id") == "fake.search"
        for call in harness.procedures.requests
    )

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(batch)

    summaries = _summary_effects(harness.scheduler.enqueued)
    assert len(summaries) == 1
    assert summaries[0].payload["branch_id"] == "zero"
    assert summaries[0].payload["reason"] == "negative_credit"
    assert _materialize_effects(harness.scheduler.enqueued) == []
    # Leaf stays negative until BranchFinalized; no grandchild leaf invented.
    assert controller.state.active_leaves.get("zero") == pytest.approx(-1.5)
    assert not any(key.startswith("zero:") for key in controller.state.active_leaves)
    assert "agent.grandchild" not in {
        (harness.manager._branches.get(harness.task_id) or {}).get(bid, {}).get("agent_id")
        for bid in controller.state.active_leaves
    }
