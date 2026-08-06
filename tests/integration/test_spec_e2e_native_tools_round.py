"""§9.3 — native_tools full research round E2E (Wave3 Task 7).

Distinct from empty-assistant E10: tool-call shaped turns carry non-empty
structured ``submit_swarm_turn`` payloads (report + delegations / terminate),
then a report epoch produces a durable body (prefer ``report_id：``).
"""

from __future__ import annotations

from typing import Any

import pytest
from fakes import FakeLLMResponse

from lunagentic_research_swarm.models import ReportKind, TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall
from lunagentic_research_swarm.runtime.turns import TurnWorker

from .test_spec_e2e_protocol_and_recovery import (
    ROOT_AGENT,
    _ChargePricing,
    _materialize_effects,
    _native_tool_call,
    _procedure_effects,
    _run_procedure_batch,
    _summary_effects,
)


def _native_response(
    *,
    report: str,
    procedures: list[dict[str, Any]] | None = None,
    delegations: list[dict[str, Any]] | None = None,
    call_id: str = "call_native_round",
    model: str = "native-round-model",
) -> FakeLLMResponse:
    """Non-empty structured native tool payload; assistant prose may be empty."""

    tool = _native_tool_call(report=report, procedures=procedures, delegations=delegations)
    tool = {**tool, "id": call_id}
    return FakeLLMResponse(
        text="",
        tool_calls=[tool],
        model=model,
        usage={
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 12,
        },
    )


@pytest.mark.asyncio
async def test_e2e_native_tools_full_research_round_durable_report(runtime_harness) -> None:
    """§9.3 — native root delegates → child terminates → report epoch durable report."""

    harness = runtime_harness
    await harness.start("native 完整调研轮", credits=60.0, time_budget=90)
    await harness.formalize("正式 native 完整调研轮")
    status = await harness.manager.status(harness.task_id)
    root_id = str(status["active_leaves"][0]["branch_id"])
    credits = float(status["active_leaves"][0]["credits"])
    controller = harness.manager._controllers[harness.task_id]
    generation = int(status["generation"])

    worker = TurnWorker(harness.llm, harness.procedures, pricing=_ChargePricing(0.2))

    # --- Root: native_tools with non-empty report + delegation (not E10 empty terminate) ---
    harness.llm.enqueue(
        _native_response(
            report="native root：委派复核要点",
            procedures=[],
            delegations=[
                {
                    "agent_id": ROOT_AGENT,
                    "task": "用 native 工具完成子调研并终结",
                    "credits": 25.0,
                }
            ],
            call_id="call_native_root",
        )
    )
    root_effect = PerformAgentCall(
        harness.task_id,
        harness.round_id,
        generation,
        event_id=f"{harness.task_id}:native-root",
        payload={
            "branch_id": root_id,
            "call_id": "call-native-root",
            "selector": "task:reasoning",
            "protocol": "native_tools",
            "messages": (
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "formalized native round"},
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

    root_completed = await worker.perform_agent_call(root_effect)
    assert root_completed.protocol_error is None
    assert root_completed.protocol == "native_tools"
    assert root_completed.protocol_result is not None
    assert root_completed.protocol_result["report"] == "native root：委派复核要点"
    assert root_completed.protocol_result["delegations"][0]["agent_id"] == ROOT_AGENT
    assert harness.llm.calls[0]["tools"] is not None
    assert harness.llm.calls[0]["tools"][0]["function"]["name"] == "submit_swarm_turn"

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(root_completed)
    root_batches = _procedure_effects(harness.scheduler.enqueued)
    assert len(root_batches) == 1
    assert root_batches[0].payload["report"] == "native root：委派复核要点"

    root_batch = await _run_procedure_batch(harness, root_batches[0])
    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(root_batch)

    children = _materialize_effects(harness.scheduler.enqueued)
    assert len(children) == 1
    assert children[0].payload["agent_id"] == ROOT_AGENT
    assert children[0].payload["parent_branch_id"] == root_id
    assert children[0].payload["credits"] == pytest.approx(25.0)

    await harness.manager.materialize_child_effect(children[0])
    child_id = str(children[0].payload["branch_id"])
    assert child_id in controller.state.active_leaves
    assert child_id in harness.coordinator.branches

    # --- Child: native_tools non-empty report + terminate (terminal leaf) ---
    harness.llm.enqueue(
        _native_response(
            report="native child：子调研结论可交付",
            procedures=[{"procedure_id": CORE_TERMINATE_ID, "arguments": {}}],
            delegations=[],
            call_id="call_native_child",
        )
    )
    child_credits = float(controller.state.active_leaves[child_id])
    child_effect = PerformAgentCall(
        harness.task_id,
        harness.round_id,
        controller.state.generation,
        event_id=f"{harness.task_id}:native-child",
        payload={
            "branch_id": child_id,
            "call_id": "call-native-child",
            "selector": "task:reasoning",
            "protocol": "native_tools",
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
    assert child_completed.protocol == "native_tools"
    assert child_completed.protocol_result is not None
    assert child_completed.protocol_result["report"] == "native child：子调研结论可交付"
    assert any(
        item.get("procedure_id") == CORE_TERMINATE_ID
        for item in child_completed.protocol_result["procedures"]
    )

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

    # Terminal summary → empty live frontier opens FINAL report epoch (synthesis).
    await harness.manager.handle_branch_summary_effect(summaries[0])
    assert harness.coordinator is not None
    await harness.coordinator.wait_for_synthesis()

    assert len(harness.llm.calls) == 2
    assert all(call.get("tools") is not None for call in harness.llm.calls)
    assert harness.reports, "report epoch must produce at least one report"
    assert harness.reports[-1].kind is ReportKind.FINAL
    assert harness.reports[-1].text and harness.reports[-1].text.strip()
    assert harness.task_status in {TaskStatus.COMPLETED, TaskStatus.FINALIZING, TaskStatus.COMPLETED_WITH_ERRORS}

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None and layer.reports
    durable = layer.reports[-1]
    assert str(durable["kind"]) == "FINAL"
    durable_text = str(durable["text"] or "")
    assert durable_text.strip()
    report_id = str(durable["report_id"])
    assert report_id
    assert f"report_id：{report_id}" in durable_text
