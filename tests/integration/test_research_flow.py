from __future__ import annotations

import pytest
from fakes import FakeLLMResponse

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.models import ReportKind, TaskStatus
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall, PerformProcedureBatch
from lunagentic_research_swarm.runtime.turns import TurnWorker


@pytest.mark.asyncio
async def test_complete_branching_research_flow(runtime_harness) -> None:
    harness = runtime_harness
    task = await harness.start("比较两个方案", credits=100.0, time_budget=120)
    await harness.formalize("正式任务")
    await harness.root_delegates({"A": 50.0, "B": 25.0, "C": 25.0})
    assert harness.raw_context_count > 0
    await harness.branch_checkpoint("A")

    harness.clock.advance(120)
    await harness.run_until_idle()

    assert task["task_id"] == harness.task_id
    assert harness.reports[0].kind is ReportKind.INTERMEDIATE
    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None and layer.reports[0]["running_branch_count"] > 0
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert await harness.pending_outbox_count() == 1

    await harness.finalize_all()

    assert harness.reports[-1].kind is ReportKind.FINAL
    assert harness.task_status is TaskStatus.COMPLETED
    assert harness.raw_context_count == 0
    assert all(not branch.messages for branch in harness.coordinator.branches.values())
    assert not harness.manager._branches.get(harness.task_id)
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE", "FINAL"]
    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None
    durable = repr(layer)
    assert "比较两个方案" not in durable
    assert "raw_payload" not in durable
    assert "transcript" not in durable
    assert "reasoning" not in durable

    assert await harness.deliver_outbox() == 4
    assert len(harness.maisaka.append_calls) == 2
    assert len(harness.maisaka.trigger_calls) == 2
    assert await harness.pending_outbox_count() == 0


@pytest.mark.asyncio
async def test_deterministic_llm_and_procedure_providers_record_scoped_calls(runtime_harness) -> None:
    harness = runtime_harness
    harness.llm.enqueue(FakeLLMResponse(payload={"report": "ok"}, model="gpt-5.6-luna-max"))

    response = await harness.llm.generate(
        selector="model:gpt-5.6-luna-max", messages=[{"role": "user", "content": "调查"}], request_id="call-1"
    )
    await harness.procedures.invoke_procedure(
        request_id="proc-1", metadata={"task_id": "task-1", "branch_id": "branch-a"}, query="资料"
    )

    assert response.payload == {"report": "ok"}
    assert harness.llm.calls[0]["selector"] == "model:gpt-5.6-luna-max"
    assert harness.procedures.requests[-1]["request_id"] == "proc-1"
    assert harness.procedures.requests[-1]["metadata"]["branch_id"] == "branch-a"


@pytest.mark.asyncio
async def test_fakes_match_turn_worker_and_procedure_executor_contracts(runtime_harness) -> None:
    harness = runtime_harness
    harness.llm.enqueue(FakeLLMResponse(payload={"report": "turn", "procedures": [], "delegations": []}))
    worker = TurnWorker(harness.llm, harness.procedures)
    agent_event = await worker.perform_agent_call(
        PerformAgentCall(
            "task-1",
            "round-1",
            0,
            event_id="agent-event",
            payload={
                "selector": "model:gpt-5.6-luna-max",
                "messages": ({"role": "user", "content": "调查"},),
                "branch_id": "branch-a",
                "call_id": "call-1",
                "credits_after_reservation": 10.0,
            },
        )
    )
    assert agent_event.protocol_result is not None
    assert agent_event.protocol_result["report"] == "turn"
    procedure_completion = await worker.perform_procedure_batch(
        PerformProcedureBatch(
            "task-1", "round-1", 0, event_id="worker-procedure", payload={"branch_id": "branch-a"}
        )
    )
    assert procedure_completion.branch_id == "branch-a"

    definition = ProcedureDefinition(
        procedure_id="fake.search",
        version="1",
        display_name="Fake search",
        description="deterministic test procedure",
        arguments_schema={"type": "object"},
        result_schema={"type": "object"},
    )
    catalog = ProcedureCatalogSnapshot(
        [ProcedureCatalogEntry(definition, "fake", "fake.invoke", "1", "fingerprint")]
    )
    executor = ProcedureExecutor(catalog, harness.procedures)
    procedure_event = await executor.invoke_many(
        PerformProcedureBatch(
            "task-1",
            "round-1",
            0,
            event_id="procedure-event",
            payload={
                "procedure_requests": ({"procedure_id": "fake.search", "arguments": {"query": "资料"}},),
                "branch_id": "branch-a",
                "call_id": "call-1",
                "turn_id": "turn-1",
                "agent_id": "builtin.researcher",
            },
        )
    )
    assert procedure_event.results[0].success is True
    call = harness.procedures.requests[-1]
    assert call["operation"] == "call"
    assert call["scoped_metadata"] == {
        "task_id": "task-1",
        "round_id": "round-1",
        "branch_id": "branch-a",
        "turn_id": "turn-1",
        "agent_id": "builtin.researcher",
    }


@pytest.mark.asyncio
async def test_fake_maisaka_keeps_append_and_trigger_failures_separate(runtime_harness) -> None:
    harness = runtime_harness
    harness.maisaka.append_error = RuntimeError("append down")
    with pytest.raises(RuntimeError, match="append down"):
        await harness.maisaka.context.append("stream", [{"type": "text", "content": "report"}])
    assert harness.maisaka.append_calls == []
    assert harness.maisaka.trigger_calls == []

    harness.maisaka.append_error = None
    await harness.maisaka.context.append("stream", [{"type": "text", "content": "report"}])
    harness.maisaka.trigger_error = RuntimeError("trigger down")
    with pytest.raises(RuntimeError, match="trigger down"):
        await harness.maisaka.proactive.trigger("stream", "review")
    assert len(harness.maisaka.append_calls) == 1
    assert harness.maisaka.trigger_calls == []


@pytest.mark.asyncio
async def test_runtime_harness_releases_sqlite_resources_on_close(tmp_path) -> None:
    from fakes import RuntimeHarness

    harness = RuntimeHarness(tmp_path)
    await harness.open()
    await harness.close()

    assert harness.resources_closed
