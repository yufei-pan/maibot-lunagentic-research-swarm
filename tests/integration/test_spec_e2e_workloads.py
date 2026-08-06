"""End-to-end simulated workloads using FakeLLMGateway / FakeSummarizer (no Host LLM).

These stitch RuntimeHarness report/outbox paths with multi-branch research shapes
from design §7 / §13 / §17 / §25.2.
"""

from __future__ import annotations

import pytest
from fakes import FakeLLMResponse

from lunagentic_research_swarm.models import ReportKind, TaskStatus
from lunagentic_research_swarm.runtime.events import ReportDeadlineReached
from lunagentic_research_swarm.storage.outbox import MaisakaOutbox


@pytest.mark.asyncio
async def test_e2e_intermediate_then_final_with_outbox_delivery(runtime_harness) -> None:
    """§13 + §17.2 — intermediate report, finalize all leaves, deliver append+trigger.

    Drive the deadline through the manager (not an early all-checkpoint open) so
    REPORTING → FinalEpochCommitted bookkeeping can reach COMPLETED.
    """

    harness = runtime_harness
    await harness.start("比较方案 A/B", credits=100.0, time_budget=120)
    await harness.formalize("正式比较任务")
    await harness.root_delegates({"A": 50.0, "B": 50.0})
    # Checkpoint only one leaf so the frontier opens via ReportDeadlineReached.
    await harness.branch_checkpoint("A")

    harness.clock.advance(120)
    await harness.run_until_idle()
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert harness.reports[0].kind is ReportKind.INTERMEDIATE

    await harness.finalize_all()
    assert harness.reports[-1].kind is ReportKind.FINAL
    assert harness.task_status is TaskStatus.COMPLETED
    assert harness.raw_context_count == 0

    delivered = await harness.deliver_outbox()
    assert delivered >= 2
    assert len(harness.maisaka.append_calls) >= 1
    assert len(harness.maisaka.trigger_calls) >= 1
    assert await harness.pending_outbox_count() == 0


@pytest.mark.asyncio
async def test_e2e_second_deadline_cycle_after_intermediate(runtime_harness) -> None:
    """§13.1 — after intermediate with live leaves, a later deadline can open again."""

    harness = runtime_harness
    await harness.start("持续调查", credits=80.0, time_budget=60)
    await harness.formalize("正式持续调查")
    await harness.root_delegates({"A": 40.0, "B": 40.0})
    await harness.branch_checkpoint("A")
    await harness.branch_checkpoint("B")
    harness.clock.advance(60)
    await harness.run_until_idle()
    assert harness.task_status is TaskStatus.RUNNING
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]

    # Keep leaves alive and force another report epoch via manager deadline.
    assert harness.manager is not None
    status = await harness.manager.status(harness.task_id)
    await harness.manager.handle_runtime_event(
        ReportDeadlineReached(
            f"{harness.task_id}:deadline-2",
            harness.task_id,
            str(status["round_id"]),
            int(status["generation"]),
            epoch=int(status.get("report_epoch") or 1) + 1
            if False
            else harness.manager._controllers[harness.task_id].state.report_epoch + 1,
        )
    )
    # Need frontier readiness again for synthesis.
    await harness.branch_checkpoint("A")
    await harness.branch_checkpoint("B")
    if harness.coordinator is not None and harness.coordinator.current_epoch is not None:
        from lunagentic_research_swarm.runtime.events import GraceExpired

        epoch = harness.coordinator.current_epoch
        status = await harness.manager.status(harness.task_id)
        await harness.manager.handle_runtime_event(
            GraceExpired(
                f"{harness.task_id}:grace-2",
                harness.task_id,
                str(status["round_id"]),
                int(status["generation"]),
                epoch=epoch.epoch,
            )
        )
        await harness.coordinator.wait_for_synthesis()

    kinds = await harness.persisted_report_kinds()
    assert kinds[0] == "INTERMEDIATE"
    assert kinds.count("INTERMEDIATE") >= 2


@pytest.mark.asyncio
async def test_e2e_outbox_trigger_failure_retries_without_duplicate_append(runtime_harness) -> None:
    """§17.2 / §25.2 — trigger fail after append does not re-append body."""

    harness = runtime_harness
    await harness.start("交付韧性", credits=50.0, time_budget=30)
    await harness.formalize("正式交付韧性")
    await harness.root_delegates({"A": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(30)
    await harness.run_until_idle()
    await harness.finalize_all()

    harness.maisaka.trigger_error = RuntimeError("transient trigger")
    outbox = MaisakaOutbox(harness.store, harness.maisaka, clock=harness.clock)
    first = await outbox.deliver_once()
    assert first >= 1
    append_after_fail = len(harness.maisaka.append_calls)
    assert append_after_fail >= 1

    harness.maisaka.trigger_error = None
    second = await outbox.deliver_once()
    assert second >= 1
    # Append count must not grow for the same report body intents.
    assert len(harness.maisaka.append_calls) == append_after_fail
    assert len(harness.maisaka.trigger_calls) >= 1


@pytest.mark.asyncio
async def test_e2e_fake_llm_turn_worker_contract_on_workload(runtime_harness) -> None:
    """§9 / §25.2 — FakeLLMGateway feeds TurnWorker without Host LLM."""

    from lunagentic_research_swarm.runtime.reducer import PerformAgentCall
    from lunagentic_research_swarm.runtime.turns import TurnWorker

    harness = runtime_harness
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "搜集完成",
                "procedures": [],
                "delegations": [{"agent_id": "builtin.researcher", "task": "复核", "credits": 0.0}],
            }
        )
    )
    worker = TurnWorker(harness.llm, harness.procedures)
    event = await worker.perform_agent_call(
        PerformAgentCall(
            "task-sim",
            "round-sim",
            0,
            event_id="agent-sim",
            payload={
                "selector": "model:gpt-5.6-luna-max",
                "messages": ({"role": "user", "content": "调查"},),
                "branch_id": "branch-a",
                "call_id": "call-sim",
                "credits_after_reservation": 10.0,
            },
        )
    )
    assert event.protocol_result is not None
    assert event.protocol_result["report"] == "搜集完成"
    assert event.protocol_result["delegations"][0]["credits"] == 0.0
    assert len(harness.llm.calls) == 1


@pytest.mark.asyncio
async def test_e2e_all_terminals_enter_completed_without_wall_clock(runtime_harness) -> None:
    """§13.4 — last leaves finalize without waiting for time budget."""

    harness = runtime_harness
    await harness.start("快速终局", credits=40.0, time_budget=3600)
    await harness.formalize("正式快速终局")
    await harness.root_delegates({"A": 20.0, "B": 20.0})
    # No clock advance — terminals alone should drive FINAL.
    await harness.finalize_all()
    assert harness.task_status is TaskStatus.COMPLETED
    kinds = await harness.persisted_report_kinds()
    assert kinds[-1] == "FINAL"


@pytest.mark.asyncio
async def test_e2e_three_branch_fanout_intermediate_then_final(runtime_harness) -> None:
    """§25.2 / §25.3#1 shape — A/B/C fan-out through report + COMPLETED."""

    harness = runtime_harness
    await harness.start("三路比较", credits=100.0, time_budget=90)
    await harness.formalize("正式三路比较")
    await harness.root_delegates({"A": 50.0, "B": 25.0, "C": 25.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(90)
    await harness.run_until_idle()
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    await harness.finalize_all()
    assert harness.task_status is TaskStatus.COMPLETED
    assert (await harness.persisted_report_kinds())[-1] == "FINAL"


@pytest.mark.asyncio
async def test_e2e_stop_releases_raw_without_final_report(runtime_harness) -> None:
    """§7.4 — stop ends the round without synthesizing a FINAL body."""

    from lunagentic_research_swarm.runtime.events import StopRequested

    harness = runtime_harness
    await harness.start("可停止任务", credits=60.0, time_budget=120)
    await harness.formalize("正式可停止任务")
    await harness.root_delegates({"A": 30.0, "B": 30.0})
    assert harness.manager is not None
    status = await harness.manager.status(harness.task_id)
    await harness.manager.handle_runtime_event(
        StopRequested(
            f"{harness.task_id}:stop",
            harness.task_id,
            str(status["round_id"]),
            int(status["generation"]),
            reason="user_stop",
        )
    )
    assert harness.task_status is TaskStatus.STOPPED
    kinds = await harness.persisted_report_kinds()
    assert "FINAL" not in kinds


@pytest.mark.asyncio
async def test_e2e_early_all_checkpoint_opens_intermediate_without_deadline(runtime_harness) -> None:
    """§13.2 / §25.2 — every active leaf checkpointed → early INTERMEDIATE."""

    harness = runtime_harness
    await harness.start("提前报告", credits=40.0, time_budget=600)
    await harness.formalize("正式提前报告")
    await harness.root_delegates({"A": 20.0, "B": 20.0})
    await harness.branch_checkpoint("A")
    await harness.branch_checkpoint("B")
    await harness.coordinator.wait_for_synthesis()
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert harness.reports[0].kind is ReportKind.INTERMEDIATE
