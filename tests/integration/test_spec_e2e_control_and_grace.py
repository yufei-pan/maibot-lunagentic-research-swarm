"""E2E report / grace / pause / stop workloads (design §7, §13, E3–E8, E11).

Uses RuntimeHarness + FakeClock + FakeLLM + ReportCoordinator. No Host LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fakes import FakeLLMResponse

import lunagentic_research_swarm.runtime.manager as manager_mod
from lunagentic_research_swarm.models import (
    BranchLifecycle,
    BranchRuntime,
    ReportKind,
    SummaryKind,
    TaskStatus,
)
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.events import AgentCallCompleted
from lunagentic_research_swarm.runtime.reducer import ArmPauseExpiry, PerformAgentCall, PerformProcedureBatch
from lunagentic_research_swarm.runtime.turns import TurnWorker


async def _override_launch_tracking(
    harness, *, materialize: bool = False
) -> list[dict[str, object]]:
    """Bypass manager materialize (catalog only knows root) and record launches.

    When ``materialize`` is True, the launch callback also inserts the child into
    the coordinator graph so a post-release safe-point can run for real.
    """

    assert harness.coordinator is not None
    launched: list[dict[str, object]] = []

    async def launch(parent: str, child: dict[str, object]) -> None:
        launched.append(dict(child))
        if not materialize:
            return
        branch_id = str(child["branch_id"])
        harness.coordinator.branches[branch_id] = BranchRuntime(
            branch_id=branch_id,
            task=harness.formalized_task,
            catalog_fingerprint="integration-catalog",
            generation=0,
            messages=[{"role": "assistant", "content": f"{branch_id} post-release evidence"}],
            credits=10.0,
            depth=2,
            parent_branch_id=str(parent),
        )

    harness.coordinator.release_held_delegations = None
    harness.coordinator.launch_delegation = launch
    return launched


# ---------------------------------------------------------------------------
# E3 — manual checkpoint holds children → deadline → INTERMEDIATE → release → FINAL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e3_manual_checkpoint_holds_children_until_deadline_release(runtime_harness) -> None:
    """§13.1 / E3 — held children stay quiet until epoch open releases them."""

    harness = runtime_harness
    await harness.start("checkpoint hold", credits=80.0, time_budget=60)
    await harness.formalize("正式 checkpoint hold")
    await harness.root_delegates({"A": 40.0, "B": 40.0})
    launched = await _override_launch_tracking(harness)

    await harness.coordinator.on_branch_safe_point(
        "A",
        checkpoint=True,
        delegations=({"branch_id": "A-child", "agent_id": "sibling", "task": "held-work"},),
    )
    assert launched == []
    assert harness.coordinator.branches["A"].lifecycle is BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT
    assert "A" in harness.coordinator._held

    harness.clock.advance(60)
    await harness.run_until_idle()

    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert harness.reports[0].kind is ReportKind.INTERMEDIATE
    # release_held → launch_delegation with the held child payload (incl. next-epoch tag).
    assert launched == [
        {
            "branch_id": "A-child",
            "agent_id": "sibling",
            "task": "held-work",
            "report_epoch": harness.coordinator.current_epoch.epoch + 1,
        }
    ]
    assert harness.coordinator.branches["A"].lifecycle is BranchLifecycle.READY
    assert harness.coordinator._held == {}


@pytest.mark.asyncio
async def test_e2e_e3_released_children_can_run_then_final(runtime_harness) -> None:
    """§13.1 / E3 — after release, launch materializes child; child safe-point then FINAL.

    Leave B in-flight so the deadline path bumps ``report_epoch`` (an early
    all-checkpoint open would synthesize without the manager epoch counter).
    Launch tracking also inserts the child so a real post-release safe-point can run.
    """

    harness = runtime_harness
    await harness.start("hold then final", credits=80.0, time_budget=45)
    await harness.formalize("正式 hold then final")
    await harness.root_delegates({"A": 40.0, "B": 40.0})
    launched = await _override_launch_tracking(harness, materialize=True)

    held_child = {"branch_id": "A-child", "task": "post-release"}
    await harness.coordinator.on_branch_safe_point(
        "A",
        checkpoint=True,
        delegations=(held_child,),
    )
    harness.clock.advance(45)
    await harness.run_until_idle()
    assert launched == [
        {
            "branch_id": "A-child",
            "task": "post-release",
            "report_epoch": harness.coordinator.current_epoch.epoch + 1,
        }
    ]
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert harness.manager._controllers[harness.task_id].state.report_epoch == 1
    assert "A-child" in harness.coordinator.branches

    # Released child takes a real terminal safe-point (not a hand-inserted finalize-only stub).
    await harness.coordinator.on_branch_safe_point("A-child", terminal=True)
    assert harness.coordinator.branches["A-child"].lifecycle is BranchLifecycle.FINALIZED

    await harness.finalize_all()
    assert harness.task_status is TaskStatus.COMPLETED
    assert (await harness.persisted_report_kinds())[-1] == "FINAL"


# ---------------------------------------------------------------------------
# E4 — mid-grace agent return auto-checkpoints; new child outside frontier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e4_mid_grace_return_auto_checkpoints_clone(runtime_harness) -> None:
    """§13.1#3 / E4 — grace return auto-checkpoints without model requesting checkpoint."""

    harness = runtime_harness
    await harness.start("mid-grace clone", credits=60.0, time_budget=120)
    await harness.formalize("正式 mid-grace clone")
    await harness.root_delegates({"A": 30.0, "B": 30.0})
    launched = await _override_launch_tracking(harness)

    epoch = await harness.coordinator.open_epoch()
    assert list(epoch.frontier) == ["A", "B"]
    assert epoch.synthesis_started is False

    await harness.coordinator.on_branch_safe_point(
        "A",
        checkpoint=False,
        delegations=({"branch_id": "A-next", "task": "next-epoch-work"},),
    )

    assert epoch.frontier["A"].checkpoint_requested is True
    assert epoch.frontier["A"].checkpoint_summary_id is not None
    assert epoch.frontier["A"].ready is True
    # Original keeps running (READY, not waiting on a manual hold).
    assert harness.coordinator.branches["A"].lifecycle is BranchLifecycle.READY
    assert harness.coordinator._held == {}


@pytest.mark.asyncio
async def test_e2e_e4_mid_grace_child_outside_current_frontier(runtime_harness) -> None:
    """§13.1#6 / E4 — child spawned in grace is tagged for the next epoch, not frontier."""

    harness = runtime_harness
    await harness.start("grace child next epoch", credits=60.0, time_budget=120)
    await harness.formalize("正式 grace child next epoch")
    await harness.root_delegates({"A": 30.0, "B": 30.0})
    launched = await _override_launch_tracking(harness)

    epoch = await harness.coordinator.open_epoch()
    await harness.coordinator.on_branch_safe_point(
        "A",
        delegations=({"branch_id": "A-next", "task": "belongs-next"},),
    )

    assert launched == [
        {"branch_id": "A-next", "task": "belongs-next", "report_epoch": epoch.epoch + 1}
    ]
    assert "A-next" not in epoch.frontier
    # Sibling still open → synthesis waits; child must not enlarge this frontier.
    assert epoch.synthesis_started is False
    assert list(epoch.frontier) == ["A", "B"]


# ---------------------------------------------------------------------------
# E5 — full grace timeout clones pre-call stable history; late FakeLLM excluded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e5_grace_timeout_clones_pre_call_stable_history(runtime_harness) -> None:
    """§13.1 / E5 — grace expiry checkpoints frozen frontier history, not late mutations.

    Late content reaches the live branch via TurnWorker → AgentCallCompleted →
    ProcedureExecutor (report folded into parent_messages) → manager handle, not a
    hand append after FakeLLM theater.
    """

    harness = runtime_harness
    await harness.start("grace timeout clone", credits=50.0, time_budget=90)
    await harness.formalize("正式 grace timeout clone")
    await harness.root_delegates({"A": 25.0, "B": 25.0})

    # Pre-call evidence is what open_epoch freezes into frontier.stable_history.
    harness.coordinator.branches["A"].messages[:] = [
        {"role": "assistant", "content": "pre-call stable A"},
    ]
    harness.coordinator.branches["B"].messages[:] = [
        {"role": "assistant", "content": "pre-call stable B"},
    ]

    epoch = await harness.coordinator.open_epoch()
    assert epoch.frontier["A"].stable_history[-1]["content"] == "pre-call stable A"

    late_marker = "LATE_FAKE_LLM_SHOULD_NOT_ENTER_CHECKPOINT"
    harness.llm.enqueue(
        FakeLLMResponse(
            text=late_marker,
            payload={"report": late_marker, "procedures": [], "delegations": []},
        )
    )
    status = await harness.manager.status(harness.task_id)
    pre_call_messages = (
        {"role": "assistant", "content": "pre-call stable A"},
        {"role": "user", "content": "late"},
    )
    worker = TurnWorker(harness.llm, harness.procedures)
    late_completed = await worker.perform_agent_call(
        PerformAgentCall(
            harness.task_id,
            harness.round_id,
            int(status["generation"]),
            event_id="late-call",
            payload={
                "selector": "model:gpt-5.6-luna-max",
                "messages": pre_call_messages,
                "branch_id": "A",
                "call_id": "late-call",
                "credits_after_reservation": 10.0,
            },
        )
    )
    assert late_completed.protocol_result is not None
    assert late_completed.protocol_result["report"] == late_marker

    harness.scheduler.enqueued.clear()
    await harness.manager.handle_runtime_event(late_completed)
    procedure_effects = [
        item for item in harness.scheduler.enqueued if isinstance(item, PerformProcedureBatch)
    ]
    assert len(procedure_effects) == 1

    # Real procedure path appends envelope report into parent_messages (§8.2/§9.1).
    executor = ProcedureExecutor(SimpleNamespace(get=lambda _pid: None), harness.procedures)
    batch_raw = await executor.invoke_many(procedure_effects[0])
    batch = replace(
        batch_raw,
        report=str(procedure_effects[0].payload.get("report", "")),
        delegations=tuple(procedure_effects[0].payload.get("delegations", ())),
        credits_after=float(procedure_effects[0].payload.get("credits_after", 0.0)),
        parent_messages=batch_raw.parent_messages
        or tuple(procedure_effects[0].payload.get("messages", ())),
        parent_depth=int(procedure_effects[0].payload.get("branch_depth", 0)),
        live_agent_ids=procedure_effects[0].payload.get("live_agent_ids"),
        agent_calls_started=int(procedure_effects[0].payload.get("agent_calls_started", 0)),
    )
    assert batch.parent_messages[-1]["content"] == late_marker

    await harness.manager.handle_runtime_event(batch)
    assert harness.coordinator.branches["A"].messages[-1]["content"] == late_marker
    # Freeze must still be the pre-call clone.
    assert epoch.frontier["A"].stable_history[-1]["content"] == "pre-call stable A"

    harness.clock.advance(60)
    await harness.coordinator.on_grace_expired(epoch.epoch)
    await harness.coordinator.wait_for_synthesis()

    assert epoch.frontier["A"].checkpoint_requested is True
    assert harness.reports[0].kind is ReportKind.INTERMEDIATE
    # Summarizer saw frozen pre-call history only.
    a_requests = [
        req
        for req in harness.summarizer.branch_requests
        if req.branch_history and req.branch_history[-1].get("content") == "pre-call stable A"
    ]
    assert a_requests, "grace clone must summarize pre-call stable history"
    for req in harness.summarizer.branch_requests:
        joined = " ".join(str(item.get("content", "")) for item in req.branch_history)
        assert late_marker not in joined
    assert late_marker not in harness.reports[0].text
    # Live branch still retains the late result for the next epoch.
    assert harness.coordinator.branches["A"].messages[-1]["content"] == late_marker


@pytest.mark.asyncio
async def test_e2e_e5_frontier_stable_history_frozen_at_open(runtime_harness) -> None:
    """§13.1 / E5 — open_epoch freezes stable_history; post-open appends do not rewrite it."""

    harness = runtime_harness
    await harness.start("freeze frontier", credits=40.0, time_budget=60)
    await harness.formalize("正式 freeze frontier")
    await harness.root_delegates({"A": 40.0})
    harness.coordinator.branches["A"].messages[:] = [{"role": "assistant", "content": "before-open"}]

    epoch = await harness.coordinator.open_epoch()
    harness.coordinator.branches["A"].messages.append({"role": "assistant", "content": "after-open"})

    assert epoch.frontier["A"].stable_history[-1]["content"] == "before-open"
    assert [item["content"] for item in epoch.frontier["A"].stable_history] == ["before-open"]


# ---------------------------------------------------------------------------
# E6 — pause → pause timeout → EXPIRED → continue from summary layer only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e6_pause_timeout_expires_and_releases_raw(runtime_harness, monkeypatch) -> None:
    """§7.3 / E6 — FakeClock-driven pause expiry marks EXPIRED and drops raw context.

    FakeScheduler records ``ArmPauseExpiry`` but does not run it; this test drains
    that effect through ``arm_pause_expiry_effect``, binds manager ``_now`` to the
    harness FakeClock, advances past ``due_at``, and awaits the armed ``_pause_jobs``
    entry (which calls ``expire_pause``).
    """

    harness = runtime_harness
    monkeypatch.setattr(manager_mod, "_now", harness.clock)
    await harness.start("pause expiry", credits=50.0, time_budget=120)
    assert harness.manager is not None
    harness.manager._pause_timeout_seconds = 30
    await harness.formalize("正式 pause expiry")
    await harness.manager.add_context(harness.task_id, "跨 round 资料")

    paused = await harness.manager.pause(harness.task_id)
    assert paused["status"] == "PAUSED"
    assert harness.manager._pause_jobs, "pause must arm an expiry job"
    arms = [item for item in harness.scheduler.enqueued if isinstance(item, ArmPauseExpiry)]
    assert arms, "PauseRequested must enqueue ArmPauseExpiry"
    due_at = float(arms[-1].payload["due_at"])
    harness.clock.advance(max(0.0, due_at - harness.clock()) + 1.0)
    await harness.manager.arm_pause_expiry_effect(arms[-1])
    await asyncio.wait_for(harness.manager._pause_jobs[harness.task_id], timeout=1.0)

    status = await harness.manager.status(harness.task_id)
    assert status["status"] == "EXPIRED"
    assert status["raw_context_released"] is True
    assert status["active_leaves"] == []


@pytest.mark.asyncio
async def test_e2e_e6_continue_after_expire_restarts_from_summary_layer(runtime_harness) -> None:
    """§7.5 / §18.3 / E6 — continue after EXPIRED builds a new round from summary layer only."""

    harness = runtime_harness
    await harness.start("expire then continue", credits=50.0, time_budget=120)
    await harness.formalize("正式 expire then continue")
    await harness.manager.add_context(harness.task_id, "跨 round 资料")
    first = await harness.manager.status(harness.task_id)

    await harness.manager.pause(harness.task_id)
    assert harness.manager._pause_jobs, "pause must arm expiry before expire_pause companion path"
    assert any(isinstance(item, ArmPauseExpiry) for item in harness.scheduler.enqueued)
    await harness.manager.expire_pause(harness.task_id)

    continued = await harness.manager.continue_task(
        harness.task_id, credit_adjustment=8.0, time_budget_seconds=90
    )
    assert continued["status"] == "RUNNING"
    assert continued["round_number"] == 2
    # Pause expiry does not bump generation; restart adds exactly one.
    assert continued["generation"] == first["generation"] + 1
    assert continued["effective_time_budget_seconds"] == 90

    root_effect = next(
        effect for effect in reversed(harness.scheduler.enqueued) if isinstance(effect, PerformAgentCall)
    )
    payload = dict(root_effect.payload)
    assert payload.get("formalized_text") == "正式 expire then continue"
    summary_layer = dict(payload.get("summary_layer") or {})
    assert "跨 round 资料" in tuple(summary_layer.get("supplied_context") or ())
    # Skeleton restart effect carries summary layer only — no raw transcript/debug.
    assert "messages" not in payload or not payload.get("messages")
    assert "debug" not in repr(payload)
    assert "transcript" not in repr(payload)
    assert harness.manager.report_coordinators[harness.task_id].round_id == continued["round_id"]
    # Manager branch cache for the new root is rebuilt from summary-layer context.
    new_root = continued["active_leaves"][0]["branch_id"]
    root_entry = harness.manager._branches[harness.task_id][new_root]
    assert "跨 round 资料" in repr(root_entry.get("messages") or ())
    assert "正式 expire then continue" in repr(root_entry.get("messages") or ())


# ---------------------------------------------------------------------------
# E7 — stop with in-flight work discarded → STOPPED (+ generation bump)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e7_stop_bumps_generation_and_discards_late_completion(runtime_harness) -> None:
    """§7.4 / E7 — stop increments generation; late AgentCallCompleted is ignored."""

    harness = runtime_harness
    await harness.start("stop late", credits=40.0, time_budget=120)
    await harness.formalize("正式 stop late")
    before = await harness.manager.status(harness.task_id)
    branch_id = before["active_leaves"][0]["branch_id"]

    harness.llm.block()
    worker = TurnWorker(harness.llm, harness.procedures)
    inflight = asyncio.create_task(
        worker.perform_agent_call(
            PerformAgentCall(
                harness.task_id,
                before["round_id"],
                before["generation"],
                event_id="inflight",
                payload={
                    "selector": "model:gpt-5.6-luna-max",
                    "messages": ({"role": "user", "content": "in flight"},),
                    "branch_id": branch_id,
                    "call_id": "inflight-call",
                    "credits_after_reservation": 10.0,
                },
            )
        )
    )
    await asyncio.sleep(0)

    stopped = await harness.manager.stop(harness.task_id, reason="用户停止")
    assert stopped["status"] == "STOPPED"
    assert stopped["generation"] == before["generation"] + 1

    harness.llm.release()
    late = await inflight
    # Feed the late worker result through the manager with the *old* generation.
    enqueued_before = len(harness.scheduler.enqueued)
    await harness.manager.handle_runtime_event(
        AgentCallCompleted(
            event_id="late-after-stop",
            task_id=harness.task_id,
            round_id=before["round_id"],
            generation=before["generation"],
            branch_id=branch_id,
            call_id="inflight-call",
            result_id="late-result",
            actual_charge=0.0,
            balance_before_reconciliation=40.0,
            protocol_result=late.protocol_result
            or {"report": "late", "procedures": (), "delegations": ()},
        )
    )
    status = await harness.manager.status(harness.task_id)
    assert status["status"] == "STOPPED"
    assert status["generation"] == before["generation"] + 1
    assert len(harness.scheduler.enqueued) == enqueued_before
    kinds = await harness.persisted_report_kinds()
    assert "FINAL" not in kinds


@pytest.mark.asyncio
async def test_e2e_e7_stop_releases_raw_without_synthesis(runtime_harness) -> None:
    """§7.4 / E7 — stop clears live leaves and does not synthesize a FINAL body."""

    harness = runtime_harness
    await harness.start("stop no final", credits=40.0, time_budget=120)
    await harness.formalize("正式 stop no final")
    await harness.root_delegates({"A": 20.0, "B": 20.0})

    stopped = await harness.manager.stop(harness.task_id, reason="planner")
    assert stopped["status"] == "STOPPED"
    assert stopped["active_leaves"] == []
    assert harness.task_status is TaskStatus.STOPPED
    assert "FINAL" not in await harness.persisted_report_kinds()


# ---------------------------------------------------------------------------
# E8 — continue with all-zero active leaves + signed adjustment at barrier
# ---------------------------------------------------------------------------


async def _seed_zero_leaves(harness, names: tuple[str, ...] = ("A", "B")) -> None:
    """Replace the root leaf with durable all-zero children (store + manager).

    ``continue_task`` persists ``update_branch_balance`` against real SQLite, so
    the zero leaves must exist as inserted branches (via ``root_delegates``).
    """

    assert harness.manager is not None
    await harness.root_delegates({name: 0.0 for name in names})
    controller = harness.manager._controllers[harness.task_id]
    leaves = {name: 0.0 for name in names}
    harness.manager._branches[harness.task_id] = {
        name: {
            "credits": 0.0,
            "pending_context": [],
            "messages": list(harness.coordinator.branches[name].messages),
            "depth": 1,
            "agent_id": f"agent.{name}",
        }
        for name in names
    }
    controller.state = replace(
        controller.state,
        active_leaves=leaves,
        credit_pool=0.0,
    )


@pytest.mark.asyncio
async def test_e2e_e8_continue_all_zero_leaves_positive_adjustment_splits(runtime_harness) -> None:
    """§11.6 / §25.3#6 / E8 — +adjustment at barrier splits evenly across zero leaves."""

    harness = runtime_harness
    await harness.start("zero barrier +", credits=0.0, time_budget=120)
    await harness.formalize("正式 zero barrier +")
    await _seed_zero_leaves(harness, ("A", "B"))

    await harness.manager.pause(harness.task_id)
    result = await harness.manager.continue_task(harness.task_id, credit_adjustment=6.0, time_budget_seconds=75)

    assert result["status"] == "RUNNING"
    assert result["effective_time_budget_seconds"] == 75
    by_id = {item["branch_id"]: item["credits"] for item in result["active_leaves"]}
    assert by_id == {"A": 3.0, "B": 3.0}


@pytest.mark.asyncio
async def test_e2e_e8_continue_all_zero_leaves_negative_adjustment_finalizes(runtime_harness) -> None:
    """§11.6 / E8 — negative adjustment on all-zero leaves finalizes them at the barrier."""

    harness = runtime_harness
    await harness.start("zero barrier -", credits=0.0, time_budget=120)
    await harness.formalize("正式 zero barrier -")
    await _seed_zero_leaves(harness, ("A", "B"))

    await harness.manager.pause(harness.task_id)
    result = await harness.manager.continue_task(harness.task_id, credit_adjustment=-2.0)

    # Negative split finalizes every leaf; no runnable balances remain.
    assert result["status"] == "RUNNING"
    assert result["active_leaves"] == []
    controller = harness.manager._controllers[harness.task_id]
    assert controller.state.active_leaves == {}
    assert controller.state.credit_pool == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# E11 — multi-epoch coverage: early terminal + later checkpoint stays INTERMEDIATE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_e11_multi_epoch_early_terminal_plus_later_checkpoint(runtime_harness) -> None:
    """§13.2 / E11 — later epoch coverage keeps early terminal + current checkpoint; not FINAL."""

    harness = runtime_harness
    await harness.start("multi-epoch coverage", credits=60.0, time_budget=120)
    await harness.formalize("正式 multi-epoch coverage")
    await harness.root_delegates({"A": 30.0, "B": 30.0})

    epoch1 = await harness.coordinator.open_epoch()
    await harness.coordinator.on_branch_safe_point("A", terminal=True)
    await harness.coordinator.on_grace_expired(epoch1.epoch)
    await harness.coordinator.wait_for_synthesis()

    first = harness.reports[0]
    assert first.kind is ReportKind.INTERMEDIATE
    assert {item.branch_id for item in first.coverage.items} == {"A", "B"}
    assert any(item.kind is SummaryKind.BRANCH_FINAL and item.branch_id == "A" for item in first.coverage.items)
    assert any(item.kind is SummaryKind.CHECKPOINT and item.branch_id == "B" for item in first.coverage.items)

    harness.clock.advance(200.0)
    harness.coordinator.branches["B"].messages.append(
        {"role": "assistant", "content": "B later checkpoint evidence"}
    )
    harness.coordinator.branches["B"].latest_checkpoint_id = None
    epoch2 = await harness.coordinator.open_epoch()
    assert list(epoch2.frontier) == ["B"]
    assert "A" not in epoch2.frontier

    await harness.coordinator.on_branch_safe_point("B", checkpoint=True)
    await harness.coordinator.wait_for_synthesis()

    later = harness.reports[-1]
    assert later.epoch == epoch2.epoch
    assert later.kind is ReportKind.INTERMEDIATE
    assert later.kind is not ReportKind.FINAL
    assert later.coverage.has_checkpoint is True
    by_branch = {item.branch_id: item for item in later.coverage.items}
    assert by_branch["A"].kind is SummaryKind.BRANCH_FINAL
    assert by_branch["B"].kind is SummaryKind.CHECKPOINT
    assert "B later checkpoint evidence" in by_branch["B"].text


@pytest.mark.asyncio
async def test_e2e_e11_all_terminals_after_checkpoint_epoch_can_final(runtime_harness) -> None:
    """§13.2 / E11 — only after every leaf terminals does a later epoch become FINAL."""

    harness = runtime_harness
    await harness.start("epoch then final", credits=40.0, time_budget=60)
    await harness.formalize("正式 epoch then final")
    await harness.root_delegates({"A": 20.0, "B": 20.0})

    await harness.branch_checkpoint("A")
    harness.clock.advance(60)
    await harness.run_until_idle()
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]

    await harness.finalize_all()
    assert harness.task_status is TaskStatus.COMPLETED
    kinds = await harness.persisted_report_kinds()
    assert kinds[-1] == "FINAL"
    assert ReportKind.INTERMEDIATE.value in kinds or "INTERMEDIATE" in kinds
