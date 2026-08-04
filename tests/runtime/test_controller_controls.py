from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.controller import TaskController
from lunagentic_research_swarm.runtime.events import (
    AgentCallCompleted,
    PauseRequested,
    ProcedureBatchCompleted,
    StopRequested,
)
from lunagentic_research_swarm.runtime.reducer import (
    Effect,
    PerformAgentCall,
    PerformProcedureBatch,
    PerformTaskSummary,
    RuntimeState,
)
from lunagentic_research_swarm.runtime.scheduler import FairScheduler
from lunagentic_research_swarm.storage.sqlite import StoreCommand

from .test_controller_start import FakeScheduler, FakeStore, harness


class ControllableScheduler(FakeScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.inflight = 0
        self.paused = []
        self.resumed = []
        self.cancelled = []
        self._idle = asyncio.Event()
        self._idle.set()

    def task_inflight_count(self, task_id: str) -> int:
        return self.inflight

    def begin(self) -> None:
        self.inflight += 1
        self._idle.clear()

    def settle(self) -> None:
        self.inflight = 0
        self._idle.set()

    async def wait_task_idle(self, task_id: str) -> None:
        await self._idle.wait()

    def pause_task(self, task_id: str) -> None:
        self.paused.append(task_id)

    def resume_task(self, task_id: str) -> None:
        self.resumed.append(task_id)

    def cancel_generation(self, task_id: str, generation: int) -> int:
        self.cancelled.append((task_id, generation))
        return 1


async def _running_manager(harness, *, effort_level: float = 1.0):
    manager, store, summarizer, _, message, config = harness
    scheduler = ControllableScheduler()
    manager.scheduler = scheduler
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120, effort_level=effort_level)
    await manager.wait_idle(result["task_id"])
    return manager, store, summarizer, scheduler, result["task_id"]


@pytest.mark.asyncio
async def test_pause_waits_for_inflight_without_summarizing_then_becomes_paused(harness) -> None:
    manager, _, summarizer, scheduler, task_id = await _running_manager(harness)
    scheduler.begin()

    result = await manager.pause(task_id)

    assert result["status"] == "PAUSING"
    assert scheduler.paused == [task_id]
    assert len(summarizer.requests) == 1
    scheduler.settle()
    await manager.wait_idle(task_id)
    assert (await manager.status(task_id))["status"] == "PAUSED"
    assert len(summarizer.requests) == 1


@pytest.mark.asyncio
async def test_stop_increments_generation_cancels_local_work_and_discards_late_result(harness) -> None:
    manager, store, _, scheduler, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    scheduler.begin()

    stopped = await manager.stop(task_id, reason="用户要求")
    assert stopped["status"] == "STOPPED"
    assert stopped["generation"] == before["generation"] + 1
    assert scheduler.cancelled == [(task_id, before["generation"])]
    command_count = len(store.commands)

    await manager.handle_runtime_event(
        AgentCallCompleted(
            event_id="late",
            task_id=task_id,
            round_id=before["round_id"],
            generation=before["generation"],
            branch_id=before["active_leaves"][0]["branch_id"],
            call_id="late-call",
            result_id="late-result",
        )
    )

    assert (await manager.status(task_id))["status"] == "STOPPED"
    assert len(store.commands) == command_count


@pytest.mark.asyncio
async def test_controller_serializes_stop_before_concurrent_completion_effects() -> None:
    """A completion submitted while stop is committing must observe STOPPED/generation+1."""

    class BlockingStore:
        def __init__(self) -> None:
            self.stop_started = asyncio.Event()
            self.release_stop = asyncio.Event()

        async def transact(self, commands) -> None:
            if any(command.kind == "update_round_generation" for command in commands):
                self.stop_started.set()
                await self.release_stop.wait()

    store = BlockingStore()
    scheduler = FakeScheduler()
    controller = TaskController(
        RuntimeState("task", TaskStatus.RUNNING, generation=0, active_round_id="round"),
        store=store,
        scheduler=scheduler,
    )
    stop = StopRequested("stop", "task", "round", 0)
    completion = AgentCallCompleted(
        event_id="completion",
        task_id="task",
        round_id="round",
        generation=0,
        branch_id="branch",
        call_id="call",
        result_id="result",
        actual_charge=0.0,
        balance_before_reconciliation=10.0,
        protocol_result={"report": "done", "procedures": (), "delegations": ()},
    )

    assert await controller.submit(stop)
    stop_drain = asyncio.create_task(controller.drain())
    await store.stop_started.wait()
    completion_submission = asyncio.create_task(controller.submit(completion))
    await asyncio.sleep(0)
    store.release_stop.set()
    assert await completion_submission
    completion_drain = asyncio.create_task(controller.drain())
    await asyncio.gather(stop_drain, completion_drain)

    assert controller.state.status is TaskStatus.STOPPED
    assert controller.state.generation == 1
    assert not any(isinstance(effect, PerformProcedureBatch) for effect in scheduler.enqueued)


@pytest.mark.asyncio
async def test_current_agent_completion_is_reduced_and_enqueues_procedure_work(harness) -> None:
    manager, store, _, scheduler, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    branch_id = before["active_leaves"][0]["branch_id"]
    command_count = len(store.commands)

    await manager.handle_runtime_event(
        AgentCallCompleted(
            event_id="current-completion",
            task_id=task_id,
            round_id=before["round_id"],
            generation=before["generation"],
            branch_id=branch_id,
            call_id="current-call",
            result_id="current-result",
            actual_charge=0.0,
            balance_before_reconciliation=100.0,
            protocol_result={"report": "完成", "procedures": (), "delegations": ()},
        )
    )

    assert len(store.commands) == command_count + 1
    assert isinstance(scheduler.enqueued[-1], PerformProcedureBatch)
    assert scheduler.enqueued[-1].payload["branch_id"] == branch_id


@pytest.mark.asyncio
async def test_add_context_is_durable_and_broadcasts_to_active_branches(harness) -> None:
    manager, store, _, _, task_id = await _running_manager(harness)

    result = await manager.add_context(task_id, "新的可靠资料")

    assert result["status"] == "RUNNING"
    layer = await store.load_summary_layer(task_id)
    assert layer.supplied_context == ("新的可靠资料",)
    status = await manager.status(task_id)
    assert status["active_leaves"][0]["pending_context"] == ["新的可靠资料"]
    with pytest.raises(LookupError):
        await manager.add_context("missing", "x")


@pytest.mark.asyncio
async def test_continue_redistributes_pool_at_barrier_including_all_zero_leaves(harness) -> None:
    manager, store, _, _, task_id = await _running_manager(harness, effort_level=0.0)
    await manager.pause(task_id)

    result = await manager.continue_task(task_id, credit_adjustment=9.0, time_budget_seconds=75)

    assert result["status"] == "RUNNING"
    assert result["effective_time_budget_seconds"] == 75
    assert result["active_leaves"] == [{"branch_id": result["active_leaves"][0]["branch_id"], "credits": 9.0}]
    continuation = [command for command in store.commands if command.kind == "update_round_continuation"]
    balances = [command for command in store.commands if command.kind == "update_branch_balance"]
    assert continuation[-1].values["time_budget_seconds"] == 75
    assert continuation[-1].values["credit_pool"] == 0.0
    assert balances[-1].values["credit_balance"] == 9.0


@pytest.mark.asyncio
async def test_continue_without_leaves_starts_new_generation_from_summary_layer_only(harness) -> None:
    manager, store, _, scheduler, task_id = await _running_manager(harness)
    await manager.add_context(task_id, "跨 round 资料")
    first = await manager.status(task_id)
    await manager.stop(task_id)

    result = await manager.continue_task(task_id, credit_adjustment=5.0)

    assert result["status"] == "RUNNING"
    assert result["round_number"] == 2
    assert result["generation"] == first["generation"] + 2
    root_effect = scheduler.enqueued[-1]
    serialized = repr(dict(root_effect.payload))
    assert "跨 round 资料" in serialized
    assert "debug" not in serialized
    assert "transcript" not in serialized


@pytest.mark.asyncio
async def test_continue_without_leaves_persists_negative_pool_and_refuses_restart(harness) -> None:
    manager, store, _, scheduler, task_id = await _running_manager(harness)
    await manager.stop(task_id)
    enqueued = len(scheduler.enqueued)
    branch_inserts = len([command for command in store.commands if command.kind == "insert_branch"])

    result = await manager.continue_task(task_id, credit_adjustment=-1.0)

    assert result["success"] is False
    assert result["error"]["code"] == "task_finished_insufficient_funds"
    assert len(scheduler.enqueued) == enqueued
    assert len([command for command in store.commands if command.kind == "insert_branch"]) == branch_inserts
    continuation = [command for command in store.commands if command.kind == "update_round_continuation"]
    assert continuation[-1].values["credit_pool"] == -1.0
    status = await manager.status(task_id)
    assert status["active_leaves"] == []
    assert status["raw_context_released"] is True


@pytest.mark.asyncio
async def test_continue_uses_reconciled_runtime_branch_credit_balance(harness) -> None:
    manager, store, _, _, task_id = await _running_manager(harness)
    before = await manager.status(task_id)
    branch_id = before["active_leaves"][0]["branch_id"]

    await manager.handle_runtime_event(
        AgentCallCompleted(
            event_id="reconcile-credit",
            task_id=task_id,
            round_id=before["round_id"],
            generation=before["generation"],
            branch_id=branch_id,
            call_id="call",
            result_id="result",
            actual_charge=20.0,
            balance_before_reconciliation=100.0,
            protocol_result={"report": "完成", "procedures": (), "delegations": ()},
        )
    )
    await manager.pause(task_id)

    result = await manager.continue_task(task_id)

    assert result["active_leaves"] == [{"branch_id": branch_id, "credits": 80.0}]
    balances = [command for command in store.commands if command.kind == "update_branch_balance"]
    assert balances[-1].values["branch_id"] == branch_id
    assert balances[-1].values["credit_balance"] == 80.0


@pytest.mark.asyncio
async def test_controller_reducer_error_rejects_manager_state_changes_and_effect_override() -> None:
    store = FakeStore()
    scheduler = FakeScheduler()
    controller = TaskController(
        RuntimeState("task", TaskStatus.PAUSED, generation=0, active_round_id="round", raw_context_released=True),
        store=store,
        scheduler=scheduler,
    )

    accepted = await controller.apply(
        PauseRequested("invalid-pause", "task", "round", 0),
        extra_commands=(StoreCommand("insert_branch", {"branch_id": "phantom"}),),
        effects=(PerformAgentCall("task", "round", 0, payload={}),),
        state_changes={"active_leaves": {"phantom": 10.0}, "raw_context_released": False},
    )

    assert accepted is False
    assert controller.state.status is TaskStatus.PAUSED
    assert controller.state.active_leaves == {}
    assert controller.state.raw_context_released is True
    assert store.commands == []
    assert len(scheduler.enqueued) == 1
    assert not isinstance(scheduler.enqueued[0], PerformAgentCall)


@pytest.mark.asyncio
async def test_pause_uses_fair_scheduler_public_stats_when_task_wait_api_is_absent(harness) -> None:
    manager, _, _, _, _, _ = harness
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(effect, token) -> None:
        started.set()
        await release.wait()

    scheduler = FairScheduler(worker=worker)
    manager.scheduler = scheduler
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await started.wait()
    await manager.wait_idle(result["task_id"])

    paused = await manager.pause(result["task_id"])
    assert paused["status"] == "PAUSING"
    release.set()
    await manager.wait_idle(result["task_id"])
    assert (await manager.status(result["task_id"]))["status"] == "PAUSED"
    await scheduler.close()


@pytest.mark.asyncio
async def test_pause_settles_with_queued_agent_then_runs_it_only_after_continue(harness) -> None:
    """A paused agent queue must not hold the pause barrier open forever."""

    manager, _, _, _, _, _ = harness
    first_agent_started = asyncio.Event()
    second_agent_started = asyncio.Event()
    release_first_agent = asyncio.Event()
    agent_starts: list[str] = []

    async def worker(effect, token) -> None:
        if not isinstance(effect, PerformAgentCall):
            return
        agent_starts.append(str(effect.payload["branch_id"]))
        if len(agent_starts) == 1:
            first_agent_started.set()
            await release_first_agent.wait()
        else:
            second_agent_started.set()

    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=worker)
    manager.scheduler = scheduler
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await first_agent_started.wait()
    before_pause = await manager.status(result["task_id"])
    first_branch_id = before_pause["active_leaves"][0]["branch_id"]
    queued_branch_id = "queued-branch"
    await scheduler.enqueue(
        PerformAgentCall(
            result["task_id"],
            before_pause["round_id"],
            before_pause["generation"],
            payload={"branch_id": queued_branch_id},
        )
    )
    assert scheduler.stats()["tasks"][result["task_id"]]["kind"]["agent"]["queued"] == 1

    paused = await manager.pause(result["task_id"])
    assert paused["status"] == "PAUSING"
    release_first_agent.set()
    await asyncio.wait_for(manager.wait_idle(result["task_id"]), timeout=0.5)

    assert (await manager.status(result["task_id"]))["status"] == "PAUSED"
    assert agent_starts == [first_branch_id]
    assert scheduler.stats()["tasks"][result["task_id"]]["kind"]["agent"]["queued"] == 1

    continued = await manager.continue_task(result["task_id"])
    assert continued["status"] == "RUNNING"
    await second_agent_started.wait()
    assert agent_starts == [first_branch_id, queued_branch_id]
    await scheduler.close()


@pytest.mark.asyncio
async def test_pause_settles_when_blocked_summary_barrier_prevents_queued_procedure_dispatch(harness) -> None:
    """A paused summarizer barrier also makes lower-priority procedure work non-runnable."""

    manager, _, _, _, _, _ = harness
    agent_started = asyncio.Event()
    release_agent = asyncio.Event()
    procedure_started = asyncio.Event()

    async def worker(effect, token) -> None:
        if isinstance(effect, PerformAgentCall):
            agent_started.set()
            await release_agent.wait()
        elif isinstance(effect, PerformProcedureBatch):
            procedure_started.set()

    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=worker)
    manager.scheduler = scheduler
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await agent_started.wait()
    before_pause = await manager.status(result["task_id"])

    await scheduler.enqueue(
        PerformTaskSummary(
            result["task_id"],
            before_pause["round_id"],
            before_pause["generation"],
            priority="barrier",
            payload={"kind": "FINAL"},
        )
    )
    await scheduler.enqueue(
        PerformProcedureBatch(
            result["task_id"],
            before_pause["round_id"],
            before_pause["generation"],
            payload={"branch_id": "queued-branch", "call_id": "call", "credits_after": 100.0},
        )
    )

    paused = await manager.pause(result["task_id"])
    assert paused["status"] == "PAUSING"
    release_agent.set()
    await asyncio.wait_for(manager.wait_idle(result["task_id"]), timeout=0.5)

    assert (await manager.status(result["task_id"]))["status"] == "PAUSED"
    assert not procedure_started.is_set()
    task_stats = scheduler.stats()["tasks"][result["task_id"]]
    assert task_stats["pause_runnable_queued"] == 0
    assert task_stats["kind"]["task_summary"]["queued"] == 1
    assert task_stats["kind"]["procedure"]["queued"] == 1
    await scheduler.close()


@pytest.mark.asyncio
async def test_pause_waits_for_queued_procedure_during_agent_to_procedure_handoff(harness) -> None:
    manager, _, _, _, _, _ = harness
    agent_started = asyncio.Event()
    release_agent = asyncio.Event()
    procedure_started = asyncio.Event()
    release_procedure = asyncio.Event()

    async def worker(effect, token):
        if isinstance(effect, PerformAgentCall):
            agent_started.set()
            await release_agent.wait()
            return AgentCallCompleted(
                event_id="agent-completed",
                task_id=effect.task_id,
                round_id=effect.round_id,
                generation=effect.generation,
                branch_id=effect.payload["branch_id"],
                call_id="call",
                result_id="result",
                actual_charge=0.0,
                balance_before_reconciliation=100.0,
                protocol_result={"report": "完成", "procedures": (), "delegations": ()},
            )
        if isinstance(effect, PerformProcedureBatch):
            procedure_started.set()
            await release_procedure.wait()
            return ProcedureBatchCompleted(
                event_id="procedures-completed",
                task_id=effect.task_id,
                round_id=effect.round_id,
                generation=effect.generation,
                branch_id=effect.payload["branch_id"],
                call_id=effect.payload["call_id"],
                credits_after=float(effect.payload["credits_after"]),
            )
        return None

    async def on_result(effect, result, token) -> None:
        if isinstance(result, AgentCallCompleted):
            await manager.handle_runtime_event(result)

    scheduler = FairScheduler(worker=worker, on_result=on_result)
    manager.scheduler = scheduler
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await agent_started.wait()

    paused = await manager.pause(result["task_id"])
    assert paused["status"] == "PAUSING"
    release_agent.set()
    await procedure_started.wait()
    await asyncio.sleep(0)
    assert (await manager.status(result["task_id"]))["status"] == "PAUSING"

    release_procedure.set()
    await manager.wait_idle(result["task_id"])
    assert (await manager.status(result["task_id"]))["status"] == "PAUSED"
    await scheduler.close()


@pytest.mark.asyncio
async def test_pause_waits_for_own_normal_procedure_despite_other_task_high_queue(harness) -> None:
    """A high-priority queue from another task must not hide this task's handoff procedure."""

    manager, _, _, _, _, _ = harness
    agent_started = asyncio.Event()
    release_agent = asyncio.Event()
    procedure_started = asyncio.Event()
    release_procedure = asyncio.Event()
    other_high_started = asyncio.Event()
    handoff_pause_queued = 0
    other_high_queued = 0
    scheduler: FairScheduler

    async def worker(effect, token):
        if isinstance(effect, PerformAgentCall):
            agent_started.set()
            await release_agent.wait()
            return AgentCallCompleted(
                event_id="agent-completed",
                task_id=effect.task_id,
                round_id=effect.round_id,
                generation=effect.generation,
                branch_id=effect.payload["branch_id"],
                call_id="call",
                result_id="result",
                actual_charge=0.0,
                balance_before_reconciliation=100.0,
                protocol_result={"report": "完成", "procedures": (), "delegations": ()},
            )
        if isinstance(effect, PerformProcedureBatch):
            procedure_started.set()
            await release_procedure.wait()
            return ProcedureBatchCompleted(
                event_id="procedures-completed",
                task_id=effect.task_id,
                round_id=effect.round_id,
                generation=effect.generation,
                branch_id=effect.payload["branch_id"],
                call_id=effect.payload["call_id"],
                credits_after=float(effect.payload["credits_after"]),
            )
        if effect.event_id == "other-high":
            other_high_started.set()
        return None

    async def on_result(effect, result, token) -> None:
        nonlocal handoff_pause_queued, other_high_queued
        if isinstance(result, AgentCallCompleted):
            await scheduler.enqueue(
                Effect(
                    task_id="other-task",
                    round_id="other-round",
                    generation=0,
                    kind="procedure",
                    priority="high",
                    event_id="other-high",
                )
            )
            await manager.handle_runtime_event(result)
            snapshot = scheduler.stats()["tasks"]
            handoff_pause_queued = snapshot[str(effect.task_id)]["pause_runnable_queued"]
            other_high_queued = snapshot["other-task"]["kind"]["procedure"]["queued"]

    scheduler = FairScheduler(worker=worker, on_result=on_result)
    manager.scheduler = scheduler
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await agent_started.wait()

    paused = await manager.pause(result["task_id"])
    assert paused["status"] == "PAUSING"
    release_agent.set()
    await other_high_started.wait()
    await procedure_started.wait()
    await asyncio.sleep(0)

    assert other_high_queued == 1
    assert handoff_pause_queued == 1
    assert (await manager.status(result["task_id"]))["status"] == "PAUSING"

    release_procedure.set()
    await manager.wait_idle(result["task_id"])
    assert (await manager.status(result["task_id"]))["status"] == "PAUSED"
    await scheduler.close()


@pytest.mark.asyncio
async def test_pause_expiry_marks_expired_and_releases_raw_context(harness) -> None:
    manager, _, _, _, task_id = await _running_manager(harness)
    await manager.pause(task_id)

    await manager.expire_pause(task_id)

    status = await manager.status(task_id)
    assert status["status"] == "EXPIRED"
    assert status["raw_context_released"] is True
