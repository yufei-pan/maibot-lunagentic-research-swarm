from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.events import AgentCallCompleted

from .test_controller_start import FakeScheduler, harness


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
    manager, _, _, _, task_id = await _running_manager(harness, effort_level=0.0)
    await manager.pause(task_id)

    result = await manager.continue_task(task_id, credit_adjustment=9.0, time_budget_seconds=75)

    assert result["status"] == "RUNNING"
    assert result["effective_time_budget_seconds"] == 75
    assert result["active_leaves"] == [{"branch_id": result["active_leaves"][0]["branch_id"], "credits": 9.0}]


@pytest.mark.asyncio
async def test_continue_without_leaves_starts_new_generation_from_summary_layer_only(harness) -> None:
    manager, _, _, scheduler, task_id = await _running_manager(harness)
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
    manager, _, _, scheduler, task_id = await _running_manager(harness)
    await manager.stop(task_id)
    enqueued = len(scheduler.enqueued)

    result = await manager.continue_task(task_id, credit_adjustment=-1.0)

    assert result["success"] is False
    assert result["error"]["code"] == "task_finished_insufficient_funds"
    assert len(scheduler.enqueued) == enqueued


@pytest.mark.asyncio
async def test_pause_expiry_marks_expired_and_releases_raw_context(harness) -> None:
    manager, _, _, _, task_id = await _running_manager(harness)
    await manager.pause(task_id)

    await manager.expire_pause(task_id)

    status = await manager.status(task_id)
    assert status["status"] == "EXPIRED"
    assert status["raw_context_released"] is True
