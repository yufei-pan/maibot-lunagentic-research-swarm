from __future__ import annotations

import asyncio

import pytest

from lunagentic_research_swarm.runtime.reducer import Effect
from lunagentic_research_swarm.runtime.scheduler import FairScheduler


def effect(
    task_id: str,
    effect_id: str,
    *,
    kind: str = "agent",
    priority: str = "normal",
    generation: int = 0,
    **payload: object,
) -> Effect:
    return Effect(
        task_id=task_id,
        round_id=f"round-{task_id}",
        generation=generation,
        kind=kind,
        priority=priority,
        event_id=effect_id,
        payload={"effect_id": effect_id, **payload},
    )


class FakeWorker:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.tokens: list[object] = []
        self._started_event = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False

    async def __call__(self, item: Effect, token: object) -> str:
        self.started.append(str(item.payload["effect_id"]))
        self.tokens.append(token)
        self._started_event.set()
        if self.block:
            await self.release.wait()
        return str(item.payload["effect_id"])

    async def wait_started(self, count: int) -> None:
        while len(self.started) < count:
            self._started_event.clear()
            await self._started_event.wait()


@pytest.fixture
def fake_worker() -> FakeWorker:
    return FakeWorker()


@pytest.mark.asyncio
async def test_wide_task_cannot_starve_other_task(fake_worker: FakeWorker) -> None:
    scheduler = FairScheduler(global_llm=2, per_task_llm=2, per_task_procedure=2, worker=fake_worker)
    await scheduler.start()
    for index in range(20):
        await scheduler.enqueue(effect("A", f"a{index}"))
    await scheduler.enqueue(effect("B", "b0"))
    await fake_worker.wait_started(3)
    assert "b0" in fake_worker.started[:3]
    await scheduler.close()


@pytest.mark.asyncio
async def test_task_joining_live_queue_gets_next_round_robin_turn(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "a0"))
    await scheduler.enqueue(effect("A", "a1"))
    await fake_worker.wait_started(1)

    await scheduler.enqueue(effect("B", "b0"))
    fake_worker.release.set()
    await fake_worker.wait_started(2)

    assert fake_worker.started[:2] == ["a0", "b0"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_control_barrier_precedes_child_launch(fake_worker: FakeWorker) -> None:
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.enqueue(effect("A", "child", kind="agent", priority="normal"))
    await scheduler.enqueue(effect("A", "continue", kind="control", priority="barrier"))
    await fake_worker.wait_started(1)
    assert fake_worker.started[0] == "continue"
    await scheduler.close()


@pytest.mark.asyncio
async def test_blocked_barrier_prevents_lower_priority_child_launch(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "inflight", kind="agent"))
    await fake_worker.wait_started(1)

    await scheduler.enqueue(effect("A", "report", kind="summarizer", priority="barrier"))
    await scheduler.enqueue(effect("A", "child", kind="procedure", priority="normal"))
    await asyncio.sleep(0)

    assert fake_worker.started == ["inflight"]
    fake_worker.release.set()
    await fake_worker.wait_started(3)
    assert fake_worker.started == ["inflight", "report", "child"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_llm_and_procedure_limits_are_independent(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=2, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "agent-0", kind="agent"))
    await scheduler.enqueue(effect("A", "agent-1", kind="agent"))
    await scheduler.enqueue(effect("A", "procedure-0", kind="procedure"))
    await scheduler.enqueue(effect("A", "procedure-1", kind="procedure"))
    await fake_worker.wait_started(3)
    assert fake_worker.started[:3] == ["agent-0", "procedure-0", "procedure-1"]
    snapshot = scheduler.stats()
    assert snapshot["global"]["llm_active"] == 1
    assert snapshot["tasks"]["A"]["llm_active"] == 1
    assert snapshot["tasks"]["A"]["procedure_active"] == 2
    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_summarizer_uses_global_but_not_task_llm_limit(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=2, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "agent", kind="agent"))
    await scheduler.enqueue(effect("A", "summary", kind="summarizer"))
    await fake_worker.wait_started(2)
    snapshot = scheduler.stats()
    assert snapshot["global"]["llm_active"] == 2
    assert snapshot["tasks"]["A"]["llm_active"] == 1
    assert snapshot["tasks"]["A"]["kind"]["summarizer"]["active"] == 1
    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_pause_blocks_new_agent_and_summarizer_but_allows_procedure(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=2, per_task_llm=2, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "agent-0", kind="agent"))
    await fake_worker.wait_started(1)
    scheduler.pause_task("A")
    await scheduler.enqueue(effect("A", "agent-1", kind="agent"))
    await scheduler.enqueue(effect("A", "summary-1", kind="summarizer"))
    await scheduler.enqueue(effect("A", "procedure-1", kind="procedure"))
    await fake_worker.wait_started(2)
    assert fake_worker.started == ["agent-0", "procedure-1"]
    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancel_generation_drops_queued_effects_and_marks_token(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "inflight", generation=0))
    await scheduler.enqueue(effect("A", "queued", generation=0))
    await fake_worker.wait_started(1)
    removed = scheduler.cancel_generation("A", generation=0)
    assert removed == 1
    assert getattr(fake_worker.tokens[0], "cancelled", False)
    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_close_waits_for_started_wrappers_and_cancels_queue(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "inflight"))
    await scheduler.enqueue(effect("A", "queued"))
    await fake_worker.wait_started(1)
    closing = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)
    assert not closing.done()
    assert scheduler.stats()["queued"] == 0
    fake_worker.release.set()
    await closing
    assert fake_worker.started == ["inflight"]
    assert scheduler.stats()["queued"] == 0


@pytest.mark.asyncio
async def test_concurrent_close_callers_wait_for_same_shutdown(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.enqueue(effect("A", "inflight"))
    await fake_worker.wait_started(1)

    first = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)
    second = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)

    assert not first.done()
    assert not second.done()
    fake_worker.release.set()
    await asyncio.gather(first, second)
    assert scheduler.stats()["closed"]


@pytest.mark.asyncio
async def test_active_worker_reentrant_close_does_not_wait_on_its_own_shutdown() -> None:
    started = asyncio.Event()
    enter_reentrant_close = asyncio.Event()
    outcomes: list[str] = []
    scheduler: FairScheduler

    async def worker(item: Effect, token: object) -> None:
        started.set()
        await enter_reentrant_close.wait()
        try:
            await asyncio.wait_for(scheduler.close(), timeout=0.05)
        except TimeoutError:
            outcomes.append("timed_out")
        else:
            outcomes.append("returned")

    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=worker)
    await scheduler.enqueue(effect("A", "worker"))
    await started.wait()
    external_close = asyncio.create_task(scheduler.close())
    while not scheduler.stats()["closing"]:
        await asyncio.sleep(0)

    enter_reentrant_close.set()
    await external_close

    assert outcomes == ["returned"]
    assert scheduler.stats()["closed"]


@pytest.mark.asyncio
async def test_active_initial_close_caller_waits_for_other_active_wrappers() -> None:
    both_started = asyncio.Event()
    start_close = asyncio.Event()
    release_other = asyncio.Event()
    close_returned = asyncio.Event()
    started_count = 0
    scheduler: FairScheduler

    async def worker(item: Effect, token: object) -> None:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            both_started.set()
        if item.payload["effect_id"] == "closer":
            await start_close.wait()
            await scheduler.close()
            close_returned.set()
        else:
            await release_other.wait()

    scheduler = FairScheduler(global_llm=2, per_task_llm=2, per_task_procedure=1, worker=worker)
    await scheduler.enqueue(effect("A", "closer"))
    await scheduler.enqueue(effect("A", "other"))
    await both_started.wait()

    start_close.set()
    await asyncio.sleep(0)
    assert not close_returned.is_set()
    release_other.set()
    await close_returned.wait()
    assert scheduler.stats()["closed"]


@pytest.mark.asyncio
async def test_global_stats_aggregate_kind_and_wait_latency(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.enqueue(effect("A", "active", kind="agent"))
    await fake_worker.wait_started(1)
    await scheduler.enqueue(effect("B", "queued", kind="summarizer"))

    snapshot = scheduler.stats()["global"]
    assert snapshot["kind"]["agent"]["active"] == 1
    assert snapshot["kind"]["agent"]["queued"] == 0
    assert snapshot["kind"]["summarizer"]["active"] == 0
    assert snapshot["kind"]["summarizer"]["queued"] == 1
    assert snapshot["active_by_kind"] == {"agent": 1, "summarizer": 0}
    assert snapshot["queued_by_kind"] == {"agent": 0, "summarizer": 1}
    assert snapshot["wait_latency"]["samples"] == 2
    assert snapshot["wait_latency"]["max_seconds"] >= snapshot["wait_latency"]["average_seconds"] >= 0.0

    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_stats_are_observable_without_prompt_payload(fake_worker: FakeWorker) -> None:
    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "safe", prompt="secret prompt", nested={"prompt": "nested secret"}))
    await fake_worker.wait_started(1)
    snapshot = scheduler.stats()
    assert "prompt" not in repr(snapshot)
    assert snapshot["tasks"]["A"]["kind"]["agent"]["active"] == 1
    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_stats_do_not_expose_worker_exception_payload() -> None:
    secret = "secret prompt from worker exception"
    failed = asyncio.Event()

    async def failing_worker(item: Effect, token: object) -> None:
        failed.set()
        raise RuntimeError(f"provider rejected {secret}")

    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=failing_worker)
    await scheduler.enqueue(effect("A", "failure", prompt=secret))
    await failed.wait()

    snapshot = scheduler.stats()
    assert secret not in repr(snapshot)
    # 只保留结构性身份，便于定位失败的 effect；异常正文可能引用 prompt，不进遥测。
    assert snapshot["errors"] == ({"kind": "RuntimeError", "effect": "agent", "task_id": "A"},)
    await scheduler.close()
