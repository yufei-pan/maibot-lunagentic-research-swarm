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
async def test_control_barrier_precedes_child_launch(fake_worker: FakeWorker) -> None:
    scheduler = FairScheduler(global_llm=1, per_task_llm=1, per_task_procedure=1, worker=fake_worker)
    await scheduler.enqueue(effect("A", "child", kind="agent", priority="normal"))
    await scheduler.enqueue(effect("A", "continue", kind="control", priority="barrier"))
    await fake_worker.wait_started(1)
    assert fake_worker.started[0] == "continue"
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
