"""Spec §22 — multi-task fairness and barrier-over-child priority smoke.

Complements ``test_scheduler`` (single ``b0 in first 3``, same-task continue/report
barrier) and ``test_spec_config_concurrency_privacy`` (pause caps) with discoverable
``test_spec_*`` pins. Offline ``FairScheduler`` only — no Host.
"""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.runtime.scheduler import FairScheduler

from .test_scheduler import FakeWorker, effect


@pytest.fixture
def fake_worker() -> FakeWorker:
    return FakeWorker()


@pytest.mark.asyncio
async def test_spec_22_wide_fanout_cannot_permanently_starve_second_task_backlog(
    fake_worker: FakeWorker,
) -> None:
    """§22 — while A's deep backlog remains, B's multiple queued LLM effects still get slots."""

    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=8, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    for index in range(8):
        await scheduler.enqueue(effect("A", f"a{index}"))
    await fake_worker.wait_started(1)

    for index in range(3):
        await scheduler.enqueue(effect("B", f"b{index}"))
    fake_worker.release.set()
    await fake_worker.wait_started(11)

    positions = {effect_id: index for index, effect_id in enumerate(fake_worker.started)}
    # Distinct from ``test_wide_task_cannot_starve_other_task`` (``b0 in started[:3]``):
    # pin that B's *backlog* is interleaved before A's queue drains.
    assert positions["b0"] < positions["a7"]
    assert positions["b1"] < positions["a7"]
    assert positions["b2"] < positions["a7"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_spec_22_barrier_ops_outrank_ordinary_child_across_tasks(
    fake_worker: FakeWorker,
) -> None:
    """§22 — stop/pause/report/continue barriers outrank ordinary child launch across tasks."""

    fake_worker.block = True
    scheduler = FairScheduler(global_llm=1, per_task_llm=4, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()
    await scheduler.enqueue(effect("A", "a-inflight", kind="agent", priority="normal"))
    await fake_worker.wait_started(1)

    for index in range(4):
        await scheduler.enqueue(effect("A", f"a-child-{index}", kind="agent", priority="normal"))

    # Spec names stop/pause/report/continue as barrier-class; pin via priority API.
    await scheduler.enqueue(effect("B", "stop", kind="control", priority="barrier"))
    await scheduler.enqueue(effect("B", "pause", kind="control", priority="barrier"))
    await scheduler.enqueue(effect("B", "report", kind="summarizer", priority="barrier"))
    await scheduler.enqueue(effect("B", "continue", kind="control", priority="barrier"))
    await scheduler.enqueue(effect("B", "b-child", kind="agent", priority="normal"))

    fake_worker.release.set()
    await fake_worker.wait_started(6)

    started = fake_worker.started
    assert started[0] == "a-inflight"
    barrier_ids = {"stop", "pause", "report", "continue"}
    first_ordinary = next(
        index
        for index, effect_id in enumerate(started)
        if effect_id.startswith("a-child") or effect_id == "b-child"
    )
    # Barrier-class work clears before any ordinary child (cross-task); exact
    # FIFO among barrier ops may skip a capacity-blocked summarizer past a
    # control ``other`` sibling — pin precedence over child, not intra-FIFO.
    assert set(started[1:first_ordinary]) == barrier_ids
    assert first_ordinary >= 5
    await scheduler.close()
