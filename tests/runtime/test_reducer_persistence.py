from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.models import TaskSnapshot, TaskStatus
from lunagentic_research_swarm.runtime.events import ContinueRequested, FormalizationSucceeded
from lunagentic_research_swarm.runtime.controller import TaskController


class FakeStore:
    def __init__(self, *, fails: bool) -> None:
        self.fails = fails
        self.commands = []

    async def transact(self, commands) -> None:
        self.commands.append(tuple(commands))
        if self.fails:
            raise RuntimeError("storage unavailable")


class FakeScheduler:
    def __init__(self) -> None:
        self.launched = []

    async def enqueue(self, effect) -> None:
        self.launched.append(effect)


class AlwaysFailStore(FakeStore):
    pass


def event_factory() -> FormalizationSucceeded:
    return FormalizationSucceeded(
        event_id="evt-formalized",
        task_id="task-1",
        round_id="round-1",
        generation=0,
        occurred_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
        formalized_text="正式任务",
        formalized_sha256="hash",
    )


@pytest.mark.asyncio
async def test_effect_is_not_launched_when_transaction_fails() -> None:
    store = FakeStore(fails=True)
    scheduler = FakeScheduler()
    controller = TaskController(
        TaskSnapshot("task-1", TaskStatus.FORMALIZING, generation=0, active_round_id="round-1"),
        store=store,
        scheduler=scheduler,
    )

    await controller.submit(event_factory())
    await controller.drain_once()

    assert scheduler.launched == []
    assert controller.state.status.value == "FAILED"


@pytest.mark.asyncio
async def test_effect_is_launched_only_after_transaction_succeeds() -> None:
    store = FakeStore(fails=False)
    scheduler = FakeScheduler()
    controller = TaskController(
        TaskSnapshot("task-1", TaskStatus.FORMALIZING, generation=0, active_round_id="round-1"),
        store=store,
        scheduler=scheduler,
    )

    await controller.submit(event_factory())
    await controller.drain_once()

    assert len(store.commands) == 1
    assert len(scheduler.launched) == 1
    assert controller.state.status.value == "RUNNING"


@pytest.mark.asyncio
async def test_controller_rejects_events_after_best_effort_failure_also_fails() -> None:
    store = AlwaysFailStore(fails=True)
    scheduler = FakeScheduler()
    health: dict[str, object] = {}
    controller = TaskController(
        TaskSnapshot("task-1", TaskStatus.FORMALIZING, generation=0, active_round_id="round-1"),
        store=store,
        scheduler=scheduler,
        health=health,
    )

    await controller.submit(event_factory())
    await controller.drain_once()

    assert controller.stopped
    assert health["runtime"]["code"] == "storage_commit_failed"  # type: ignore[index]

    accepted = await controller.submit(
        ContinueRequested(
            "evt-late-continue",
            "task-1",
            "round-1",
            0,
            next_round_id="round-2",
            next_generation=1,
        )
    )

    assert accepted is False
    assert await controller.drain_once() is False
    assert scheduler.launched == []
