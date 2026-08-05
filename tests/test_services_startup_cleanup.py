from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.services import LRSServiceContainer


class FailingStore:
    def __init__(self) -> None:
        self.opened = False
        self.close_count = 0

    async def open(self) -> None:
        self.opened = True

    async def mark_active_rounds_interrupted(self, now: float) -> int:
        del now
        raise RuntimeError("startup interruption failure")

    async def close(self) -> None:
        self.close_count += 1


class RecordingOutbox:
    def __init__(self, store: Any, maisaka: Any, **kwargs: Any) -> None:
        del store, maisaka, kwargs
        self.start_count = 0
        self.close_count = 0

    async def start(self) -> None:
        self.start_count += 1

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_start_failure_closes_outbox_and_store(monkeypatch, tmp_path: Path) -> None:
    import lunagentic_research_swarm.services as services_module

    store = FailingStore()
    outboxes: list[RecordingOutbox] = []

    def outbox_factory(store_arg: Any, maisaka: Any, **kwargs: Any) -> RecordingOutbox:
        outbox = RecordingOutbox(store_arg, maisaka, **kwargs)
        outboxes.append(outbox)
        return outbox

    monkeypatch.setattr(services_module, "MaisakaOutbox", outbox_factory)
    context = SimpleNamespace(
        paths=SimpleNamespace(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
        logger=logging.getLogger("lrs-startup-cleanup-test"),
        maisaka=object(),
    )
    container = LRSServiceContainer(
        context,
        LRSConfig(),
        store_factory=lambda path: store,
        host_snapshot_loader=lambda: None,
    )

    with pytest.raises(RuntimeError, match="startup interruption failure"):
        await container.start()

    assert len(outboxes) == 1
    assert outboxes[0].start_count == 1
    assert outboxes[0].close_count == 1
    assert store.opened
    assert store.close_count == 1
    assert container._state == "failed"
