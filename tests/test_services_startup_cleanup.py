from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
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


class RecordingProvider(BundledProcedureProvider):
    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self.aclose_count = 0

    async def aclose(self) -> None:
        self.aclose_count += 1
        await super().aclose()


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


@pytest.mark.asyncio
async def test_start_failure_after_providers_closes_bundled_provider(
    monkeypatch, tmp_path: Path
) -> None:
    class LateFailingStore:
        def __init__(self) -> None:
            self.close_count = 0

        async def open(self) -> None:
            return None

        async def mark_active_rounds_interrupted(self, now: float) -> int:
            del now
            return 0

        async def close(self) -> None:
            self.close_count += 1

    class FakeDiscovery:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.closed = False

        async def refresh(self) -> None:
            return None

        def start(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    providers: list[RecordingProvider] = []

    def loader(
        agents: Any,
        procedures: Any,
        *,
        ctx: Any | None = None,
        web_search_config: Any | None = None,
        **kwargs: Any,
    ) -> RecordingProvider:
        del web_search_config, kwargs
        from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions

        provider = RecordingProvider(ctx)
        providers.append(provider)
        agents.replace_provider(
            "builtin",
            [definition.model_dump(mode="json") for definition in bundled_agent_definitions()],
        )
        procedures.replace_provider("builtin", provider.describe())
        return provider

    async def boom(self: LRSServiceContainer) -> None:
        raise RuntimeError("runtime start failure")

    monkeypatch.setattr(LRSServiceContainer, "_start_runtime", boom)

    store = LateFailingStore()
    context = SimpleNamespace(
        paths=SimpleNamespace(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
        logger=logging.getLogger("lrs-startup-provider-cleanup-test"),
        maisaka=None,
    )
    container = LRSServiceContainer(
        context,
        LRSConfig(),
        store_factory=lambda path: store,
        discovery_factory=FakeDiscovery,
        host_snapshot_loader=lambda: None,
        builtin_provider_loader=loader,
    )

    with pytest.raises(RuntimeError, match="runtime start failure"):
        await container.start()

    assert len(providers) == 1
    assert providers[0].aclose_count == 1
    assert container._bundled_procedure_provider is None
    assert container._state == "failed"
    assert store.close_count == 1


@pytest.mark.asyncio
async def test_ensure_ready_failure_degrades_vector_without_aborting_start(
    monkeypatch, tmp_path: Path
) -> None:
    """unsupported selector / ensure_ready 失败时 vector_index 降级，LRS 其余部分仍可启动。"""
    import lunagentic_research_swarm.services as services_module
    from lunagentic_research_swarm.errors import LRSError
    from lunagentic_research_swarm.storage.vectors import VectorOpResult

    class FakeDiscovery:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def refresh(self) -> None:
            return None

        def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    def loader(
        agents: Any,
        procedures: Any,
        *,
        ctx: Any | None = None,
        web_search_config: Any | None = None,
        **kwargs: Any,
    ) -> RecordingProvider:
        del web_search_config, kwargs
        from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions

        provider = RecordingProvider(ctx)
        agents.replace_provider(
            "builtin",
            [definition.model_dump(mode="json") for definition in bundled_agent_definitions()],
        )
        procedures.replace_provider("builtin", provider.describe())
        return provider

    async def boom_ready(self: Any) -> VectorOpResult:
        return VectorOpResult.fail(
            LRSError(
                "physical_embedding_selector_unsupported",
                "首发 embedding 不支持 model: 物理 pinning，请使用 task: selector",
            )
        )

    monkeypatch.setattr(services_module.VectorIndex, "ensure_ready", boom_ready)

    context = SimpleNamespace(
        paths=SimpleNamespace(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
        logger=logging.getLogger("lrs-vector-degrade-test"),
        llm=object(),
        maisaka=None,
    )
    container = LRSServiceContainer(
        context,
        LRSConfig(),
        discovery_factory=FakeDiscovery,
        host_snapshot_loader=lambda: {"models": [], "model_task_config": {}},
        builtin_provider_loader=loader,
    )

    await container.start()
    try:
        assert container._state == "running"
        vector_status = container._status["vector_index"]
        assert vector_status["status"] == "degraded"
        assert vector_status["code"] == "physical_embedding_selector_unsupported"
        assert vector_status["status"] != "healthy"
    finally:
        await container.close()
