from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF, ON_MODEL_CONFIG_RELOAD
from maibot_sdk.context import PluginContext, PluginPaths

from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.extensions.contracts import AgentDefinition, ProcedureDefinition
from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningStatus


class FakeStore:
    def __init__(self, events: list[str], *, open_error: Exception | None = None) -> None:
        self.events = events
        self.open_error = open_error
        self.data_dir: Path | None = None
        self.runtime_dir: Path | None = None
        self.close_count = 0

    async def open(self) -> None:
        assert self.data_dir is not None and self.data_dir.is_dir()
        assert self.runtime_dir is not None and self.runtime_dir.is_dir()
        self.events.append("store.open")
        if self.open_error is not None:
            raise self.open_error

    async def mark_active_rounds_interrupted(self, now: float) -> int:
        assert now > 0
        self.events.append("store.interrupt")
        return 2

    async def close(self) -> None:
        self.events.append("store.close")
        self.close_count += 1

    async def transact(self, commands: Any) -> None:
        assert commands


class FakeDiscovery:
    def __init__(
        self,
        events: list[str],
        agents: Any,
        procedures: Any,
        interval: float,
        on_refresh: Callable[[Any, Any], None] | None,
    ) -> None:
        self.events = events
        self.agents = agents
        self.procedures = procedures
        self.interval = interval
        self.on_refresh = on_refresh
        self.closed = False
        self.background_task: asyncio.Task[None] | None = None
        self.extension_fingerprints: dict[str, str] = {}
        self.close_count = 0
        self.close_started: asyncio.Event | None = None
        self.close_release: asyncio.Event | None = None
        self.close_error: Exception | None = None

    async def refresh(self) -> None:
        self.events.append("extensions.refresh")
        if self.on_refresh is not None:
            self.on_refresh(self.agents, self.procedures)

    def start(self) -> None:
        self.events.append("extensions.start")
        self.background_task = asyncio.create_task(self._wait_forever(), name="fake-extension-refresh")

    async def _wait_forever(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.events.append("extensions.cancelled")
            raise

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_count += 1
        self.events.append("extensions.close")
        if self.close_started is not None:
            self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.background_task is not None:
            self.background_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self.background_task
            self.background_task = None
        if self.close_error is not None:
            raise self.close_error


class DiscoveryFactory:
    def __init__(
        self,
        events: list[str],
        on_refresh: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self.events = events
        self.on_refresh = on_refresh
        self.instances: list[FakeDiscovery] = []

    def __call__(
        self,
        ctx: Any,
        agents: Any,
        procedures: Any,
        *,
        refresh_interval_seconds: float,
        refresh_listener: Any = None,
    ) -> FakeDiscovery:
        assert ctx.api is not None
        del refresh_listener
        instance = FakeDiscovery(
            self.events,
            agents,
            procedures,
            refresh_interval_seconds,
            self.on_refresh,
        )
        self.instances.append(instance)
        return instance


class FakePinning:
    def __init__(self, available: bool) -> None:
        self.available = available

    def check_compatibility(self) -> PhysicalPinningStatus:
        return PhysicalPinningStatus(
            self.available,
            None if self.available else "physical_pinning_unsupported",
        )


def config_with(**sections: dict[str, Any]) -> LRSConfig:
    raw = LRSConfig().model_dump(mode="python")
    for section, updates in sections.items():
        raw[section].update(updates)
    return LRSConfig.model_validate(raw)


def agent_payload(agent_id: str, *, selector: str = "task:utils") -> AgentDefinition:
    return AgentDefinition.model_validate(
        {
            "agent_id": agent_id,
            "version": "1",
            "display_name": "测试根智能体",
            "description": "用于基础服务测试",
            "character_prompt": "只执行测试任务。",
            "model_selector": selector,
            "can_be_root": True,
        }
    )


def procedure_payload(procedure_id: str) -> ProcedureDefinition:
    return ProcedureDefinition.model_validate(
        {
            "procedure_id": procedure_id,
            "version": "1",
            "display_name": "测试 Procedure",
            "description": "用于基础服务测试",
            "arguments_schema": {"type": "object"},
            "result_schema": {"type": "object"},
        }
    )


def make_context(tmp_path: Path) -> PluginContext:
    async def rpc(
        method: str,
        plugin_id: str,
        payload: dict[str, Any] | None,
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        assert method == "cap.call"
        assert plugin_id == "com.0-hz.lunagentic-research-swarm"
        assert timeout_ms is None
        assert payload is not None
        capability = payload["capability"]
        if capability == "api.list":
            return {"success": True, "apis": []}
        raise AssertionError(f"未声明的测试 capability：{capability}")

    return PluginContext(
        "com.0-hz.lunagentic-research-swarm",
        rpc_call=rpc,
        paths=PluginPaths(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
    )


def build_container(
    plugin_module: Any,
    tmp_path: Path,
    *,
    config: LRSConfig | None = None,
    events: list[str] | None = None,
    store: FakeStore | None = None,
    discovery_factory: DiscoveryFactory | None = None,
    snapshot_loader: Callable[[], dict[str, Any] | None] | None = None,
    pinning_available: bool = False,
) -> tuple[Any, PluginContext, FakeStore, DiscoveryFactory, list[str]]:
    event_log = events if events is not None else []
    context = make_context(tmp_path)
    fake_store = store or FakeStore(event_log)
    fake_store.data_dir = context.paths.data_dir
    fake_store.runtime_dir = context.paths.runtime_dir
    factory = discovery_factory or DiscoveryFactory(event_log)
    loader = snapshot_loader or (lambda: None)
    container_cls = getattr(plugin_module, "LRSServiceContainer")
    container = container_cls(
        context,
        config or LRSConfig(),
        store_factory=lambda path: fake_store,
        discovery_factory=factory,
        host_snapshot_loader=loader,
        physical_pinning=FakePinning(pinning_available),
    )
    return container, context, fake_store, factory, event_log


@pytest.mark.asyncio
async def test_start_uses_strict_foundation_order_and_close_is_idempotent(plugin_module, tmp_path: Path) -> None:
    events: list[str] = []

    def snapshot_loader() -> dict[str, Any]:
        events.append("snapshot")
        return {"models": [], "model_task_config": {}}

    def builtin_loader(agents: Any, procedures: Any) -> None:
        assert not agents.provider_ids
        assert not procedures.provider_ids
        events.append("builtin")

    context = make_context(tmp_path)
    store = FakeStore(events)
    store.data_dir = context.paths.data_dir
    store.runtime_dir = context.paths.runtime_dir
    factory = DiscoveryFactory(events)
    container_cls = getattr(plugin_module, "LRSServiceContainer")
    container = container_cls(
        context,
        LRSConfig(),
        store_factory=lambda path: store,
        discovery_factory=factory,
        host_snapshot_loader=snapshot_loader,
        physical_pinning=FakePinning(False),
        builtin_provider_loader=builtin_loader,
    )

    await container.start()

    assert events == [
        "store.open",
        "store.interrupt",
        "snapshot",
        "builtin",
        "extensions.refresh",
        "extensions.start",
    ]
    await asyncio.sleep(0)
    await container.close()
    await container.close()
    assert factory.instances[0].close_count == 1
    assert store.close_count == 1
    assert events[-3:] == ["extensions.close", "extensions.cancelled", "store.close"]
    with pytest.raises(RuntimeError, match="已关闭"):
        container.health()
    with pytest.raises(RuntimeError, match="已关闭"):
        await container.refresh_extensions(reason="provider_request")


@pytest.mark.asyncio
async def test_started_service_exposes_production_research_manager(plugin_module, tmp_path: Path) -> None:
    container, _, _, _, _ = build_container(plugin_module, tmp_path)

    assert getattr(container, "manager", None) is None
    await container.start()

    assert container.manager is not None
    assert container.manager.scheduler is container.scheduler
    await container.close()
    assert container.manager is None


@pytest.mark.asyncio
async def test_sqlite_failure_propagates_through_plugin_load(plugin_module, tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    store = FakeStore(events, open_error=RuntimeError("字面 SQLite 打开失败"))
    container, context, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        events=events,
        store=store,
    )
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    monkeypatch.setattr(plugin_module, "LRSServiceContainer", lambda ctx, config: container)

    with pytest.raises(RuntimeError, match="字面 SQLite 打开失败"):
        await plugin.on_load()
    assert events[0] == "store.open"
    assert "extensions.refresh" not in events


@pytest.mark.asyncio
async def test_snapshot_and_provider_degradation_do_not_block_load(plugin_module, tmp_path: Path) -> None:
    events: list[str] = []

    def broken_snapshot() -> dict[str, Any]:
        raise RuntimeError("字面 Host snapshot 失败")

    def providers(agents: Any, procedures: Any) -> None:
        agents.replace_provider("provider.good", [agent_payload("good.root")])
        agents.reject_provider("provider.bad", ["字面 provider 失败"])

    factory = DiscoveryFactory(events, providers)
    config = config_with(plugin={"root_agent": "good.root"})
    container, _, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        config=config,
        events=events,
        discovery_factory=factory,
        snapshot_loader=broken_snapshot,
        pinning_available=False,
    )

    await container.start()
    health = container.health()

    assert health["sqlite"]["status"] == "healthy"
    assert health["initial_price_snapshot"] == {
        "status": "degraded",
        "code": "host_model_snapshot_unavailable",
    }
    assert health["extension_providers"]["agents"]["provider.good"]["status"] == "healthy"
    assert health["extension_providers"]["agents"]["provider.bad"] == {
        "status": "invalid",
        "code": "extension_provider_invalid",
    }
    await container.close()


@pytest.mark.asyncio
async def test_raised_extension_refresh_is_degraded_in_health_and_public_result(plugin_module, tmp_path: Path) -> None:
    def broken_refresh(agents: Any, procedures: Any) -> None:
        raise RuntimeError("字面扩展刷新抛错")

    container, _, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        discovery_factory=DiscoveryFactory([], broken_refresh),
    )

    await container.start()

    assert container.health()["extension_discovery"] == {
        "status": "degraded",
        "code": "extension_discovery_failed",
    }
    result = await container.refresh_extensions(reason="provider_request")
    assert result["status"] == "degraded"
    await container.close()


@pytest.mark.asyncio
async def test_real_discovery_health_tracks_periodic_failure_and_recovery(plugin_module, tmp_path: Path) -> None:
    fail_listing = True

    async def rpc(
        method: str,
        plugin_id: str,
        payload: dict[str, Any] | None,
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        assert method == "cap.call"
        assert plugin_id == "com.0-hz.lunagentic-research-swarm"
        assert timeout_ms is None
        assert payload is not None and payload["capability"] == "api.list"
        if fail_listing:
            return {"success": False, "error": {"code": "list_failed", "message": "字面列表失败"}}
        return {"success": True, "apis": []}

    context = PluginContext(
        "com.0-hz.lunagentic-research-swarm",
        rpc_call=rpc,
        paths=PluginPaths(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
    )
    events: list[str] = []
    store = FakeStore(events)
    store.data_dir = context.paths.data_dir
    store.runtime_dir = context.paths.runtime_dir
    container = plugin_module.LRSServiceContainer(
        context,
        LRSConfig(),
        store_factory=lambda path: store,
        host_snapshot_loader=lambda: {"models": [], "model_task_config": {}},
        physical_pinning=FakePinning(False),
    )
    await container.start()
    assert container.health()["extension_discovery"] == {
        "status": "degraded",
        "code": "extension_discovery_failed",
    }

    fail_listing = False
    await container._discovery.refresh()
    assert container.health()["extension_discovery"] == {"status": "healthy"}

    fail_listing = True
    await container._discovery.refresh()
    assert container.health()["extension_discovery"] == {
        "status": "degraded",
        "code": "extension_discovery_failed",
    }

    fail_listing = False
    await container._discovery.refresh()
    assert container.health()["extension_discovery"] == {"status": "healthy"}
    await container.close()


@pytest.mark.asyncio
async def test_periodic_persistence_failure_recovers_without_stopping_background_loop(
    plugin_module,
    tmp_path: Path,
) -> None:
    list_count = 0

    async def rpc(
        method: str,
        plugin_id: str,
        payload: dict[str, Any] | None,
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        nonlocal list_count
        assert method == "cap.call"
        assert plugin_id == "com.0-hz.lunagentic-research-swarm"
        assert timeout_ms is None
        assert payload is not None and payload["capability"] == "api.list"
        list_count += 1
        return {"success": True, "apis": []}

    class FlakyStore(FakeStore):
        def __init__(self, events: list[str]) -> None:
            super().__init__(events)
            self.fail_next = False
            self.failed = asyncio.Event()
            self.persisted = asyncio.Event()

        async def transact(self, commands: Any) -> None:
            assert commands
            if self.fail_next:
                self.fail_next = False
                self.failed.set()
                raise RuntimeError("字面 periodic SQLite persistence 失败")
            self.persisted.set()

    context = PluginContext(
        "com.0-hz.lunagentic-research-swarm",
        rpc_call=rpc,
        paths=PluginPaths(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
    )
    events: list[str] = []
    store = FlakyStore(events)
    store.data_dir = context.paths.data_dir
    store.runtime_dir = context.paths.runtime_dir
    container = plugin_module.LRSServiceContainer(
        context,
        LRSConfig(),
        store_factory=lambda path: store,
        host_snapshot_loader=lambda: {"models": [], "model_task_config": {}},
        physical_pinning=FakePinning(False),
    )
    await container.start()
    try:
        store.fail_next = True
        container._discovery.request_refresh()
        await asyncio.wait_for(store.failed.wait(), timeout=1)
        for _ in range(20):
            if container._discovery._inflight_refresh is None:
                break
            await asyncio.sleep(0)
        assert container.health()["extension_discovery"] == {
            "status": "critical",
            "code": "extension_fingerprint_persistence_failed",
        }

        store.persisted = asyncio.Event()
        recovered = await container.refresh_extensions(reason="provider_request")
        assert recovered["status"] == "healthy"
        assert container.health()["extension_discovery"] == {"status": "healthy"}
        assert not container._discovery.background_task.done()

        store.persisted = asyncio.Event()
        scans_before = list_count
        container._discovery.request_refresh()
        await asyncio.wait_for(store.persisted.wait(), timeout=1)
        assert list_count == scans_before + 1
        assert container.health()["extension_discovery"] == {"status": "healthy"}
    finally:
        with suppress(Exception):
            await container.close()


@pytest.mark.asyncio
async def test_health_reports_unexpectedly_finished_extension_background_task(
    plugin_module,
    tmp_path: Path,
) -> None:
    container, _, _, factory, _ = build_container(plugin_module, tmp_path)
    await container.start()
    discovery = factory.instances[0]
    assert discovery.background_task is not None
    discovery.background_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await discovery.background_task

    async def crash() -> None:
        raise RuntimeError("字面 background crash")

    discovery.background_task = asyncio.create_task(crash())
    await asyncio.sleep(0)

    assert container.health()["extension_discovery"] == {
        "status": "critical",
        "code": "extension_refresh_background_failed",
    }
    with suppress(RuntimeError):
        await discovery.background_task
    discovery.background_task = None
    await container.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_infos",
    [
        [None],
        [
            {
                "name": "describe_agents",
                "full_name": ".describe_agents",
                "plugin_id": "",
                "description": "无效 provider 身份",
                "version": "1",
                "public": True,
                "enabled": True,
                "dynamic": False,
                "offline_reason": "",
                "metadata": {
                    "description": "无效 provider 身份",
                    "version": "1",
                    "public": True,
                    "enabled": True,
                    "handler_name": "describe_agents",
                    "dynamic": False,
                    "lunagentic_extension": "agents",
                    "lunagentic_contract": "1",
                },
            }
        ],
    ],
)
async def test_real_descriptor_validation_failure_degrades_and_clean_scan_recovers(
    plugin_module,
    tmp_path: Path,
    malformed_infos: list[Any],
) -> None:
    visible_infos = malformed_infos

    async def rpc(
        method: str,
        plugin_id: str,
        payload: dict[str, Any] | None,
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        assert method == "cap.call"
        assert plugin_id == "com.0-hz.lunagentic-research-swarm"
        assert timeout_ms is None
        assert payload is not None and payload["capability"] == "api.list"
        return {"success": True, "apis": list(visible_infos)}

    context = PluginContext(
        "com.0-hz.lunagentic-research-swarm",
        rpc_call=rpc,
        paths=PluginPaths(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
    )
    events: list[str] = []
    store = FakeStore(events)
    store.data_dir = context.paths.data_dir
    store.runtime_dir = context.paths.runtime_dir
    container = plugin_module.LRSServiceContainer(
        context,
        LRSConfig(),
        store_factory=lambda path: store,
        host_snapshot_loader=lambda: {"models": [], "model_task_config": {}},
        physical_pinning=FakePinning(False),
    )
    await container.start()

    assert container.health()["extension_discovery"] == {
        "status": "degraded",
        "code": "extension_descriptor_invalid",
    }
    assert (await container.refresh_extensions(reason="provider_request"))["status"] == "degraded"

    visible_infos = []
    await container._discovery.refresh()
    assert container.health()["extension_discovery"] == {"status": "healthy"}
    assert (await container.refresh_extensions(reason="provider_request"))["status"] == "healthy"
    await container.close()


@pytest.mark.asyncio
async def test_health_is_explicit_for_root_summarizer_and_physical_pinning(plugin_module, tmp_path: Path) -> None:
    unavailable, _, _, _, _ = build_container(
        plugin_module,
        tmp_path / "unavailable",
        snapshot_loader=lambda: {
            "models": [{"name": "m", "price_in": 1, "price_out": 2}],
            "model_task_config": {"utils": {"model_list": ["m"]}},
        },
        pinning_available=False,
    )
    await unavailable.start()
    health = unavailable.health()
    assert health["root_agent"] == {
        "status": "degraded",
        "code": "root_agent_unavailable",
        "agent_id": "builtin.quick_thinker",
    }
    assert health["root_selector"] == {
        "status": "degraded",
        "code": "root_agent_unavailable",
    }
    assert health["summarizer_selector"] == {
        "status": "degraded",
        "code": "summarizer_selector_unavailable",
        "selector": "task:mid_memory",
    }
    assert health["physical_pinning"] == {
        "status": "degraded",
        "code": "physical_pinning_unsupported",
    }
    await unavailable.close()

    def root_provider(agents: Any, procedures: Any) -> None:
        agents.replace_provider("provider.agents", [agent_payload("agents.root")])

    healthy_config = config_with(plugin={"root_agent": "agents.root"})
    healthy, _, _, _, _ = build_container(
        plugin_module,
        tmp_path / "healthy",
        config=healthy_config,
        discovery_factory=DiscoveryFactory([], root_provider),
        snapshot_loader=lambda: {
            "models": [{"name": "m", "price_in": 1, "price_out": 2}],
            "model_task_config": {
                "utils": {"model_list": ["m"]},
                "mid_memory": {"model_list": ["m"]},
            },
        },
        pinning_available=True,
    )
    await healthy.start()
    health = healthy.health()
    assert health["root_agent"]["status"] == "healthy"
    assert health["root_selector"] == {
        "status": "healthy",
        "selector": "task:utils",
    }
    assert health["summarizer_selector"] == {
        "status": "healthy",
        "selector": "task:mid_memory",
    }
    assert health["physical_pinning"] == {"status": "healthy"}
    await healthy.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "budget", "expect_warning"),
    [
        ({"name": "priced", "price_in": 1, "price_out": 2}, 59.0, True),
        ({"name": "free"}, 0.0, False),
    ],
)
async def test_load_warns_for_low_priced_budget_but_not_free_models(
    plugin_module,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    model: dict[str, Any],
    budget: float,
    expect_warning: bool,
) -> None:
    def root_provider(agents: Any, procedures: Any) -> None:
        agents.replace_provider("provider.agents", [agent_payload("agents.root")])

    config = config_with(
        plugin={"root_agent": "agents.root"},
        budget={"default_effort_credits": budget},
    )
    container, _, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        config=config,
        discovery_factory=DiscoveryFactory([], root_provider),
        snapshot_loader=lambda: {
            "models": [model],
            "model_task_config": {"utils": {"model_list": [model["name"]]}},
        },
    )

    await container.start()

    warning_records = [record for record in caplog.records if "根调用保守估算" in record.message]
    assert bool(warning_records) is expect_warning
    await container.close()


@pytest.mark.asyncio
async def test_model_reload_keeps_only_safe_pricing_fields_and_detaches_input(plugin_module, tmp_path: Path) -> None:
    container, context, _, _, _ = build_container(plugin_module, tmp_path)
    await container.start()
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    plugin._services = container
    snapshot = {
        "api_providers": [{"name": "secret", "api_key": "never-store"}],
        "models": [{"name": "m", "price_in": 1, "price_out": 2, "api_provider": "secret"}],
        "model_task_config": {"utils": {"model_list": ["m"]}},
    }

    await plugin.on_config_update(ON_MODEL_CONFIG_RELOAD, snapshot, "v2")
    snapshot["models"][0]["price_in"] = 99
    snapshot["model_task_config"]["utils"]["model_list"].append("secret-model")

    assert plugin._services.price_catalog.debug_snapshot() == {
        "models": {"m": {"price_in": 1.0, "cache": False, "cache_price_in": 0.0, "price_out": 2.0}},
        "tasks": {"utils": ["m"]},
    }
    assert "never-store" not in repr(plugin._services)
    assert plugin._services.health()["initial_price_snapshot"] == {"status": "healthy"}
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_invalid_model_reload_keeps_catalog_and_does_not_log_success(
    plugin_module,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    container, context, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        snapshot_loader=lambda: {
            "models": [{"name": "m", "price_in": 1}],
            "model_task_config": {},
        },
    )
    await container.start()
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    plugin._services = container

    await plugin.on_config_update(ON_MODEL_CONFIG_RELOAD, {"models": "invalid"}, "bad-v2")

    assert container.price_catalog.debug_snapshot()["models"]["m"]["price_in"] == 1.0
    assert container.health()["initial_price_snapshot"] == {
        "status": "degraded",
        "code": "host_model_snapshot_invalid",
    }
    assert not any("快照已更新" in record.message for record in caplog.records)
    assert any("未应用" in record.message for record in caplog.records)
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_initial_snapshot_loader_does_not_retain_raw_snapshot(plugin_module, tmp_path: Path) -> None:
    class Snapshot(dict[str, Any]):
        pass

    snapshot = Snapshot(
        {
            "api_providers": [{"api_key": "never-retain"}],
            "models": [{"name": "m", "price_in": 1}],
            "model_task_config": {},
        }
    )
    snapshot_ref = weakref.ref(snapshot)

    def loader(captured: Snapshot = snapshot) -> Snapshot:
        return captured

    container, _, _, _, _ = build_container(plugin_module, tmp_path, snapshot_loader=loader)
    await container.start()
    del loader
    del snapshot
    gc.collect()

    assert snapshot_ref() is None
    assert container.price_catalog.debug_snapshot()["models"]["m"]["price_in"] == 1.0
    await container.close()


@pytest.mark.asyncio
async def test_self_reload_updates_only_live_safety_refresh_and_boundary_health(plugin_module, tmp_path: Path) -> None:
    container, context, _, factory, _ = build_container(plugin_module, tmp_path)
    await container.start()
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    plugin._services = container
    updated = config_with(
        plugin={"root_agent": "other.root"},
        summarizer={"selector": "task:new_summary"},
        scheduler={"max_global_llm_concurrency": 3},
        extensions={"refresh_interval_seconds": 120},
    ).model_dump(mode="python")

    await plugin.on_config_update(CONFIG_RELOAD_SCOPE_SELF, updated, "self-v2")
    updated["scheduler"]["max_global_llm_concurrency"] = 999

    assert plugin._services.safety_limits["max_global_llm_concurrency"] == 3
    assert len(factory.instances) == 2
    assert factory.instances[0].closed
    assert factory.instances[1].interval == 120.0
    assert plugin._services.health()["config_reload"] == {
        "status": "healthy",
        "code": "live_limits_updated",
        "version": "self-v2",
        "next_round": ["catalog", "selectors", "prompt"],
    }
    assert plugin._services.health()["root_agent"]["agent_id"] == "other.root"
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_self_reload_atomically_replaces_detached_next_round_snapshot(plugin_module, tmp_path: Path) -> None:
    """若 reload 仍只改 live limits，新 round 会继续使用旧目录、selector、预算和价格。"""

    def providers(agents: Any, procedures: Any) -> None:
        agents.replace_provider(
            "provider.agents",
            [agent_payload("agents.old_root"), agent_payload("agents.new_root")],
        )
        procedures.replace_provider("provider.tools", [procedure_payload("tools.fetch")])

    initial = config_with(
        plugin={"root_agent": "agents.old_root"},
        llm={"force_selector": "task:old_force"},
        summarizer={"selector": "task:old_summary"},
        budget={"default_effort_credits": 25.0},
        pricing={"models": {"priced": {"price_in": 1.0}}},
        agents={"agents.old_root": {"selector": "task:old_agent"}},
        procedures={"tools.fetch": {"timeout_seconds": 11.0}},
    )
    container, context, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        config=initial,
        discovery_factory=DiscoveryFactory([], providers),
        snapshot_loader=lambda: {
            "models": [{"name": "priced", "price_in": 9.0}],
            "model_task_config": {
                "old_force": {"model_list": ["priced"]},
                "new_force": {"model_list": ["priced"]},
            },
        },
    )
    await container.start()
    before = await container.snapshot_next_round()
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(initial.model_dump(mode="python"))
    plugin._services = container
    updated = config_with(
        plugin={"root_agent": "agents.new_root"},
        llm={"force_selector": "task:new_force"},
        summarizer={"selector": "task:new_summary"},
        budget={"default_effort_credits": 250.0},
        pricing={"models": {"priced": {"price_in": 2.0, "price_out": 3.0}}},
        agents={"agents.new_root": {"selector": "task:new_agent"}},
        procedures={"tools.fetch": {"timeout_seconds": 22.0}},
    ).model_dump(mode="python")

    await plugin.on_config_update(CONFIG_RELOAD_SCOPE_SELF, updated, "self-v3")
    updated["plugin"]["root_agent"] = "mutated.root"
    updated["agents"]["agents.new_root"]["selector"] = "task:mutated"
    updated["pricing"]["models"]["priced"]["price_in"] = 99.0
    after = await container.snapshot_next_round()

    assert before.root_agent == "agents.old_root"
    assert before.root_force_selector == "task:old_force"
    assert before.summarizer_selector == "task:old_summary"
    assert before.default_effort_credits == 25.0
    assert before.agent_catalog.get("agents.old_root").definition.model_selector == "task:old_agent"
    assert before.procedure_catalog.get("tools.fetch").definition.timeout_seconds == 11.0
    assert before.price_catalog.resolve_model("priced").profile.price_in == 1.0

    assert after.root_agent == "agents.new_root"
    assert after.root_force_selector == "task:new_force"
    assert after.summarizer_selector == "task:new_summary"
    assert after.default_effort_credits == 250.0
    assert after.agent_catalog.get("agents.new_root").definition.model_selector == "task:new_agent"
    assert after.procedure_catalog.get("tools.fetch").definition.timeout_seconds == 22.0
    assert after.price_catalog.resolve_model("priced").profile.price_in == 2.0
    assert after.price_catalog.resolve_model("priced").profile.price_out == 3.0
    assert after.price_catalog is not before.price_catalog
    assert container.agent_registry.snapshot({}).root_agent == "agents.new_root"
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_self_reload_and_unload_are_serialized_without_refresh_task_leak(plugin_module, tmp_path: Path) -> None:
    container, context, store, factory, _ = build_container(plugin_module, tmp_path)
    await container.start()
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    plugin._services = container
    first = factory.instances[0]
    first.close_started = asyncio.Event()
    first.close_release = asyncio.Event()
    updated = config_with(extensions={"refresh_interval_seconds": 120}).model_dump(mode="python")

    reload_task = asyncio.create_task(plugin.on_config_update(CONFIG_RELOAD_SCOPE_SELF, updated, "self-v2"))
    await first.close_started.wait()
    unload_task = asyncio.create_task(plugin.on_unload())
    await asyncio.sleep(0)
    assert not unload_task.done()
    first.close_release.set()
    await reload_task
    await unload_task

    assert len(factory.instances) == 2
    assert factory.instances[1].closed
    assert factory.instances[1].background_task is None
    assert store.close_count == 1
    with pytest.raises(RuntimeError, match="已关闭"):
        await plugin.refresh_extensions()


@pytest.mark.asyncio
async def test_close_still_closes_sqlite_and_becomes_idempotent_when_refresh_shutdown_fails(
    plugin_module,
    tmp_path: Path,
) -> None:
    container, _, store, factory, _ = build_container(plugin_module, tmp_path)
    await container.start()
    factory.instances[0].close_error = RuntimeError("字面 refresh shutdown 失败")

    with pytest.raises(RuntimeError, match="字面 refresh shutdown 失败"):
        await container.close()

    assert store.close_count == 1
    assert container.closed
    await container.close()
    assert store.close_count == 1


@pytest.mark.asyncio
async def test_refresh_api_fails_outside_running_lifecycle_and_calls_service(plugin_module, tmp_path: Path) -> None:
    plugin = plugin_module.create_plugin()
    with pytest.raises(RuntimeError, match="尚未初始化"):
        await plugin.refresh_extensions()

    container, context, _, _, events = build_container(plugin_module, tmp_path)
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    plugin._services = container
    await container.start()

    result = await plugin.refresh_extensions()

    assert result["status"] == "healthy"
    assert result["reason"] == "provider_request"
    assert events.count("extensions.refresh") == 2
    await plugin.on_unload()
    with pytest.raises(RuntimeError, match="已关闭"):
        await plugin.refresh_extensions()
