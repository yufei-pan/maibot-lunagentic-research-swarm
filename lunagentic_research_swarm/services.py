"""LRS 基础服务生命周期、热更新与健康状态编排。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from lunagentic_research_swarm.agents.registry import AgentRegistry, RootAgentUnavailableError
from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.extensions.discovery import ExtensionDiscovery
from lunagentic_research_swarm.llm.gateway import HostModelSnapshotReader, ModelSelector, resolve_generation_selector
from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningAdapter, PhysicalPinningStatus
from lunagentic_research_swarm.llm.pricing import PriceCatalog
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore

StoreFactory = Callable[[Path], Any]
DiscoveryFactory = Callable[..., Any]
BuiltinProviderLoader = Callable[[AgentRegistry, ProcedureRegistry], None]


def _empty_builtin_provider_loader(agents: AgentRegistry, procedures: ProcedureRegistry) -> None:
    """本阶段只保留内置 provider 接线点，不注册任何默认定义。"""

    del agents, procedures


class LRSServiceContainer:
    """组合基础服务，并冻结跨生命周期边界的安全值快照。"""

    def __init__(
        self,
        ctx: Any,
        config: LRSConfig,
        *,
        store_factory: StoreFactory = SQLiteStateStore,
        discovery_factory: DiscoveryFactory = ExtensionDiscovery,
        host_snapshot_loader: Callable[[], dict[str, Any] | None] | None = None,
        physical_pinning: Any | None = None,
        builtin_provider_loader: BuiltinProviderLoader = _empty_builtin_provider_loader,
    ) -> None:
        self._ctx = ctx
        self._store = store_factory(Path(ctx.paths.data_dir) / "lrs-state.sqlite3")
        self._discovery_factory = discovery_factory
        self._host_snapshot_loader = host_snapshot_loader
        self._physical_pinning = physical_pinning or PhysicalPinningAdapter()
        self._builtin_provider_loader = builtin_provider_loader

        self.agent_registry = AgentRegistry(root_agent=config.plugin.root_agent)
        self.procedure_registry = ProcedureRegistry()
        self._agent_overrides = {name: value.model_copy(deep=True) for name, value in config.agents.items()}
        self._root_agent = str(config.plugin.root_agent)
        self._root_force_selector = str(config.llm.force_selector)
        self._summarizer_selector = str(config.summarizer.selector)
        self._default_effort_credits = float(config.budget.default_effort_credits)
        self._plugin_price_overrides = {
            name: value.model_dump(mode="python") for name, value in config.pricing.models.items()
        }
        self._refresh_interval_seconds = float(config.extensions.refresh_interval_seconds)
        self._safety_limits = self._extract_safety_limits(config)

        self.price_catalog: PriceCatalog | None = None
        self._discovery: Any | None = None
        self._state = "new"
        self._lifecycle_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._status: dict[str, dict[str, Any]] = {
            "sqlite": {"status": "pending"},
            "initial_price_snapshot": {"status": "pending"},
            "physical_pinning": {"status": "pending"},
            "config_reload": {"status": "healthy", "code": "not_reloaded"},
        }

    @staticmethod
    def _extract_safety_limits(config: LRSConfig) -> dict[str, int | float]:
        return {
            "default_time_budget_seconds": int(config.timing.default_time_budget_seconds),
            "grace_period_seconds": int(config.timing.grace_period_seconds),
            "pause_timeout_seconds": int(config.timing.pause_timeout_seconds),
            "feedback_wait_seconds": int(config.timing.feedback_wait_seconds),
            "max_task_llm_concurrency": int(config.scheduler.max_task_llm_concurrency),
            "max_global_llm_concurrency": int(config.scheduler.max_global_llm_concurrency),
            "max_task_procedure_concurrency": int(config.scheduler.max_task_procedure_concurrency),
            "max_delegations_per_turn": int(config.scheduler.max_delegations_per_turn),
            "max_branch_depth": int(config.scheduler.max_branch_depth),
            "max_agent_calls_per_task": int(config.scheduler.max_agent_calls_per_task),
            "auto_compact_tokens": int(config.context.auto_compact_tokens),
            "reserved_output_tokens": int(config.context.reserved_output_tokens),
            "safety_margin_tokens": int(config.context.safety_margin_tokens),
            "max_correction_turns": int(config.protocol.max_correction_turns),
        }

    @property
    def safety_limits(self) -> Mapping[str, int | float]:
        return dict(self._safety_limits)

    @property
    def closed(self) -> bool:
        return self._state == "closed"

    def __repr__(self) -> str:
        return f"LRSServiceContainer(state={self._state!r})"

    async def start(self) -> None:
        if self._state != "new":
            raise RuntimeError(f"LRS 基础服务无法从 {self._state} 状态启动")
        self._state = "starting"
        try:
            self._ctx.paths.data_dir.mkdir(parents=True, exist_ok=True)
            self._ctx.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
            await self._store.open()
            self._status["sqlite"] = {"status": "healthy"}
            interrupted = await self._store.mark_active_rounds_interrupted(time.time())
            self._status["legacy_rounds"] = {
                "status": "healthy",
                "interrupted": int(interrupted),
            }
        except BaseException:
            self._status["sqlite"] = {"status": "critical", "code": "sqlite_initialization_failed"}
            self._state = "failed"
            raise

        self._load_initial_price_catalog()
        self._builtin_provider_loader(self.agent_registry, self.procedure_registry)
        self._discovery = self._new_discovery(self._refresh_interval_seconds)
        await self._refresh_external_extensions()
        self._record_physical_pinning_health()
        self._warn_for_low_root_budget()
        self._discovery.start()
        self._state = "running"

    def _load_initial_price_catalog(self) -> None:
        loader = self._host_snapshot_loader
        self._host_snapshot_loader = None
        reader = (
            HostModelSnapshotReader(loader=loader)
            if loader is not None
            else HostModelSnapshotReader()
        )
        snapshot = reader.read()
        self.price_catalog = PriceCatalog.from_host_snapshot(snapshot, self._plugin_price_overrides)
        self._status["initial_price_snapshot"] = (
            {"status": "healthy"}
            if snapshot is not None
            else {"status": "degraded", "code": "host_model_snapshot_unavailable"}
        )

    def _new_discovery(self, interval: float) -> Any:
        return self._discovery_factory(
            self._ctx,
            self.agent_registry,
            self.procedure_registry,
            refresh_interval_seconds=interval,
        )

    async def _refresh_external_extensions(self) -> None:
        assert self._discovery is not None
        try:
            await self._discovery.refresh()
        except Exception:
            self._status["extension_discovery"] = {
                "status": "degraded",
                "code": "extension_discovery_failed",
            }
            self._ctx.logger.warning("LRS 扩展初始刷新失败；插件将以降级目录继续加载", exc_info=True)
        else:
            # Discovery 会把可恢复的 Host/list 失败写入 live fingerprint；正常返回本身不缓存降级。
            self._status["extension_discovery"] = {"status": "healthy"}

    def _record_physical_pinning_health(self) -> None:
        try:
            status = self._physical_pinning.check_compatibility()
        except Exception:
            status = PhysicalPinningStatus(False, "physical_pinning_unsupported")
        self._status["physical_pinning"] = (
            {"status": "healthy"}
            if status.available
            else {"status": "degraded", "code": status.error_code or "physical_pinning_unsupported"}
        )

    def _root_selector(self) -> str | None:
        try:
            snapshot = self.agent_registry.snapshot(self._agent_overrides)
        except RootAgentUnavailableError:
            return None
        entry = snapshot.get(self._root_agent)
        if entry is None:
            return None
        try:
            return resolve_generation_selector(entry.definition.model_selector, self._root_force_selector).raw
        except Exception:
            return None

    def _warn_for_low_root_budget(self) -> None:
        assert self.price_catalog is not None
        selector = self._root_selector()
        if selector is None:
            return
        warning = self.price_catalog.low_budget_warning(selector, self._default_effort_credits)
        if warning is not None:
            self._ctx.logger.warning(
                "%s（按 500000 个 cache miss 输入 token 与 50000 个输出 token 估算）",
                warning,
            )

    def _ensure_running(self) -> None:
        if self._state == "closed":
            raise RuntimeError("LRS 基础服务已关闭")
        if self._state != "running":
            raise RuntimeError("LRS 基础服务尚未初始化")

    async def refresh_extensions(self, *, reason: str) -> dict[str, Any]:
        self._ensure_running()
        async with self._refresh_lock:
            self._ensure_running()
            await self._refresh_external_extensions()
        provider_health = self._extension_provider_health()
        degraded = self._extension_discovery_health()["status"] != "healthy" or any(
            item["status"] != "healthy"
            for kind in provider_health.values()
            for item in kind.values()
            if item["status"] != "removed"
        )
        return {
            "status": "degraded" if degraded else "healthy",
            "reason": str(reason),
            "extension_providers": provider_health,
        }

    def update_model_snapshot(self, snapshot: Mapping[str, Any]) -> bool:
        self._ensure_running()
        assert self.price_catalog is not None
        try:
            self.price_catalog.replace_host_snapshot(snapshot)
        except Exception:
            self._status["initial_price_snapshot"] = {
                "status": "degraded",
                "code": "host_model_snapshot_invalid",
            }
            self._ctx.logger.warning("LRS 模型热更新快照无效；继续使用上一个安全价格快照", exc_info=True)
            return False
        self._status["initial_price_snapshot"] = {"status": "healthy"}
        return True

    async def update_self_config(self, config: LRSConfig, *, version: str) -> None:
        async with self._lifecycle_lock:
            self._ensure_running()
            detached_limits = self._extract_safety_limits(config)
            new_interval = float(config.extensions.refresh_interval_seconds)
            self._safety_limits = detached_limits
            if new_interval != self._refresh_interval_seconds:
                async with self._refresh_lock:
                    self._ensure_running()
                    assert self._discovery is not None
                    old_discovery = self._discovery
                    await old_discovery.close()
                    self._refresh_interval_seconds = new_interval
                    self._discovery = self._new_discovery(new_interval)
                    self._discovery.start()
            self._status["config_reload"] = {
                "status": "healthy",
                "code": "live_limits_updated",
                "version": str(version),
                "next_round": ["catalog", "selectors", "prompt"],
            }

    def _selector_health(self, selector: str) -> dict[str, Any]:
        assert self.price_catalog is not None
        try:
            parsed = ModelSelector.parse(selector)
        except Exception:
            return {"status": "degraded", "code": "selector_invalid", "selector": selector}
        debug = self.price_catalog.debug_snapshot()
        if parsed.scheme == "task":
            available = bool(debug["tasks"].get(parsed.name))
        else:
            available = parsed.name in debug["models"]
            if available and self._status["physical_pinning"]["status"] != "healthy":
                return {
                    "status": "degraded",
                    "code": "physical_pinning_unsupported",
                    "selector": selector,
                }
        return (
            {"status": "healthy", "selector": selector}
            if available
            else {"status": "degraded", "code": "selector_unavailable", "selector": selector}
        )

    def _root_health(self) -> tuple[dict[str, Any], dict[str, Any]]:
        selector = self._root_selector()
        if selector is None:
            return (
                {
                    "status": "degraded",
                    "code": "root_agent_unavailable",
                    "agent_id": self._root_agent,
                },
                {"status": "degraded", "code": "root_agent_unavailable"},
            )
        selector_health = self._selector_health(selector)
        if selector_health.get("code") == "selector_unavailable":
            selector_health["code"] = "root_selector_unavailable"
        return ({"status": "healthy", "agent_id": self._root_agent}, selector_health)

    def _summarizer_health(self) -> dict[str, Any]:
        try:
            selector = resolve_generation_selector(self._summarizer_selector, self._root_force_selector).raw
        except Exception:
            return {
                "status": "degraded",
                "code": "summarizer_selector_invalid",
                "selector": self._summarizer_selector,
            }
        result = self._selector_health(selector)
        if result.get("code") == "selector_unavailable":
            result["code"] = "summarizer_selector_unavailable"
        return result

    @staticmethod
    def _provider_map(source: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for provider_id, item in sorted(source.items()):
            if item.status == "healthy":
                result[provider_id] = {"status": "healthy"}
            elif item.status == "removed":
                result[provider_id] = {"status": "removed", "code": "extension_provider_removed"}
            else:
                result[provider_id] = {"status": "invalid", "code": "extension_provider_invalid"}
        return result

    def _extension_provider_health(self) -> dict[str, dict[str, dict[str, str]]]:
        return {
            "agents": self._provider_map(self.agent_registry.health),
            "procedures": self._provider_map(self.procedure_registry.health),
        }

    def _extension_discovery_health(self) -> dict[str, str]:
        assert self._discovery is not None
        fingerprints = self._discovery.extension_fingerprints
        if "discovery" in fingerprints:
            return {"status": "degraded", "code": "extension_discovery_failed"}
        if any(key.startswith("descriptor:") for key in fingerprints):
            return {"status": "degraded", "code": "extension_descriptor_invalid"}
        return dict(self._status["extension_discovery"])

    def health(self) -> dict[str, Any]:
        self._ensure_running()
        root_agent, root_selector = self._root_health()
        return {
            "sqlite": dict(self._status["sqlite"]),
            "legacy_rounds": dict(self._status["legacy_rounds"]),
            "initial_price_snapshot": dict(self._status["initial_price_snapshot"]),
            "physical_pinning": dict(self._status["physical_pinning"]),
            "extension_discovery": self._extension_discovery_health(),
            "extension_providers": self._extension_provider_health(),
            "root_agent": root_agent,
            "root_selector": root_selector,
            "summarizer_selector": self._summarizer_health(),
            "config_reload": dict(self._status["config_reload"]),
        }

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._state == "closed":
                return
            self._state = "closing"
            discovery_error: BaseException | None = None
            if self._discovery is not None:
                try:
                    await self._discovery.close()
                except BaseException as exc:
                    discovery_error = exc
            try:
                await self._store.close()
            except BaseException as store_error:
                self._state = "closed"
                if discovery_error is not None:
                    raise discovery_error from store_error
                raise
            self._state = "closed"
            if discovery_error is not None:
                raise discovery_error
