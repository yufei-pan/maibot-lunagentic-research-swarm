"""LRS 基础服务生命周期、热更新与健康状态编排。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.agents.registry import (
    AgentCatalogSnapshot,
    AgentRegistry,
    RootAgentUnavailableError,
)
from lunagentic_research_swarm.config import AgentOverride, LRSConfig, ProcedureOverride
from lunagentic_research_swarm.extensions.contracts import ExtensionRefreshDelta
from lunagentic_research_swarm.extensions.discovery import ExtensionDiscovery
from lunagentic_research_swarm.llm.gateway import HostModelSnapshotReader, ModelSelector, resolve_generation_selector
from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningAdapter, PhysicalPinningStatus
from lunagentic_research_swarm.llm.pricing import PriceCatalog
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogSnapshot, ProcedureRegistry
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand

StoreFactory = Callable[[Path], Any]
DiscoveryFactory = Callable[..., Any]
BuiltinProviderLoader = Callable[[AgentRegistry, ProcedureRegistry], None]


@dataclass(frozen=True, slots=True)
class NextRoundSnapshot:
    """启动一个 round 时一次性冻结的配置与目录边界。"""

    root_agent: str
    root_force_selector: str
    summarizer_selector: str
    default_effort_credits: float
    agent_catalog: AgentCatalogSnapshot
    procedure_catalog: ProcedureCatalogSnapshot
    price_catalog: PriceCatalog


@dataclass(frozen=True, slots=True)
class _NextRoundState:
    root_agent: str
    root_force_selector: str
    summarizer_selector: str
    default_effort_credits: float
    agent_overrides: tuple[tuple[str, AgentOverride], ...]
    procedure_overrides: tuple[tuple[str, ProcedureOverride], ...]
    plugin_price_overrides: tuple[tuple[str, Mapping[str, Any]], ...]
    price_catalog: PriceCatalog | None

    @classmethod
    def from_config(cls, config: LRSConfig, *, price_catalog: PriceCatalog | None) -> _NextRoundState:
        return cls(
            root_agent=str(config.plugin.root_agent),
            root_force_selector=str(config.llm.force_selector),
            summarizer_selector=str(config.summarizer.selector),
            default_effort_credits=float(config.budget.default_effort_credits),
            agent_overrides=tuple(
                sorted((name, value.model_copy(deep=True)) for name, value in config.agents.items())
            ),
            procedure_overrides=tuple(
                sorted((name, value.model_copy(deep=True)) for name, value in config.procedures.items())
            ),
            plugin_price_overrides=tuple(
                sorted(
                    (
                        name,
                        MappingProxyType(dict(value.model_dump(mode="python"))),
                    )
                    for name, value in config.pricing.models.items()
                )
            ),
            price_catalog=price_catalog,
        )

    def detached_agent_overrides(self) -> dict[str, AgentOverride]:
        return {name: value.model_copy(deep=True) for name, value in self.agent_overrides}

    def detached_procedure_overrides(self) -> dict[str, ProcedureOverride]:
        return {name: value.model_copy(deep=True) for name, value in self.procedure_overrides}

    def detached_price_overrides(self) -> dict[str, dict[str, Any]]:
        return {name: dict(value) for name, value in self.plugin_price_overrides}


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
        self._next_round_state = _NextRoundState.from_config(config, price_catalog=None)
        self._refresh_interval_seconds = float(config.extensions.refresh_interval_seconds)
        self._safety_limits = self._extract_safety_limits(config)

        self._discovery: Any | None = None
        self._state = "new"
        self._lifecycle_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._last_extension_persistence_error: Exception | None = None
        self._status: dict[str, dict[str, Any]] = {
            "sqlite": {"status": "pending"},
            "initial_price_snapshot": {"status": "pending"},
            "physical_pinning": {"status": "pending"},
            "extension_fingerprint_store": {"status": "pending"},
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
    def price_catalog(self) -> PriceCatalog | None:
        return self._next_round_state.price_catalog

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
        catalog = PriceCatalog.from_host_snapshot(
            snapshot,
            self._next_round_state.detached_price_overrides(),
        )
        self._next_round_state = replace(self._next_round_state, price_catalog=catalog)
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
            refresh_listener=self._persist_extension_refresh,
        )

    async def _persist_extension_refresh(self, delta: ExtensionRefreshDelta) -> None:
        commands = [
            StoreCommand(
                "insert_extension_fingerprint",
                {
                    "event_id": f"ext_{uuid.uuid4().hex}",
                    "provider_plugin_id": event.provider_plugin_id,
                    "extension_kind": event.extension_kind,
                    "fingerprint": event.fingerprint,
                    "availability": event.availability,
                    "error_json": (
                        json.dumps(
                            {"errors": list(event.errors)},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if event.errors
                        else None
                    ),
                    "created_at": event.created_at,
                },
            )
            for event in delta.events
        ]
        if not commands:
            return
        try:
            await self._store.transact(commands)
        except Exception as exc:
            self._last_extension_persistence_error = exc
            self._status["extension_fingerprint_store"] = {
                "status": "critical",
                "code": "extension_fingerprint_persistence_failed",
            }
            raise
        self._last_extension_persistence_error = None
        self._status["extension_fingerprint_store"] = {
            "status": "healthy",
            "events": len(commands),
        }

    async def _refresh_external_extensions(self) -> None:
        assert self._discovery is not None
        try:
            await self._discovery.refresh()
        except Exception as exc:
            if exc is self._last_extension_persistence_error:
                self._status["extension_discovery"] = {
                    "status": "critical",
                    "code": "extension_fingerprint_persistence_failed",
                }
                raise
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
        state = self._next_round_state
        try:
            snapshot = self.agent_registry.snapshot(state.detached_agent_overrides())
        except RootAgentUnavailableError:
            return None
        entry = snapshot.get(state.root_agent)
        if entry is None:
            return None
        try:
            return resolve_generation_selector(entry.definition.model_selector, state.root_force_selector).raw
        except Exception:
            return None

    def _warn_for_low_root_budget(self) -> None:
        assert self.price_catalog is not None
        selector = self._root_selector()
        if selector is None:
            return
        warning = self.price_catalog.low_budget_warning(
            selector,
            self._next_round_state.default_effort_credits,
        )
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
            replacement = PriceCatalog.from_host_snapshot(
                snapshot,
                self._next_round_state.detached_price_overrides(),
            )
        except Exception:
            self._status["initial_price_snapshot"] = {
                "status": "degraded",
                "code": "host_model_snapshot_invalid",
            }
            self._ctx.logger.warning("LRS 模型热更新快照无效；继续使用上一个安全价格快照", exc_info=True)
            return False
        self._next_round_state = replace(self._next_round_state, price_catalog=replacement)
        self._status["initial_price_snapshot"] = {"status": "healthy"}
        return True

    async def snapshot_next_round(self) -> NextRoundSnapshot:
        """在生命周期锁内把 live registry 与 next-round 配置冻结成独立对象。"""

        async with self._lifecycle_lock:
            self._ensure_running()
            state = self._next_round_state
            assert state.price_catalog is not None
            return NextRoundSnapshot(
                root_agent=state.root_agent,
                root_force_selector=state.root_force_selector,
                summarizer_selector=state.summarizer_selector,
                default_effort_credits=state.default_effort_credits,
                agent_catalog=self.agent_registry.snapshot(state.detached_agent_overrides()),
                procedure_catalog=self.procedure_registry.snapshot(
                    state.detached_procedure_overrides()
                ),
                price_catalog=state.price_catalog,
            )

    async def update_self_config(self, config: LRSConfig, *, version: str) -> None:
        async with self._lifecycle_lock:
            self._ensure_running()
            detached_limits = self._extract_safety_limits(config)
            new_interval = float(config.extensions.refresh_interval_seconds)
            current = self._next_round_state
            assert current.price_catalog is not None
            replacement = _NextRoundState.from_config(config, price_catalog=None)
            replacement = replace(
                replacement,
                price_catalog=current.price_catalog.with_plugin_overrides(
                    replacement.detached_price_overrides()
                ),
            )
            if new_interval != self._refresh_interval_seconds:
                async with self._refresh_lock:
                    self._ensure_running()
                    assert self._discovery is not None
                    old_discovery = self._discovery
                    await old_discovery.close()
                    self._refresh_interval_seconds = new_interval
                    self._discovery = self._new_discovery(new_interval)
                    self._discovery.start()
            self.agent_registry.set_root_agent(replacement.root_agent)
            self._next_round_state = replacement
            self._safety_limits = detached_limits
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
        state = self._next_round_state
        selector = self._root_selector()
        if selector is None:
            return (
                {
                    "status": "degraded",
                    "code": "root_agent_unavailable",
                    "agent_id": state.root_agent,
                },
                {"status": "degraded", "code": "root_agent_unavailable"},
            )
        selector_health = self._selector_health(selector)
        if selector_health.get("code") == "selector_unavailable":
            selector_health["code"] = "root_selector_unavailable"
        return ({"status": "healthy", "agent_id": state.root_agent}, selector_health)

    def _summarizer_health(self) -> dict[str, Any]:
        state = self._next_round_state
        try:
            selector = resolve_generation_selector(
                state.summarizer_selector,
                state.root_force_selector,
            ).raw
        except Exception:
            return {
                "status": "degraded",
                "code": "summarizer_selector_invalid",
                "selector": state.summarizer_selector,
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
        persistence = self._status["extension_fingerprint_store"]
        if persistence["status"] == "critical":
            return {
                "status": "critical",
                "code": "extension_fingerprint_persistence_failed",
            }
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
