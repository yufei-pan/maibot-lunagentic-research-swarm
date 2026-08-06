"""LRS 基础服务生命周期、热更新与健康状态编排。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
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
from lunagentic_research_swarm.feedback import FeedbackService
from lunagentic_research_swarm.llm.gateway import (
    HostModelSnapshotReader,
    LLMGateway,
    ModelSelector,
    resolve_generation_selector,
)
from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningAdapter, PhysicalPinningStatus
from lunagentic_research_swarm.llm.pricing import PriceCatalog
from lunagentic_research_swarm.llm.summarizer import SummarizerService
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.procedures.executor import (
    ProcedureExecutor,
    bundled_procedure_invoker,
)
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogSnapshot, ProcedureRegistry
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.manager import ResearchManager
from lunagentic_research_swarm.runtime.scheduler import FairScheduler
from lunagentic_research_swarm.runtime.turns import TurnWorker
from lunagentic_research_swarm.statistics import StatisticsService
from lunagentic_research_swarm.storage.debug import DebugStore
from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from lunagentic_research_swarm.storage.vectors import VectorIndex

StoreFactory = Callable[[Path], Any]
DiscoveryFactory = Callable[..., Any]
BuiltinProviderLoader = Callable[..., Any]


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


def _load_builtin_providers(
    agents: AgentRegistry,
    procedures: ProcedureRegistry,
    *,
    ctx: Any | None = None,
    web_search_config: Any | None = None,
) -> BundledProcedureProvider:
    """通过与第三方相同的 replace_provider 路径装入内置默认智能体与 Procedures。

    返回带 ``ctx`` 的持久 ``BundledProcedureProvider``，供 executor 本地 invoker 调用；
    不为 builtin 写 scheduler / procedure_id shortcut。
    """

    from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions

    provider = BundledProcedureProvider(ctx, web_search_config=web_search_config)
    agents.replace_provider(
        "builtin",
        [definition.model_dump(mode="json") for definition in bundled_agent_definitions()],
    )
    procedures.replace_provider("builtin", provider.describe())
    return provider


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
        builtin_provider_loader: BuiltinProviderLoader = _load_builtin_providers,
    ) -> None:
        self._ctx = ctx
        self._config = config.model_copy(deep=True)
        self._store = store_factory(Path(ctx.paths.data_dir) / "lrs-state.sqlite3")
        self.outbox: MaisakaOutbox | None = None
        self.feedback: FeedbackService | None = None
        self.vector_index: VectorIndex | None = None
        self.debug_store: DebugStore | None = None
        self.statistics: StatisticsService | None = None
        self.manager: ResearchManager | None = None
        self.scheduler: FairScheduler | None = None
        self._effect_runner: RuntimeEffectRunner | None = None
        self._outbox_poll_seconds = float(config.reporting.outbox_poll_seconds)
        self._discovery_factory = discovery_factory
        self._host_snapshot_loader = host_snapshot_loader
        self._physical_pinning = physical_pinning or PhysicalPinningAdapter()
        self._builtin_provider_loader = builtin_provider_loader
        self._bundled_procedure_provider: BundledProcedureProvider | None = None

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
            "vector_index": {"status": "pending"},
            "initial_price_snapshot": {"status": "pending"},
            "physical_pinning": {"status": "pending"},
            "extension_fingerprint_store": {"status": "pending"},
            "config_reload": {"status": "healthy", "code": "not_reloaded"},
        }

    @staticmethod
    def _extract_safety_limits(config: LRSConfig) -> dict[str, Any]:
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
            "model_context_window": (
                int(config.context.model_context_window)
                if config.context.model_context_window is not None
                else None
            ),
            "max_correction_turns": int(config.protocol.max_correction_turns),
            "warning_miss_input_tokens": int(config.budget.warning_miss_input_tokens),
            "warning_output_tokens": int(config.budget.warning_output_tokens),
            "default_protocol": str(config.protocol.default_mode),
            "max_report_chars": int(config.reporting.max_report_chars),
            "max_stats_chars": int(config.reporting.max_stats_chars),
            "deliver_intermediate": bool(config.reporting.deliver_intermediate),
            "deliver_final": bool(config.reporting.deliver_final),
        }

    def _default_protocol(self) -> str:
        """`[protocol] default_mode`：provider 未声明 protocol 时的回退协议。"""

        return str(self._config.protocol.default_mode)

    @property
    def safety_limits(self) -> Mapping[str, Any]:
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
            llm = getattr(self._ctx, "llm", None)
            if llm is not None:
                # 向量层是可重建派生索引：start / ensure_ready / LanceDB IO 失败只降级
                # vector_index，绝不拖垮整个 LRS 启动（勿把异常误标为 sqlite_initialization_failed）。
                try:
                    self.vector_index = VectorIndex(
                        self._store,
                        llm,
                        self._config.embedding,
                        Path(self._ctx.paths.data_dir) / "vectors" / "lancedb",
                    )
                    await self.vector_index.start()
                    ready = await self.vector_index.ensure_ready()
                    if not ready.success:
                        err = ready.error
                        self._status["vector_index"] = {
                            "status": "degraded",
                            "code": err.code if err is not None else "vector_index_unavailable",
                            "message": err.message if err is not None else "ensure_ready failed",
                        }
                    else:
                        vector_status = await self.vector_index.status()
                        self._status["vector_index"] = {
                            "status": "healthy",
                            "idle": vector_status.idle,
                            "active_generation": vector_status.active_generation,
                            "dimension": vector_status.dimension,
                            "rebuilding": vector_status.rebuilding,
                        }
                except Exception as exc:
                    if self.vector_index is not None:
                        with suppress(Exception):
                            await self.vector_index.close()
                    self.vector_index = None
                    self._status["vector_index"] = {
                        "status": "unavailable",
                        "code": getattr(exc, "code", None) or "vector_index_unavailable",
                        "message": getattr(exc, "message", None) or str(exc),
                    }
                    self._ctx.logger.warning(
                        "LRS 向量索引启动失败；插件将以降级向量层继续加载",
                        exc_info=True,
                    )
            else:
                self._status["vector_index"] = {"status": "degraded", "code": "llm_unavailable"}
            maisaka = getattr(self._ctx, "maisaka", None)
            if maisaka is not None:
                self.outbox = MaisakaOutbox(
                    self._store,
                    maisaka,
                    poll_interval_seconds=self._outbox_poll_seconds,
                )
                await self.outbox.start()
                self._status["maisaka_outbox"] = {
                    "status": "healthy",
                    "delivery": "at_least_once_stable_idempotency",
                }
            else:
                self._status["maisaka_outbox"] = {"status": "degraded", "code": "maisaka_unavailable"}
            self.feedback = FeedbackService(
                self._store,
                vector_index=self.vector_index,
                outbox=self.outbox,
                feedback_wait_seconds=int(self._config.timing.feedback_wait_seconds),
                reminders_enabled=bool(self._config.feedback.reminders_enabled),
                index_lessons=bool(self._config.feedback.index_lessons),
                max_lesson_chars=int(self._config.feedback.max_lesson_chars),
                poll_interval_seconds=self._outbox_poll_seconds,
            )
            await self.feedback.start()
            self._status["feedback"] = {
                "status": "healthy",
                "reminders_enabled": bool(self._config.feedback.reminders_enabled),
                "index_lessons": bool(self._config.feedback.index_lessons),
            }
            self.debug_store = DebugStore(
                Path(self._ctx.paths.data_dir),
                store_agent_transcripts=bool(self._config.storage.store_agent_transcripts),
                store_raw_procedure_payloads=bool(self._config.storage.store_raw_procedure_payloads),
                authority_store=self._store,
            )
            await self.debug_store.open()
            self.statistics = StatisticsService(self._store)
            self._status["debug_storage"] = {
                "status": "healthy",
                "store_agent_transcripts": bool(self._config.storage.store_agent_transcripts),
                "store_raw_procedure_payloads": bool(self._config.storage.store_raw_procedure_payloads),
                "enabled": self.debug_store.enabled,
            }
            interrupted = await self._store.mark_active_rounds_interrupted(time.time())
            self._status["legacy_rounds"] = {
                "status": "healthy",
                "interrupted": int(interrupted),
            }
            self._load_initial_price_catalog()
            self._install_builtin_providers()
            self._discovery = self._new_discovery(self._refresh_interval_seconds)
            await self._refresh_external_extensions()
            self._ensure_configured_root_available()
            self._record_physical_pinning_health()
            self._warn_for_low_root_budget()
            await self._start_runtime()
            assert self.manager is not None
            restored = await self.manager.restore_tasks()
            self._status["legacy_rounds"] = {
                **self._status["legacy_rounds"],
                "restored_tasks": int(restored),
            }
            self._discovery.start()
            self._state = "running"
        except BaseException as exc:
            # Only a genuine store failure is a SQLite failure; a root-agent or
            # extension error must not be reported as `sqlite_initialization_failed`.
            self._record_start_failure(exc)
            await self._cleanup_start_failure()
            self._state = "failed"
            raise

    _START_STAGE_CODES: Mapping[str, str] = MappingProxyType(
        {
            "sqlite": "sqlite_initialization_failed",
            "root_agent": "root_agent_unavailable",
            "extension_fingerprint_store": "extension_fingerprint_persistence_failed",
            "runtime": "runtime_start_failed",
        }
    )

    def _record_start_failure(self, exc: BaseException) -> None:
        """Attribute a startup failure to the stage that actually failed."""

        if isinstance(exc, RootAgentUnavailableError):
            self._status["root_agent"] = {
                "status": "critical",
                "code": self._START_STAGE_CODES["root_agent"],
            }
            return
        if self._status["extension_fingerprint_store"]["status"] == "critical":
            return
        if self._status["sqlite"]["status"] != "healthy":
            self._status["sqlite"] = {
                "status": "critical",
                "code": self._START_STAGE_CODES["sqlite"],
            }
            return
        self._status["runtime"] = {
            "status": "critical",
            "code": self._START_STAGE_CODES["runtime"],
            "error_type": type(exc).__name__,
        }

    def _install_builtin_providers(self) -> None:
        """装入内置 agents/procedures，并保留带真实 ctx 的 durable procedure provider。"""

        loader = self._builtin_provider_loader
        web_search_config = self._config.web_search
        try:
            loaded = loader(
                self.agent_registry,
                self.procedure_registry,
                ctx=self._ctx,
                web_search_config=web_search_config,
            )
        except TypeError:
            try:
                loaded = loader(self.agent_registry, self.procedure_registry, ctx=self._ctx)
            except TypeError:
                # 测试用两参数 loader 不接受 ctx。
                loaded = loader(self.agent_registry, self.procedure_registry)
        if isinstance(loaded, BundledProcedureProvider):
            if loaded.ctx is None:
                loaded.ctx = self._ctx
            loaded.bind_case_index(store=self._store, vector_index=self.vector_index)
            self._bundled_procedure_provider = loaded
            return
        # 自定义 loader 未返回 provider 时仍创建 durable 实例，供 executor 本地 invoker 使用。
        provider = BundledProcedureProvider(
            self._ctx,
            web_search_config=web_search_config,
            store=self._store,
            vector_index=self.vector_index,
        )
        if "builtin" not in self.procedure_registry.provider_ids:
            self.procedure_registry.replace_provider("builtin", provider.describe())
        self._bundled_procedure_provider = provider

    def _procedure_local_invokers(self) -> dict[str, Any]:
        provider = self._bundled_procedure_provider
        if provider is None:
            return {}
        return {"builtin.invoke_procedure": bundled_procedure_invoker(provider)}

    def _bind_contractor_runtime(
        self,
        *,
        gateway: Any,
        price_catalog: Any,
        summarizer: Any | None = None,
    ) -> None:
        """把 LLM / 冻结 round 查找 / live 回落解析 / 嵌套 invoker 注入 builtin.contractor。"""

        provider = self._bundled_procedure_provider
        if provider is None:
            return
        from lunagentic_research_swarm.procedures.bundled.contractor import (
            ContractorDeps,
            make_nested_procedure_invoker,
        )

        def round_snapshot_for_task(task_id: str) -> Any | None:
            """按 task_id 取 ResearchManager 冻结的 round snapshot（若有）。"""

            manager = self.manager
            if manager is None or not task_id:
                return None
            snapshots = getattr(manager, "_round_snapshots", None)
            if not isinstance(snapshots, Mapping):
                return None
            return snapshots.get(task_id)

        def resolve_agent(agent_id: str) -> Any:
            # 仅在无冻结 snapshot 时作为回落（单测 / 无 task 上下文）。
            snapshot = self.agent_registry.snapshot(self._next_round_state.detached_agent_overrides())
            entry = snapshot.get(agent_id)
            return None if entry is None else entry.definition

        def resolve_procedure_catalog(agent_id: str) -> list[dict[str, Any]]:
            agent_snap = self.agent_registry.snapshot(self._next_round_state.detached_agent_overrides())
            proc_snap = self.procedure_registry.snapshot(self._next_round_state.detached_procedure_overrides())
            allowed = agent_snap.resolve_allowed_procedures(agent_id, proc_snap)
            items: list[dict[str, Any]] = []
            for procedure_id in allowed:
                if procedure_id == "builtin.contractor":
                    continue
                entry = proc_snap.get(procedure_id)
                if entry is None:
                    continue
                definition = entry.definition
                items.append(
                    {
                        "procedure_id": procedure_id,
                        "display_name": str(getattr(definition, "display_name", "") or ""),
                        "description": str(getattr(definition, "description", "") or ""),
                    }
                )
            return items

        invoke_nested = make_nested_procedure_invoker(
            provider=provider,
            summarizer=summarizer,
            price_catalog=price_catalog,
        )
        provider.bind_contractor_deps(
            ContractorDeps(
                llm=gateway,
                prices=price_catalog,
                resolve_agent=resolve_agent,
                invoke_nested_procedure=invoke_nested,
                resolve_procedure_catalog=resolve_procedure_catalog,
                round_snapshot_for_task=round_snapshot_for_task,
            )
        )

    async def _cleanup_start_failure(self) -> None:
        """释放启动阶段已经创建的资源而不遮蔽原始异常。"""

        resources: tuple[tuple[str, Any | None], ...] = (
            ("runtime_scheduler", self.scheduler),
            ("extension_discovery", self._discovery),
            ("feedback", self.feedback),
            ("debug_store", self.debug_store),
            ("vector_index", self.vector_index),
            ("maisaka_outbox", self.outbox),
            ("sqlite", self._store),
        )
        self._discovery = None
        self.feedback = None
        self.debug_store = None
        self.statistics = None
        self.vector_index = None
        self.outbox = None
        manager = self.manager
        self.manager = None
        self.scheduler = None
        self._effect_runner = None
        provider = self._bundled_procedure_provider
        self._bundled_procedure_provider = None
        if manager is not None:
            try:
                await manager.shutdown()
            except BaseException as exc:
                self._ctx.logger.error(
                    "LRS 启动失败后的 manager 清理异常：resource=runtime_manager；error_type=%s",
                    type(exc).__name__,
                )
        if provider is not None:
            try:
                await provider.aclose()
            except BaseException as exc:
                self._ctx.logger.error(
                    "LRS 启动失败后的资源清理异常：resource=bundled_procedure_provider；error_type=%s",
                    type(exc).__name__,
                )
        for resource_name, resource in resources:
            if resource is None:
                continue
            try:
                await resource.close()
            except BaseException as exc:
                # 只记录安全的错误类型，清理异常不能覆盖启动原始异常。
                self._ctx.logger.error(
                    "LRS 启动失败后的资源清理异常：resource=%s；error_type=%s",
                    resource_name,
                    type(exc).__name__,
                )

    async def _start_runtime(self) -> None:
        """Compose the production worker cycle after catalogs/pricing exist."""

        assert self.price_catalog is not None
        config = self._config
        gateway = LLMGateway(self._ctx, physical_pinning=self._physical_pinning)
        summarizer_selector = resolve_generation_selector(
            config.summarizer.selector,
            config.llm.force_selector,
        ).raw
        summarizer = SummarizerService(
            gateway,
            selector=summarizer_selector,
            temperature=float(config.summarizer.temperature),
            max_tokens=int(config.summarizer.max_tokens),
        )
        procedure_catalog = self.procedure_registry.snapshot(
            self._next_round_state.detached_procedure_overrides()
        )
        local_invokers = self._procedure_local_invokers()
        procedures = ProcedureExecutor(
            procedure_catalog,
            ctx=self._ctx,
            summarizer=summarizer,
            local_invokers=local_invokers,
            debug_store=self.debug_store,
        )

        def procedure_factory(catalog: Any) -> ProcedureExecutor:
            return ProcedureExecutor(
                catalog,
                ctx=self._ctx,
                summarizer=summarizer,
                local_invokers=local_invokers,
                debug_store=self.debug_store,
            )

        turn_worker = TurnWorker(
            gateway,
            procedures,
            pricing=self.price_catalog,
            procedure_factory=procedure_factory,
            debug_store=self.debug_store,
        )
        self._bind_contractor_runtime(
            gateway=gateway,
            price_catalog=self.price_catalog,
            summarizer=summarizer,
        )
        runner = RuntimeEffectRunner(turn_worker)
        scheduler = FairScheduler(
            global_llm=int(config.scheduler.max_global_llm_concurrency),
            per_task_llm=int(config.scheduler.max_task_llm_concurrency),
            per_task_procedure=int(config.scheduler.max_task_procedure_concurrency),
            worker=runner,
        )
        manager = ResearchManager(
            ctx=self._ctx,
            store=self._store,
            summarizer=summarizer,
            scheduler=scheduler,
            snapshot_provider=self.snapshot_next_round,
            recent_message_limit=20,
            pause_timeout_seconds=int(config.timing.pause_timeout_seconds),
            grace_period_seconds=int(config.timing.grace_period_seconds),
            agent_live_provider=self.agent_registry.is_live,
            runtime_limits=self._safety_limits,
            feedback_service=self.feedback,
            statistics=self.statistics,
            summarizer_selector=summarizer_selector,
            outbox=self.outbox,
        )
        runner.bind_manager(manager)
        self._effect_runner = runner
        self.scheduler = scheduler
        self.manager = manager
        await scheduler.start()

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

    def _ensure_configured_root_available(self) -> None:
        """在暴露 running runtime 前确认配置的 root 能形成有效 catalog snapshot。"""

        self.agent_registry.snapshot(
            self._next_round_state.detached_agent_overrides(),
            default_protocol=self._default_protocol(),
        )

    def _root_selector(self) -> str | None:
        state = self._next_round_state
        try:
            snapshot = self.agent_registry.snapshot(
                state.detached_agent_overrides(), default_protocol=self._default_protocol()
            )
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
        miss_tokens = int(self._config.budget.warning_miss_input_tokens)
        output_tokens = int(self._config.budget.warning_output_tokens)
        warning = self.price_catalog.low_budget_warning(
            selector,
            self._next_round_state.default_effort_credits,
            miss_input_tokens=miss_tokens,
            output_tokens=output_tokens,
        )
        if warning is not None:
            self._ctx.logger.warning(
                "%s（按 %d 个 cache miss 输入 token 与 %d 个输出 token 估算）",
                warning,
                miss_tokens,
                output_tokens,
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
                agent_catalog=self.agent_registry.snapshot(
                    state.detached_agent_overrides(), default_protocol=self._default_protocol()
                ),
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
            self._config = config.model_copy(deep=True)
            if self._bundled_procedure_provider is not None:
                self._bundled_procedure_provider.web_search.replace_config(self._config.web_search)
                self.procedure_registry.replace_provider(
                    "builtin",
                    self._bundled_procedure_provider.describe(),
                )
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
        background_task = self._discovery.background_task
        if background_task is None or background_task.cancelled():
            return {
                "status": "critical",
                "code": "extension_refresh_background_stopped",
            }
        if background_task.done():
            return {
                "status": "critical",
                "code": (
                    "extension_refresh_background_failed"
                    if background_task.exception() is not None
                    else "extension_refresh_background_stopped"
                ),
            }
        fingerprints = self._discovery.extension_fingerprints
        if "discovery" in fingerprints:
            return {"status": "degraded", "code": "extension_discovery_failed"}
        if any(key.startswith("descriptor:") for key in fingerprints):
            return {"status": "degraded", "code": "extension_descriptor_invalid"}
        return dict(self._status["extension_discovery"])

    def _recommended_fetch_health(self) -> dict[str, Any]:
        """推荐的 fetch-url provider 健康度；缺失时为 recommended_missing，不阻断加载。"""

        procedure_id = "fetch_url.fetch"
        provider_id = "com.0-hz.fetch-url"
        if self.procedure_registry.is_live(procedure_id):
            return {"status": "healthy", "detail": procedure_id}
        provider_health = self.procedure_registry.health.get(provider_id)
        if provider_health is not None and provider_health.status == "invalid":
            return {"status": "degraded", "code": "fetch_provider_invalid", "detail": procedure_id}
        return {"status": "recommended_missing", "code": "fetch_url_missing", "detail": procedure_id}

    @staticmethod
    def _vector_health_from_status(status: Any) -> dict[str, Any]:
        """由 live VectorIndex.status() 构造 health 条目，避免永久冻结启动快照。"""

        payload: dict[str, Any] = {
            "idle": getattr(status, "idle", None),
            "active_generation": getattr(status, "active_generation", None),
            "dimension": getattr(status, "dimension", None),
            "rebuilding": getattr(status, "rebuilding", None),
        }
        last_error_code = getattr(status, "last_error_code", None)
        last_error_message = getattr(status, "last_error_message", None)
        failed_candidate = getattr(status, "failed_candidate", None)
        if last_error_code or failed_candidate is not None:
            payload["status"] = "degraded"
            payload["code"] = last_error_code or "vector_rebuild_failed"
            if last_error_message:
                payload["message"] = last_error_message
            if failed_candidate is not None:
                payload["failed_candidate"] = failed_candidate
        elif getattr(status, "rebuilding", False):
            payload["status"] = "degraded"
            payload["code"] = "vector_index_rebuilding"
        else:
            payload["status"] = "healthy"
        return payload

    async def refresh_vector_index_health(self) -> dict[str, Any]:
        """用 live VectorIndex.status() 刷新 `_status['vector_index']`。"""

        self._ensure_running()
        if self.vector_index is None:
            current = dict(self._status.get("vector_index") or {"status": "unavailable", "code": "vector_index_unavailable"})
            self._status["vector_index"] = current
            return current
        try:
            status = await self.vector_index.status()
            self._status["vector_index"] = self._vector_health_from_status(status)
        except Exception as exc:
            self._status["vector_index"] = {
                "status": "degraded",
                "code": getattr(exc, "code", None) or "vector_index_unavailable",
                "message": getattr(exc, "message", None) or str(exc),
            }
        return dict(self._status["vector_index"])

    def health(self) -> dict[str, Any]:
        self._ensure_running()
        root_agent, root_selector = self._root_health()
        return {
            "sqlite": dict(self._status["sqlite"]),
            "vector_index": dict(self._status.get("vector_index", {"status": "unknown"})),
            "maisaka_outbox": dict(self._status.get("maisaka_outbox", {"status": "unknown"})),
            "legacy_rounds": dict(self._status["legacy_rounds"]),
            "initial_price_snapshot": dict(self._status["initial_price_snapshot"]),
            "physical_pinning": dict(self._status["physical_pinning"]),
            "extension_discovery": self._extension_discovery_health(),
            "extension_providers": self._extension_provider_health(),
            "recommended_fetch": self._recommended_fetch_health(),
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
            close_error: BaseException | None = None
            if self.manager is not None:
                try:
                    await self.manager.shutdown()
                except BaseException as exc:
                    close_error = exc
                self.manager = None
            if self.scheduler is not None:
                try:
                    await self.scheduler.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                self.scheduler = None
                self._effect_runner = None
            if self._discovery is not None:
                try:
                    await self._discovery.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
            if self.feedback is not None:
                try:
                    await self.feedback.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                self.feedback = None
            if self.debug_store is not None:
                try:
                    await self.debug_store.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                self.debug_store = None
            self.statistics = None
            if self.outbox is not None:
                try:
                    await self.outbox.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                self.outbox = None
            if self.vector_index is not None:
                try:
                    await self.vector_index.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                self.vector_index = None
            if self._bundled_procedure_provider is not None:
                try:
                    await self._bundled_procedure_provider.aclose()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                self._bundled_procedure_provider = None
            try:
                await self._store.close()
            except BaseException as store_error:
                self._state = "closed"
                if close_error is not None:
                    raise close_error from store_error
                raise
            self._state = "closed"
            if close_error is not None:
                raise close_error
