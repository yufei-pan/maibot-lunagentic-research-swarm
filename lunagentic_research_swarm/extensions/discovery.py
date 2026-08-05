"""通过 Host 公共 API metadata 发现外部 agent 与 Procedure provider。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

from .contracts import (
    AgentDefinition,
    ExtensionRefreshDelta,
    ExtensionRefreshEvent,
    ProcedureDefinition,
)
from .validation import canonical_fingerprint


def _is_agent_descriptor(info: Mapping[str, Any]) -> bool:
    metadata = info.get("metadata")
    return (
        info.get("name") == "describe_agents"
        and info.get("version") == "1"
        and info.get("public") is True
        and isinstance(metadata, Mapping)
        and metadata.get("lunagentic_extension") == "agents"
        and metadata.get("lunagentic_contract") == "1"
    )


def _is_procedure_descriptor(info: Mapping[str, Any]) -> bool:
    metadata = info.get("metadata")
    return (
        info.get("name") == "describe_procedures"
        and info.get("version") == "1"
        and info.get("public") is True
        and isinstance(metadata, Mapping)
        and metadata.get("lunagentic_extension") == "procedures"
        and metadata.get("lunagentic_contract") == "1"
    )


class ExtensionDiscovery:
    """只在 Host RPC 边界保留动态对象，目录内部全部转换为值契约。"""

    def __init__(
        self,
        ctx: Any,
        agent_registry: AgentRegistry,
        procedure_registry: ProcedureRegistry,
        *,
        refresh_interval_seconds: float,
        refresh_listener: Callable[[ExtensionRefreshDelta], Awaitable[None]] | None = None,
    ) -> None:
        if refresh_interval_seconds <= 0:
            raise ValueError("扩展刷新间隔必须大于 0")
        self._ctx = ctx
        self._agents = agent_registry
        self._procedures = procedure_registry
        self._refresh_interval_seconds = float(refresh_interval_seconds)
        self._refresh_listener = refresh_listener
        self._refresh_lock = asyncio.Lock()
        self._inflight_refresh: asyncio.Task[ExtensionRefreshDelta] | None = None
        self._refresh_requested = asyncio.Event()
        self._background_task: asyncio.Task[None] | None = None
        self._closed = False
        self._extension_fingerprints: dict[str, str] = {}

    @property
    def extension_fingerprints(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._extension_fingerprints))

    @property
    def background_task(self) -> asyncio.Task[None] | None:
        return self._background_task

    @property
    def closed(self) -> bool:
        return self._closed

    async def refresh(self) -> ExtensionRefreshDelta:
        """共享同一个在途扫描；单个 provider 失败不阻断其他批次。"""

        if self._closed:
            return ExtensionRefreshDelta()
        task = self._inflight_refresh
        if task is None or task.done():
            task = asyncio.create_task(self._refresh_once(), name="lrs-extension-refresh-once")
            self._inflight_refresh = task
        try:
            return await asyncio.shield(task)
        finally:
            if self._inflight_refresh is task and task.done():
                self._inflight_refresh = None

    async def _refresh_once(self) -> ExtensionRefreshDelta:
        delta = await self._scan_once()
        if self._refresh_listener is not None and delta.events:
            await self._refresh_listener(delta)
        return delta

    async def _scan_once(self) -> ExtensionRefreshDelta:
        """执行一次实际 Host 扫描并产生不可变审计增量。"""

        async with self._refresh_lock:
            if self._closed:
                return ExtensionRefreshDelta()
            refreshed_at = time.time()
            events: list[ExtensionRefreshEvent] = []
            try:
                raw_infos = await self._ctx.api.list()
                if not isinstance(raw_infos, list):
                    raise TypeError("ctx.api.list() 必须返回 API metadata 列表")
            except Exception as exc:
                events.append(self._record_global_error(exc, refreshed_at))
                return ExtensionRefreshDelta(tuple(events))
            self._extension_fingerprints.pop("discovery", None)
            for key in tuple(self._extension_fingerprints):
                if key.startswith("descriptor:"):
                    self._extension_fingerprints.pop(key, None)
            if self._closed:
                return ExtensionRefreshDelta()
            events.append(
                ExtensionRefreshEvent(
                    provider_plugin_id="host-api",
                    extension_kind="discovery",
                    availability="available",
                    fingerprint=canonical_fingerprint({"status": "available"}),
                    errors=(),
                    created_at=refreshed_at,
                )
            )

            agent_descriptors: dict[str, Mapping[str, Any]] = {}
            procedure_descriptors: dict[str, Mapping[str, Any]] = {}
            invalid_candidates: list[tuple[int, str, str, str]] = []
            for index, raw_info in enumerate(raw_infos):
                if not isinstance(raw_info, Mapping):
                    events.append(
                        self._record_descriptor_error(
                            index,
                            "API metadata 必须为 Mapping",
                            refreshed_at,
                        )
                    )
                    continue
                metadata = raw_info.get("metadata")
                tagged_kind = metadata.get("lunagentic_extension") if isinstance(metadata, Mapping) else None
                if tagged_kind in {"agents", "procedures"}:
                    provider_id = raw_info.get("plugin_id")
                    if (
                        not isinstance(provider_id, str)
                        or not provider_id
                        or provider_id != provider_id.strip()
                    ):
                        events.append(
                            self._record_descriptor_error(
                                index,
                                "descriptor plugin_id 无效",
                                refreshed_at,
                            )
                        )
                        continue
                if _is_agent_descriptor(raw_info):
                    self._collect_descriptor(agent_descriptors, raw_info, "agents", index, invalid_candidates)
                elif _is_procedure_descriptor(raw_info):
                    self._collect_descriptor(procedure_descriptors, raw_info, "procedures", index, invalid_candidates)
                elif tagged_kind in {"agents", "procedures"}:
                    provider_id = raw_info.get("plugin_id")
                    assert isinstance(provider_id, str)
                    invalid_candidates.append(
                        (index, tagged_kind, provider_id, "descriptor metadata 与契约不完整匹配")
                    )

            visible_agents = set(agent_descriptors)
            visible_procedures = set(procedure_descriptors)
            for index, kind, provider_id, error in invalid_candidates:
                events.append(self._record_descriptor_error(index, error, refreshed_at))
                if kind == "agents":
                    visible_agents.add(provider_id)
                    events.append(self._reject_agents(provider_id, [error], refreshed_at))
                else:
                    visible_procedures.add(provider_id)
                    events.append(self._reject_procedures(provider_id, [error], refreshed_at))

            for provider_id, descriptor in sorted(agent_descriptors.items()):
                if self._closed:
                    return ExtensionRefreshDelta(tuple(events))
                events.append(await self._refresh_agents(provider_id, descriptor, refreshed_at))
            for provider_id, descriptor in sorted(procedure_descriptors.items()):
                if self._closed:
                    return ExtensionRefreshDelta(tuple(events))
                events.append(await self._refresh_procedures(provider_id, descriptor, refreshed_at))

            for provider_id in self._agents.provider_ids - visible_agents:
                if provider_id == "builtin":
                    # 本地装入的 builtin provider 不经由 Host API 发现，不能被扫描误删。
                    continue
                self._agents.remove_provider(provider_id)
                events.append(self._record_health("agents", provider_id, refreshed_at))
            for provider_id in self._procedures.provider_ids - visible_procedures:
                if provider_id == "builtin":
                    continue
                self._procedures.remove_provider(provider_id)
                events.append(self._record_health("procedures", provider_id, refreshed_at))
            return ExtensionRefreshDelta(tuple(events))

    @staticmethod
    def _collect_descriptor(
        target: dict[str, Mapping[str, Any]],
        info: Mapping[str, Any],
        kind: str,
        index: int,
        invalid_candidates: list[tuple[int, str, str, str]],
    ) -> None:
        provider_id = info.get("plugin_id")
        assert isinstance(provider_id, str) and provider_id and provider_id == provider_id.strip()
        expected_name = f"{provider_id}.describe_{kind}"
        if info.get("full_name") != expected_name:
            invalid_candidates.append((index, kind, provider_id, "Host API full_name 与 plugin_id 不一致"))
            return
        if provider_id in target:
            target.pop(provider_id, None)
            invalid_candidates.append((index, kind, provider_id, "同一 provider 存在重复 descriptor"))
            return
        target[provider_id] = info

    async def _refresh_agents(
        self,
        provider_id: str,
        descriptor: Mapping[str, Any],
        refreshed_at: float,
    ) -> ExtensionRefreshEvent:
        del descriptor
        api_name = f"{provider_id}.describe_agents"
        try:
            envelope = await self._ctx.api.call(api_name, version="1")
            payloads = self._validate_envelope(envelope, "agents")
            definitions = tuple(AgentDefinition.model_validate(payload) for payload in payloads)
            self._agents.replace_provider(provider_id, definitions)
        except Exception as exc:
            return self._reject_agents(provider_id, [str(exc)], refreshed_at)
        return self._record_health("agents", provider_id, refreshed_at)

    async def _refresh_procedures(
        self,
        provider_id: str,
        descriptor: Mapping[str, Any],
        refreshed_at: float,
    ) -> ExtensionRefreshEvent:
        del descriptor
        api_name = f"{provider_id}.describe_procedures"
        try:
            envelope = await self._ctx.api.call(api_name, version="1")
            payloads = self._validate_envelope(envelope, "procedures")
            definitions = tuple(ProcedureDefinition.model_validate(payload) for payload in payloads)
            self._procedures.replace_provider(provider_id, definitions)
        except Exception as exc:
            return self._reject_procedures(provider_id, [str(exc)], refreshed_at)
        return self._record_health("procedures", provider_id, refreshed_at)

    @staticmethod
    def _validate_envelope(envelope: Any, item_key: str) -> Sequence[Mapping[str, Any]]:
        if not isinstance(envelope, Mapping):
            raise ValueError("descriptor 返回值必须为 Mapping 信封")
        expected_keys = {"contract_version", item_key}
        if set(envelope) != expected_keys:
            raise ValueError(f"descriptor 信封字段必须且只能为 {sorted(expected_keys)}")
        if envelope["contract_version"] != "1":
            raise ValueError("descriptor contract_version 必须为 1")
        payloads = envelope[item_key]
        if not isinstance(payloads, list):
            raise ValueError(f"descriptor {item_key} 必须为 list")
        if any(not isinstance(payload, Mapping) for payload in payloads):
            raise ValueError(f"descriptor {item_key} 的每一项必须为 Mapping")
        return payloads

    def _reject_agents(
        self,
        provider_id: str,
        errors: Sequence[str],
        refreshed_at: float,
    ) -> ExtensionRefreshEvent:
        self._agents.reject_provider(provider_id, errors)
        return self._record_health("agents", provider_id, refreshed_at)

    def _reject_procedures(
        self,
        provider_id: str,
        errors: Sequence[str],
        refreshed_at: float,
    ) -> ExtensionRefreshEvent:
        self._procedures.reject_provider(provider_id, errors)
        return self._record_health("procedures", provider_id, refreshed_at)

    def _record_health(
        self,
        kind: str,
        provider_id: str,
        refreshed_at: float,
    ) -> ExtensionRefreshEvent:
        registry = self._agents if kind == "agents" else self._procedures
        health = registry.health[provider_id]
        self._extension_fingerprints[f"{kind}:{provider_id}"] = health.fingerprint
        availability = "available" if health.status == "healthy" else health.status
        return ExtensionRefreshEvent(
            provider_plugin_id=provider_id,
            extension_kind=kind,  # type: ignore[arg-type]
            availability=availability,  # type: ignore[arg-type]
            fingerprint=health.fingerprint,
            errors=tuple(health.errors),
            created_at=refreshed_at,
        )

    def _record_global_error(self, exc: Exception, refreshed_at: float) -> ExtensionRefreshEvent:
        fingerprint = canonical_fingerprint(
            {"status": "invalid", "error": str(exc)}
        )
        self._extension_fingerprints["discovery"] = fingerprint
        return ExtensionRefreshEvent(
            provider_plugin_id="host-api",
            extension_kind="discovery",
            availability="invalid",
            fingerprint=fingerprint,
            errors=(str(exc),),
            created_at=refreshed_at,
        )

    def _record_descriptor_error(
        self,
        index: int,
        error: str,
        refreshed_at: float,
    ) -> ExtensionRefreshEvent:
        fingerprint = canonical_fingerprint(
            {"status": "invalid", "error": str(error)}
        )
        provider_id = f"descriptor:{index}"
        self._extension_fingerprints[provider_id] = fingerprint
        return ExtensionRefreshEvent(
            provider_plugin_id=provider_id,
            extension_kind="discovery",
            availability="invalid",
            fingerprint=fingerprint,
            errors=(str(error),),
            created_at=refreshed_at,
        )

    def start(self) -> None:
        """启动周期 refresh 协调器；首次扫描由 load 路径显式调用 refresh。"""

        if self._closed:
            raise RuntimeError("ExtensionDiscovery 已关闭")
        if self._background_task is None:
            self._background_task = asyncio.create_task(self._run(), name="lrs-extension-refresh")

    def request_refresh(self) -> None:
        """合并任意数量尚未消费的刷新请求。"""

        if not self._closed:
            self._refresh_requested.set()

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._refresh_requested.wait(), timeout=self._refresh_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
            self._refresh_requested.clear()
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._ctx.logger.error("LRS 周期扩展刷新失败；将在下一周期重试", exc_info=True)

    async def close(self) -> None:
        """取消并等待周期 task，保证卸载后不会再触发 Host RPC。"""

        if self._closed:
            return
        self._closed = True
        task = self._background_task
        self._background_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        async with self._refresh_lock:
            pass
