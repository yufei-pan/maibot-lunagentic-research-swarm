"""通过 Host 公共 API metadata 发现外部 agent 与 Procedure provider。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

from .contracts import AgentDefinition, ProcedureDefinition
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
    ) -> None:
        if refresh_interval_seconds <= 0:
            raise ValueError("扩展刷新间隔必须大于 0")
        self._ctx = ctx
        self._agents = agent_registry
        self._procedures = procedure_registry
        self._refresh_interval_seconds = float(refresh_interval_seconds)
        self._refresh_lock = asyncio.Lock()
        self._inflight_refresh: asyncio.Task[None] | None = None
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

    async def refresh(self) -> None:
        """共享同一个在途扫描；单个 provider 失败不阻断其他批次。"""

        if self._closed:
            return
        task = self._inflight_refresh
        if task is None or task.done():
            task = asyncio.create_task(self._refresh_once(), name="lrs-extension-refresh-once")
            self._inflight_refresh = task
        try:
            await asyncio.shield(task)
        finally:
            if self._inflight_refresh is task and task.done():
                self._inflight_refresh = None

    async def _refresh_once(self) -> None:
        """执行一次实际 Host 扫描。"""

        async with self._refresh_lock:
            if self._closed:
                return
            try:
                raw_infos = await self._ctx.api.list()
                if not isinstance(raw_infos, list):
                    raise TypeError("ctx.api.list() 必须返回 API metadata 列表")
            except Exception as exc:
                self._record_global_error(exc)
                return
            self._extension_fingerprints.pop("discovery", None)
            if self._closed:
                return

            agent_descriptors: dict[str, Mapping[str, Any]] = {}
            procedure_descriptors: dict[str, Mapping[str, Any]] = {}
            invalid_candidates: list[tuple[str, str, str]] = []
            for index, raw_info in enumerate(raw_infos):
                if not isinstance(raw_info, Mapping):
                    self._extension_fingerprints[f"descriptor:{index}"] = canonical_fingerprint(
                        {"status": "invalid", "error": "API metadata 必须为 Mapping"}
                    )
                    continue
                metadata = raw_info.get("metadata")
                tagged_kind = metadata.get("lunagentic_extension") if isinstance(metadata, Mapping) else None
                if _is_agent_descriptor(raw_info):
                    self._collect_descriptor(agent_descriptors, raw_info, "agents", invalid_candidates)
                elif _is_procedure_descriptor(raw_info):
                    self._collect_descriptor(procedure_descriptors, raw_info, "procedures", invalid_candidates)
                elif tagged_kind in {"agents", "procedures"}:
                    provider_id = raw_info.get("plugin_id")
                    if isinstance(provider_id, str) and provider_id:
                        invalid_candidates.append((tagged_kind, provider_id, "descriptor metadata 与契约不完整匹配"))

            visible_agents = set(agent_descriptors)
            visible_procedures = set(procedure_descriptors)
            for kind, provider_id, error in invalid_candidates:
                if kind == "agents":
                    visible_agents.add(provider_id)
                    self._reject_agents(provider_id, [error])
                else:
                    visible_procedures.add(provider_id)
                    self._reject_procedures(provider_id, [error])

            for provider_id, descriptor in sorted(agent_descriptors.items()):
                if self._closed:
                    return
                await self._refresh_agents(provider_id, descriptor)
            for provider_id, descriptor in sorted(procedure_descriptors.items()):
                if self._closed:
                    return
                await self._refresh_procedures(provider_id, descriptor)

            for provider_id in self._agents.provider_ids - visible_agents:
                self._agents.remove_provider(provider_id)
                self._record_health("agents", provider_id, self._agents.health[provider_id].fingerprint)
            for provider_id in self._procedures.provider_ids - visible_procedures:
                self._procedures.remove_provider(provider_id)
                self._record_health("procedures", provider_id, self._procedures.health[provider_id].fingerprint)

    @staticmethod
    def _collect_descriptor(
        target: dict[str, Mapping[str, Any]],
        info: Mapping[str, Any],
        kind: str,
        invalid_candidates: list[tuple[str, str, str]],
    ) -> None:
        provider_id = info.get("plugin_id")
        if not isinstance(provider_id, str) or not provider_id or provider_id != provider_id.strip():
            return
        expected_name = f"{provider_id}.describe_{kind}"
        if info.get("full_name") != expected_name:
            invalid_candidates.append((kind, provider_id, "Host API full_name 与 plugin_id 不一致"))
            return
        if provider_id in target:
            target.pop(provider_id, None)
            invalid_candidates.append((kind, provider_id, "同一 provider 存在重复 descriptor"))
            return
        target[provider_id] = info

    async def _refresh_agents(self, provider_id: str, descriptor: Mapping[str, Any]) -> None:
        del descriptor
        api_name = f"{provider_id}.describe_agents"
        try:
            envelope = await self._ctx.api.call(api_name, version="1")
            payloads = self._validate_envelope(envelope, "agents")
            definitions = tuple(AgentDefinition.model_validate(payload) for payload in payloads)
            self._agents.replace_provider(provider_id, definitions)
        except Exception as exc:
            self._reject_agents(provider_id, [str(exc)])
            return
        self._record_health("agents", provider_id, self._agents.health[provider_id].fingerprint)

    async def _refresh_procedures(self, provider_id: str, descriptor: Mapping[str, Any]) -> None:
        del descriptor
        api_name = f"{provider_id}.describe_procedures"
        try:
            envelope = await self._ctx.api.call(api_name, version="1")
            payloads = self._validate_envelope(envelope, "procedures")
            definitions = tuple(ProcedureDefinition.model_validate(payload) for payload in payloads)
            self._procedures.replace_provider(provider_id, definitions)
        except Exception as exc:
            self._reject_procedures(provider_id, [str(exc)])
            return
        self._record_health("procedures", provider_id, self._procedures.health[provider_id].fingerprint)

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

    def _reject_agents(self, provider_id: str, errors: Sequence[str]) -> None:
        self._agents.reject_provider(provider_id, errors)
        self._record_health("agents", provider_id, self._agents.health[provider_id].fingerprint)

    def _reject_procedures(self, provider_id: str, errors: Sequence[str]) -> None:
        self._procedures.reject_provider(provider_id, errors)
        self._record_health("procedures", provider_id, self._procedures.health[provider_id].fingerprint)

    def _record_health(self, kind: str, provider_id: str, fingerprint: str) -> None:
        self._extension_fingerprints[f"{kind}:{provider_id}"] = fingerprint

    def _record_global_error(self, exc: Exception) -> None:
        self._extension_fingerprints["discovery"] = canonical_fingerprint(
            {"status": "invalid", "error": str(exc)}
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
            await self.refresh()

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
