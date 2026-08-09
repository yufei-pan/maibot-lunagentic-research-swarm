"""按 provider 原子替换的 agent 目录与冻结 round snapshot。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.config import AgentOverride
from lunagentic_research_swarm.extensions.contracts import AgentDefinition, CatalogDelta, ProviderHealth
from lunagentic_research_swarm.extensions.validation import (
    authorized_provider_namespace,
    canonical_fingerprint,
    validate_model_selector,
)


class RootAgentUnavailableError(ValueError):
    """配置的根智能体不能进入新 round。"""


@dataclass(frozen=True, slots=True)
class AgentCatalogEntry:
    definition: AgentDefinition
    provider_plugin_id: str
    api_name: str
    api_version: str
    fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class AgentCatalogSnapshot:
    """一个 round 持有的不可替换 agent 目录。"""

    entries: tuple[AgentCatalogEntry, ...]
    fingerprint: str
    root_agent: str
    _by_id: Mapping[str, AgentCatalogEntry]

    def __init__(self, entries: Sequence[AgentCatalogEntry], *, root_agent: str) -> None:
        ordered = tuple(sorted(entries, key=lambda entry: entry.definition.agent_id))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "root_agent", root_agent)
        object.__setattr__(
            self, "_by_id", MappingProxyType({entry.definition.agent_id: entry for entry in ordered})
        )
        object.__setattr__(self, "fingerprint", canonical_fingerprint(
            [
                {
                    "definition": entry.definition,
                    "provider_plugin_id": entry.provider_plugin_id,
                    "api_name": entry.api_name,
                    "api_version": entry.api_version,
                    "fingerprint": entry.fingerprint,
                }
                for entry in ordered
            ]
        ))

    def get(self, agent_id: str) -> AgentCatalogEntry | None:
        return self._by_id.get(agent_id)

    def resolve_allowed_procedures(self, agent_id: str, procedures: Any) -> tuple[str, ...]:
        """解析本智能体可调用的研究 Procedure（由 Procedure.allowed_agents 决定）。

        Agent 侧 ``allowed_procedures`` 不再参与裁决；保留本方法名以兼容旧调用方。
        ``core.*`` 由执行器控制路径单独处理，不出现在本返回值中——runtime header
        会另行并入 core ID。
        """

        if self.get(agent_id) is None:
            return ()
        resolve = getattr(procedures, "resolve_callable_procedures", None)
        if callable(resolve):
            return tuple(str(item) for item in resolve(agent_id))
        # 兼容仅有 ids 的假目录：视为全部开放。
        ids = getattr(procedures, "ids", ())
        if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes, bytearray)):
            return tuple(str(item) for item in ids)
        return ()


@dataclass(frozen=True, slots=True)
class _ProviderBatch:
    definitions: tuple[AgentDefinition, ...]
    fingerprint: str


class AgentRegistry:
    """live agent 目录；每个 provider 的批次只能整批进入或整批离开。"""

    def __init__(self, *, root_agent: str) -> None:
        self._root_agent = root_agent
        self._providers: dict[str, _ProviderBatch] = {}
        self._owners: dict[str, str] = {}
        self._health: dict[str, ProviderHealth] = {}
        # agent_id 集合：provider 显式声明了 protocol，不接受全局 default_mode。
        self._explicit_protocol: set[str] = set()

    @property
    def health(self) -> Mapping[str, ProviderHealth]:
        return MappingProxyType(dict(self._health))

    @property
    def provider_ids(self) -> frozenset[str]:
        return frozenset(
            provider_id for provider_id, health in self._health.items() if health.status != "removed"
        )

    def set_root_agent(self, agent_id: str) -> None:
        self._root_agent = agent_id

    def replace_provider(
        self, provider_id: str, definitions: Sequence[AgentDefinition | Mapping[str, Any]]
    ) -> CatalogDelta:
        try:
            checked = self._validate_batch(provider_id, definitions)
        except (TypeError, ValueError) as exc:
            self.reject_provider(provider_id, [str(exc)])
            raise

        old = self._providers.get(provider_id)
        old_by_id = {item.agent_id: item for item in old.definitions} if old else {}
        new_by_id = {item.agent_id: item for item in checked}
        for agent_id in old_by_id:
            self._owners.pop(agent_id, None)
        fingerprint = canonical_fingerprint(list(checked), id_field="agent_id")
        self._providers[provider_id] = _ProviderBatch(checked, fingerprint)
        for agent_id in new_by_id:
            self._owners[agent_id] = provider_id
        self._health[provider_id] = ProviderHealth(provider_id, "healthy", fingerprint=fingerprint)
        return CatalogDelta(
            added=tuple(sorted(new_by_id.keys() - old_by_id.keys())),
            removed=tuple(sorted(old_by_id.keys() - new_by_id.keys())),
            updated=tuple(
                sorted(
                    agent_id
                    for agent_id in old_by_id.keys() & new_by_id.keys()
                    if old_by_id[agent_id] != new_by_id[agent_id]
                )
            ),
        )

    def _validate_batch(
        self, provider_id: str, definitions: Sequence[AgentDefinition | Mapping[str, Any]]
    ) -> tuple[AgentDefinition, ...]:
        authorized_namespace = authorized_provider_namespace(provider_id)
        raw_definitions: list[Mapping[str, Any]] = []
        for item in definitions:
            if isinstance(item, AgentDefinition):
                raw_definitions.append(item.model_dump(mode="python"))
            elif isinstance(item, Mapping):
                raw_definitions.append(item)
            else:
                raise TypeError("agent provider 批次只能包含 AgentDefinition 或 Mapping")
        checked = tuple(AgentDefinition.model_validate(item) for item in raw_definitions)
        # `[protocol] default_mode` only applies to definitions that never stated a
        # protocol; an explicit per-agent protocol always wins.
        for raw, definition in zip(raw_definitions, checked):
            if "protocol" in raw:
                self._explicit_protocol.add(definition.agent_id)
            else:
                self._explicit_protocol.discard(definition.agent_id)
        ids = [item.agent_id for item in checked]
        if len(set(ids)) != len(ids):
            raise ValueError("agent provider 批次包含重复 ID")
        for agent_id in ids:
            if agent_id.partition(".")[0] != authorized_namespace:
                raise ValueError(
                    f"agent_id {agent_id} 不属于 provider {provider_id} 获授权的命名空间 {authorized_namespace}"
                )
            owner = self._owners.get(agent_id)
            if owner is not None and owner != provider_id:
                raise ValueError(f"agent_id {agent_id} 已属于 provider {owner}，不得冒充")
        return checked

    def reject_provider(self, provider_id: str, errors: Sequence[str]) -> None:
        self._drop_provider(provider_id)
        normalized = tuple(str(error) for error in errors) or ("provider 批次无效",)
        fingerprint = canonical_fingerprint({"provider_plugin_id": provider_id, "errors": list(normalized)})
        self._health[provider_id] = ProviderHealth(provider_id, "invalid", normalized, fingerprint)

    def remove_provider(self, provider_id: str) -> CatalogDelta:
        removed = self._drop_provider(provider_id)
        fingerprint = canonical_fingerprint({"provider_plugin_id": provider_id, "status": "removed"})
        self._health[provider_id] = ProviderHealth(provider_id, "removed", fingerprint=fingerprint)
        return CatalogDelta(removed=removed)

    def _drop_provider(self, provider_id: str) -> tuple[str, ...]:
        old = self._providers.pop(provider_id, None)
        if old is None:
            return ()
        removed = tuple(sorted(item.agent_id for item in old.definitions))
        for agent_id in removed:
            self._owners.pop(agent_id, None)
        return removed

    def is_live(self, agent_id: str) -> bool:
        provider_id = self._owners.get(agent_id)
        if provider_id is None:
            return False
        batch = self._providers[provider_id]
        return any(item.agent_id == agent_id and item.enabled for item in batch.definitions)

    def root_capable_agent_ids(
        self,
        overrides: Mapping[str, AgentOverride] | None = None,
        *,
        default_protocol: str | None = None,
    ) -> tuple[str, ...]:
        """返回启用且 ``can_be_root`` 的智能体 ID（不校验当前配置 root 是否可用）。"""

        overrides = overrides or {}
        chosen: list[str] = []
        for provider_id, batch in self._providers.items():
            del provider_id
            for original in batch.definitions:
                fallback = (
                    default_protocol
                    if default_protocol and original.agent_id not in self._explicit_protocol
                    else None
                )
                definition = self._apply_override(
                    original, overrides.get(original.agent_id), default_protocol=fallback
                )
                if definition.enabled and definition.can_be_root:
                    chosen.append(definition.agent_id)
        return tuple(sorted(set(chosen)))

    def snapshot(
        self,
        overrides: Mapping[str, AgentOverride],
        *,
        default_protocol: str | None = None,
    ) -> AgentCatalogSnapshot:
        entries: list[AgentCatalogEntry] = []
        for provider_id, batch in self._providers.items():
            for original in batch.definitions:
                fallback = (
                    default_protocol
                    if default_protocol and original.agent_id not in self._explicit_protocol
                    else None
                )
                definition = self._apply_override(
                    original, overrides.get(original.agent_id), default_protocol=fallback
                )
                if not definition.enabled:
                    continue
                fingerprint = canonical_fingerprint(definition)
                entries.append(
                    AgentCatalogEntry(
                        definition=definition,
                        provider_plugin_id=provider_id,
                        api_name=f"{provider_id}.describe_agents",
                        api_version="1",
                        fingerprint=fingerprint,
                    )
                )
        snapshot = AgentCatalogSnapshot(entries, root_agent=self._root_agent)
        root = snapshot.get(self._root_agent)
        if root is None or not root.definition.can_be_root:
            raise RootAgentUnavailableError(f"配置的根智能体 {self._root_agent} 不存在、未启用或不允许成为 root")
        return snapshot

    @staticmethod
    def _apply_override(
        definition: AgentDefinition,
        override: AgentOverride | None,
        *,
        default_protocol: str | None = None,
    ) -> AgentDefinition:
        explicit = override.model_dump(mode="python", exclude_unset=True) if override is not None else {}
        if override is None and not default_protocol:
            return definition.model_copy(deep=True)
        raw = definition.model_dump(mode="python")
        if default_protocol and explicit.get("protocol") is None:
            raw["protocol"] = default_protocol
        if explicit.get("enabled") is not None:
            raw["enabled"] = explicit["enabled"]
        if explicit.get("protocol") is not None:
            raw["protocol"] = explicit["protocol"]
        if explicit.get("auto_compact_tokens") is not None:
            raw["auto_compact_tokens"] = explicit["auto_compact_tokens"]
        if explicit.get("selector") is not None:
            raw["model_selector"] = validate_model_selector(explicit["selector"])
        return AgentDefinition.model_validate(raw)
