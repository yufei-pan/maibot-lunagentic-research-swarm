"""按 provider 原子替换的 agent 目录与冻结 round snapshot。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.config import AgentOverride
from lunagentic_research_swarm.extensions.contracts import AgentDefinition, CatalogDelta, ProviderHealth
from lunagentic_research_swarm.extensions.validation import canonical_fingerprint, validate_model_selector


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
        """在冻结 round 时才把语法有效的 allowlist 与 Procedure snapshot 求交。"""

        entry = self.get(agent_id)
        if entry is None:
            return ()
        available = set(procedures.ids)
        allowlist = entry.definition.allowed_procedures
        if allowlist == ["*"]:
            return tuple(sorted(available))
        return tuple(procedure_id for procedure_id in allowlist if procedure_id in available)


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
        self._namespace_owners: dict[str, str] = {}
        self._provider_namespaces: dict[str, str] = {}
        self._health: dict[str, ProviderHealth] = {}

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
            checked, namespace = self._validate_batch(provider_id, definitions)
        except (TypeError, ValueError) as exc:
            self.reject_provider(provider_id, [str(exc)])
            raise

        old = self._providers.get(provider_id)
        old_by_id = {item.agent_id: item for item in old.definitions} if old else {}
        new_by_id = {item.agent_id: item for item in checked}
        for agent_id in old_by_id:
            self._owners.pop(agent_id, None)
        fingerprint = canonical_fingerprint(checked, id_field="agent_id")
        self._providers[provider_id] = _ProviderBatch(checked, fingerprint)
        if namespace is not None:
            self._namespace_owners[namespace] = provider_id
            self._provider_namespaces[provider_id] = namespace
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
    ) -> tuple[tuple[AgentDefinition, ...], str | None]:
        if not isinstance(provider_id, str) or not provider_id.strip() or provider_id != provider_id.strip():
            raise ValueError("provider plugin ID 不能为空或包含首尾空白")
        raw_definitions: list[Mapping[str, Any]] = []
        for item in definitions:
            if isinstance(item, AgentDefinition):
                raw_definitions.append(item.model_dump(mode="python"))
            elif isinstance(item, Mapping):
                raw_definitions.append(item)
            else:
                raise TypeError("agent provider 批次只能包含 AgentDefinition 或 Mapping")
        checked = tuple(AgentDefinition.model_validate(item) for item in raw_definitions)
        ids = [item.agent_id for item in checked]
        if len(set(ids)) != len(ids):
            raise ValueError("agent provider 批次包含重复 ID")
        namespaces = {agent_id.partition(".")[0] for agent_id in ids}
        if len(namespaces) > 1:
            raise ValueError("同一 agent provider 批次只能声明一个命名空间")
        namespace = next(iter(namespaces), None)
        established = self._provider_namespaces.get(provider_id)
        if namespace is not None and established is not None and namespace != established:
            raise ValueError(f"provider {provider_id} 已绑定命名空间 {established}，不得改为 {namespace}")
        namespace_owner = self._namespace_owners.get(namespace) if namespace is not None else None
        if namespace_owner is not None and namespace_owner != provider_id:
            raise ValueError(f"命名空间 {namespace} 已属于 provider {namespace_owner}，不得冒充")
        for agent_id in ids:
            owner = self._owners.get(agent_id)
            if owner is not None and owner != provider_id:
                raise ValueError(f"agent_id {agent_id} 已属于 provider {owner}，不得冒充")
        return checked, namespace

    def reject_provider(self, provider_id: str, errors: Sequence[str]) -> None:
        self._drop_provider(provider_id)
        normalized = tuple(str(error) for error in errors) or ("provider 批次无效",)
        fingerprint = canonical_fingerprint({"provider_plugin_id": provider_id, "errors": normalized})
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

    def snapshot(self, overrides: Mapping[str, AgentOverride]) -> AgentCatalogSnapshot:
        entries: list[AgentCatalogEntry] = []
        for provider_id, batch in self._providers.items():
            for original in batch.definitions:
                definition = self._apply_override(original, overrides.get(original.agent_id))
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
    def _apply_override(definition: AgentDefinition, override: AgentOverride | None) -> AgentDefinition:
        if override is None:
            return definition.model_copy(deep=True)
        explicit = override.model_dump(mode="python", exclude_unset=True)
        raw = definition.model_dump(mode="python")
        if explicit.get("enabled") is not None:
            raw["enabled"] = explicit["enabled"]
        if explicit.get("protocol") is not None:
            raw["protocol"] = explicit["protocol"]
        if explicit.get("auto_compact_tokens") is not None:
            raw["auto_compact_tokens"] = explicit["auto_compact_tokens"]
        if explicit.get("selector") is not None:
            raw["model_selector"] = validate_model_selector(explicit["selector"])
        return AgentDefinition.model_validate(raw)
