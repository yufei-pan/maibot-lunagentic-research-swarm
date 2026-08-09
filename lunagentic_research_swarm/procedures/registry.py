"""按 provider 原子替换的 Procedure 目录与冻结 round snapshot。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.config import ProcedureOverride
from lunagentic_research_swarm.extensions.contracts import CatalogDelta, ProcedureDefinition, ProviderHealth
from lunagentic_research_swarm.extensions.validation import authorized_provider_namespace, canonical_fingerprint


@dataclass(frozen=True, slots=True)
class ProcedureCatalogEntry:
    definition: ProcedureDefinition
    provider_plugin_id: str
    api_name: str
    api_version: str
    fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class ProcedureCatalogSnapshot:
    entries: tuple[ProcedureCatalogEntry, ...]
    fingerprint: str
    ids: tuple[str, ...]
    _by_id: Mapping[str, ProcedureCatalogEntry]

    def __init__(self, entries: Sequence[ProcedureCatalogEntry]) -> None:
        ordered = tuple(sorted(entries, key=lambda entry: entry.definition.procedure_id))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "ids", tuple(entry.definition.procedure_id for entry in ordered))
        object.__setattr__(
            self, "_by_id", MappingProxyType({entry.definition.procedure_id: entry for entry in ordered})
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

    def get(self, procedure_id: str) -> ProcedureCatalogEntry | None:
        return self._by_id.get(procedure_id)

    def resolve_callable_procedures(self, agent_id: str) -> tuple[str, ...]:
        """按 Procedure.allowed_agents 解析该智能体可调用的研究 Procedure ID。"""

        agent = str(agent_id or "").strip()
        if not agent:
            return ()
        chosen: list[str] = []
        for entry in self.entries:
            allowed = entry.definition.allowed_agents
            if allowed == ["*"] or agent in allowed:
                chosen.append(entry.definition.procedure_id)
        return tuple(chosen)


@dataclass(frozen=True, slots=True)
class _ProviderBatch:
    definitions: tuple[ProcedureDefinition, ...]
    fingerprint: str


class ProcedureRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, _ProviderBatch] = {}
        self._owners: dict[str, str] = {}
        self._health: dict[str, ProviderHealth] = {}

    @property
    def health(self) -> Mapping[str, ProviderHealth]:
        return MappingProxyType(dict(self._health))

    @property
    def provider_ids(self) -> frozenset[str]:
        return frozenset(
            provider_id for provider_id, health in self._health.items() if health.status != "removed"
        )

    def replace_provider(
        self, provider_id: str, definitions: Sequence[ProcedureDefinition | Mapping[str, Any]]
    ) -> CatalogDelta:
        try:
            checked = self._validate_batch(provider_id, definitions)
        except (TypeError, ValueError) as exc:
            self.reject_provider(provider_id, [str(exc)])
            raise
        old = self._providers.get(provider_id)
        old_by_id = {item.procedure_id: item for item in old.definitions} if old else {}
        new_by_id = {item.procedure_id: item for item in checked}
        for procedure_id in old_by_id:
            self._owners.pop(procedure_id, None)
        fingerprint = canonical_fingerprint(list(checked), id_field="procedure_id")
        self._providers[provider_id] = _ProviderBatch(checked, fingerprint)
        for procedure_id in new_by_id:
            self._owners[procedure_id] = provider_id
        self._health[provider_id] = ProviderHealth(provider_id, "healthy", fingerprint=fingerprint)
        return CatalogDelta(
            added=tuple(sorted(new_by_id.keys() - old_by_id.keys())),
            removed=tuple(sorted(old_by_id.keys() - new_by_id.keys())),
            updated=tuple(
                sorted(
                    procedure_id
                    for procedure_id in old_by_id.keys() & new_by_id.keys()
                    if old_by_id[procedure_id] != new_by_id[procedure_id]
                )
            ),
        )

    def _validate_batch(
        self, provider_id: str, definitions: Sequence[ProcedureDefinition | Mapping[str, Any]]
    ) -> tuple[ProcedureDefinition, ...]:
        authorized_namespace = authorized_provider_namespace(provider_id)
        raw_definitions: list[Mapping[str, Any]] = []
        for item in definitions:
            if isinstance(item, ProcedureDefinition):
                raw_definitions.append(item.model_dump(mode="python"))
            elif isinstance(item, Mapping):
                raw_definitions.append(item)
            else:
                raise TypeError("Procedure provider 批次只能包含 ProcedureDefinition 或 Mapping")
        checked = tuple(ProcedureDefinition.model_validate(item) for item in raw_definitions)
        ids = [item.procedure_id for item in checked]
        if len(set(ids)) != len(ids):
            raise ValueError("Procedure provider 批次包含重复 ID")
        for procedure_id in ids:
            if procedure_id.startswith("core."):
                raise ValueError(f"procedure_id {procedure_id} 不得使用保留的 core 命名空间")
            if procedure_id.partition(".")[0] != authorized_namespace:
                raise ValueError(
                    f"procedure_id {procedure_id} 不属于 provider {provider_id} 获授权的命名空间 {authorized_namespace}"
                )
            owner = self._owners.get(procedure_id)
            if owner is not None and owner != provider_id:
                raise ValueError(f"procedure_id {procedure_id} 已属于 provider {owner}，不得冒充")
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
        removed = tuple(sorted(item.procedure_id for item in old.definitions))
        for procedure_id in removed:
            self._owners.pop(procedure_id, None)
        return removed

    def is_live(self, procedure_id: str) -> bool:
        provider_id = self._owners.get(procedure_id)
        if provider_id is None:
            return False
        batch = self._providers[provider_id]
        return any(item.procedure_id == procedure_id and item.enabled for item in batch.definitions)

    def snapshot(self, overrides: Mapping[str, ProcedureOverride]) -> ProcedureCatalogSnapshot:
        entries: list[ProcedureCatalogEntry] = []
        for provider_id, batch in self._providers.items():
            for original in batch.definitions:
                definition = self._apply_override(original, overrides.get(original.procedure_id))
                if not definition.enabled:
                    continue
                fingerprint = canonical_fingerprint(definition)
                entries.append(
                    ProcedureCatalogEntry(
                        definition=definition,
                        provider_plugin_id=provider_id,
                        api_name=f"{provider_id}.invoke_procedure",
                        api_version="1",
                        fingerprint=fingerprint,
                    )
                )
        return ProcedureCatalogSnapshot(entries)

    @staticmethod
    def _apply_override(
        definition: ProcedureDefinition, override: ProcedureOverride | None
    ) -> ProcedureDefinition:
        if override is None:
            return definition.model_copy(deep=True)
        explicit = override.model_dump(mode="python", exclude_unset=True)
        raw = definition.model_dump(mode="python")
        if explicit.get("enabled") is not None:
            raw["enabled"] = explicit["enabled"]
        if explicit.get("timeout_seconds") is not None:
            raw["timeout_seconds"] = explicit["timeout_seconds"]
        return ProcedureDefinition.model_validate(raw)
