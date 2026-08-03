from __future__ import annotations

import pytest

from lunagentic_research_swarm.agents.registry import AgentRegistry, RootAgentUnavailableError
from lunagentic_research_swarm.config import AgentOverride, ProcedureOverride
from lunagentic_research_swarm.extensions.contracts import AgentDefinition, ProcedureDefinition
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry


def agent(agent_id: str, **changes: object) -> AgentDefinition:
    payload: dict[str, object] = {
        "agent_id": agent_id,
        "version": "1",
        "display_name": agent_id,
        "description": "测试智能体",
        "character_prompt": "按要求工作。",
        "model_selector": "task:utils",
        "can_be_root": True,
    }
    payload.update(changes)
    return AgentDefinition.model_validate(payload)


def procedure(procedure_id: str, **changes: object) -> ProcedureDefinition:
    payload: dict[str, object] = {
        "procedure_id": procedure_id,
        "version": "1",
        "display_name": procedure_id,
        "description": "测试流程",
        "arguments_schema": {"type": "object"},
        "result_schema": {"type": "object"},
    }
    payload.update(changes)
    return ProcedureDefinition.model_validate(payload)


def test_provider_replacement_is_atomic_and_rejected_batch_removes_stale_catalog() -> None:
    registry = AgentRegistry(root_agent="one.root")
    registry.replace_provider("provider.one", [agent("one.root"), agent("one.reader")])

    delta = registry.replace_provider("provider.one", [agent("one.root"), agent("one.writer")])
    assert delta.added == ("one.writer",)
    assert delta.removed == ("one.reader",)
    assert registry.is_live("one.writer")

    with pytest.raises(ValueError, match="重复"):
        registry.replace_provider("provider.one", [agent("one.root"), agent("one.root")])

    assert not registry.is_live("one.root")
    assert not registry.is_live("one.writer")
    assert registry.health["provider.one"].status == "invalid"
    assert registry.health["provider.one"].errors


def test_provider_cannot_claim_definition_owned_by_another_provider() -> None:
    registry = AgentRegistry(root_agent="one.root")
    registry.replace_provider("provider.one", [agent("one.root")])

    with pytest.raises(ValueError, match="provider.one"):
        registry.replace_provider("other.one", [agent("one.root")])

    assert registry.health["other.one"].status == "invalid"
    assert registry.is_live("one.root")


def test_provider_cannot_publish_outside_its_own_namespace() -> None:
    agents = AgentRegistry(root_agent="one.root")
    procedures = ProcedureRegistry()
    agents.replace_provider("provider.tools", [agent("tools.reader")])
    procedures.replace_provider("provider.tools", [procedure("tools.fetch")])

    with pytest.raises(ValueError, match="provider.tools"):
        agents.replace_provider("provider.evil", [agent("tools.writer")])
    with pytest.raises(ValueError, match="provider.tools"):
        procedures.replace_provider("provider.evil", [procedure("tools.search")])

    assert agents.health["provider.evil"].status == "invalid"
    assert procedures.health["provider.evil"].status == "invalid"


def test_registry_normalizes_mapping_payloads_through_strict_contracts() -> None:
    agents = AgentRegistry(root_agent="builtin.root")
    procedures = ProcedureRegistry()

    agents.replace_provider("builtin", [agent("builtin.root").model_dump(mode="json")])
    procedures.replace_provider("com.0-hz.fetch-url", [procedure("fetch_url.fetch").model_dump(mode="json")])

    assert agents.snapshot({}).get("builtin.root") is not None
    assert procedures.snapshot({}).get("fetch_url.fetch") is not None

    bad = agent("builtin.bad").model_dump(mode="json")
    bad["model_selector"] = "silent-fallback"
    with pytest.raises(ValueError, match="task:|model:"):
        agents.replace_provider("builtin", [bad])


def test_snapshot_applies_only_explicit_overrides_and_validates_root() -> None:
    registry = AgentRegistry(root_agent="one.root")
    registry.replace_provider(
        "provider.one",
        [agent("one.root", model_selector="task:planner"), agent("one.disabled", enabled=False)],
    )

    snapshot = registry.snapshot(
        {
            "one.root": AgentOverride(selector="model:gpt-test", protocol=None, enabled=None),
            "missing.agent": AgentOverride(enabled=False),
        }
    )
    assert snapshot.get("one.root").definition.model_selector == "model:gpt-test"
    assert snapshot.get("one.root").definition.protocol == "json_envelope"
    assert snapshot.get("one.disabled") is None

    with pytest.raises(ValueError, match="task:|model:"):
        registry.snapshot({"one.root": AgentOverride(selector="invalid")})

    registry.set_root_agent("one.disabled")
    with pytest.raises(RootAgentUnavailableError, match="根智能体"):
        registry.snapshot({})


def test_snapshot_is_frozen_while_live_check_reflects_provider_removal() -> None:
    registry = AgentRegistry(root_agent="one.root")
    registry.replace_provider("provider.one", [agent("one.root")])
    snapshot = registry.snapshot({})

    registry.remove_provider("provider.one")

    assert snapshot.get("one.root") is not None
    assert not registry.is_live("one.root")
    with pytest.raises(AttributeError):
        snapshot.entries.append(snapshot.entries[0])  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        snapshot.fingerprint = "mutated"  # type: ignore[misc]


def test_snapshot_entries_keep_provider_api_identity_and_stable_fingerprint() -> None:
    first = AgentRegistry(root_agent="one.root")
    second = AgentRegistry(root_agent="one.root")
    first.replace_provider("provider.one", [agent("one.reader"), agent("one.root")])
    second.replace_provider("provider.one", [agent("one.root"), agent("one.reader")])

    first_snapshot = first.snapshot({})
    second_snapshot = second.snapshot({})
    entry = first_snapshot.get("one.root")
    assert entry.provider_plugin_id == "provider.one"
    assert entry.api_name == "provider.one.describe_agents"
    assert entry.api_version == "1"
    assert entry.fingerprint
    assert first_snapshot.fingerprint == second_snapshot.fingerprint


def test_agent_allowlist_is_intersected_only_when_round_catalog_is_frozen() -> None:
    agents = AgentRegistry(root_agent="one.root")
    procedures = ProcedureRegistry()
    agents.replace_provider(
        "provider.one",
        [agent("one.root", allowed_procedures=["tools.fetch", "missing.optional"])],
    )
    procedures.replace_provider("provider.tools", [procedure("tools.fetch"), procedure("tools.off", enabled=False)])

    agent_snapshot = agents.snapshot({})
    procedure_snapshot = procedures.snapshot({})

    assert agent_snapshot.get("one.root").definition.allowed_procedures == ["tools.fetch", "missing.optional"]
    assert agent_snapshot.resolve_allowed_procedures("one.root", procedure_snapshot) == ("tools.fetch",)


def test_procedure_registry_replacement_overrides_and_removal() -> None:
    registry = ProcedureRegistry()
    registry.replace_provider("provider.tools", [procedure("tools.fetch"), procedure("tools.search")])
    snapshot = registry.snapshot({"tools.fetch": ProcedureOverride(timeout_seconds=12.5)})

    assert snapshot.get("tools.fetch").definition.timeout_seconds == 12.5
    assert snapshot.get("tools.fetch").api_name == "provider.tools.invoke_procedure"
    assert registry.is_live("tools.search")

    registry.remove_provider("provider.tools")
    assert snapshot.get("tools.search") is not None
    assert not registry.is_live("tools.search")
