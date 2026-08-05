"""内置 Procedure provider：describe / registry 装入 / invoke 分发。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.extensions.validation import validate_procedure_batch
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry
from lunagentic_research_swarm.services import _load_builtin_providers


_EXPECTED_MEMORY_IDS = {
    "builtin.chat_streams",
    "builtin.message_recent",
    "builtin.message_by_id",
    "builtin.message_time_range",
    "builtin.person_lookup",
    "builtin.knowledge_search",
}


class _Knowledge:
    async def search(self, query: str, limit: int = 5):
        return [{"title": "hit", "snippet": query, "limit": limit}]


def _provider() -> BundledProcedureProvider:
    return BundledProcedureProvider(SimpleNamespace(knowledge=_Knowledge()))


def test_describe_returns_six_valid_memory_procedures() -> None:
    payloads = _provider().describe()
    definitions = validate_procedure_batch("builtin", payloads)
    assert {item.procedure_id for item in definitions} == _EXPECTED_MEMORY_IDS
    assert all(item.idempotent is True for item in definitions)
    assert all(item.timeout_seconds == 30.0 for item in definitions)
    assert all(item.external_cost_kind == "none" for item in definitions)
    assert all(item.enabled is True for item in definitions)


def test_validate_procedure_batch_rejects_foreign_namespace() -> None:
    payloads = _provider().describe()
    payloads[0]["procedure_id"] = "evil.chat_streams"
    with pytest.raises(ValueError, match="命名空间"):
        validate_procedure_batch("builtin", payloads)


def test_registry_loads_builtin_procedure_provider() -> None:
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", _provider().describe())
    snapshot = registry.snapshot({})
    for procedure_id in _EXPECTED_MEMORY_IDS:
        entry = snapshot.get(procedure_id)
        assert entry is not None
        assert entry.provider_plugin_id == "builtin"
        assert entry.api_name == "builtin.invoke_procedure"


def test_builtin_loader_registers_procedures_without_scheduler_shortcut() -> None:
    agents = AgentRegistry(root_agent="builtin.quick_thinker")
    procedures = ProcedureRegistry()
    _load_builtin_providers(agents, procedures)
    snapshot = procedures.snapshot({})
    assert set(snapshot.ids) >= _EXPECTED_MEMORY_IDS
    for procedure_id in _EXPECTED_MEMORY_IDS:
        assert snapshot.get(procedure_id).api_name == "builtin.invoke_procedure"


@pytest.mark.asyncio
async def test_invoke_unknown_procedure_is_invalid_arguments() -> None:
    result = await _provider().invoke("builtin.not_a_thing", {})
    assert not result.success
    assert result.error.code == "invalid_arguments"


@pytest.mark.asyncio
async def test_invoke_dispatches_handler_map() -> None:
    result = await _provider().invoke("builtin.knowledge_search", {"query": "q", "limit": 2})
    assert result.success
    assert result.data["query"] == "q"
    assert isinstance(result.data, dict)
    assert "items" in result.data
    assert "truncated" in result.data
