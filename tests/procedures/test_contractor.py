"""builtin.contractor：定义默认值、目录注册、禁用覆写与 stub 行为。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.config import ProcedureOverride
from lunagentic_research_swarm.procedures.bundled.contractor import (
    CONTRACTOR_PROCEDURE_ID,
    contractor_procedure_definitions,
)
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

_PLUGIN_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DEFAULT = _PLUGIN_ROOT / "config.default.toml"

_EXPECTED_BUNDLED_PROCEDURE_IDS = (
    "builtin.chat_streams",
    "builtin.message_recent",
    "builtin.message_by_id",
    "builtin.message_time_range",
    "builtin.person_lookup",
    "builtin.knowledge_search",
    "builtin.calculate",
    "builtin.statistics",
    "builtin.convert_units",
    "builtin.normalize_urls",
    "builtin.organize_provenance",
    "builtin.web_search",
    "builtin.past_cases",
    "builtin.contractor",
)


def test_contractor_definition_defaults() -> None:
    defs = contractor_procedure_definitions()
    assert len(defs) == 1
    d = defs[0]
    assert d.procedure_id == CONTRACTOR_PROCEDURE_ID
    assert d.timeout_seconds == 0.0
    assert d.enabled is True
    assert d.idempotent is False
    props = d.arguments_schema["properties"]
    assert "agent_id" in props
    assert "question" in props
    assert "temperature" in props
    assert "personality" in props
    assert "time_budget_seconds" in props
    assert set(d.arguments_schema["required"]) == {"agent_id", "question"}
    assert d.result_schema == {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    }


def test_bundled_provider_includes_contractor_when_described() -> None:
    payloads = BundledProcedureProvider(SimpleNamespace()).describe()
    ids = {item["procedure_id"] for item in payloads}
    assert CONTRACTOR_PROCEDURE_ID in ids
    contractor = next(item for item in payloads if item["procedure_id"] == CONTRACTOR_PROCEDURE_ID)
    assert contractor["timeout_seconds"] == 0.0
    assert contractor["enabled"] is True
    assert contractor["idempotent"] is False


@pytest.mark.asyncio
async def test_contractor_stub_returns_runtime_missing_without_deps() -> None:
    result = await BundledProcedureProvider(SimpleNamespace()).invoke(
        CONTRACTOR_PROCEDURE_ID,
        {"agent_id": "builtin.quick_thinker", "question": "1+1?"},
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.code == "contractor_runtime_missing"


def test_contractor_disabled_by_override_removed_from_snapshot() -> None:
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", contractor_procedure_definitions())
    enabled = registry.snapshot({})
    assert enabled.get(CONTRACTOR_PROCEDURE_ID) is not None

    disabled = registry.snapshot({CONTRACTOR_PROCEDURE_ID: ProcedureOverride(enabled=False)})
    assert disabled.get(CONTRACTOR_PROCEDURE_ID) is None


def test_config_default_toml_lists_all_bundled_procedure_toggles() -> None:
    text = _CONFIG_DEFAULT.read_text(encoding="utf-8")
    for procedure_id in _EXPECTED_BUNDLED_PROCEDURE_IDS:
        assert f'[procedures."{procedure_id}"]' in text
    assert "timeout_seconds = 0" in text
    contractor_block = text.split('[procedures."builtin.contractor"]', 1)[1]
    assert "enabled = true" in contractor_block.split("[", 1)[0]
    assert "timeout_seconds = 0" in contractor_block.split("[", 1)[0]
