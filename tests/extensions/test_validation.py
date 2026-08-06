from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from lunagentic_research_swarm.extensions.contracts import (
    AgentDefinition,
    ProcedureDefinition,
    ProcedureInvocation,
    ProcedureResult,
)
from lunagentic_research_swarm.extensions.validation import canonical_fingerprint


def agent_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "agent_id": "example.reader",
        "version": "1",
        "display_name": "阅读者",
        "description": "读取材料",
        "character_prompt": "只基于材料报告。",
        "model_selector": "task:utils",
    }
    payload.update(changes)
    return payload


def procedure_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "procedure_id": "example.fetch",
        "version": "1",
        "display_name": "抓取",
        "description": "抓取指定资源",
        "arguments_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
        "result_schema": {"type": "object"},
    }
    payload.update(changes)
    return payload


def test_missing_optional_agent_fields_use_documented_defaults() -> None:
    definition = AgentDefinition.model_validate(agent_payload())

    assert definition.protocol == "json_envelope"
    assert definition.allowed_procedures == ["*"]
    assert definition.can_be_root is False
    assert definition.auto_compact_tokens is None
    assert definition.enabled is True


def test_agent_public_fields_are_exact() -> None:
    assert set(AgentDefinition.model_fields) == {
        "agent_id",
        "version",
        "display_name",
        "description",
        "character_prompt",
        "model_selector",
        "protocol",
        "allowed_procedures",
        "can_be_root",
        "auto_compact_tokens",
        "enabled",
    }


def test_contract_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AgentDefinition.model_validate(agent_payload(reasoning="high"))

    with pytest.raises(ValidationError, match="extra"):
        ProcedureDefinition.model_validate(procedure_payload(provider_plugin_id="payload-spoof"))


@pytest.mark.parametrize("agent_id", ["core.summarizer", "summarizer", "vendor.summarizer", "bad id", "a" * 129])
def test_agent_id_cannot_impersonate_core_or_be_invalid(agent_id: str) -> None:
    with pytest.raises(ValidationError):
        AgentDefinition.model_validate(agent_payload(agent_id=agent_id))


@pytest.mark.parametrize("selector", ["utils", "task:", "model:", " task:utils", "task:utils ", 123])
def test_explicit_invalid_selector_is_not_replaced(selector: object) -> None:
    with pytest.raises(ValidationError, match="task:|model:"):
        AgentDefinition.model_validate(agent_payload(model_selector=selector))


@pytest.mark.parametrize("allowed", [["missing.optional"], ["*"], ["example.fetch", "missing.optional"]])
def test_agent_allowlist_checks_syntax_without_requiring_availability(allowed: list[str]) -> None:
    assert AgentDefinition.model_validate(agent_payload(allowed_procedures=allowed)).allowed_procedures == allowed


@pytest.mark.parametrize("allowed", [[], ["bad id"], ["*", "example.fetch"], ["example.fetch", "example.fetch"]])
def test_agent_allowlist_rejects_empty_invalid_mixed_or_duplicate_values(allowed: list[str]) -> None:
    with pytest.raises(ValidationError):
        AgentDefinition.model_validate(agent_payload(allowed_procedures=allowed))


@pytest.mark.parametrize("field", ["arguments_schema", "result_schema"])
@pytest.mark.parametrize("schema", [{}, {"type": "array"}, [], "object"])
def test_procedure_schemas_must_be_object_schemas(field: str, schema: object) -> None:
    with pytest.raises(ValidationError, match="object"):
        ProcedureDefinition.model_validate(procedure_payload(**{field: schema}))


@pytest.mark.parametrize("timeout", [-1, 600.1, "30"])
def test_procedure_timeout_is_strictly_bounded(timeout: object) -> None:
    with pytest.raises(ValidationError):
        ProcedureDefinition.model_validate(procedure_payload(timeout_seconds=timeout))


def test_procedure_timeout_zero_means_disabled() -> None:
    definition = ProcedureDefinition.model_validate(procedure_payload(timeout_seconds=0.0))
    assert definition.timeout_seconds == 0.0


def test_procedure_defaults_and_public_fields_are_exact() -> None:
    definition = ProcedureDefinition.model_validate(procedure_payload())

    assert set(ProcedureDefinition.model_fields) == {
        "procedure_id",
        "version",
        "display_name",
        "description",
        "arguments_schema",
        "result_schema",
        "idempotent",
        "timeout_seconds",
        "external_cost_kind",
        "enabled",
    }
    assert definition.idempotent is False
    assert definition.timeout_seconds == 30.0
    assert definition.external_cost_kind == "none"
    assert definition.enabled is True


def test_invocation_and_result_have_fixed_strict_envelopes() -> None:
    invocation = ProcedureInvocation.model_validate(
        {
            "request_id": "req_1",
            "task_id": "lrs_1",
            "round_id": "rnd_1",
            "branch_id": "br_1",
            "turn_id": "turn_1",
            "agent_id": "example.reader",
            "arguments": {"url": "https://example.test"},
            "scoped_metadata": {"locale": "zh-CN"},
        }
    )
    result = ProcedureResult.model_validate(
        {
            "success": True,
            "data": {"title": "示例"},
            "error": None,
            "metadata": {"provider_plugin_id": "provider.example", "duration_ms": 2},
        }
    )

    assert set(ProcedureInvocation.model_fields) == {
        "request_id",
        "task_id",
        "round_id",
        "branch_id",
        "turn_id",
        "agent_id",
        "arguments",
        "scoped_metadata",
    }
    assert set(ProcedureResult.model_fields) == {
        "success",
        "data",
        "error",
        "metadata",
        "research_credits_charged",
    }
    assert result.research_credits_charged == 0.0
    with pytest.raises(ValidationError):
        ProcedureResult.model_validate({**result.model_dump(), "raw_payload": "secret"})
    assert invocation.arguments["url"] == "https://example.test"


def test_procedure_result_research_credits_charged_default() -> None:
    result = ProcedureResult.model_validate(
        {"success": True, "data": {}, "error": None, "metadata": {}}
    )
    assert result.research_credits_charged == 0.0
    assert set(ProcedureResult.model_fields) == {
        "success",
        "data",
        "error",
        "metadata",
        "research_credits_charged",
    }


def test_contract_nested_payloads_are_immutable_after_validation() -> None:
    agent = AgentDefinition.model_validate(agent_payload(allowed_procedures=["example.fetch"]))
    procedure = ProcedureDefinition.model_validate(
        procedure_payload(arguments_schema={"type": "object", "properties": {"tags": {"type": "array"}}})
    )
    invocation = ProcedureInvocation.model_validate(
        {
            "request_id": "req_1",
            "task_id": "lrs_1",
            "round_id": "rnd_1",
            "branch_id": "br_1",
            "turn_id": "turn_1",
            "agent_id": "example.reader",
            "arguments": {"tags": ["initial"]},
            "scoped_metadata": {},
        }
    )

    with pytest.raises(TypeError):
        agent.allowed_procedures.append("example.other")
    with pytest.raises(TypeError):
        procedure.arguments_schema["properties"]["tags"]["type"] = "string"
    with pytest.raises(TypeError):
        invocation.arguments["tags"].append("mutated")


@pytest.mark.parametrize(
    "bad_value", [{"bad": {1, 2}}, {"bad": object()}, {"bad": float("nan")}, {"bad": float("inf")}]
)
def test_invocation_and_result_reject_non_json_values(bad_value: dict[str, object]) -> None:
    invocation = {
        "request_id": "req_1",
        "task_id": "lrs_1",
        "round_id": "rnd_1",
        "branch_id": "br_1",
        "turn_id": "turn_1",
        "agent_id": "example.reader",
        "arguments": bad_value,
        "scoped_metadata": {},
    }
    with pytest.raises(ValidationError, match="JSON"):
        ProcedureInvocation.model_validate(invocation)
    with pytest.raises(ValidationError, match="JSON"):
        ProcedureResult.model_validate({"success": True, "data": bad_value, "error": None, "metadata": {}})


def test_canonical_fingerprint_uses_sorted_utf8_json_and_catalog_id_order() -> None:
    reader = AgentDefinition.model_validate(agent_payload(agent_id="example.reader"))
    writer = AgentDefinition.model_validate(agent_payload(agent_id="example.writer", display_name="写作者"))
    canonical = [reader.model_dump(mode="json"), writer.model_dump(mode="json")]
    expected = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert canonical_fingerprint([writer, reader], id_field="agent_id") == expected
    assert canonical_fingerprint([reader, writer], id_field="agent_id") == expected


@pytest.mark.parametrize(
    "invalid",
    [
        {1: "x"},
        ("x",),
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": b"x"},
        {"value": object()},
    ],
)
def test_canonical_fingerprint_rejects_non_json_values(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError), match="JSON"):
        canonical_fingerprint(invalid)


def test_canonical_fingerprint_does_not_collapse_key_or_sequence_types() -> None:
    with pytest.raises((TypeError, ValueError), match="JSON"):
        canonical_fingerprint({1: "x"})
    with pytest.raises((TypeError, ValueError), match="JSON"):
        canonical_fingerprint(("x",))

    assert canonical_fingerprint({"1": "x"})
    assert canonical_fingerprint(["x"])
