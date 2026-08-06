from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.procedures.core import split_procedure_requests
from lunagentic_research_swarm.runtime.events import event_from_json, event_to_json


def definition(procedure_id: str, *, idempotent: bool = False, timeout_seconds: float = 30.0) -> ProcedureDefinition:
    return ProcedureDefinition.model_validate(
        {
            "procedure_id": procedure_id,
            "version": "7",
            "display_name": procedure_id,
            "description": "测试 Procedure",
            "arguments_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "idempotent": idempotent,
            "timeout_seconds": timeout_seconds,
        }
    )


def catalog(*definitions: ProcedureDefinition) -> ProcedureCatalogSnapshot:
    return ProcedureCatalogSnapshot(
        [
            ProcedureCatalogEntry(
                definition=item,
                provider_plugin_id="provider.tools",
                api_name="provider.tools.invoke_procedure",
                api_version="1",
                fingerprint=f"fingerprint:{item.procedure_id}",
            )
            for item in definitions
        ]
    )


@dataclass
class FakeAPI:
    responses: dict[str, Any]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def call(self, name: str, *, version: str = "", **kwargs: Any) -> Any:
        self.calls.append((name, version, kwargs))
        response = self.responses[kwargs["procedure_id"]]
        if isinstance(response, list):
            value = response.pop(0)
        else:
            value = response
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value()
        if asyncio.iscoroutine(value):
            return await value
        return value


def effect(requests: list[ProcedureRequest]) -> PerformProcedureBatch:
    return PerformProcedureBatch(
        task_id="task-1",
        round_id="round-1",
        generation=4,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "agent_id": "agent.reader",
            "requests": requests,
        },
    )


@pytest.mark.asyncio
async def test_executor_runs_only_ordinary_batch_and_returns_event() -> None:
    ordinary, controls = split_procedure_requests(
        [
            ProcedureRequest(procedure_id="builtin.search", arguments={"q": "x"}),
            ProcedureRequest(procedure_id="core.checkpoint"),
        ]
    )
    api = FakeAPI({"builtin.search": {"success": True, "data": {"ok": True}, "error": None, "metadata": {}}})
    executor = ProcedureExecutor(catalog(definition("builtin.search")), api=api)

    event = await executor.invoke_many(effect(ordinary))

    assert isinstance(event, ProcedureBatchCompleted)
    assert [item.procedure_id for item in event.results] == ["builtin.search"]
    assert controls.checkpoint
    assert len(api.calls) == 1
    assert api.calls[0][2]["scoped_metadata"] == {
        "task_id": "task-1",
        "round_id": "round-1",
        "branch_id": "branch-1",
        "turn_id": "turn-1",
        "agent_id": "agent.reader",
    }


@pytest.mark.asyncio
async def test_executor_keeps_result_order_when_calls_complete_out_of_order() -> None:
    async def slow() -> dict[str, Any]:
        await asyncio.sleep(0.02)
        return {"success": True, "data": {"index": 0}, "error": None, "metadata": {}}

    async def fast() -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"success": True, "data": {"index": 1}, "error": None, "metadata": {}}

    api = FakeAPI({"builtin.first": slow, "builtin.second": fast})
    executor = ProcedureExecutor(catalog(definition("builtin.first"), definition("builtin.second")), api=api)
    event = await executor.invoke_many(
        effect(
            [
                ProcedureRequest(procedure_id="builtin.first"),
                ProcedureRequest(procedure_id="builtin.second"),
            ]
        )
    )

    assert [item.procedure_id for item in event.results] == ["builtin.first", "builtin.second"]
    assert [item.result.data["index"] for item in event.results] == [0, 1]


@pytest.mark.asyncio
async def test_idempotent_retry_reuses_request_id_only_for_explicit_retryable_error() -> None:
    api = FakeAPI(
        {
            "builtin.retry": [
                {"success": False, "data": None, "error": {"code": "busy", "retryable": True}, "metadata": {}},
                {"success": True, "data": {"ok": True}, "error": None, "metadata": {}},
            ]
        }
    )
    executor = ProcedureExecutor(catalog(definition("builtin.retry", idempotent=True)), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.retry")]))

    assert len(api.calls) == 2
    assert api.calls[0][2]["request_id"] == api.calls[1][2]["request_id"]
    assert event.results[0].attempts == 2
    assert event.results[0].result.success


@pytest.mark.asyncio
async def test_non_idempotent_timeout_is_not_retried() -> None:
    async def hangs() -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"success": True, "data": {}, "error": None, "metadata": {}}

    api = FakeAPI({"builtin.write": hangs})
    executor = ProcedureExecutor(catalog(definition("builtin.write", timeout_seconds=0.01)), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.write")]))

    assert len(api.calls) == 1
    assert event.results[0].result.error["code"] == "procedure_timeout"


@pytest.mark.asyncio
async def test_timeout_seconds_zero_disables_hard_wait() -> None:
    async def slow() -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"success": True, "data": {"ok": True}, "error": None, "metadata": {}}

    api = FakeAPI({"builtin.slow": slow})
    executor = ProcedureExecutor(catalog(definition("builtin.slow", timeout_seconds=0.0)), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.slow")]))

    assert len(api.calls) == 1
    assert event.results[0].result.success
    assert event.results[0].result.error is None


@pytest.mark.asyncio
async def test_invalid_provider_result_is_structured_without_raw_payload() -> None:
    api = FakeAPI({"builtin.bad": {"success": "yes", "secret": "do-not-persist"}})
    executor = ProcedureExecutor(catalog(definition("builtin.bad")), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.bad")]))

    result = event.results[0].result
    assert not result.success
    assert result.error["code"] == "provider_contract_invalid"
    assert "do-not-persist" not in repr(result)


@pytest.mark.asyncio
async def test_batch_event_round_trip_preserves_order_and_frozen_result_metadata() -> None:
    api = FakeAPI({"builtin.echo": {"success": True, "data": {"ok": True}, "error": None, "metadata": {}}})
    executor = ProcedureExecutor(catalog(definition("builtin.echo")), api=api)
    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.echo")]))

    decoded = event_from_json(event_to_json(event))

    assert decoded == event
    with pytest.raises(TypeError):
        decoded.results[0].result.metadata["mutated"] = True  # type: ignore[index]


@pytest.mark.asyncio
async def test_provider_payload_and_error_sensitive_fields_are_removed_before_event_json() -> None:
    api = FakeAPI(
        {
            "builtin.safe": {
                "success": True,
                "data": {
                    "answer": "业务字段保留",
                    "nested": {
                        "normal": "保留",
                        "reasoning": "secret-reasoning",
                        "raw_payload": {"token": "secret-raw"},
                        "provenance": {"secret": "secret-provenance"},
                        "payload": {"secret": "secret-payload"},
                    },
                },
                "error": None,
                "metadata": {},
            },
            "builtin.safe_error": {
                "success": False,
                "data": None,
                "error": {
                    "code": "provider_failed",
                    "message": "业务错误",
                    "reasoning": "secret-error-reasoning",
                    "raw_payload": {"token": "secret-error-raw"},
                    "provenance": {"secret": "secret-error-provenance"},
                    "payload": {"secret": "secret-error-payload"},
                },
                "metadata": {},
            },
        }
    )
    executor = ProcedureExecutor(catalog(definition("builtin.safe"), definition("builtin.safe_error")), api=api)

    event = await executor.invoke_many(
        effect([ProcedureRequest(procedure_id="builtin.safe"), ProcedureRequest(procedure_id="builtin.safe_error")])
    )
    encoded = event_to_json(event)

    assert "业务字段保留" in encoded and "保留" in encoded
    for secret in (
        "secret-reasoning",
        "secret-raw",
        "secret-provenance",
        "secret-payload",
        "secret-error-reasoning",
        "secret-error-raw",
        "secret-error-provenance",
        "secret-error-payload",
    ):
        assert secret not in encoded


@pytest.mark.asyncio
async def test_raw_result_arguments_transcript_and_messages_are_not_persisted() -> None:
    api = FakeAPI(
        {
            "builtin.transcript_safe": {
                "success": True,
                "data": {
                    "answer": "普通业务字段",
                    "raw_result": {"secret": "secret-raw-result"},
                    "raw_arguments": {"secret": "secret-raw-arguments"},
                    "transcript": [{"content": "secret-transcript"}],
                    "messages": [{"content": "secret-messages"}],
                },
                "error": None,
                "metadata": {},
            }
        }
    )
    executor = ProcedureExecutor(catalog(definition("builtin.transcript_safe")), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.transcript_safe")]))
    encoded = event_to_json(event)

    assert "普通业务字段" in encoded
    for secret in (
        "secret-raw-result",
        "secret-raw-arguments",
        "secret-transcript",
        "secret-messages",
    ):
        assert secret not in encoded


@pytest.mark.asyncio
async def test_catalog_provenance_overwrites_provider_spoofed_result_metadata() -> None:
    api = FakeAPI(
        {
            "builtin.identity": {
                "success": True,
                "data": {"ok": True},
                "error": None,
                "metadata": {
                    "provider_plugin_id": "evil.provider",
                    "api_name": "evil.invoke_procedure",
                    "api_version": "999",
                    "request_id": "evil-request",
                },
            }
        }
    )
    executor = ProcedureExecutor(catalog(definition("builtin.identity")), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.identity")]))
    item = event.results[0]

    assert item.result.metadata["provider_plugin_id"] == "provider.tools"
    assert item.result.metadata["api_name"] == "provider.tools.invoke_procedure"
    assert item.result.metadata["api_version"] == "1"
    assert item.result.metadata["request_id"] == item.request_id


@pytest.mark.asyncio
async def test_string_retryable_exception_is_not_treated_as_explicit_retry() -> None:
    class StringRetryableError(RuntimeError):
        retryable = "true"

    api = FakeAPI(
        {
            "builtin.once": [
                StringRetryableError(),
                {"success": True, "data": {}, "error": None, "metadata": {}},
            ]
        }
    )
    executor = ProcedureExecutor(catalog(definition("builtin.once", idempotent=True)), api=api)

    event = await executor.invoke_many(effect([ProcedureRequest(procedure_id="builtin.once")]))

    assert len(api.calls) == 1
    assert event.results[0].attempts == 1
    assert event.results[0].result.error["code"] == "provider_call_failed"
