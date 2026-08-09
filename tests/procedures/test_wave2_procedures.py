"""Wave-2 procedure fixes: B-C1 + B-I1..B-I5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from lunagentic_research_swarm.config import WebSearchSection
from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.extensions.validation import validate_procedure_batch
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.bundled.provenance import normalize_url
from lunagentic_research_swarm.procedures.bundled.web_search import WebSearchService, web_search_procedure_definitions
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot, ProcedureRegistry
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import NotifyToolWaiter, PerformProcedureBatch, RuntimeState, reduce_event


def definition(procedure_id: str) -> ProcedureDefinition:
    return ProcedureDefinition.model_validate(
        {
            "procedure_id": procedure_id,
            "version": "1",
            "display_name": procedure_id,
            "description": "测试",
            "arguments_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "idempotent": True,
            "timeout_seconds": 30.0,
        }
    )


def catalog(*definitions: ProcedureDefinition) -> ProcedureCatalogSnapshot:
    return ProcedureCatalogSnapshot(
        [
            ProcedureCatalogEntry(
                definition=item,
                provider_plugin_id="builtin",
                api_name="builtin.invoke_procedure",
                api_version="1",
                fingerprint=f"fp:{item.procedure_id}",
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
        return self.responses[kwargs["procedure_id"]]


def _effect(
    requests: list[ProcedureRequest],
    *,
    messages: tuple[dict[str, Any], ...] = ({"role": "user", "content": "formalized"},),
    allowed_procedures: tuple[str, ...] | None = None,
    agent_id: str = "builtin.memory_researcher",
) -> PerformProcedureBatch:
    payload: dict[str, Any] = {
        "branch_id": "branch-1",
        "call_id": "call-1",
        "turn_id": "turn-1",
        "agent_id": agent_id,
        "requests": requests,
        "messages": messages,
        "delegations": ({"agent_id": "builtin.child", "task": "继续调查", "credits": 1.0},),
        "credits_after": 2.0,
    }
    if allowed_procedures is not None:
        payload["allowed_procedures"] = allowed_procedures
    return PerformProcedureBatch(task_id="task-1", round_id="round-1", generation=0, payload=payload)


@pytest.mark.asyncio
async def test_bc1_procedure_results_fold_into_parent_and_child_messages() -> None:
    api = FakeAPI(
        {
            "builtin.web_search": {
                "success": True,
                "data": {
                    "engine": "duckduckgo",
                    "query": "lunar research",
                    "results": [{"url": "https://example.com", "title": "Hit", "snippet": "snippet"}],
                },
                "error": None,
                "metadata": {},
            }
        }
    )
    executor = ProcedureExecutor(catalog(definition("builtin.web_search")), api=api)
    completed = await executor.invoke_many(
        _effect([ProcedureRequest(procedure_id="builtin.web_search", arguments={"engine": "duckduckgo", "query": "q"})])
    )

    assert any(
        "【Procedure 结果 · builtin.web_search】" in str(item.get("content", ""))
        for item in completed.parent_messages
    )
    assert any("lunar research" in str(item.get("content", "")) for item in completed.parent_messages)
    # 纯审计字段不进入被后代继承的消息（只留在事件/存储里）。
    assert "provider_plugin_id" not in repr(completed.parent_messages)
    assert "raw_payload" not in repr(completed.parent_messages)

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, active_round_id="round-1", active_leaves={"branch-1": 2.0}),
        ProcedureBatchCompleted(
            "evt",
            "task-1",
            "round-1",
            0,
            branch_id="branch-1",
            credits_after=2.0,
            results=completed.results,
            parent_messages=completed.parent_messages,
            delegations=({"agent_id": "builtin.child", "task": "继续调查", "credits": 1.0},),
            live_agent_ids=("builtin.child",),
        ),
    )
    children = [
        item
        for item in transition.effects
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]
    assert len(children) == 1
    child_blob = repr(children[0].payload["messages"])
    assert "lunar research" in child_blob
    assert "Procedure 结果" in child_blob


@pytest.mark.asyncio
async def test_bi1_allowed_procedures_rejected_at_invoke() -> None:
    api = FakeAPI(
        {
            "builtin.web_search": {"success": True, "data": {"ok": True}, "error": None, "metadata": {}},
            "builtin.knowledge_search": {"success": True, "data": {"hits": []}, "error": None, "metadata": {}},
        }
    )
    executor = ProcedureExecutor(
        catalog(definition("builtin.web_search"), definition("builtin.knowledge_search")),
        api=api,
    )
    event = await executor.invoke_many(
        _effect(
            [ProcedureRequest(procedure_id="builtin.web_search", arguments={"engine": "duckduckgo", "query": "x"})],
            allowed_procedures=("builtin.knowledge_search",),
        )
    )

    assert not event.results[0].result.success
    assert event.results[0].result.error["code"] == "procedure_not_allowed"
    assert api.calls == []


@pytest.mark.asyncio
async def test_bi5_requesting_agent_id_preserved_in_result_metadata() -> None:
    api = FakeAPI({"builtin.knowledge_search": {"success": True, "data": {"hits": [1]}, "error": None, "metadata": {}}})
    executor = ProcedureExecutor(catalog(definition("builtin.knowledge_search")), api=api)
    event = await executor.invoke_many(
        _effect(
            [ProcedureRequest(procedure_id="builtin.knowledge_search", arguments={"query": "q"})],
            agent_id="builtin.memory_researcher",
        )
    )

    assert event.results[0].result.metadata["agent_id"] == "builtin.memory_researcher"

    transition = reduce_event(
        RuntimeState("task-1", TaskStatus.RUNNING, active_round_id="round-1", active_leaves={"branch-1": 2.0}),
        ProcedureBatchCompleted(
            "evt",
            "task-1",
            "round-1",
            0,
            branch_id="branch-1",
            credits_after=2.0,
            results=event.results,
            parent_messages=event.parent_messages,
        ),
    )
    audit = [cmd for cmd in transition.commands if getattr(cmd, "kind", None) == "insert_procedure_call"]
    assert audit
    assert audit[0].values["agent_id"] == "builtin.memory_researcher"


def test_bi2_web_search_omitted_when_no_engines_available() -> None:
    section = WebSearchSection(enabled_engines=[])
    service = WebSearchService(section, object())
    assert service.available_engines == ()
    assert web_search_procedure_definitions(service) == []

    section2 = WebSearchSection(enabled_engines=["searxng", "tavily", "you"])
    service2 = WebSearchService(section2, object())
    assert service2.available_engines == ()
    assert web_search_procedure_definitions(service2) == []


def test_bi3_third_party_cannot_register_core_procedure_ids() -> None:
    with pytest.raises(ValidationError, match="core"):
        ProcedureDefinition.model_validate(
            {
                "procedure_id": "core.exfiltrate",
                "version": "1",
                "display_name": "evil",
                "description": "evil",
                "arguments_schema": {"type": "object"},
                "result_schema": {"type": "object"},
            }
        )

    with pytest.raises(ValueError, match="core"):
        validate_procedure_batch(
            "com.evil.core",
            [
                {
                    "procedure_id": "core.exfiltrate",
                    "version": "1",
                    "display_name": "evil",
                    "description": "evil",
                    "arguments_schema": {"type": "object"},
                    "result_schema": {"type": "object"},
                }
            ],
        )

    registry = ProcedureRegistry()
    with pytest.raises(ValueError, match="core"):
        registry.replace_provider(
            "com.evil.core",
            [
                {
                    "procedure_id": "core.exfiltrate",
                    "version": "1",
                    "display_name": "evil",
                    "description": "evil",
                    "arguments_schema": {"type": "object"},
                    "result_schema": {"type": "object"},
                }
            ],
        )


def test_bi4_normalize_url_strips_userinfo() -> None:
    assert normalize_url("https://user:secret@example.com/a?b=2&a=1#frag") == "https://example.com/a?b=2&a=1"
    assert "secret" not in normalize_url("https://user:secret@example.com/path")
    assert normalize_url("https://onlyuser@example.com/x") == "https://example.com/x"
