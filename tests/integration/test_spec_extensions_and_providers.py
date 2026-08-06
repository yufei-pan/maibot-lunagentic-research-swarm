"""Spec §14–16 / §25.4 — extension mid-task removal, missing optional providers, pinning smoke.

Builds on ``test_extension_removal`` / ``test_optional_providers`` / ``test_physical_pinning``
with discoverable ``test_spec_*`` names. Offline only (Fake Host modules for pinning).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.commands import _recommended_fetch_status
from lunagentic_research_swarm.config import WebSearchSection
from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningAdapter
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.bundled.web_search import (
    WebSearchService,
    web_search_procedure_definitions,
)
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformAgentCall,
    PerformBranchSummary,
    PerformProcedureBatch,
    RuntimeState,
    reduce_event,
)
from test_lifecycle import build_container

FETCH_PROCEDURE = "fetch_url.fetch"
WEB_SEARCH_PROCEDURE = "builtin.web_search"


def _procedure_event(*, delegations: tuple[dict[str, Any], ...], live_agent_ids: tuple[str, ...]):
    return ProcedureBatchCompleted(
        "procedure",
        "task",
        "round",
        0,
        branch_id="parent",
        call_id="call",
        result_id="result",
        results=(),
        report="evidence",
        delegations=delegations,
        credits_after=10.0,
        parent_messages=({"role": "assistant", "content": "parent"},),
        parent_depth=0,
        live_agent_ids=live_agent_ids,
        max_delegations_per_turn=8,
        max_branch_depth=8,
        max_agent_calls_per_task=32,
        agent_calls_started=1,
    )


def _install_host_fakes(monkeypatch: pytest.MonkeyPatch, orchestrator_type: type) -> None:
    modules = {
        "src": ModuleType("src"),
        "src.config": ModuleType("src.config"),
        "src.config.model_configs": ModuleType("src.config.model_configs"),
        "src.llm_models": ModuleType("src.llm_models"),
        "src.llm_models.utils_model": ModuleType("src.llm_models.utils_model"),
    }

    class TaskConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.model_list = kwargs["model_list"]

    modules["src.config.model_configs"].TaskConfig = TaskConfig  # type: ignore[attr-defined]
    modules["src.llm_models.utils_model"].LLMOrchestrator = orchestrator_type  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# §14.3 — extension / agent removal mid-task
# ---------------------------------------------------------------------------


def test_spec_14_3_removed_extension_edge_finalizes_while_valid_sibling_materializes() -> None:
    """§14.3 — mid-task removal: unavailable edge → branch summary; live sibling continues."""

    state = RuntimeState(
        "task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 10.0}
    )
    transition = reduce_event(
        state,
        _procedure_event(
            delegations=(
                {"agent_id": "extension.removed", "task": "removed work", "credits": 5.0},
                {"agent_id": "builtin.researcher", "task": "valid work", "credits": 5.0},
            ),
            live_agent_ids=("builtin.researcher",),
        ),
    )

    edges = [
        effect
        for effect in transition.effects
        if isinstance(effect, PerformBranchSummary) and effect.payload.get("reason") == "agent_unavailable"
    ]
    siblings = [
        effect
        for effect in transition.effects
        if isinstance(effect, NotifyToolWaiter) and effect.payload.get("agent_id") == "builtin.researcher"
    ]
    assert len(edges) == 1
    assert edges[0].payload["agent_id"] == "extension.removed"
    assert edges[0].payload["credits"] == 0.0
    assert len(siblings) == 1
    assert siblings[0].payload.get("action") == "materialize_child"
    assert siblings[0].payload["credits"] == pytest.approx(5.0)


def test_spec_14_3_registry_provider_removal_blocks_new_edge_and_retries_parent() -> None:
    """§14.3 — remove_provider drops live agents; sole in-flight edge → parent retry, no invent."""

    registry = AgentRegistry(root_agent="extension.root")
    registry.replace_provider(
        "provider.extension",
        [
            {
                "agent_id": "extension.root",
                "version": "1",
                "display_name": "root",
                "description": "root",
                "character_prompt": "root",
                "model_selector": "task:utils",
                "can_be_root": True,
            },
            {
                "agent_id": "extension.sibling",
                "version": "1",
                "display_name": "sibling",
                "description": "sibling",
                "character_prompt": "sibling",
                "model_selector": "task:utils",
            },
        ],
    )
    assert registry.is_live("extension.sibling")
    registry.remove_provider("provider.extension")
    assert not registry.is_live("extension.sibling")

    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 1.0}),
        _procedure_event(
            delegations=({"agent_id": "extension.sibling", "task": "in-flight", "credits": 1.0},),
            live_agent_ids=(),
        ),
    )
    assert not [effect for effect in transition.effects if isinstance(effect, NotifyToolWaiter)]
    retries = [effect for effect in transition.effects if isinstance(effect, PerformAgentCall)]
    assert len(retries) == 1
    assert retries[0].payload["branch_id"] == "parent"
    notice = retries[0].payload["appended_messages"][0]["content"]
    assert "extension.sibling" in notice and "agent_unavailable" in notice
    assert transition.next_state.active_leaves == {"parent": 10.0}
    assert transition.next_state.credit_pool == 0.0
    assert transition.next_state.agent_calls_started == 2


# ---------------------------------------------------------------------------
# §15.4 / §15.5 — missing optional fetch / search → visible failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spec_15_5_missing_fetch_provider_visible_not_silent_success(
    plugin_module, tmp_path: Path
) -> None:
    """§15.5 — missing fetch-url: health recommended_missing; invoke → procedure_unavailable."""

    container, _, _, _, _ = build_container(plugin_module, tmp_path)
    await container.start()
    try:
        health = container.health()
        assert health["sqlite"]["status"] == "healthy"
        recommended = health.get("recommended_fetch") or _recommended_fetch_status(container)
        assert recommended["status"] == "recommended_missing"
        assert recommended.get("code") == "fetch_url_missing"
        assert recommended["detail"] == FETCH_PROCEDURE
        assert container.procedure_registry.snapshot({}).get(FETCH_PROCEDURE) is None
        assert not container.procedure_registry.is_live(FETCH_PROCEDURE)
    finally:
        await container.close()

    empty = ProcedureRegistry()
    assert empty.snapshot({}).get(FETCH_PROCEDURE) is None
    executor = ProcedureExecutor(empty.snapshot({}), api=object())
    missing = await executor.invoke_many(
        PerformProcedureBatch(
            task_id="task-fetch-miss",
            round_id="round-1",
            generation=0,
            payload={
                "branch_id": "branch-1",
                "call_id": "call-1",
                "turn_id": "turn-1",
                "agent_id": "builtin.researcher",
                "requests": [{"procedure_id": FETCH_PROCEDURE, "arguments": {"url": "https://example.com"}}],
            },
        )
    )
    assert missing.results[0].success is False
    assert missing.results[0].result.error["code"] == "procedure_unavailable"
    # No silent empty success payload.
    assert missing.results[0].result.data in (None, {})


@pytest.mark.asyncio
async def test_spec_15_4_missing_search_engines_omits_procedure_invoke_fails_visibly() -> None:
    """§15.4 — no configured engines → web_search omitted; catalog invoke fails visibly."""

    section = WebSearchSection(enabled_engines=[])
    service = WebSearchService(section, object())
    assert service.available_engines == ()
    assert web_search_procedure_definitions(service) == []

    # Misconfigured paid engines only (no duckduckgo) also omit the procedure.
    section2 = WebSearchSection(enabled_engines=["searxng", "tavily", "you"])
    service2 = WebSearchService(section2, object())
    assert service2.available_engines == ()
    assert web_search_procedure_definitions(service2) == []

    executor = ProcedureExecutor(ProcedureRegistry().snapshot({}), api=object())
    missing = await executor.invoke_many(
        PerformProcedureBatch(
            task_id="task-search-miss",
            round_id="round-1",
            generation=0,
            payload={
                "branch_id": "branch-1",
                "call_id": "call-1",
                "turn_id": "turn-1",
                "agent_id": "builtin.researcher",
                "requests": [
                    {
                        "procedure_id": WEB_SEARCH_PROCEDURE,
                        "arguments": {"engine": "duckduckgo", "query": "lrs"},
                    }
                ],
            },
        )
    )
    assert missing.results[0].success is False
    assert missing.results[0].result.error["code"] == "procedure_unavailable"
    assert missing.results[0].result.data in (None, {})


# ---------------------------------------------------------------------------
# §16.3 — physical pinning health / explicit reject smoke (no real Host)
# ---------------------------------------------------------------------------


def test_spec_16_3_physical_pinning_compatibility_smoke_without_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§16.3 — capability check reports healthy vs unsupported via faked Host modules."""

    class CompatibleOrchestrator:
        def __init__(self, task_name: str, request_type: str = "", session_id: str = "") -> None:
            pass

        async def generate_response_async(
            self,
            prompt: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            return None

        async def generate_response_with_message_async(
            self,
            message_factory: Any,
            temperature: float | None = None,
            max_tokens: int | None = None,
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            return None

    _install_host_fakes(monkeypatch, CompatibleOrchestrator)
    ok = PhysicalPinningAdapter().check_compatibility()
    assert ok.available is True
    assert ok.error_code is None

    class MissingMessageRoute(CompatibleOrchestrator):
        generate_response_with_message_async = None  # type: ignore[assignment]

    _install_host_fakes(monkeypatch, MissingMessageRoute)
    bad = PhysicalPinningAdapter().check_compatibility()
    assert bad.available is False
    assert bad.error_code == "physical_pinning_unsupported"


@pytest.mark.asyncio
async def test_spec_16_3_physical_pinning_incompatible_rejects_without_alternate_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§16.3 — signature mismatch → explicit error; never attempts an alternate model route."""

    class IncompatibleOrchestrator:
        def __init__(self, renamed_task: str) -> None:
            raise AssertionError("签名检查后不应实例化")

        async def generate_response_async(self, renamed_prompt: str) -> Any:
            raise AssertionError("签名检查后不应调用")

    _install_host_fakes(monkeypatch, IncompatibleOrchestrator)
    result = await PhysicalPinningAdapter().generate(
        physical_name="physical",
        prompt="hello",
        tools=None,
        temperature=None,
        max_tokens=None,
    )
    assert result["success"] is False
    assert result["error"]["code"] == "physical_pinning_unsupported"
