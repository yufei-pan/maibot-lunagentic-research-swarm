"""可选 provider 兼容矩阵：fetch-url / 文件仓库 / 无效第三方批次。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.commands import _recommended_fetch_status
from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import (
    ProcedureCatalogEntry,
    ProcedureCatalogSnapshot,
    ProcedureRegistry,
)
from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch
from test_lifecycle import DiscoveryFactory, agent_payload, build_container, config_with


FETCH_PROVIDER = "com.0-hz.fetch-url"
FETCH_PROCEDURE = "fetch_url.fetch"
FILE_PROVIDER = "com.0-hz.file-depot"


def fetch_procedure_payload() -> dict[str, Any]:
    return {
        "procedure_id": FETCH_PROCEDURE,
        "version": "1",
        "display_name": "抓取网页",
        "description": "抓取网页全文",
        "arguments_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        "result_schema": {"type": "object"},
        "idempotent": True,
        "timeout_seconds": 30,
        "external_cost_kind": "none",
        "enabled": True,
    }


def procedure_definition(procedure_id: str = FETCH_PROCEDURE) -> ProcedureDefinition:
    return ProcedureDefinition.model_validate(fetch_procedure_payload() if procedure_id == FETCH_PROCEDURE else {
        "procedure_id": procedure_id,
        "version": "1",
        "display_name": procedure_id,
        "description": "测试",
        "arguments_schema": {"type": "object"},
        "result_schema": {"type": "object"},
    })


@pytest.mark.asyncio
async def test_missing_fetch_url_loads_and_reports_recommended_missing(plugin_module, tmp_path: Path) -> None:
    container, _, _, _, _ = build_container(plugin_module, tmp_path)

    await container.start()
    try:
        health = container.health()
        assert health["sqlite"]["status"] == "healthy"
        assert health["root_agent"]["status"] == "healthy"
        assert health["root_agent"]["agent_id"] == "builtin.quick_thinker"

        recommended = health.get("recommended_fetch") or _recommended_fetch_status(container)
        assert recommended["status"] == "recommended_missing"
        assert recommended["detail"] == FETCH_PROCEDURE

        snapshot = container.procedure_registry.snapshot({})
        assert snapshot.get(FETCH_PROCEDURE) is None
        # core + builtin 仍可用
        assert snapshot.get("builtin.web_search") is not None or any(
            entry.definition.procedure_id.startswith("builtin.") for entry in snapshot.entries
        )
    finally:
        await container.close()


@pytest.mark.asyncio
async def test_valid_fetch_provider_appears_in_catalog_and_healthy_fetch(plugin_module, tmp_path: Path) -> None:
    def providers(agents: Any, procedures: Any) -> None:
        procedures.replace_provider(FETCH_PROVIDER, [fetch_procedure_payload()])

    container, _, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        discovery_factory=DiscoveryFactory([], providers),
    )

    await container.start()
    try:
        snapshot = container.procedure_registry.snapshot({})
        assert snapshot.get(FETCH_PROCEDURE) is not None
        recommended = container.health().get("recommended_fetch") or _recommended_fetch_status(container)
        assert recommended["status"] == "healthy"
        assert recommended["detail"] == FETCH_PROCEDURE
    finally:
        await container.close()


@pytest.mark.asyncio
async def test_removed_fetch_provider_blocks_new_calls_but_inflight_completes() -> None:
    registry = ProcedureRegistry()
    registry.replace_provider(FETCH_PROVIDER, [fetch_procedure_payload()])
    frozen = registry.snapshot({})
    entry = frozen.get(FETCH_PROCEDURE)
    assert entry is not None

    gate = asyncio.Event()
    gate.clear()
    completed: list[dict[str, Any]] = []

    class SlowAPI:
        async def call(self, name: str, *, version: str = "", **kwargs: Any) -> Any:
            del name, version
            await gate.wait()
            payload = {
                "success": True,
                "data": {"body": "ok"},
                "error": None,
                "metadata": {},
            }
            completed.append(payload)
            return payload

    executor = ProcedureExecutor(frozen, api=SlowAPI())
    effect = PerformProcedureBatch(
        task_id="task-1",
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
    inflight = asyncio.create_task(executor.invoke_many(effect))

    # 卸载 live provider：新目录不再含 fetch
    registry.remove_provider(FETCH_PROVIDER)
    assert not registry.is_live(FETCH_PROCEDURE)
    empty = registry.snapshot({})
    assert empty.get(FETCH_PROCEDURE) is None
    new_executor = ProcedureExecutor(empty, api=SlowAPI())
    missing = await new_executor.invoke_many(
        PerformProcedureBatch(
            task_id="task-2",
            round_id="round-2",
            generation=0,
            payload={
                "branch_id": "branch-2",
                "call_id": "call-2",
                "turn_id": "turn-2",
                "agent_id": "builtin.researcher",
                "requests": [{"procedure_id": FETCH_PROCEDURE, "arguments": {"url": "https://example.com"}}],
            },
        )
    )
    assert missing.results[0].success is False
    assert missing.results[0].result.error["code"] == "procedure_unavailable"

    gate.set()
    done = await inflight
    assert done.results[0].success is True
    assert done.results[0].result.data == {"body": "ok"}
    assert completed


@pytest.mark.asyncio
async def test_missing_file_depot_does_not_affect_core_or_builtin(plugin_module, tmp_path: Path) -> None:
    container, _, _, _, _ = build_container(plugin_module, tmp_path)
    await container.start()
    try:
        assert FILE_PROVIDER not in container.procedure_registry.provider_ids
        snapshot = container.procedure_registry.snapshot({})
        # core 由 executor 本地处理，不进 registry；builtin 必须齐全
        bundled = {entry.definition.procedure_id for entry in snapshot.entries}
        assert "builtin.calculate" in bundled
        assert "builtin.web_search" in bundled
        assert "builtin.past_cases" in bundled
        health = container.health()
        assert health["sqlite"]["status"] == "healthy"
        assert health["root_agent"]["status"] == "healthy"
    finally:
        await container.close()


@pytest.mark.asyncio
async def test_invalid_third_party_batches_are_visible_in_health(plugin_module, tmp_path: Path) -> None:
    def providers(agents: Any, procedures: Any) -> None:
        agents.replace_provider("provider.good", [agent_payload("good.root")])
        agents.reject_provider("provider.bad-agents", ["字面 agent 批次无效"])
        procedures.reject_provider("provider.bad-procs", ["字面 procedure 批次无效"])

    container, _, _, _, _ = build_container(
        plugin_module,
        tmp_path,
        config=config_with(plugin={"root_agent": "good.root"}),
        discovery_factory=DiscoveryFactory([], providers),
    )
    await container.start()
    try:
        health = container.health()
        assert health["extension_providers"]["agents"]["provider.bad-agents"] == {
            "status": "invalid",
            "code": "extension_provider_invalid",
        }
        assert health["extension_providers"]["procedures"]["provider.bad-procs"] == {
            "status": "invalid",
            "code": "extension_provider_invalid",
        }
        assert health["extension_providers"]["agents"]["provider.good"]["status"] == "healthy"
        # 无效批次不得阻断加载
        assert health["sqlite"]["status"] == "healthy"
    finally:
        await container.close()


def test_catalog_entry_helper_keeps_fetch_identity() -> None:
    """sanity：冻结目录条目保留 provider API 身份，供在途调用路由。"""

    definition = procedure_definition()
    entry = ProcedureCatalogEntry(
        definition=definition,
        provider_plugin_id=FETCH_PROVIDER,
        api_name=f"{FETCH_PROVIDER}.invoke_procedure",
        api_version="1",
        fingerprint="fp",
    )
    snapshot = ProcedureCatalogSnapshot([entry])
    assert snapshot.get(FETCH_PROCEDURE) is entry
