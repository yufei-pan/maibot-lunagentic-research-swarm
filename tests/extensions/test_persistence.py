from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningStatus
from lunagentic_research_swarm.services import LRSServiceContainer
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


def _api_info(*, public: bool = True) -> dict[str, Any]:
    metadata = {
        "lunagentic_extension": "agents",
        "lunagentic_contract": "1",
    }
    return {
        "name": "describe_agents",
        "full_name": "provider.agents.describe_agents",
        "plugin_id": "provider.agents",
        "version": "1",
        "public": public,
        "metadata": metadata,
    }


def _agent_envelope() -> dict[str, Any]:
    return {
        "contract_version": "1",
        "agents": [
            {
                "agent_id": "agents.root",
                "version": "1",
                "display_name": "测试根智能体",
                "description": "测试扩展持久化",
                "character_prompt": "只执行测试任务。",
                "model_selector": "task:utils",
                "can_be_root": True,
            }
        ],
    }


class MutableAPI:
    def __init__(self) -> None:
        self.infos = [_api_info()]
        self.response: Any = _agent_envelope()
        self.list_count = 0

    async def list(self) -> list[dict[str, Any]]:
        self.list_count += 1
        return list(self.infos)

    async def call(self, api_name: str, *, version: str = "", **kwargs: Any) -> Any:
        assert api_name == "provider.agents.describe_agents"
        assert version == "1"
        assert kwargs == {}
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class CompatiblePinning:
    def check_compatibility(self) -> PhysicalPinningStatus:
        return PhysicalPinningStatus(True)


class ObservingStore(SQLiteStateStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.fingerprint_persisted = asyncio.Event()

    async def transact(self, commands: tuple[StoreCommand, ...] | list[StoreCommand]) -> None:
        await super().transact(commands)
        if any(command.kind == "insert_extension_fingerprint" for command in commands):
            self.fingerprint_persisted.set()


def _build_container(tmp_path: Path) -> tuple[LRSServiceContainer, MutableAPI, ObservingStore, Path]:
    database_path = tmp_path / "data" / "lrs-state.sqlite3"
    api = MutableAPI()
    context = SimpleNamespace(
        api=api,
        paths=SimpleNamespace(data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"),
        logger=logging.getLogger("lrs-extension-persistence-test"),
    )
    store = ObservingStore(database_path)
    config = LRSConfig.model_validate(
        {
            "plugin": {"root_agent": "agents.root"},
            "extensions": {"refresh_interval_seconds": 10},
        }
    )
    container = LRSServiceContainer(
        context,
        config,
        store_factory=lambda path: store,
        host_snapshot_loader=lambda: {"models": [], "model_task_config": {}},
        physical_pinning=CompatiblePinning(),
    )
    return container, api, store, database_path


def _stored_events(database_path: Path) -> list[tuple[str, str, str, str | None]]:
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            """
            SELECT provider_plugin_id, extension_kind, availability, error_json
            FROM extension_fingerprints
            ORDER BY created_at, rowid
            """
        ).fetchall()


@pytest.mark.asyncio
async def test_initial_success_and_public_invalid_descriptor_are_persisted(tmp_path: Path) -> None:
    container, api, _, database_path = _build_container(tmp_path)
    await container.start()
    api.infos = [_api_info(public=False)]

    result = await container.refresh_extensions(reason="provider_request")
    await container.close()

    events = _stored_events(database_path)
    assert result["status"] == "degraded"
    assert ("provider.agents", "agents", "available", None) in events
    provider_invalid = next(
        event for event in events if event[:3] == ("provider.agents", "agents", "invalid")
    )
    assert json.loads(provider_invalid[3])["errors"] == ["descriptor metadata 与契约不完整匹配"]
    descriptor_invalid = next(
        event for event in events if event[:3] == ("descriptor:0", "discovery", "invalid")
    )
    assert "descriptor metadata" in json.loads(descriptor_invalid[3])["errors"][0]


@pytest.mark.asyncio
async def test_public_removal_and_periodic_recovery_share_authoritative_persistence(tmp_path: Path) -> None:
    container, api, store, database_path = _build_container(tmp_path)
    await container.start()
    api.infos = []
    await container.refresh_extensions(reason="provider_request")
    api.infos = [_api_info()]
    store.fingerprint_persisted = asyncio.Event()
    scans_before = api.list_count

    container._discovery.request_refresh()
    await asyncio.wait_for(store.fingerprint_persisted.wait(), timeout=2)
    assert api.list_count == scans_before + 1
    await container.close()

    provider_events = [
        availability
        for provider, kind, availability, _ in _stored_events(database_path)
        if (provider, kind) == ("provider.agents", "agents")
    ]
    assert provider_events == ["available", "removed", "available"]


@pytest.mark.asyncio
async def test_initial_fingerprint_persistence_failure_propagates(tmp_path: Path) -> None:
    container, _, store, _ = _build_container(tmp_path)

    async def fail_transact(commands: tuple[StoreCommand, ...] | list[StoreCommand]) -> None:
        assert any(command.kind == "insert_extension_fingerprint" for command in commands)
        raise sqlite3.OperationalError("字面 extension audit 写失败")

    store.transact = fail_transact  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError, match="字面 extension audit 写失败"):
        await container.start()
    assert container._status["extension_fingerprint_store"] == {
        "status": "critical",
        "code": "extension_fingerprint_persistence_failed",
    }
    await container.close()
