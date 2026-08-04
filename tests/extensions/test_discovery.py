from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.extensions.contracts import ExtensionRefreshDelta, ExtensionRefreshEvent
from lunagentic_research_swarm.extensions.discovery import ExtensionDiscovery
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry


def api_info(
    plugin_id: str,
    kind: str,
    *,
    tagged: bool = True,
    name: str | None = None,
    version: str = "1",
    public: bool = True,
) -> dict[str, Any]:
    api_name = name or ("describe_agents" if kind == "agents" else "describe_procedures")
    metadata: dict[str, Any] = {
        "description": f"{kind} descriptor",
        "version": version,
        "public": public,
        "enabled": True,
        "handler_name": api_name,
        "dynamic": False,
    }
    if tagged:
        metadata.update({"lunagentic_extension": kind, "lunagentic_contract": "1"})
    return {
        "name": api_name,
        "full_name": f"{plugin_id}.{api_name}",
        "plugin_id": plugin_id,
        "description": f"{kind} descriptor",
        "version": version,
        "public": public,
        "enabled": True,
        "dynamic": False,
        "offline_reason": "",
        "metadata": metadata,
    }


def agent_payload(agent_id: str, *, can_be_root: bool = False) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "version": "1",
        "display_name": agent_id,
        "description": "测试智能体",
        "character_prompt": "按要求工作。",
        "model_selector": "task:utils",
        "can_be_root": can_be_root,
    }


def procedure_payload(procedure_id: str) -> dict[str, Any]:
    return {
        "procedure_id": procedure_id,
        "version": "1",
        "display_name": procedure_id,
        "description": "测试流程",
        "arguments_schema": {"type": "object"},
        "result_schema": {"type": "object"},
    }


class FakeAPI:
    """仅替代 Host RPC 边界；响应完整复现 APICapability 解包后的结构。"""

    def __init__(self, infos: list[dict[str, Any]], responses: dict[str, Any]) -> None:
        self.infos = infos
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.list_count = 0
        self.call_started: asyncio.Event | None = None
        self.release_call: asyncio.Event | None = None
        self.list_started: asyncio.Event | None = None
        self.release_list: asyncio.Event | None = None

    async def list(self, *, plugin_id: str = "") -> list[dict[str, Any]]:
        assert plugin_id == ""
        self.list_count += 1
        if self.list_started is not None:
            self.list_started.set()
        if self.release_list is not None:
            await self.release_list.wait()
        return list(self.infos)

    async def get(self, api_name: str, *, version: str = "") -> dict[str, Any] | None:
        for info in self.infos:
            if info["full_name"] == api_name and (not version or info["version"] == version):
                return dict(info)
        return None

    async def call(self, api_name: str, *, version: str = "", **kwargs: Any) -> Any:
        self.calls.append((api_name, version, kwargs))
        if self.call_started is not None:
            self.call_started.set()
        if self.release_call is not None:
            await self.release_call.wait()
        response = self.responses[api_name]
        if isinstance(response, Exception):
            raise response
        return response


class FakeContext:
    def __init__(self, api: FakeAPI) -> None:
        self.api = api


def test_refresh_delta_detaches_mutable_sequences() -> None:
    errors = ["字面错误"]
    events = [
        ExtensionRefreshEvent(
            provider_plugin_id="provider.agents",
            extension_kind="agents",
            availability="invalid",
            fingerprint="literal_fingerprint",
            errors=errors,  # type: ignore[arg-type]
            created_at=1.0,
        )
    ]
    delta = ExtensionRefreshDelta(events)  # type: ignore[arg-type]

    errors.append("外部修改")
    events.clear()

    assert delta.events[0].errors == ("字面错误",)


@pytest.mark.asyncio
async def test_discovery_filters_tags_uses_full_names_and_host_provider_identity() -> None:
    infos = [
        api_info("provider.agents", "agents"),
        api_info("provider.tools", "procedures"),
        api_info("ignored.untagged", "agents", tagged=False),
        api_info("ignored.private", "agents", public=False),
        api_info("ignored.v2", "agents", version="2"),
    ]
    api = FakeAPI(
        infos,
        {
            "provider.agents.describe_agents": {
                "contract_version": "1",
                "agents": [agent_payload("agents.reader"), agent_payload("agents.root", can_be_root=True)],
            },
            "provider.tools.describe_procedures": {
                "contract_version": "1",
                "procedures": [procedure_payload("tools.fetch")],
            },
        },
    )
    agents = AgentRegistry(root_agent="agents.root")
    procedures = ProcedureRegistry()
    discovery = ExtensionDiscovery(FakeContext(api), agents, procedures, refresh_interval_seconds=60)

    await discovery.refresh()

    assert api.calls == [
        ("provider.agents.describe_agents", "1", {}),
        ("provider.tools.describe_procedures", "1", {}),
    ]
    agent_entry = agents.snapshot({}).get("agents.reader")
    assert agent_entry.provider_plugin_id == "provider.agents"
    assert procedures.snapshot({}).get("tools.fetch").provider_plugin_id == "provider.tools"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_response",
    [
        None,
        {},
        {"contract_version": "2", "agents": []},
        {"contract_version": "1", "agents": {}},
        {"contract_version": "1", "agents": [], "provider_plugin_id": "spoof"},
        {"contract_version": "1", "agents": [agent_payload("agents.good"), {"agent_id": "bad"}]},
    ],
)
async def test_bad_envelope_or_definition_rejects_whole_provider_batch(bad_response: Any) -> None:
    info = api_info("provider.agents", "agents")
    api = FakeAPI(
        [info],
        {
            "provider.agents.describe_agents": {
                "contract_version": "1",
                "agents": [agent_payload("agents.root", can_be_root=True)],
            }
        },
    )
    agents = AgentRegistry(root_agent="agents.root")
    discovery = ExtensionDiscovery(FakeContext(api), agents, ProcedureRegistry(), refresh_interval_seconds=60)
    await discovery.refresh()
    assert agents.is_live("agents.root")

    api.responses["provider.agents.describe_agents"] = bad_response
    await discovery.refresh()

    assert not agents.is_live("agents.root")
    assert agents.health["provider.agents"].status == "invalid"
    assert agents.health["provider.agents"].errors
    assert "agents:provider.agents" in discovery.extension_fingerprints


@pytest.mark.asyncio
async def test_one_provider_failure_does_not_block_another_and_missing_provider_is_removed() -> None:
    good = api_info("provider.good", "agents")
    bad = api_info("provider.bad", "agents")
    api = FakeAPI(
        [good, bad],
        {
            "provider.good.describe_agents": {
                "contract_version": "1",
                "agents": [agent_payload("good.root", can_be_root=True)],
            },
            "provider.bad.describe_agents": RuntimeError("字面 provider 调用失败"),
        },
    )
    agents = AgentRegistry(root_agent="good.root")
    discovery = ExtensionDiscovery(FakeContext(api), agents, ProcedureRegistry(), refresh_interval_seconds=60)

    await discovery.refresh()
    assert agents.is_live("good.root")
    assert agents.health["provider.bad"].status == "invalid"

    api.infos = [bad]
    await discovery.refresh()
    assert not agents.is_live("good.root")
    assert agents.health["provider.good"].status == "removed"


@pytest.mark.asyncio
async def test_invalid_provider_becomes_removed_when_descriptor_disappears() -> None:
    bad = api_info("provider.bad", "agents")
    api = FakeAPI([bad], {"provider.bad.describe_agents": RuntimeError("字面失败")})
    agents = AgentRegistry(root_agent="unused.root")
    discovery = ExtensionDiscovery(FakeContext(api), agents, ProcedureRegistry(), refresh_interval_seconds=60)

    await discovery.refresh()
    assert agents.health["provider.bad"].status == "invalid"

    api.infos = []
    await discovery.refresh()
    assert agents.health["provider.bad"].status == "removed"


@pytest.mark.asyncio
async def test_request_refresh_coalesces_and_close_cleans_background_task() -> None:
    api = FakeAPI([], {})
    api.call_started = asyncio.Event()
    discovery = ExtensionDiscovery(
        FakeContext(api), AgentRegistry(root_agent="unused.root"), ProcedureRegistry(), refresh_interval_seconds=3600
    )
    discovery.start()

    discovery.request_refresh()
    discovery.request_refresh()
    discovery.request_refresh()
    for _ in range(50):
        if api.list_count:
            break
        await asyncio.sleep(0)
    assert api.list_count == 1


@pytest.mark.asyncio
async def test_overlapping_direct_refresh_calls_share_one_scan_and_result() -> None:
    info = api_info("provider.agents", "agents")
    api = FakeAPI(
        [info],
        {"provider.agents.describe_agents": RuntimeError("字面共享 provider 失败")},
    )
    api.list_started = asyncio.Event()
    api.release_list = asyncio.Event()
    agents = AgentRegistry(root_agent="agents.root")
    discovery = ExtensionDiscovery(FakeContext(api), agents, ProcedureRegistry(), refresh_interval_seconds=60)

    first = asyncio.create_task(discovery.refresh())
    await api.list_started.wait()
    second = asyncio.create_task(discovery.refresh())
    await asyncio.sleep(0)
    assert api.list_count == 1

    api.release_list.set()
    results = await asyncio.gather(first, second)

    assert results[0] == results[1]
    assert results[0].events[1].availability == "invalid"
    assert api.list_count == 1
    assert api.calls == [("provider.agents.describe_agents", "1", {})]
    assert agents.health["provider.agents"].status == "invalid"
    assert "字面共享 provider 失败" in agents.health["provider.agents"].errors[0]

    await discovery.close()
    assert discovery.closed
    assert discovery.background_task is None
    discovery.request_refresh()
    await asyncio.sleep(0)
    assert api.list_count == 1


@pytest.mark.asyncio
async def test_close_waits_for_active_refresh_and_prevents_followup_host_calls() -> None:
    info = api_info("provider.agents", "agents")
    api = FakeAPI(
        [info],
        {
            "provider.agents.describe_agents": {
                "contract_version": "1",
                "agents": [agent_payload("agents.root", can_be_root=True)],
            }
        },
    )
    api.list_started = asyncio.Event()
    api.release_list = asyncio.Event()
    discovery = ExtensionDiscovery(
        FakeContext(api), AgentRegistry(root_agent="agents.root"), ProcedureRegistry(), refresh_interval_seconds=60
    )

    refresh_task = asyncio.create_task(discovery.refresh())
    await api.list_started.wait()
    close_task = asyncio.create_task(discovery.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    api.release_list.set()
    await close_task
    await refresh_task
    assert api.calls == []


@pytest.mark.asyncio
async def test_close_during_first_provider_call_does_not_launch_second_provider_call() -> None:
    infos = [api_info("provider.a", "agents"), api_info("provider.b", "agents")]
    api = FakeAPI(
        infos,
        {
            "provider.a.describe_agents": {
                "contract_version": "1",
                "agents": [agent_payload("a.root", can_be_root=True)],
            },
            "provider.b.describe_agents": {
                "contract_version": "1",
                "agents": [agent_payload("b.root", can_be_root=True)],
            },
        },
    )
    api.call_started = asyncio.Event()
    api.release_call = asyncio.Event()
    discovery = ExtensionDiscovery(
        FakeContext(api), AgentRegistry(root_agent="a.root"), ProcedureRegistry(), refresh_interval_seconds=60
    )

    refresh_task = asyncio.create_task(discovery.refresh())
    await api.call_started.wait()
    close_task = asyncio.create_task(discovery.close())
    await asyncio.sleep(0)
    api.release_call.set()
    await close_task
    await refresh_task

    assert api.calls == [("provider.a.describe_agents", "1", {})]
