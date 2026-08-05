"""registry snapshot → ProcedureExecutor → 本地 builtin memory handler 集成路径。"""

from __future__ import annotations

from typing import Any

import pytest

from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.procedures.executor import (
    CompositeProcedureAPI,
    ProcedureExecutor,
    bundled_procedure_invoker,
)
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry
from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch

from test_memory import FakeCtx


def _effect(requests: list[ProcedureRequest]) -> PerformProcedureBatch:
    return PerformProcedureBatch(
        task_id="task-1",
        round_id="round-1",
        generation=1,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "agent_id": "builtin.quick_thinker",
            "requests": requests,
        },
    )


@pytest.mark.asyncio
async def test_registry_snapshot_reaches_local_memory_via_executor() -> None:
    ctx = FakeCtx()
    provider = BundledProcedureProvider(ctx)
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", provider.describe())
    catalog = registry.snapshot({})

    entry = catalog.get("builtin.message_recent")
    assert entry is not None
    assert entry.api_name == "builtin.invoke_procedure"

    # Host API 故意缺失：只能走本地 invoker，证明无 Host 注册也能成功。
    executor = ProcedureExecutor(
        catalog,
        api=None,
        local_invokers={"builtin.invoke_procedure": bundled_procedure_invoker(provider)},
    )
    event = await executor.invoke_many(
        _effect([ProcedureRequest(procedure_id="builtin.message_recent", arguments={"stream_id": "s", "limit": 2})])
    )

    assert len(event.results) == 1
    item = event.results[0]
    assert item.success
    assert item.api_name == "builtin.invoke_procedure"
    assert item.procedure_id == "builtin.message_recent"
    assert item.data is not None
    assert item.data["stream_id"] == "s"
    assert len(item.data["items"]) == 2
    assert ctx.message.recent_calls == [("s", 2)]


@pytest.mark.asyncio
async def test_composite_api_prefers_local_invoker_over_host() -> None:
    ctx = FakeCtx()
    provider = BundledProcedureProvider(ctx)
    host_calls: list[str] = []

    class HostAPI:
        async def call(self, name: str, *, version: str = "1", **kwargs: Any) -> Any:
            host_calls.append(name)
            raise AssertionError("不应回退到 Host API")

    api = CompositeProcedureAPI(
        host_api=HostAPI(),
        local_invokers={"builtin.invoke_procedure": bundled_procedure_invoker(provider)},
    )
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", provider.describe())
    executor = ProcedureExecutor(registry.snapshot({}), api=api)

    event = await executor.invoke_many(
        _effect(
            [
                ProcedureRequest(
                    procedure_id="builtin.knowledge_search",
                    arguments={"query": "alpha", "limit": 1},
                )
            ]
        )
    )

    assert host_calls == []
    assert event.results[0].success
    assert event.results[0].data["query"] == "alpha"
    assert ctx.knowledge.calls == [("alpha", 1)]
