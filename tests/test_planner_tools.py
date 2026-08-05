from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeResearchManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def start(self, **kwargs):
        self.calls.append(("start", (), dict(kwargs)))
        return {"task_id": "lrs_fake", "status": "FORMALIZING", "initial_credits": 100.0}

    async def pause(self, task_id, **kwargs):
        self.calls.append(("pause", (task_id,), dict(kwargs)))
        return {"task_id": task_id, "status": "PAUSED", "round_id": "rnd-1"}

    async def continue_task(self, task_id, **kwargs):
        self.calls.append(("continue", (task_id,), dict(kwargs)))
        return {"task_id": task_id, "status": "RUNNING", "round_id": "rnd-2"}

    async def stop(self, task_id, **kwargs):
        self.calls.append(("stop", (task_id,), dict(kwargs)))
        return {"task_id": task_id, "status": "STOPPED", "round_id": "rnd-1"}

    async def add_context(self, task_id, context, **kwargs):
        self.calls.append(("add_context", (task_id, context), dict(kwargs)))
        return {"task_id": task_id, "status": "RUNNING", "round_id": "rnd-1"}

    async def status(self, task_id, **kwargs):
        self.calls.append(("status", (task_id,), dict(kwargs)))
        return {
            "task_id": task_id, "status": "RUNNING", "round_id": "rnd-1",
            "active_leaves": [{"branch_id": "br-1", "credits": 1, "pending_context": ["secret"]}],
        }

    async def list_tasks(self, **kwargs):
        self.calls.append(("list_tasks", (), dict(kwargs)))
        return [{
            "task_id": "lrs_fake", "status": "RUNNING", "round_id": "rnd-1",
            "created_at": "2026-08-04T12:00:00Z",
            "active_leaves": [{"branch_id": "br-1", "credits": 1, "pending_context": ["secret"]}],
        }]


@pytest.fixture
def fake_plugin(plugin_module):
    plugin = plugin_module.create_plugin()
    manager = FakeResearchManager()
    plugin._manager = manager
    return plugin, manager


def test_runtime_planner_tool_names_are_plain(plugin_module) -> None:
    names = {
        item["name"] for item in plugin_module.create_plugin().get_components()
        if item["type"] == "TOOL"
    }
    assert names == {
        "start_deep_research", "pause_deep_research", "continue_deep_research",
        "stop_deep_research", "add_research_context", "get_research_status",
        "list_research_tasks", "submit_research_feedback",
    }


def test_planner_schemas_do_not_allow_planner_to_select_stream(plugin_module) -> None:
    component = next(
        item for item in plugin_module.create_plugin().get_components()
        if item["name"] == "start_deep_research" and item["type"] == "TOOL"
    )
    schema = component["metadata"]["parameters_raw"]
    assert schema["additionalProperties"] is False
    assert "stream_id" not in schema["properties"]


@pytest.mark.asyncio
async def test_start_tool_forwards_stream_and_returns_immediately(fake_plugin) -> None:
    plugin, manager = fake_plugin
    result = await plugin.start_deep_research(
        objective="调查", time_budget_seconds=60, effort_level=1.0, stream_id="s1"
    )
    assert result["success"]
    assert result["task_id"].startswith("lrs_")
    assert result["status"] == "FORMALIZING"
    assert manager.calls == [
        (
            "start", (),
            {
                "objective": "调查", "stream_id": "s1", "time_budget_seconds": 60,
                "effort_level": 1.0, "planner_context": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_mutating_tools_forward_controls_and_add_stable_fields(fake_plugin) -> None:
    plugin, manager = fake_plugin
    paused = await plugin.pause_deep_research("lrs_fake", stream_id="s1")
    continued = await plugin.continue_deep_research(
        "lrs_fake", time_budget_seconds=90, credit_adjustment=-2.5, stream_id="s1"
    )
    stopped = await plugin.stop_deep_research("lrs_fake", reason="已满足", stream_id="s1")
    supplied = await plugin.add_research_context("lrs_fake", information="补充条件", stream_id="s1")

    assert all(item["success"] for item in (paused, continued, stopped, supplied))
    assert paused["effective_time_budget_seconds"] is None
    assert continued["effective_time_budget_seconds"] == 90
    assert continued["effective_credits_or_adjustment"] == -2.5
    assert manager.calls == [
        ("pause", ("lrs_fake",), {"stream_id": "s1"}),
        ("continue", ("lrs_fake",), {"time_budget_seconds": 90, "credit_adjustment": -2.5, "stream_id": "s1"}),
        ("stop", ("lrs_fake",), {"reason": "已满足", "stream_id": "s1"}),
        ("add_context", ("lrs_fake", "补充条件"), {"stream_id": "s1"}),
    ]


@pytest.mark.asyncio
async def test_status_and_list_tools_return_stable_shapes(fake_plugin) -> None:
    plugin, manager = fake_plugin
    status = await plugin.get_research_status("lrs_fake", stream_id="s1")
    listed = await plugin.list_research_tasks(status="RUNNING", limit=20, stream_id="s1")
    assert status == {
        "success": True, "task_id": "lrs_fake", "status": "RUNNING", "round_id": "rnd-1",
        "active_leaves": [{"branch_id": "br-1", "credits": 1}],
    }
    assert listed["success"] is True
    assert listed["tasks"][0]["task_id"] == "lrs_fake"
    assert [call[0] for call in manager.calls] == ["status", "list_tasks"]
    assert manager.calls[0][2] == {"stream_id": "s1"}
    assert manager.calls[1][2] == {"stream_id": "s1"}


@pytest.mark.asyncio
async def test_start_requires_host_stream_id_and_manager_errors_are_structured(fake_plugin) -> None:
    plugin, manager = fake_plugin
    missing = await plugin.start_deep_research(objective="调查")
    assert missing["success"] is False
    assert missing["error"] == {"code": "stream_id_required", "message": "stream_id 不能为空"}
    assert all(key in missing for key in (
        "task_id", "round", "status", "effective_time_budget_seconds", "effective_credits_or_adjustment",
    ))
    assert manager.calls == []

    plugin._manager = SimpleNamespace(start=lambda **kwargs: (_ for _ in ()).throw(LookupError("not found")))
    result = await plugin.start_deep_research(objective="调查", stream_id="s1")
    assert result["success"] is False
    assert result["error"]["code"] == "task_not_found"

    missing_status = await plugin.get_research_status("lrs_fake")
    assert missing_status["error"]["code"] == "stream_id_required"


@pytest.mark.asyncio
async def test_structured_manager_failure_and_strict_time_filtering(fake_plugin) -> None:
    plugin, manager = fake_plugin

    async def failed_continue(*args, **kwargs):
        return {
            "success": False,
            "error": {"code": "task_finished_insufficient_funds", "message": "没有可用 credits"},
            "task_id": args[0],
        }

    manager.continue_task = failed_continue
    failed = await plugin.continue_deep_research("lrs_fake", stream_id="s1")
    assert failed["success"] is False
    assert failed["error"]["code"] == "task_finished_insufficient_funds"
    assert all(key in failed for key in (
        "task_id", "round", "status", "effective_time_budget_seconds", "effective_credits_or_adjustment",
    ))

    before = await plugin.list_research_tasks(
        created_before="2026-08-04T11:00:00Z", stream_id="s1"
    )
    assert before["tasks"] == []
    invalid = await plugin.list_research_tasks(created_after="2026-08-04", stream_id="s1")
    assert invalid["error"]["code"] == "invalid_argument"

    plugin._manager = None
    unavailable = await plugin.stop_deep_research("lrs_fake", stream_id="s1")
    assert unavailable["error"]["code"] == "manager_unavailable"
    assert all(key in unavailable for key in (
        "task_id", "round", "status", "effective_time_budget_seconds", "effective_credits_or_adjustment",
    ))
