"""`/swarm` 用户命令注册与 health/status 输出。"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.storage.vectors import VectorOpResult


class _FakeSend:
    def __init__(self) -> None:
        self.texts: list[tuple[str, str]] = []

    async def text(self, text: str, stream_id: str) -> bool:
        self.texts.append((str(text), str(stream_id)))
        return True


class _FakeVector:
    def __init__(self) -> None:
        self.rebuild_calls: list[bool] = []
        self._status = SimpleNamespace(
            idle=False,
            rebuilding=False,
            active_generation=1,
            dimension=8,
            selector="task:embedding",
            model_fingerprint="fp1",
            last_error_code=None,
            last_error_message=None,
            retired_generations=(),
            failed_candidate=None,
        )

    async def status(self) -> Any:
        return self._status

    async def list_jobs(self, *, status: str | None = None) -> list[dict[str, Any]]:
        del status
        return [{"job_id": "vj1", "status": "PENDING", "source_kind": "formalized_task"}]

    async def rebuild(self, *, force: bool = False) -> VectorOpResult:
        self.rebuild_calls.append(bool(force))
        return VectorOpResult.ok(
            code="rebuilt",
            data={"generation": 2, "dimension": 8, "count": 3},
        )


class _FakeStatistics:
    async def task(self, task_id: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "agent_calls": 2,
            "summarizer_calls": 1,
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "cache_hit_tokens": 10,
            "cache_miss_tokens": 90,
            "cache_hit_rate": 0.1,
            "estimated_credits": 5.0,
            "actual_credits": 4.5,
            "unreconciled_credits": 0.0,
            "cost_equivalent_credits": 0.2,
            "credit_pool": 95.0,
            "credit_debt": 0.0,
            "branches_total": 2,
            "branches_active": 1,
            "branches_finalized": 1,
            "max_branch_depth": 1,
            "compact_count": 0,
            "checkpoint_count": 0,
            "protocol_correction_count": 0,
            "continue_count": 0,
            "procedures_success": 1,
            "procedures_error": 0,
            "external_cost_credits": 0.0,
            "duration_ms_total": 1200,
            "error_count": 0,
        }

    async def plugin(self) -> dict[str, Any]:
        return {
            "models": {"m1": {"calls": 3, "prompt_tokens": 100, "completion_tokens": 40}},
            "agents": {"builtin.quick_thinker": {"branches": 2, "finalized": 1}},
            "procedures": {"builtin.web_search": {"calls": 1, "success": 1, "error": 0}},
            "tasks": {"lrs_a": await self.task("lrs_a")},
        }


class _FakeFeedback:
    def __init__(self) -> None:
        self.submits: list[dict[str, Any]] = []

    async def submit(self, **kwargs: Any) -> Any:
        self.submits.append(dict(kwargs))
        return SimpleNamespace(
            feedback_id="fb_1",
            lesson_id="les_1",
            disposition=kwargs["disposition"],
            round_id="rnd_1",
            lesson_indexing="queued",
            lesson_index_error=None,
        )


class _FakeStore:
    def __init__(self) -> None:
        self.layers: dict[str, Any] = {
            "lrs_a": SimpleNamespace(reports=[{}, {}, {}]),
            "lrs_b": SimpleNamespace(reports=[]),
        }

    async def load_summary_layer(self, task_id: str) -> Any | None:
        return self.layers.get(task_id)


class _FakeManager:
    def __init__(self) -> None:
        self.tasks = [
            {
                "task_id": "lrs_a",
                "status": "RUNNING",
                "round_id": "rnd_1",
                "round_number": 1,
                "generation": 0,
                "active_leaves": [{"branch_id": "br_1", "credits": 50.0}],
                "created_at": "2026-08-04T12:00:00Z",
            },
            {
                "task_id": "lrs_b",
                "status": "SUCCEEDED",
                "round_id": "rnd_2",
                "round_number": 2,
                "generation": 1,
                "active_leaves": [],
                "created_at": "2026-08-04T11:00:00Z",
            },
        ]
        self.store = _FakeStore()
        self.report_coordinators: dict[str, Any] = {}

    async def status(self, task_id: str, *, stream_id: str | None = None) -> dict[str, Any]:
        del stream_id
        for item in self.tasks:
            if item["task_id"] == task_id:
                return dict(item)
        raise LookupError(f"task {task_id} 不存在")

    async def list_tasks(self, *, stream_id: str | None = None) -> list[dict[str, Any]]:
        del stream_id
        return [dict(item) for item in self.tasks]


class _FakeScheduler:
    def stats(self) -> dict[str, Any]:
        return {
            "tasks": {
                "lrs_a": {"active": 1, "queued": 2, "paused": False},
            },
            "global": {"active": 1, "queued": 2},
        }


class _FakeRegistry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._health = {
            "com.0-hz.lrs-builtin": SimpleNamespace(
                provider_plugin_id="com.0-hz.lrs-builtin",
                status="healthy",
                errors=(),
            )
        }
        if kind == "agents":
            self._rows = [
                {
                    "id": "builtin.quick_thinker",
                    "provider": "com.0-hz.lrs-builtin",
                    "enabled": True,
                    "selector": "task:fast",
                    "protocol": "json_envelope",
                }
            ]
        else:
            self._rows = [
                {
                    "id": "builtin.web_search",
                    "provider": "com.0-hz.lrs-builtin",
                    "enabled": True,
                },
                {
                    "id": "fetch_url.fetch",
                    "provider": "com.0-hz.fetch-url",
                    "enabled": True,
                },
            ]
            self._health["com.0-hz.fetch-url"] = SimpleNamespace(
                provider_plugin_id="com.0-hz.fetch-url",
                status="healthy",
                errors=(),
            )

    @property
    def health(self) -> dict[str, Any]:
        return dict(self._health)

    @property
    def _providers(self) -> dict[str, Any]:
        # commands.py 遍历 registry._providers；测试用简易结构。
        batches: dict[str, Any] = {}
        for row in self._rows:
            provider = row["provider"]
            if provider not in batches:
                batches[provider] = SimpleNamespace(definitions=[])
            if self.kind == "agents":
                batches[provider].definitions.append(
                    SimpleNamespace(
                        agent_id=row["id"],
                        enabled=row["enabled"],
                        model_selector=row.get("selector", ""),
                        protocol=row.get("protocol", "json_envelope"),
                        display_name=row["id"],
                        character_prompt="SECRET_PROMPT",
                    )
                )
            else:
                batches[provider].definitions.append(
                    SimpleNamespace(
                        procedure_id=row["id"],
                        enabled=row["enabled"],
                        display_name=row["id"],
                        description="desc",
                    )
                )
        return batches


class _FakeServices:
    def __init__(self) -> None:
        self.vector_index = _FakeVector()
        self.statistics = _FakeStatistics()
        self.feedback = _FakeFeedback()
        self.scheduler = _FakeScheduler()
        self.manager = _FakeManager()
        self.agent_registry = _FakeRegistry("agents")
        self.procedure_registry = _FakeRegistry("procedures")
        self._config = SimpleNamespace(
            commands=SimpleNamespace(
                enabled=True,
                max_output_chars=12000,
                maintenance_allowed_person_ids=[],
                allow_vector_rebuild=True,
            ),
            agents={},
            procedures={},
            plugin=SimpleNamespace(root_agent="builtin.quick_thinker", enabled=True),
            feedback=SimpleNamespace(reminders_enabled=True),
        )
        self._status = {
            "sqlite": {"status": "healthy"},
            "vector_index": {"status": "healthy", "active_generation": 1},
            "physical_pinning": {"status": "healthy"},
            "extension_discovery": {"status": "healthy"},
            "maisaka_outbox": {"status": "healthy"},
            "feedback": {"status": "healthy", "reminders_enabled": True},
        }

    def health(self) -> dict[str, Any]:
        return {
            "sqlite": dict(self._status["sqlite"]),
            "vector_index": dict(self._status["vector_index"]),
            "physical_pinning": dict(self._status["physical_pinning"]),
            "extension_discovery": dict(self._status["extension_discovery"]),
            "maisaka_outbox": dict(self._status["maisaka_outbox"]),
            "extension_providers": {
                "agents": {"com.0-hz.lrs-builtin": {"status": "healthy"}},
                "procedures": {
                    "com.0-hz.lrs-builtin": {"status": "healthy"},
                    "com.0-hz.fetch-url": {"status": "healthy"},
                },
            },
            "root_agent": {"status": "healthy", "agent_id": "builtin.quick_thinker"},
            "config_reload": {"status": "healthy"},
        }


class CommandHarness:
    """匹配已注册 COMMAND 正则并调用插件 handler。"""

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self.send = _FakeSend()
        self.services = _FakeServices()
        self.vector = self.services.vector_index
        plugin._services = self.services
        plugin._manager = self.services.manager
        plugin._ctx = SimpleNamespace(send=self.send, logger=SimpleNamespace(info=lambda *a, **k: None))
        # config property via _plugin_config_instance if present
        plugin._plugin_config_instance = self.services._config

    @property
    def sent_text(self) -> str:
        return "\n".join(text for text, _ in self.send.texts)

    async def invoke(self, text: str, *, stream_id: str = "s", **extra: Any) -> Any:
        from maibot_sdk.components import _COMPONENT_INFO_ATTR

        components = [item for item in self.plugin.get_components() if item.get("type") == "COMMAND"]
        # Prefer longer / more specific patterns first (vectors status before status).
        components.sort(
            key=lambda item: len(str((item.get("metadata") or {}).get("command_pattern") or "")),
            reverse=True,
        )
        matched = None
        matched_groups: dict[str, str] = {}
        for item in components:
            meta = item.get("metadata") or {}
            pattern = str(meta.get("command_pattern") or "")
            handler_name = str(meta.get("handler_name") or "")
            if not pattern:
                continue
            match = re.match(pattern, text)
            if match is None:
                continue
            if handler_name and hasattr(self.plugin, handler_name):
                matched = getattr(self.plugin, handler_name)
            else:
                for name in dir(type(self.plugin)):
                    unbound = getattr(type(self.plugin), name, None)
                    if not callable(unbound):
                        continue
                    comp = getattr(unbound, _COMPONENT_INFO_ATTR, None)
                    if comp is not None and getattr(comp, "name", None) == item["name"]:
                        matched = getattr(self.plugin, name)
                        break
            matched_groups = {k: v for k, v in match.groupdict().items() if v is not None}
            break
        if matched is None:
            raise AssertionError(f"no COMMAND matched text={text!r}")
        kwargs = {
            "stream_id": stream_id,
            "text": text,
            "matched_groups": matched_groups,
            **extra,
        }
        return await matched(**kwargs)


@pytest.fixture
def command_harness(plugin_module) -> CommandHarness:
    return CommandHarness(plugin_module.create_plugin())


def test_swarm_command_patterns_are_registered(plugin_module) -> None:
    names = {
        item["name"] for item in plugin_module.create_plugin().get_components()
        if item["type"] == "COMMAND"
    }
    assert names == {
        "swarm_status", "swarm_tasks", "swarm_stats", "swarm_agents",
        "swarm_procedures", "swarm_health", "swarm_vectors_status",
        "swarm_vectors_rebuild", "swarm_feedback",
    }


@pytest.mark.asyncio
async def test_force_vector_rebuild_command(command_harness) -> None:
    await command_harness.invoke("/swarm vectors rebuild --force", stream_id="s")
    assert command_harness.vector.rebuild_calls == [True]
    assert "已创建新的向量 generation" in command_harness.sent_text


@pytest.mark.asyncio
async def test_swarm_status_overview_and_task(command_harness) -> None:
    await command_harness.invoke("/swarm status", stream_id="s")
    assert "RUNNING" in command_harness.sent_text or "运行" in command_harness.sent_text
    command_harness.send.texts.clear()
    await command_harness.invoke("/swarm status lrs_a", stream_id="s")
    text = command_harness.sent_text
    assert "lrs_a" in text
    assert "RUNNING" in text


@pytest.mark.asyncio
async def test_swarm_status_includes_report_count(command_harness) -> None:
    await command_harness.invoke("/swarm status lrs_a", stream_id="s")
    assert "报告数：3" in command_harness.sent_text
    command_harness.send.texts.clear()
    # Live coordinator reports take precedence over persisted layer.
    command_harness.plugin._manager.report_coordinators["lrs_a"] = SimpleNamespace(
        deadline_at="2026-08-04T13:00:00Z",
        reports=[object(), object()],
    )
    await command_harness.invoke("/swarm status lrs_a", stream_id="s")
    text = command_harness.sent_text
    assert "报告数：2" in text
    assert "deadline：2026-08-04T13:00:00Z" in text


def test_clip_command_output_honors_limit_with_many_errors() -> None:
    from lunagentic_research_swarm.commands import clip_command_output

    body = "正文内容" * 20
    errors = [f"错误条目-{i:03d}" for i in range(100)]
    out = clip_command_output(body, 1000, error_lines=errors)
    assert len(out) <= 1000
    match = re.search(r"共\s*(\d+)\s*/\s*显示\s*(\d+)", out)
    assert match is not None, out
    total, shown = int(match.group(1)), int(match.group(2))
    assert total == 100
    assert 0 <= shown < total
    assert out.count("错误条目-") == shown


def test_clip_command_output_keeps_all_errors_when_budget_allows() -> None:
    from lunagentic_research_swarm.commands import clip_command_output

    out = clip_command_output("ok", 1000, error_lines=["a", "b", "c"])
    assert len(out) <= 1000
    assert "共 3 / 显示 3" in out
    assert "- a" in out and "- b" in out and "- c" in out


@pytest.mark.asyncio
async def test_swarm_stats_and_feedback(command_harness) -> None:
    await command_harness.invoke("/swarm stats lrs_a", stream_id="s")
    assert "prompt_tokens" in command_harness.sent_text or "prompt" in command_harness.sent_text.lower() or "token" in command_harness.sent_text.lower()
    command_harness.send.texts.clear()
    await command_harness.invoke("/swarm feedback lrs_a accepted 不错", stream_id="s")
    assert command_harness.services.feedback.submits
    assert command_harness.services.feedback.submits[0]["disposition"] == "accepted"
    assert "fb_1" in command_harness.sent_text or "反馈" in command_harness.sent_text


@pytest.mark.asyncio
async def test_swarm_health_mentions_fetch_and_sqlite(command_harness) -> None:
    await command_harness.invoke("/swarm health", stream_id="s")
    text = command_harness.sent_text
    assert "sqlite" in text.lower() or "SQLite" in text
    assert "fetch" in text.lower() or "推荐" in text


@pytest.mark.asyncio
async def test_swarm_agents_hides_prompt_secrets(command_harness) -> None:
    await command_harness.invoke("/swarm agents", stream_id="s")
    assert "SECRET_PROMPT" not in command_harness.sent_text
    assert "builtin.quick_thinker" in command_harness.sent_text


@pytest.mark.asyncio
async def test_missing_stream_id_errors(command_harness) -> None:
    result = await command_harness.invoke("/swarm health", stream_id="")
    assert result[0] is False or "stream_id" in command_harness.sent_text or (
        isinstance(result, tuple) and "stream" in str(result[1]).lower()
    )
