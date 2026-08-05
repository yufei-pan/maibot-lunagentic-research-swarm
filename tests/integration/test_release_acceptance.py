"""0.1.0 发布验收：组件完整性与端到端 release flow 断言。"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions
from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.models import ReportKind, TaskStatus
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.procedures.core import CORE_PROCEDURE_IDS
from lunagentic_research_swarm.runtime.manager import ResearchManager
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from fakes import FakeClock


EXPECTED_AGENTS = {
    "builtin.quick_thinker",
    "builtin.deep_thinker",
    "builtin.debater",
    "builtin.researcher",
    "builtin.memory_researcher",
    "builtin.knowledge_reporter",
    "builtin.past_case_researcher",
    "builtin.evidence_verifier",
    "builtin.quantitative_analyst",
}

EXPECTED_TOOLS = {
    "start_deep_research",
    "pause_deep_research",
    "continue_deep_research",
    "stop_deep_research",
    "add_research_context",
    "get_research_status",
    "list_research_tasks",
    "submit_research_feedback",
}

EXPECTED_COMMANDS = {
    "swarm_status",
    "swarm_tasks",
    "swarm_stats",
    "swarm_agents",
    "swarm_procedures",
    "swarm_health",
    "swarm_vectors_status",
    "swarm_vectors_rebuild",
    "swarm_feedback",
}

EXPECTED_BUNDLED_PROCEDURES = {
    "builtin.chat_streams",
    "builtin.message_recent",
    "builtin.message_by_id",
    "builtin.message_time_range",
    "builtin.person_lookup",
    "builtin.knowledge_search",
    "builtin.web_search",
    "builtin.past_cases",
    "builtin.calculate",
    "builtin.statistics",
    "builtin.convert_units",
    "builtin.normalize_urls",
    "builtin.organize_provenance",
}


def test_release_catalog_has_nine_agents_eight_tools_nine_commands(plugin_module) -> None:
    agents = bundled_agent_definitions()
    assert {item.agent_id for item in agents} == EXPECTED_AGENTS
    assert next(item for item in agents if item.agent_id == "builtin.quick_thinker").can_be_root

    components = plugin_module.create_plugin().get_components()
    tools = {item["name"] for item in components if item["type"] == "TOOL"}
    commands = {item["name"] for item in components if item["type"] == "COMMAND"}
    assert tools == EXPECTED_TOOLS
    assert commands == EXPECTED_COMMANDS


def test_release_bundled_and_core_procedures_are_complete() -> None:
    provider = BundledProcedureProvider(object())
    bundled = {item["procedure_id"] for item in provider.describe()}
    assert bundled == EXPECTED_BUNDLED_PROCEDURES
    assert CORE_PROCEDURE_IDS == frozenset({"core.compact", "core.checkpoint", "core.terminate"})


def test_default_config_disables_raw_storage_and_keeps_quick_root() -> None:
    config = LRSConfig()
    assert config.plugin.root_agent == "builtin.quick_thinker"
    assert config.storage.store_agent_transcripts is False
    assert config.storage.store_raw_procedure_payloads is False
    assert config.budget.default_effort_credits == 100.0
    assert config.timing.grace_period_seconds == 60
    assert config.timing.feedback_wait_seconds == 600


def test_default_config_template_covers_required_sections() -> None:
    text = (Path(__file__).resolve().parents[2] / "config.default.toml").read_text(encoding="utf-8")
    for section in (
        "[plugin]",
        "[llm]",
        "[summarizer]",
        "[embedding]",
        "[timing]",
        "[budget]",
        "[scheduler]",
        "[context]",
        "[protocol]",
        "[storage]",
        "[extensions]",
        "[pricing]",
        "[reporting]",
        "[feedback]",
        "[commands]",
        "[web_search]",
        "[agents]",
        "[procedures]",
    ):
        assert section in text
    assert "builtin.quick_thinker" in text
    assert "searxng_url" in text
    assert "tavily_api_key" in text
    assert "you_api_key" in text
    assert "you_base_url" in text
    assert "Note" in text or "注意" in text
    for agent_id in EXPECTED_AGENTS:
        assert agent_id in text


def test_continue_restart_uses_summary_layer_only() -> None:
    source = inspect.getsource(ResearchManager._restart_round)
    assert "load_summary_layer" in source
    assert "summary_layers" in source
    assert "objective" not in source


@pytest.mark.asyncio
async def test_release_flow_start_formalize_report_privacy_and_stats(runtime_harness) -> None:
    harness = runtime_harness
    started = await harness.start("比较两个方案", credits=100.0, time_budget=120)
    assert started["task_id"].startswith("lrs_")
    assert harness.formalized_task is None

    await harness.formalize("正式任务")
    task_row = await harness.store.load_task(harness.task_id)
    assert task_row.formalized_task is not None
    assert task_row.formalized_task.text == "正式任务"

    jobs = await harness.store.run_locked(
        lambda connection: [
            dict(row)
            for row in connection.execute(
                "SELECT source_kind, source_id, status FROM vector_jobs WHERE source_id = ?",
                (harness.task_id,),
            ).fetchall()
        ]
    )
    assert any(row["source_kind"] == "formalized_task" for row in jobs)

    await harness.root_delegates({"A": 50.0, "B": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(120)
    await harness.run_until_idle()

    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert await harness.pending_outbox_count() >= 1
    assert harness.maisaka.append_calls == []

    await harness.finalize_all()
    assert harness.reports[-1].kind is ReportKind.FINAL
    assert harness.task_status is TaskStatus.COMPLETED
    final = harness.reports[-1]
    assert "统计" in final.text or "credits" in final.text.lower() or "token" in final.text.lower()

    # stats 亦写入 SQLite
    report_rows = await harness.store.run_locked(
        lambda connection: [
            dict(row)
            for row in connection.execute(
                "SELECT kind, stats_json FROM reports WHERE task_id = ? AND kind = ?",
                (harness.task_id, ReportKind.FINAL.value),
            ).fetchall()
        ]
    )
    assert report_rows
    assert report_rows[0]["stats_json"]
    assert harness.raw_context_count == 0
    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None
    durable = repr(layer)
    assert "raw_payload" not in durable
    assert "transcript" not in durable
    assert "reasoning" not in durable

    await harness.deliver_outbox()
    assert len(harness.maisaka.append_calls) >= 1


@pytest.mark.asyncio
async def test_feedback_reminder_persists_for_completed_round(tmp_path: Path) -> None:
    from lunagentic_research_swarm.feedback import FeedbackService
    from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
    from types import SimpleNamespace

    store = SQLiteStateStore(tmp_path / "remind.sqlite3")
    await store.open()
    clock = FakeClock(1_000.0)
    try:
        await store.transact(
            [
                StoreCommand(
                    "insert_task",
                    {
                        "task_id": "lrs_release",
                        "stream_id": "s",
                        "current_round_number": 1,
                        "created_at": clock(),
                        "updated_at": clock(),
                    },
                ),
                StoreCommand(
                    "insert_round",
                    {
                        "round_id": "rnd1",
                        "task_id": "lrs_release",
                        "round_number": 1,
                        "generation": 0,
                        "status": "COMPLETED",
                        "time_budget_seconds": 120,
                        "credit_pool": 0.0,
                        "started_at": clock(),
                    },
                ),
            ]
        )
        maisaka = SimpleNamespace(trigger_calls=0)

        class _Proactive:
            async def trigger(self, *args, **kwargs):
                del args, kwargs
                maisaka.trigger_calls += 1

        class _Context:
            async def append(self, *args, **kwargs):
                del args, kwargs

        maisaka.proactive = _Proactive()
        maisaka.context = _Context()
        outbox = MaisakaOutbox(store, maisaka, clock=clock, poll_interval_seconds=0.01)
        service = FeedbackService(
            store,
            outbox=outbox,
            clock=clock,
            feedback_wait_seconds=600,
            reminders_enabled=True,
            index_lessons=False,
        )
        await service.schedule(task_id="lrs_release", round_id="rnd1", ended_at=clock())
        pending = await store.run_locked(
            lambda connection: connection.execute(
                "SELECT status FROM feedback_reminders WHERE task_id = ?",
                ("lrs_release",),
            ).fetchone()
        )
        assert pending is not None
        assert str(pending["status"]).lower() in {"pending", "due", "scheduled"}
    finally:
        await store.close()
