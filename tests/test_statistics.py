"""确定性统计：从权威账本重算，与报告 snapshot 一致，且不读 raw debug。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lunagentic_research_swarm.statistics import StatisticsService
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


async def _open_seeded(tmp_path: Path) -> SQLiteStateStore:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand(
                "insert_task",
                {
                    "task_id": "lrs_stats",
                    "stream_id": "s",
                    "formalized_text": "formal",
                    "formalized_sha256": "abc",
                    "created_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "rnd_1",
                    "task_id": "lrs_stats",
                    "round_number": 1,
                    "generation": 1,
                    "status": "COMPLETED",
                    "time_budget_seconds": 120,
                    "credit_pool": -2.5,
                    "started_at": 1.0,
                    "ended_at": 40.0,
                },
            ),
            StoreCommand(
                "insert_branch",
                {
                    "branch_id": "br_root",
                    "round_id": "rnd_1",
                    "parent_branch_id": None,
                    "agent_id": "builtin.quick_thinker",
                    "lifecycle": "FINALIZED",
                    "depth": 0,
                    "credit_balance": 0.0,
                    "generation": 1,
                    "created_at": 1.0,
                    "finalized_at": 30.0,
                },
            ),
            StoreCommand(
                "insert_branch",
                {
                    "branch_id": "br_child",
                    "round_id": "rnd_1",
                    "parent_branch_id": "br_root",
                    "agent_id": "builtin.researcher",
                    "lifecycle": "IN_FLIGHT",
                    "depth": 2,
                    "credit_balance": 1.0,
                    "generation": 1,
                    "created_at": 2.0,
                },
            ),
            StoreCommand(
                "insert_llm_usage",
                {
                    "usage_id": "u_agent",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "call_id": "c_agent",
                    "role": "agent",
                    "selector": "task:utils",
                    "estimated_model_name": "m1",
                    "actual_model_name": "m1-phys",
                    "price_source": "host_config",
                    "price_fingerprint": "fp",
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cache_hit_tokens": 40,
                    "cache_miss_tokens": 60,
                    "estimated_charge": 1.5,
                    "actual_charge": 1.2,
                    "adjustment": 0.3,
                    "reconciliation_status": "actual",
                    "duration_ms": 50,
                    "created_at": 3.0,
                },
            ),
            StoreCommand(
                "insert_llm_usage",
                {
                    "usage_id": "u_sum",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "call_id": "c_sum",
                    "role": "summarizer",
                    "selector": "task:mid_memory",
                    "estimated_model_name": "m1",
                    "actual_model_name": "m1-phys",
                    "price_source": "host_config",
                    "price_fingerprint": "fp",
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 0,
                    "estimated_charge": 0.8,
                    "actual_charge": 0.8,
                    "adjustment": 0.0,
                    "reconciliation_status": "actual",
                    "duration_ms": 40,
                    "created_at": 4.0,
                },
            ),
            StoreCommand(
                "insert_llm_usage",
                {
                    "usage_id": "u_unrec",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_child",
                    "call_id": "c_unrec",
                    "role": "agent",
                    "selector": "task:utils",
                    "estimated_model_name": "m1",
                    "actual_model_name": None,
                    "price_source": "host_config",
                    "price_fingerprint": "fp",
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 10,
                    "estimated_charge": 0.4,
                    "actual_charge": None,
                    "adjustment": None,
                    "reconciliation_status": "estimated_unreconciled",
                    "duration_ms": 0,
                    "created_at": 5.0,
                },
            ),
            StoreCommand(
                "insert_credit_ledger",
                {
                    "ledger_id": "led_1",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "call_id": "c_agent",
                    "entry_kind": "actual_charge",
                    "amount": -1.2,
                    "balance_after": 8.8,
                    "metadata_json": "{}",
                    "created_at": 3.0,
                },
            ),
            StoreCommand(
                "insert_procedure_call",
                {
                    "request_id": "p1",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "turn_id": "c_agent",
                    "agent_id": "builtin.quick_thinker",
                    "procedure_id": "builtin.web_search",
                    "provider_plugin_id": "builtin",
                    "status": "SUCCEEDED",
                    "duration_ms": 12,
                    "error_code": None,
                    "provenance_json": "{}",
                    "external_cost_json": json.dumps({"credits": 0.25}, ensure_ascii=False),
                    "created_at": 6.0,
                },
            ),
            StoreCommand(
                "insert_procedure_call",
                {
                    "request_id": "p2",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "turn_id": "c_agent",
                    "agent_id": "builtin.quick_thinker",
                    "procedure_id": "core.compact",
                    "provider_plugin_id": "core",
                    "status": "SUCCEEDED",
                    "duration_ms": 5,
                    "error_code": None,
                    "provenance_json": "{}",
                    "external_cost_json": None,
                    "created_at": 7.0,
                },
            ),
            StoreCommand(
                "insert_procedure_call",
                {
                    "request_id": "p3",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_child",
                    "turn_id": "c_unrec",
                    "agent_id": "builtin.researcher",
                    "procedure_id": "core.checkpoint",
                    "provider_plugin_id": "core",
                    "status": "ERROR",
                    "duration_ms": 3,
                    "error_code": "checkpoint_failed",
                    "provenance_json": "{}",
                    "external_cost_json": None,
                    "created_at": 8.0,
                },
            ),
            StoreCommand(
                "insert_lifecycle_event",
                {
                    "event_id": "e_corr",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "event_type": "ProtocolCorrection",
                    "from_status": "RUNNING",
                    "to_status": "RUNNING",
                    "metadata_json": json.dumps({"correction_count": 1}, ensure_ascii=False),
                    "created_at": 9.0,
                },
            ),
            StoreCommand(
                "insert_lifecycle_event",
                {
                    "event_id": "e_cont",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "event_type": "ContinueRequested",
                    "from_status": "PAUSED",
                    "to_status": "RUNNING",
                    "metadata_json": "{}",
                    "created_at": 10.0,
                },
            ),
            StoreCommand(
                "insert_report",
                {
                    "report_id": "rpt_1",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "epoch": 1,
                    "kind": "FINAL",
                    "text": "结论",
                    "status": "SUCCEEDED",
                    "running_branch_count": 1,
                    "stats_json": "{}",  # filled after compute
                    "created_at": 40.0,
                },
            ),
        ]
    )
    return store


@pytest.mark.asyncio
async def test_task_stats_recomputed_from_ledger_equal_report_snapshot(tmp_path: Path) -> None:
    store = await _open_seeded(tmp_path)
    service = StatisticsService(store)
    stats = await service.task("lrs_stats")
    assert stats["agent_calls"] == 2
    assert stats["summarizer_calls"] == 1
    assert stats["prompt_tokens"] == 160
    assert stats["completion_tokens"] == 30
    assert stats["cache_hit_tokens"] == 40
    assert stats["cache_miss_tokens"] == 70
    assert stats["cache_hit_rate"] == pytest.approx(40 / 110)
    assert stats["estimated_credits"] == pytest.approx(1.9)
    assert stats["actual_credits"] == pytest.approx(1.2)
    assert stats["unreconciled_credits"] == pytest.approx(0.4)
    assert stats["cost_equivalent_credits"] == pytest.approx(0.8)
    assert stats["credit_pool"] == pytest.approx(-2.5)
    assert stats["credit_debt"] == pytest.approx(2.5)
    assert stats["branches_total"] == 2
    assert stats["branches_active"] == 1
    assert stats["branches_finalized"] == 1
    assert stats["max_branch_depth"] == 2
    assert stats["compact_count"] == 1
    assert stats["checkpoint_count"] == 1
    assert stats["protocol_correction_count"] == 1
    assert stats["continue_count"] == 1
    assert stats["procedures_success"] == 2
    assert stats["procedures_error"] == 1
    assert stats["external_cost_credits"] == pytest.approx(0.25)
    assert stats["duration_ms_total"] == 50 + 40 + 0 + 12 + 5 + 3
    assert stats["error_count"] >= 1

    # 同 transaction snapshot：写入报告后从账本重算应完全相等。
    await store.transact(
        [
            StoreCommand(
                "insert_report",
                {
                    "report_id": "rpt_snap",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "epoch": 2,
                    "kind": "FINAL",
                    "text": "snap",
                    "status": "SUCCEEDED",
                    "running_branch_count": 1,
                    "stats_json": json.dumps(stats, ensure_ascii=False, sort_keys=True),
                    "created_at": 41.0,
                },
            )
        ]
    )
    recomputed = await service.task("lrs_stats")
    layer = await store.load_summary_layer("lrs_stats")
    assert layer is not None
    saved = next(item["stats"] for item in layer.reports if item["report_id"] == "rpt_snap")
    assert dict(saved) == stats
    assert recomputed == await service.task("lrs_stats")
    cache_a = await service.cache("lrs_stats")
    cache_b = await service.cache("lrs_stats")
    assert cache_a == cache_b
    assert cache_a["hit"] == 40
    assert cache_a["miss"] == 70
    assert cache_a["hit_rate"] == pytest.approx(40 / 110)
    await store.close()


@pytest.mark.asyncio
async def test_cache_hit_rate_null_when_denominator_zero(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "lrs_empty", "stream_id": "s", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "rnd_e",
                    "task_id": "lrs_empty",
                    "round_number": 1,
                    "generation": 1,
                    "status": "RUNNING",
                    "time_budget_seconds": 10,
                    "credit_pool": 0.0,
                    "started_at": 1.0,
                },
            ),
        ]
    )
    service = StatisticsService(store)
    stats = await service.task("lrs_empty")
    assert stats["cache_hit_rate"] is None
    assert (await service.cache("lrs_empty"))["hit_rate"] is None
    await store.close()


@pytest.mark.asyncio
async def test_plugin_stats_aggregate_without_reading_raw(tmp_path: Path) -> None:
    store = await _open_seeded(tmp_path)
    # 写入独立 debug DB；统计服务不得依赖它。
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    with sqlite3.connect(debug_dir / "raw.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE agent_transcripts (id INTEGER PRIMARY KEY, messages TEXT)"
        )
        connection.execute(
            "INSERT INTO agent_transcripts(messages) VALUES (?)",
            ("raw-agent-secret",),
        )
    service = StatisticsService(store)
    plugin = await service.plugin()
    assert "models" in plugin
    assert "m1-phys" in plugin["models"] or "m1" in plugin["models"]
    assert "agents" in plugin
    assert "builtin.quick_thinker" in plugin["agents"]
    assert "procedures" in plugin
    assert "builtin.web_search" in plugin["procedures"]
    assert "tasks" in plugin
    assert "lrs_stats" in plugin["tasks"]
    encoded = json.dumps(plugin, ensure_ascii=False)
    assert "raw-agent-secret" not in encoded
    await store.close()


@pytest.mark.asyncio
async def test_statistics_ignore_missing_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    service = StatisticsService(store)
    stats = await service.task("missing")
    assert stats["agent_calls"] == 0
    assert stats["cache_hit_rate"] is None
    await store.close()
