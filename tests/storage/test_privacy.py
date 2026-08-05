from __future__ import annotations

import sqlite3

import pytest

from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore


EXPECTED_COLUMNS = {
    "tasks": (
        "task_id", "stream_id", "formalized_text", "formalized_sha256",
        "current_round_number", "created_at", "updated_at",
    ),
    "investigation_rounds": (
        "round_id", "task_id", "round_number", "generation", "status", "time_budget_seconds",
        "grace_period_seconds", "credit_pool", "catalog_fingerprint", "started_at", "report_deadline_at", "ended_at",
    ),
    "lifecycle_events": (
        "event_id", "task_id", "round_id", "event_type", "from_status", "to_status", "metadata_json", "created_at",
    ),
    "branches": (
        "branch_id", "round_id", "parent_branch_id", "agent_id", "lifecycle", "depth", "credit_balance", "generation",
        "latest_checkpoint_summary_id", "terminal_summary_id", "created_at", "finalized_at",
    ),
    "summaries": (
        "summary_id", "task_id", "round_id", "branch_id", "kind", "report_epoch",
        "text", "status", "error_code", "created_at",
    ),
    "reports": (
        "report_id", "task_id", "round_id", "epoch", "kind", "text",
        "status", "running_branch_count", "stats_json", "created_at",
    ),
    "credit_ledger": (
        "ledger_id", "task_id", "round_id", "branch_id", "call_id", "entry_kind",
        "amount", "balance_after", "metadata_json", "created_at",
    ),
    "llm_usage": (
        "usage_id", "task_id", "round_id", "branch_id", "call_id", "role", "selector", "estimated_model_name",
        "actual_model_name", "price_source", "price_fingerprint",
        "prompt_tokens", "completion_tokens", "cache_hit_tokens",
        "cache_miss_tokens", "estimated_charge", "actual_charge", "adjustment",
        "reconciliation_status", "duration_ms", "created_at",
    ),
    "procedure_calls": (
        "request_id", "task_id", "round_id", "branch_id", "turn_id", "agent_id", "procedure_id", "provider_plugin_id",
        "status", "duration_ms", "error_code", "provenance_json", "external_cost_json", "created_at",
    ),
    "extension_fingerprints": (
        "event_id", "provider_plugin_id", "extension_kind", "fingerprint", "availability", "error_json", "created_at",
    ),
    "feedback_events": ("feedback_id", "task_id", "round_id", "disposition", "payload_json", "created_at"),
    "feedback_reminders": ("reminder_id", "task_id", "round_id", "due_at", "status", "triggered_at"),
    "maisaka_outbox": (
        "outbox_id", "task_id", "round_id", "report_id", "delivery_kind", "idempotency_key", "payload_json", "status",
        "attempt_count", "next_attempt_at", "last_error", "created_at", "delivered_at",
    ),
    "vector_jobs": (
        "job_id", "source_kind", "source_id", "generation", "status",
        "attempt_count", "error_code", "error_json", "created_at", "updated_at",
    ),
    "vector_generations": (
        "generation", "selector", "actual_model_name", "model_fingerprint", "dimension",
        "table_name", "schema_version", "status", "created_at", "activated_at", "retired_at",
    ),
    "vector_documents": (
        "source_kind", "source_id", "generation", "actual_model_name",
        "model_fingerprint", "dimension", "indexed_at",
    ),
    "schema_migrations": ("version", "name", "applied_at"),
}

EXPECTED_INDEXES = {
    "idx_rounds_round_status",
    "idx_lifecycle_events_task_created",
    "idx_summaries_task_created",
    "idx_summaries_branch_kind_created",
    "idx_vector_jobs_status",
    "idx_outbox_status_next_attempt",
    "one_active_vector_generation",
}


@pytest.mark.asyncio
async def test_default_storage_has_no_raw_transcript_or_payload_tables_or_columns(tmp_path) -> None:
    """若默认 migration 引入 raw 表或字段，隐私开关关闭也会留下持久化入口。"""

    path = tmp_path / "state.sqlite3"
    store = SQLiteStateStore(path)
    await store.open()
    names = await store.table_names()
    await store.close()

    assert "agent_transcripts" not in names
    assert "procedure_raw_payloads" not in names
    with sqlite3.connect(path) as connection:
        columns = {
            column[1]
            for table_name in names
            for column in connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        }
    forbidden_fragments = ("transcript", "raw_payload", "raw_arguments", "raw_result", "messages")
    assert not any(fragment in column.lower() for column in columns for fragment in forbidden_fragments)


@pytest.mark.asyncio
async def test_migration_creates_exact_authoritative_tables_columns_and_indexes(tmp_path) -> None:
    """若 migration 漏建或擅自扩展权威 schema，后续 reducer 与隐私边界会漂移。"""

    path = tmp_path / "state.sqlite3"
    store = SQLiteStateStore(path)
    await store.open()
    await store.close()

    with sqlite3.connect(path) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            table_name: tuple(row[1] for row in connection.execute(f'PRAGMA table_info("{table_name}")'))
            for table_name in names
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert columns == EXPECTED_COLUMNS
    assert indexes == EXPECTED_INDEXES
