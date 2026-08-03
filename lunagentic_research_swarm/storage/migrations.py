"""SQLite 权威状态库的有序迁移。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: Sequence[Migration] = (
    Migration(
        version=1,
        name="authoritative_state",
        statements=(
            """
            CREATE TABLE tasks (
              task_id TEXT PRIMARY KEY,
              stream_id TEXT NOT NULL,
              formalized_text TEXT,
              formalized_sha256 TEXT,
              current_round_number INTEGER NOT NULL DEFAULT 1,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE investigation_rounds (
              round_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(task_id),
              round_number INTEGER NOT NULL,
              generation INTEGER NOT NULL,
              status TEXT NOT NULL,
              time_budget_seconds INTEGER NOT NULL,
              grace_period_seconds INTEGER NOT NULL DEFAULT 60,
              credit_pool REAL NOT NULL,
              catalog_fingerprint TEXT,
              started_at REAL NOT NULL,
              report_deadline_at REAL,
              ended_at REAL,
              UNIQUE(task_id, round_number)
            )
            """,
            """
            CREATE TABLE lifecycle_events (
              event_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(task_id),
              round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
              event_type TEXT NOT NULL,
              from_status TEXT,
              to_status TEXT,
              metadata_json TEXT NOT NULL,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE branches (
              branch_id TEXT PRIMARY KEY,
              round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
              parent_branch_id TEXT,
              agent_id TEXT NOT NULL,
              lifecycle TEXT NOT NULL,
              depth INTEGER NOT NULL,
              credit_balance REAL NOT NULL,
              generation INTEGER NOT NULL,
              latest_checkpoint_summary_id TEXT,
              terminal_summary_id TEXT,
              created_at REAL NOT NULL,
              finalized_at REAL
            )
            """,
            """
            CREATE TABLE summaries (
              summary_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(task_id),
              round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
              branch_id TEXT,
              kind TEXT NOT NULL,
              report_epoch INTEGER,
              text TEXT,
              status TEXT NOT NULL,
              error_code TEXT,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE reports (
              report_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(task_id),
              round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
              epoch INTEGER NOT NULL,
              kind TEXT NOT NULL,
              text TEXT,
              status TEXT NOT NULL,
              running_branch_count INTEGER NOT NULL,
              stats_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              UNIQUE(round_id, epoch)
            )
            """,
            """
            CREATE TABLE credit_ledger (
              ledger_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              round_id TEXT NOT NULL,
              branch_id TEXT,
              call_id TEXT,
              entry_kind TEXT NOT NULL,
              amount REAL NOT NULL,
              balance_after REAL,
              metadata_json TEXT NOT NULL,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE llm_usage (
              usage_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              round_id TEXT NOT NULL,
              branch_id TEXT,
              call_id TEXT NOT NULL,
              role TEXT NOT NULL,
              selector TEXT NOT NULL,
              estimated_model_name TEXT,
              actual_model_name TEXT,
              price_source TEXT NOT NULL,
              price_fingerprint TEXT NOT NULL,
              prompt_tokens INTEGER NOT NULL,
              completion_tokens INTEGER NOT NULL,
              cache_hit_tokens INTEGER NOT NULL,
              cache_miss_tokens INTEGER NOT NULL,
              estimated_charge REAL NOT NULL,
              actual_charge REAL,
              adjustment REAL,
              reconciliation_status TEXT NOT NULL,
              duration_ms INTEGER NOT NULL,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE procedure_calls (
              request_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              round_id TEXT NOT NULL,
              branch_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              procedure_id TEXT NOT NULL,
              provider_plugin_id TEXT NOT NULL,
              status TEXT NOT NULL,
              duration_ms INTEGER NOT NULL,
              error_code TEXT,
              provenance_json TEXT NOT NULL,
              external_cost_json TEXT,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE extension_fingerprints (
              event_id TEXT PRIMARY KEY,
              provider_plugin_id TEXT NOT NULL,
              extension_kind TEXT NOT NULL,
              fingerprint TEXT NOT NULL,
              availability TEXT NOT NULL,
              error_json TEXT,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE feedback_events (
              feedback_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              round_id TEXT NOT NULL,
              disposition TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE feedback_reminders (
              reminder_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              round_id TEXT NOT NULL,
              due_at REAL NOT NULL,
              status TEXT NOT NULL,
              triggered_at REAL,
              UNIQUE(round_id)
            )
            """,
            """
            CREATE TABLE maisaka_outbox (
              outbox_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              round_id TEXT NOT NULL,
              report_id TEXT,
              delivery_kind TEXT NOT NULL,
              idempotency_key TEXT NOT NULL UNIQUE,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL,
              attempt_count INTEGER NOT NULL DEFAULT 0,
              next_attempt_at REAL NOT NULL,
              last_error TEXT,
              created_at REAL NOT NULL,
              delivered_at REAL
            )
            """,
            """
            CREATE TABLE vector_jobs (
              job_id TEXT PRIMARY KEY,
              source_kind TEXT NOT NULL,
              source_id TEXT NOT NULL,
              generation INTEGER,
              status TEXT NOT NULL,
              attempt_count INTEGER NOT NULL DEFAULT 0,
              error_code TEXT,
              error_json TEXT,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE schema_migrations (
              version INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              applied_at REAL NOT NULL
            )
            """,
            "CREATE INDEX idx_rounds_round_status ON investigation_rounds(round_id, status)",
            "CREATE INDEX idx_lifecycle_events_task_created ON lifecycle_events(task_id, created_at)",
            "CREATE INDEX idx_summaries_task_created ON summaries(task_id, created_at)",
            "CREATE INDEX idx_summaries_branch_kind_created ON summaries(branch_id, kind, created_at)",
            "CREATE INDEX idx_vector_jobs_status ON vector_jobs(status)",
            "CREATE INDEX idx_outbox_status_next_attempt ON maisaka_outbox(status, next_attempt_at)",
        ),
    ),
)
