"""以单连接和显式 transaction 实现 SQLite 权威状态存储。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.models import FormalizedTask, TaskStatus
from lunagentic_research_swarm.storage.migrations import MIGRATIONS


def _freeze_json(value: Any) -> Any:
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class StoreCommand:
    """一次状态转移中的不可变数据库命令。"""

    kind: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_json(self.values))


@dataclass(frozen=True, slots=True)
class StoredRound:
    round_id: str
    task_id: str
    round_number: int
    generation: int
    status: TaskStatus
    time_budget_seconds: int
    grace_period_seconds: int
    credit_pool: float
    catalog_fingerprint: str | None
    started_at: float
    report_deadline_at: float | None
    ended_at: float | None


@dataclass(frozen=True, slots=True)
class StoredTask:
    task_id: str
    stream_id: str
    formalized_task: FormalizedTask | None
    current_round_number: int
    current_round: StoredRound | None
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class SummaryLayer:
    task_id: str
    formalized_task: FormalizedTask | None
    summaries: tuple[Mapping[str, Any], ...]
    reports: tuple[Mapping[str, Any], ...]
    feedback: tuple[Mapping[str, Any], ...]
    supplied_context: tuple[str, ...]


ACTIVE_STATUSES = (
    TaskStatus.FORMALIZING,
    TaskStatus.RUNNING,
    TaskStatus.REPORTING,
    TaskStatus.PAUSING,
    TaskStatus.PAUSED,
    TaskStatus.FINALIZING,
)


class MissingStoreTargetError(LookupError):
    """单目标状态 mutation 未匹配到唯一权威记录。"""


def _require_single_target(cursor: sqlite3.Cursor, *, target_kind: str, target_id: Any) -> None:
    if cursor.rowcount != 1:
        raise MissingStoreTargetError(f"{target_kind} {target_id} 不存在或不唯一")


class SQLiteStateStore:
    """所有数据库访问经同一个 asyncio lock 串行提交到工作线程。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._command_handlers: Mapping[str, Callable[[sqlite3.Connection, Mapping[str, Any]], None]] = (
            MappingProxyType(
                {
                    "insert_task": self._insert_task,
                    "insert_round": self._insert_round,
                    "insert_lifecycle_event": self._insert_lifecycle_event,
                    "insert_branch": self._insert_branch,
                    "insert_summary": self._insert_summary,
                    "insert_report": self._insert_report,
                    "insert_credit_ledger": self._insert_credit_ledger,
                    "insert_llm_usage": self._insert_llm_usage,
                    "insert_procedure_call": self._insert_procedure_call,
                    "insert_extension_fingerprint": self._insert_extension_fingerprint,
                    "insert_feedback_event": self._insert_feedback_event,
                    "insert_feedback_reminder": self._insert_feedback_reminder,
                    "insert_outbox": self._insert_outbox,
                    "insert_vector_job": self._insert_vector_job,
                    "update_task_formalization": self._update_task_formalization,
                    "set_task_current_round": self._set_task_current_round,
                    "update_round_status": self._update_round_status,
                    "update_round_generation": self._update_round_generation,
                    "update_round_continuation": self._update_round_continuation,
                    "update_branch_balance": self._update_branch_balance,
                }
            )
        )

    async def _call(self, function: Callable[..., Any], *args: Any) -> Any:
        async with self._lock:
            operation = asyncio.create_task(asyncio.to_thread(function, *args))
            cancellation: asyncio.CancelledError | None = None
            while not operation.done():
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError as exc:
                    # worker 无法被取消；必须继续持锁直到它退出，避免同一连接并发使用。
                    if cancellation is None:
                        cancellation = exc
            result = operation.result()
            if cancellation is not None:
                raise cancellation
            return result

    async def open(self) -> None:
        await self._call(self._open_sync)

    async def close(self) -> None:
        await self._call(self._close_sync)

    async def transact(self, commands: Sequence[StoreCommand]) -> None:
        await self._call(self._transact_sync, tuple(commands))

    async def load_task(self, task_id: str) -> StoredTask | None:
        return await self._call(self._load_task_sync, task_id)

    async def list_active_rounds(self) -> tuple[StoredRound, ...]:
        return await self._call(self._list_active_rounds_sync)

    async def load_summary_layer(self, task_id: str) -> SummaryLayer | None:
        return await self._call(self._load_summary_layer_sync, task_id)

    async def mark_active_rounds_interrupted(self, now: float) -> int:
        return await self._call(self._mark_active_rounds_interrupted_sync, now)

    async def table_names(self) -> tuple[str, ...]:
        return await self._call(self._table_names_sync)

    async def pragma_settings(self) -> dict[str, int | str]:
        return await self._call(self._pragma_settings_sync)

    def _open_sync(self) -> None:
        if self._connection is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            connection.execute("PRAGMA synchronous=FULL")
            if str(journal_mode).lower() != "wal":
                raise RuntimeError("无法启用 SQLite WAL 模式")
            self._apply_migrations(connection)
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def _close_sync(self) -> None:
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None

    @staticmethod
    def _apply_migrations(connection: sqlite3.Connection) -> None:
        has_migration_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone()
        applied_rows = (
            connection.execute("SELECT version, name FROM schema_migrations ORDER BY version").fetchall()
            if has_migration_table
            else ()
        )
        known = {migration.version: migration.name for migration in MIGRATIONS}
        applied: set[int] = set()
        for row in applied_rows:
            version = int(row[0])
            name = str(row[1])
            if known.get(version) != name:
                raise RuntimeError(f"数据库迁移版本不受支持：{version}/{name}")
            applied.add(version)

        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, time.time()),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("SQLite 状态存储尚未打开")
        return self._connection

    def _transact_sync(self, commands: tuple[StoreCommand, ...]) -> None:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            for command in commands:
                if not isinstance(command, StoreCommand):
                    raise TypeError("存储 transaction 只接受 StoreCommand")
                handler = self._command_handlers.get(command.kind)
                if handler is None:
                    raise ValueError(f"未知存储命令：{command.kind}")
                handler(connection, command.values)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    def _insert_task(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        created_at = values["created_at"]
        connection.execute(
            """
            INSERT INTO tasks(
                task_id, stream_id, formalized_text, formalized_sha256,
                current_round_number, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["task_id"],
                values["stream_id"],
                values.get("formalized_text"),
                values.get("formalized_sha256"),
                values.get("current_round_number", 1),
                created_at,
                values.get("updated_at", created_at),
            ),
        )

    @staticmethod
    def _insert_round(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO investigation_rounds(
                round_id, task_id, round_number, generation, status,
                time_budget_seconds, grace_period_seconds, credit_pool,
                catalog_fingerprint, started_at, report_deadline_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["round_id"],
                values["task_id"],
                values["round_number"],
                values["generation"],
                values["status"],
                values["time_budget_seconds"],
                values.get("grace_period_seconds", 60),
                values["credit_pool"],
                values.get("catalog_fingerprint"),
                values["started_at"],
                values.get("report_deadline_at"),
                values.get("ended_at"),
            ),
        )

    @staticmethod
    def _insert_lifecycle_event(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO lifecycle_events(
                event_id, task_id, round_id, event_type, from_status,
                to_status, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["event_id"],
                values["task_id"],
                values["round_id"],
                values["event_type"],
                values.get("from_status"),
                values.get("to_status"),
                values["metadata_json"],
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_branch(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO branches(
                branch_id, round_id, parent_branch_id, agent_id, lifecycle, depth,
                credit_balance, generation, latest_checkpoint_summary_id,
                terminal_summary_id, created_at, finalized_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["branch_id"],
                values["round_id"],
                values.get("parent_branch_id"),
                values["agent_id"],
                values["lifecycle"],
                values["depth"],
                values["credit_balance"],
                values["generation"],
                values.get("latest_checkpoint_summary_id"),
                values.get("terminal_summary_id"),
                values["created_at"],
                values.get("finalized_at"),
            ),
        )

    @staticmethod
    def _insert_summary(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO summaries(
                summary_id, task_id, round_id, branch_id, kind, report_epoch,
                text, status, error_code, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["summary_id"],
                values["task_id"],
                values["round_id"],
                values.get("branch_id"),
                values["kind"],
                values.get("report_epoch"),
                values.get("text"),
                values["status"],
                values.get("error_code"),
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_report(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO reports(
                report_id, task_id, round_id, epoch, kind, text, status,
                running_branch_count, stats_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["report_id"],
                values["task_id"],
                values["round_id"],
                values["epoch"],
                values["kind"],
                values.get("text"),
                values["status"],
                values["running_branch_count"],
                values["stats_json"],
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_credit_ledger(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO credit_ledger(
                ledger_id, task_id, round_id, branch_id, call_id, entry_kind,
                amount, balance_after, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["ledger_id"],
                values["task_id"],
                values["round_id"],
                values.get("branch_id"),
                values.get("call_id"),
                values["entry_kind"],
                values["amount"],
                values.get("balance_after"),
                values["metadata_json"],
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_llm_usage(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO llm_usage(
                usage_id, task_id, round_id, branch_id, call_id, role, selector,
                estimated_model_name, actual_model_name, price_source, price_fingerprint,
                prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens,
                estimated_charge, actual_charge, adjustment, reconciliation_status,
                duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["usage_id"],
                values["task_id"],
                values["round_id"],
                values.get("branch_id"),
                values["call_id"],
                values["role"],
                values["selector"],
                values.get("estimated_model_name"),
                values.get("actual_model_name"),
                values["price_source"],
                values["price_fingerprint"],
                values["prompt_tokens"],
                values["completion_tokens"],
                values["cache_hit_tokens"],
                values["cache_miss_tokens"],
                values["estimated_charge"],
                values.get("actual_charge"),
                values.get("adjustment"),
                values["reconciliation_status"],
                values["duration_ms"],
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_procedure_call(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO procedure_calls(
                request_id, task_id, round_id, branch_id, turn_id, agent_id,
                procedure_id, provider_plugin_id, status, duration_ms, error_code,
                provenance_json, external_cost_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["request_id"],
                values["task_id"],
                values["round_id"],
                values["branch_id"],
                values["turn_id"],
                values["agent_id"],
                values["procedure_id"],
                values["provider_plugin_id"],
                values["status"],
                values["duration_ms"],
                values.get("error_code"),
                values["provenance_json"],
                values.get("external_cost_json"),
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_extension_fingerprint(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO extension_fingerprints(
                event_id, provider_plugin_id, extension_kind, fingerprint,
                availability, error_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["event_id"],
                values["provider_plugin_id"],
                values["extension_kind"],
                values["fingerprint"],
                values["availability"],
                values.get("error_json"),
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_feedback_event(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO feedback_events(
                feedback_id, task_id, round_id, disposition, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                values["feedback_id"],
                values["task_id"],
                values["round_id"],
                values["disposition"],
                values["payload_json"],
                values["created_at"],
            ),
        )

    @staticmethod
    def _insert_feedback_reminder(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO feedback_reminders(
                reminder_id, task_id, round_id, due_at, status, triggered_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                values["reminder_id"],
                values["task_id"],
                values["round_id"],
                values["due_at"],
                values["status"],
                values.get("triggered_at"),
            ),
        )

    @staticmethod
    def _insert_outbox(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO maisaka_outbox(
                outbox_id, task_id, round_id, report_id, delivery_kind,
                idempotency_key, payload_json, status, attempt_count,
                next_attempt_at, last_error, created_at, delivered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["outbox_id"],
                values["task_id"],
                values["round_id"],
                values.get("report_id"),
                values["delivery_kind"],
                values["idempotency_key"],
                values["payload_json"],
                values["status"],
                values.get("attempt_count", 0),
                values["next_attempt_at"],
                values.get("last_error"),
                values["created_at"],
                values.get("delivered_at"),
            ),
        )

    @staticmethod
    def _insert_vector_job(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        created_at = values["created_at"]
        connection.execute(
            """
            INSERT INTO vector_jobs(
                job_id, source_kind, source_id, generation, status, attempt_count,
                error_code, error_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["job_id"],
                values["source_kind"],
                values["source_id"],
                values.get("generation"),
                values["status"],
                values.get("attempt_count", 0),
                values.get("error_code"),
                values.get("error_json"),
                created_at,
                values.get("updated_at", created_at),
            ),
        )

    @staticmethod
    def _update_task_formalization(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        cursor = connection.execute(
            """
            UPDATE tasks
            SET formalized_text = ?, formalized_sha256 = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (values["formalized_text"], values["formalized_sha256"], values["updated_at"], values["task_id"]),
        )
        _require_single_target(cursor, target_kind="Task", target_id=values["task_id"])

    @staticmethod
    def _set_task_current_round(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        cursor = connection.execute(
            """
            UPDATE tasks SET current_round_number = ?, updated_at = ? WHERE task_id = ?
            """,
            (values["current_round_number"], values["updated_at"], values["task_id"]),
        )
        _require_single_target(cursor, target_kind="Task", target_id=values["task_id"])

    @staticmethod
    def _update_round_status(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        cursor = connection.execute(
            """
            UPDATE investigation_rounds
            SET status = ?, report_deadline_at = ?, ended_at = ?
            WHERE round_id = ?
            """,
            (
                values["status"],
                values.get("report_deadline_at"),
                values.get("ended_at"),
                values["round_id"],
            ),
        )
        _require_single_target(cursor, target_kind="Round", target_id=values["round_id"])

    @staticmethod
    def _update_round_generation(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        """持久化 stop 产生的新 generation，令重启后迟到结果仍会被拒绝。"""

        cursor = connection.execute(
            "UPDATE investigation_rounds SET generation = ? WHERE round_id = ?",
            (values["generation"], values["round_id"]),
        )
        _require_single_target(cursor, target_kind="Round", target_id=values["round_id"])

    @staticmethod
    def _update_round_continuation(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        """Persist the pause-barrier budget/pool change in the same transaction."""

        cursor = connection.execute(
            """
            UPDATE investigation_rounds
            SET credit_pool = ?, time_budget_seconds = ?, report_deadline_at = ?
            WHERE round_id = ?
            """,
            (
                values["credit_pool"],
                values["time_budget_seconds"],
                values.get("report_deadline_at"),
                values["round_id"],
            ),
        )
        _require_single_target(cursor, target_kind="Round", target_id=values["round_id"])

    @staticmethod
    def _update_branch_balance(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
        cursor = connection.execute(
            "UPDATE branches SET credit_balance = ?, lifecycle = ? WHERE branch_id = ?",
            (values["credit_balance"], values["lifecycle"], values["branch_id"]),
        )
        _require_single_target(cursor, target_kind="Branch", target_id=values["branch_id"])

    @staticmethod
    def _row_to_round(row: sqlite3.Row) -> StoredRound:
        return StoredRound(
            round_id=row["round_id"],
            task_id=row["round_task_id"] if "round_task_id" in row.keys() else row["task_id"],
            round_number=row["round_number"],
            generation=row["generation"],
            status=TaskStatus(row["status"]),
            time_budget_seconds=row["time_budget_seconds"],
            grace_period_seconds=row["grace_period_seconds"],
            credit_pool=row["credit_pool"],
            catalog_fingerprint=row["catalog_fingerprint"],
            started_at=row["started_at"],
            report_deadline_at=row["report_deadline_at"],
            ended_at=row["ended_at"],
        )

    @staticmethod
    def _formalized_task(text: Any, sha256: Any) -> FormalizedTask | None:
        if text is None and sha256 is None:
            return None
        if not isinstance(text, str) or not isinstance(sha256, str):
            raise ValueError("权威库中的正式任务字段不完整")
        return FormalizedTask(text=text, sha256=sha256)

    def _load_task_sync(self, task_id: str) -> StoredTask | None:
        connection = self._require_connection()
        row = connection.execute(
            """
            SELECT
                t.task_id, t.stream_id, t.formalized_text, t.formalized_sha256,
                t.current_round_number, t.created_at, t.updated_at,
                r.round_id, r.task_id AS round_task_id, r.round_number, r.generation,
                r.status, r.time_budget_seconds, r.grace_period_seconds,
                r.credit_pool, r.catalog_fingerprint, r.started_at,
                r.report_deadline_at, r.ended_at
            FROM tasks AS t
            LEFT JOIN investigation_rounds AS r
              ON r.task_id = t.task_id AND r.round_number = t.current_round_number
            WHERE t.task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        current_round = self._row_to_round(row) if row["round_id"] is not None else None
        return StoredTask(
            task_id=row["task_id"],
            stream_id=row["stream_id"],
            formalized_task=self._formalized_task(row["formalized_text"], row["formalized_sha256"]),
            current_round_number=row["current_round_number"],
            current_round=current_round,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _list_active_rounds_sync(self) -> tuple[StoredRound, ...]:
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT
                round_id, task_id, round_number, generation, status,
                time_budget_seconds, grace_period_seconds, credit_pool,
                catalog_fingerprint, started_at, report_deadline_at, ended_at
            FROM investigation_rounds
            WHERE status IN (?, ?, ?, ?, ?, ?)
            ORDER BY started_at, round_id
            """,
            tuple(status.value for status in ACTIVE_STATUSES),
        ).fetchall()
        return tuple(self._row_to_round(row) for row in rows)

    def _load_summary_layer_sync(self, task_id: str) -> SummaryLayer | None:
        connection = self._require_connection()
        task = connection.execute(
            "SELECT formalized_text, formalized_sha256 FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if task is None:
            return None
        summary_rows = connection.execute(
            """
            SELECT summary_id, task_id, round_id, branch_id, kind, report_epoch,
                   text, status, error_code, created_at
            FROM summaries WHERE task_id = ? ORDER BY created_at, summary_id
            """,
            (task_id,),
        ).fetchall()
        report_rows = connection.execute(
            """
            SELECT report_id, task_id, round_id, epoch, kind, text, status,
                   running_branch_count, stats_json, created_at
            FROM reports WHERE task_id = ? ORDER BY created_at, report_id
            """,
            (task_id,),
        ).fetchall()
        feedback_rows = connection.execute(
            """
            SELECT feedback_id, task_id, round_id, disposition, payload_json, created_at
            FROM feedback_events WHERE task_id = ? ORDER BY created_at, feedback_id
            """,
            (task_id,),
        ).fetchall()
        context_rows = connection.execute(
            """
            SELECT metadata_json FROM lifecycle_events
            WHERE task_id = ? AND event_type = 'ContextSupplied'
            ORDER BY created_at, event_id
            """,
            (task_id,),
        ).fetchall()

        summaries = tuple(MappingProxyType(dict(row)) for row in summary_rows)
        reports = []
        for row in report_rows:
            item = dict(row)
            item["stats"] = _freeze_json(json.loads(item.pop("stats_json")))
            reports.append(MappingProxyType(item))
        feedback = []
        for row in feedback_rows:
            item = dict(row)
            item["payload"] = _freeze_json(json.loads(item.pop("payload_json")))
            feedback.append(MappingProxyType(item))
        supplied_context = []
        for row in context_rows:
            metadata = json.loads(row["metadata_json"])
            context = metadata.get("context") if isinstance(metadata, dict) else None
            if not isinstance(context, str):
                raise ValueError("ContextSupplied lifecycle event 缺少 context 字符串")
            supplied_context.append(context)
        return SummaryLayer(
            task_id=task_id,
            formalized_task=self._formalized_task(task["formalized_text"], task["formalized_sha256"]),
            summaries=summaries,
            reports=tuple(reports),
            feedback=tuple(feedback),
            supplied_context=tuple(supplied_context),
        )

    def _mark_active_rounds_interrupted_sync(self, now: float) -> int:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                """
                UPDATE investigation_rounds
                SET status = ?, ended_at = ?
                WHERE status IN (?, ?, ?, ?, ?, ?)
                """,
                (
                    TaskStatus.INTERRUPTED.value,
                    now,
                    *(status.value for status in ACTIVE_STATUSES),
                ),
            )
            connection.commit()
            return cursor.rowcount
        except BaseException:
            connection.rollback()
            raise

    def _table_names_sync(self) -> tuple[str, ...]:
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        return tuple(row["name"] for row in rows)

    def _pragma_settings_sync(self) -> dict[str, int | str]:
        connection = self._require_connection()
        return {
            "foreign_keys": int(connection.execute("PRAGMA foreign_keys").fetchone()[0]),
            "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            "synchronous": int(connection.execute("PRAGMA synchronous").fetchone()[0]),
        }
