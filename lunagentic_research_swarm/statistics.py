"""从权威 SQLite 账本确定性重算 Task / Plugin 统计。"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from lunagentic_research_swarm.runtime.credits import is_summarizer_role


def _cache_hit_rate(hit: int, miss: int) -> float | None:
    total = hit + miss
    if total <= 0:
        return None
    return hit / total


def _external_cost(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return 0.0
    if isinstance(value, Mapping):
        for key in ("credits", "amount", "cost", "external_cost"):
            item = value.get(key)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                return float(item)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _is_reserved_status(status: str) -> bool:
    return str(status or "") == "reserved"


def _usage_rows_for_stats(usage_rows: Sequence[Any]) -> list[Any]:
    """每个 call_id 优先取非 reserved；仅当无核销行时保留 orphan reserved。"""

    by_call: dict[str, list[Any]] = defaultdict(list)
    for row in usage_rows:
        by_call[str(row["call_id"] or "")].append(row)
    selected: list[Any] = []
    for rows in by_call.values():
        non_reserved = [row for row in rows if not _is_reserved_status(str(row["reconciliation_status"] or ""))]
        selected.extend(non_reserved if non_reserved else rows)
    return selected


def _credits_from_ledger(connection: sqlite3.Connection, task_id: str) -> tuple[float, float, float] | None:
    """从 ``credit_ledger`` 汇总研究积分；无行时返回 None（回退到 usage）。

    - ``estimated``：各 call 的 ``input_reservation`` 绝对值之和
    - ``actual``：已核销 call 的净支出 ``-sum(amounts)``
    - ``unreconciled``：仅有 reservation、尚无 reconciliation 的 call（含 crash orphan）
    """

    rows = connection.execute(
        """
        SELECT call_id, entry_kind, amount
        FROM credit_ledger WHERE task_id = ?
        """,
        (task_id,),
    ).fetchall()
    if not rows:
        return None
    by_call: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_call[str(row["call_id"] or "")].append(row)
    estimated = actual = unreconciled = 0.0
    for entries in by_call.values():
        reservation = sum(
            -float(entry["amount"] or 0.0)
            for entry in entries
            if str(entry["entry_kind"] or "") == "input_reservation"
        )
        has_recon = any(str(entry["entry_kind"] or "") == "input_reconciliation" for entry in entries)
        estimated += reservation
        if has_recon:
            actual += -sum(float(entry["amount"] or 0.0) for entry in entries)
        else:
            unreconciled += reservation
    return estimated, actual, unreconciled


def compute_task_stats(connection: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    """纯函数：从已打开的权威连接重算 task 统计（可嵌入同一 transaction）。"""

    task_id = str(task_id)
    usage_rows = connection.execute(
        """
        SELECT role, call_id, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens,
               estimated_charge, actual_charge, reconciliation_status, duration_ms
        FROM llm_usage WHERE task_id = ?
        """,
        (task_id,),
    ).fetchall()
    agent_calls = 0
    summarizer_calls = 0
    prompt = completion = hit = miss = 0
    estimated = actual = unreconciled = cost_equivalent = 0.0
    duration_ms = 0
    for row in _usage_rows_for_stats(usage_rows):
        status = str(row["reconciliation_status"] or "")
        role = str(row["role"] or "")
        prompt += int(row["prompt_tokens"] or 0)
        completion += int(row["completion_tokens"] or 0)
        hit += int(row["cache_hit_tokens"] or 0)
        miss += int(row["cache_miss_tokens"] or 0)
        duration_ms += int(row["duration_ms"] or 0)
        est = float(row["estimated_charge"] or 0.0)
        act = row["actual_charge"]
        if is_summarizer_role(role):
            summarizer_calls += 1
            cost_equivalent += float(act if act is not None else est)
        else:
            agent_calls += 1
            estimated += est
            if status == "estimated_unreconciled" or _is_reserved_status(status):
                unreconciled += est
            elif act is not None:
                actual += float(act)

    ledger_credits = _credits_from_ledger(connection, task_id)
    if ledger_credits is not None:
        estimated, actual, unreconciled = ledger_credits

    procedure_rows = connection.execute(
        """
        SELECT procedure_id, status, duration_ms, error_code, external_cost_json
        FROM procedure_calls WHERE task_id = ?
        """,
        (task_id,),
    ).fetchall()
    procedures_success = procedures_error = 0
    compact_count = 0
    external_cost = 0.0
    for row in procedure_rows:
        procedure_id = str(row["procedure_id"] or "")
        status = str(row["status"] or "").upper()
        duration_ms += int(row["duration_ms"] or 0)
        external_cost += _external_cost(row["external_cost_json"])
        if procedure_id == "core.compact":
            compact_count += 1
        if status in {"SUCCEEDED", "SUCCESS", "OK"}:
            procedures_success += 1
        elif status in {"ERROR", "FAILED", "FAILURE"} or row["error_code"]:
            procedures_error += 1

    branch_rows = connection.execute(
        """
        SELECT b.lifecycle, b.depth
        FROM branches b
        JOIN investigation_rounds r ON r.round_id = b.round_id
        WHERE r.task_id = ?
        """,
        (task_id,),
    ).fetchall()
    branches_total = len(branch_rows)
    branches_finalized = sum(1 for row in branch_rows if str(row["lifecycle"]) == "FINALIZED")
    branches_active = branches_total - branches_finalized
    max_depth = max((int(row["depth"] or 0) for row in branch_rows), default=0)

    pool_row = connection.execute(
        """
        SELECT credit_pool FROM investigation_rounds
        WHERE task_id = ?
        ORDER BY round_number DESC, started_at DESC
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    credit_pool = float(pool_row["credit_pool"]) if pool_row is not None else 0.0
    credit_debt = abs(min(0.0, credit_pool))

    # checkpoint 由 ReportCoordinator 直接调用总结器，不经过 procedure_calls；
    # 权威计数来自 summaries 表，否则 core.checkpoint 行数恒为 0。
    checkpoint_count = int(
        connection.execute(
            """
            SELECT COUNT(*) AS n FROM summaries
            WHERE task_id = ? AND kind = 'CHECKPOINT'
            """,
            (task_id,),
        ).fetchone()["n"]
    )

    # 协议纠错：reducer 以 ``{call_id}:correction`` 再 reserve；不依赖虚构的 lifecycle 事件名。
    protocol_correction_count = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT call_id) AS n FROM llm_usage
            WHERE task_id = ? AND call_id LIKE '%:correction'
            """,
            (task_id,),
        ).fetchone()["n"]
    )
    continue_count = int(
        connection.execute(
            """
            SELECT COUNT(*) AS n FROM lifecycle_events
            WHERE task_id = ? AND event_type = 'ContinueRequested'
            """,
            (task_id,),
        ).fetchone()["n"]
    )
    error_count = procedures_error + int(
        connection.execute(
            """
            SELECT COUNT(*) AS n FROM llm_usage
            WHERE task_id = ? AND reconciliation_status = 'estimated_unreconciled'
            """,
            (task_id,),
        ).fetchone()["n"]
    )

    return {
        "task_id": task_id,
        "agent_calls": agent_calls,
        "summarizer_calls": summarizer_calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_hit_rate": _cache_hit_rate(hit, miss),
        "estimated_credits": estimated,
        "actual_credits": actual,
        "unreconciled_credits": unreconciled,
        "cost_equivalent_credits": cost_equivalent,
        "credit_pool": credit_pool,
        "credit_debt": credit_debt,
        "branches_total": branches_total,
        "branches_active": branches_active,
        "branches_finalized": branches_finalized,
        "max_branch_depth": max_depth,
        "compact_count": compact_count,
        "checkpoint_count": checkpoint_count,
        "protocol_correction_count": protocol_correction_count,
        "continue_count": continue_count,
        "procedures_success": procedures_success,
        "procedures_error": procedures_error,
        "external_cost_credits": external_cost,
        "duration_ms_total": duration_ms,
        "error_count": error_count,
    }


class StatisticsService:
    """``task`` / ``plugin`` / ``cache``：只读权威库，永不读 debug raw。"""

    def __init__(self, store: Any) -> None:
        self.store = store

    async def task(self, task_id: str) -> dict[str, Any]:
        task_id = str(task_id)
        return await self.store.run_locked(lambda connection: compute_task_stats(connection, task_id))

    async def cache(self, task_id: str) -> dict[str, Any]:
        stats = await self.task(task_id)
        return {
            "hit": int(stats["cache_hit_tokens"]),
            "miss": int(stats["cache_miss_tokens"]),
            "hit_rate": stats["cache_hit_rate"],
        }

    async def plugin(self) -> dict[str, Any]:
        return await self.store.run_locked(self._compute_plugin)

    def _compute_task(self, connection: sqlite3.Connection, task_id: str) -> dict[str, Any]:
        return compute_task_stats(connection, task_id)

    def _compute_plugin(self, connection: sqlite3.Connection) -> dict[str, Any]:
        models: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 0,
                "actual_credits": 0.0,
                "cost_equivalent_credits": 0.0,
            }
        )
        usage_rows = connection.execute(
            """
            SELECT role, call_id, actual_model_name, estimated_model_name, prompt_tokens, completion_tokens,
                   cache_hit_tokens, cache_miss_tokens, actual_charge, estimated_charge,
                   reconciliation_status
            FROM llm_usage
            """
        ).fetchall()
        for row in _usage_rows_for_stats(usage_rows):
            name = str(row["actual_model_name"] or row["estimated_model_name"] or "unknown")
            bucket = models[name]
            bucket["calls"] += 1
            bucket["prompt_tokens"] += int(row["prompt_tokens"] or 0)
            bucket["completion_tokens"] += int(row["completion_tokens"] or 0)
            bucket["cache_hit_tokens"] += int(row["cache_hit_tokens"] or 0)
            bucket["cache_miss_tokens"] += int(row["cache_miss_tokens"] or 0)
            charge = float(row["actual_charge"] if row["actual_charge"] is not None else row["estimated_charge"] or 0.0)
            role = str(row["role"] or "")
            if is_summarizer_role(role):
                bucket["cost_equivalent_credits"] += charge
            else:
                bucket["actual_credits"] += charge

        agents: dict[str, dict[str, Any]] = defaultdict(lambda: {"branches": 0, "finalized": 0})
        for row in connection.execute("SELECT agent_id, lifecycle FROM branches"):
            agent_id = str(row["agent_id"] or "unknown")
            agents[agent_id]["branches"] += 1
            if str(row["lifecycle"]) == "FINALIZED":
                agents[agent_id]["finalized"] += 1

        procedures: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "success": 0, "error": 0, "external_cost_credits": 0.0}
        )
        for row in connection.execute(
            "SELECT procedure_id, status, error_code, external_cost_json FROM procedure_calls"
        ):
            procedure_id = str(row["procedure_id"] or "unknown")
            bucket = procedures[procedure_id]
            bucket["calls"] += 1
            bucket["external_cost_credits"] += _external_cost(row["external_cost_json"])
            status = str(row["status"] or "").upper()
            if status in {"SUCCEEDED", "SUCCESS", "OK"}:
                bucket["success"] += 1
            elif status in {"ERROR", "FAILED", "FAILURE"} or row["error_code"]:
                bucket["error"] += 1

        tasks: dict[str, dict[str, Any]] = {}
        for row in connection.execute("SELECT task_id FROM tasks ORDER BY created_at, task_id"):
            task_id = str(row["task_id"])
            tasks[task_id] = compute_task_stats(connection, task_id)

        return {
            "models": dict(models),
            "agents": dict(agents),
            "procedures": dict(procedures),
            "tasks": tasks,
        }


__all__ = ["StatisticsService", "compute_task_stats"]
