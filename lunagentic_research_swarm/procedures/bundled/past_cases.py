"""内置历史案例检索 Procedure：VectorIndex + SQLite feedback 真值。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from lunagentic_research_swarm.errors import (
    VECTOR_INDEX_REBUILDING,
    VECTOR_INDEX_UNAVAILABLE,
    VECTOR_REBUILD_FAILED,
)
from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult

Handler = Callable[[Any, Mapping[str, Any]], Awaitable[ProcedureResult]]

PAST_CASES_PROCEDURE_ID = "builtin.past_cases"

RERANK_FORMULA = (
    "similarity + accepted*0.15 + mixed*0.05 + outcome_confirmed*0.10 - rejected*0.20"
)

_ACCEPTED_BONUS = 0.15
_MIXED_BONUS = 0.05
_OUTCOME_CONFIRMED_BONUS = 0.10
_REJECTED_PENALTY = 0.20

_USE_AS = {
    "accepted": "success_pattern",
    "mixed": "risk_reminder",
    "rejected": "anti_pattern",
    "unreviewed": "unverified",
    "outcome_correction": "outcome_correction",
}

_VECTOR_FAIL_CODES = frozenset(
    {
        VECTOR_INDEX_REBUILDING,
        VECTOR_INDEX_UNAVAILABLE,
        VECTOR_REBUILD_FAILED,
        "embedding_generation_mismatch",
    }
)


def past_cases_procedure_definitions() -> list[ProcedureDefinition]:
    """构造 builtin.past_cases 定义。"""

    payload = {
        "procedure_id": PAST_CASES_PROCEDURE_ID,
        "version": "1",
        "display_name": "历史案例检索",
        "description": (
            "按正式任务描述检索相似历史案例；显式区分 accepted/mixed/rejected/unreviewed"
            "与 outcome correction，透明返回 rerank 分量；不把无反馈案例视为已验证成功。"
        ),
        "arguments_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 8000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "exclude_task_id": {"type": "string"},
                "created_after": {"type": "number"},
                "created_before": {"type": "number"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "cases": {"type": "array"},
                "rerank_formula": {"type": "string"},
                "truncated": {"type": "integer"},
            },
        },
        "idempotent": True,
        "timeout_seconds": 60.0,
        "external_cost_kind": "none",
        "enabled": True,
    }
    return [ProcedureDefinition.model_validate(payload)]


def _failure(code: str, message: str, *, metadata: Mapping[str, Any] | None = None) -> ProcedureResult:
    error: dict[str, Any] = {"code": code, "message": message}
    if metadata:
        error["metadata"] = dict(metadata)
    return ProcedureResult(success=False, data=None, error=error, metadata={})


def _success(data: Mapping[str, Any]) -> ProcedureResult:
    return ProcedureResult(success=True, data=dict(data), error=None, metadata={})


def _similarity_from_hit(hit: Mapping[str, Any]) -> float:
    raw = hit.get("similarity")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    distance = hit.get("_distance")
    if isinstance(distance, int | float) and not isinstance(distance, bool):
        return 1.0 / (1.0 + max(0.0, float(distance)))
    return 0.0


def _normalize_disposition(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"accepted", "accept", "approved"}:
        return "accepted"
    if text in {"mixed", "partial"}:
        return "mixed"
    if text in {"rejected", "reject", "denied"}:
        return "rejected"
    if text in {"superseded", "outcome_correction"}:
        return text
    return text


def _latest_feedback(feedback_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """取时间序上最新且未被后续 superseded 的反馈；全被 supersede 则返回 None。"""

    if not feedback_rows:
        return None
    superseded: set[str] = set()
    for row in feedback_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        parent = payload.get("supersedes_feedback_id") if isinstance(payload, Mapping) else None
        if parent:
            superseded.add(str(parent))
        if str(row.get("disposition", "")).lower() == "superseded":
            sid = row.get("feedback_id")
            if sid:
                superseded.add(str(sid))
    for row in reversed(list(feedback_rows)):
        fid = str(row.get("feedback_id") or "")
        disposition = _normalize_disposition(row.get("disposition"))
        if disposition == "superseded" or fid in superseded:
            continue
        return row
    return None


def _validation_status(feedback: Mapping[str, Any] | None) -> str:
    if feedback is None:
        return "unreviewed"
    disposition = _normalize_disposition(feedback.get("disposition"))
    if disposition in {"accepted", "mixed", "rejected"}:
        return disposition
    payload = feedback.get("payload") if isinstance(feedback.get("payload"), Mapping) else {}
    corrections = payload.get("corrections") if isinstance(payload, Mapping) else None
    outcome = payload.get("outcome") if isinstance(payload, Mapping) else None
    if (isinstance(corrections, list) and corrections) or outcome:
        return "outcome_correction"
    return "unreviewed"


def _outcome_confirmed(feedback: Mapping[str, Any] | None) -> bool:
    if feedback is None:
        return False
    payload = feedback.get("payload") if isinstance(feedback.get("payload"), Mapping) else {}
    if not isinstance(payload, Mapping):
        return False
    if payload.get("outcome_confirmed") is True:
        return True
    # 明确 outcome 且 accepted 视为 outcome_confirmed
    return bool(payload.get("outcome")) and _normalize_disposition(feedback.get("disposition")) == "accepted"


def _rerank_components(similarity: float, status: str, *, outcome_confirmed: bool) -> dict[str, float]:
    accepted_bonus = _ACCEPTED_BONUS if status == "accepted" else 0.0
    mixed_bonus = _MIXED_BONUS if status == "mixed" else 0.0
    rejected_penalty = _REJECTED_PENALTY if status == "rejected" else 0.0
    outcome_bonus = _OUTCOME_CONFIRMED_BONUS if outcome_confirmed else 0.0
    score = similarity + accepted_bonus + mixed_bonus + outcome_bonus - rejected_penalty
    return {
        "similarity": similarity,
        "accepted_bonus": accepted_bonus,
        "mixed_bonus": mixed_bonus,
        "outcome_confirmed_bonus": outcome_bonus,
        "rejected_penalty": rejected_penalty,
        "rerank_score": score,
    }


def _use_as(status: str) -> str:
    return _USE_AS.get(status, "unverified")


def _parse_args(arguments: Mapping[str, Any]) -> dict[str, Any] | ProcedureResult:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _failure("invalid_arguments", "query 必须为非空字符串")
    if len(query) > 8000:
        return _failure("invalid_arguments", "query 长度不能超过 8000")

    limit_raw = arguments.get("limit", 10)
    if isinstance(limit_raw, bool) or not isinstance(limit_raw, int):
        return _failure("invalid_arguments", "limit 必须为整数")
    if limit_raw < 1 or limit_raw > 20:
        return _failure("invalid_arguments", "limit 必须在 1..20")

    exclude = arguments.get("exclude_task_id")
    if exclude is not None and not isinstance(exclude, str):
        return _failure("invalid_arguments", "exclude_task_id 必须为字符串")

    created_after = arguments.get("created_after")
    created_before = arguments.get("created_before")
    for key, value in (("created_after", created_after), ("created_before", created_before)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            return _failure("invalid_arguments", f"{key} 必须为数字时间戳")

    return {
        "query": query.strip(),
        "limit": limit_raw,
        "exclude_task_id": exclude.strip() if isinstance(exclude, str) and exclude.strip() else None,
        "created_after": float(created_after) if created_after is not None else None,
        "created_before": float(created_before) if created_before is not None else None,
    }


async def _past_cases(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    """检索相似历史案例并按 feedback 真值透明 rerank。"""

    parsed = _parse_args(arguments)
    if isinstance(parsed, ProcedureResult):
        return parsed

    vector_index = getattr(ctx, "vector_index", None)
    store = getattr(ctx, "store", None)
    # BundledProcedureProvider 把依赖挂在自身，handler 的 ctx 是 provider 代理
    if vector_index is None or store is None:
        return _failure(VECTOR_INDEX_UNAVAILABLE, "历史案例索引或状态库不可用")

    search_limit = min(50, int(parsed["limit"]) * 5)
    search_result = await vector_index.search(parsed["query"], limit=search_limit)
    if not search_result.success:
        error = search_result.error
        code = error.code if error is not None else VECTOR_INDEX_UNAVAILABLE
        message = error.message if error is not None else "向量检索失败"
        metadata = error.metadata if error is not None else None
        if code not in _VECTOR_FAIL_CODES:
            code = VECTOR_INDEX_UNAVAILABLE
        return _failure(code, message, metadata=metadata)

    hits = []
    if search_result.data and isinstance(search_result.data.get("hits"), list):
        hits = list(search_result.data["hits"])

    # 按 task 聚合：保留最高 similarity，合并 source_ids
    by_task: dict[str, dict[str, Any]] = {}
    for hit in hits:
        if not isinstance(hit, Mapping):
            continue
        task_id = hit.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        if parsed["exclude_task_id"] and task_id == parsed["exclude_task_id"]:
            continue
        similarity = _similarity_from_hit(hit)
        source_id = str(hit.get("source_id") or "")
        bucket = by_task.get(task_id)
        if bucket is None:
            by_task[task_id] = {
                "task_id": task_id,
                "similarity": similarity,
                "source_ids": [source_id] if source_id else [],
                "hit_texts": [str(hit.get("text") or "")],
            }
        else:
            if similarity > bucket["similarity"]:
                bucket["similarity"] = similarity
            if source_id and source_id not in bucket["source_ids"]:
                bucket["source_ids"].append(source_id)
            text = str(hit.get("text") or "")
            if text and text not in bucket["hit_texts"]:
                bucket["hit_texts"].append(text)

    cases: list[dict[str, Any]] = []
    for task_id, bucket in by_task.items():
        layer = await store.load_summary_layer(task_id)
        if layer is None:
            continue
        created_at = None
        task_row = await store.load_task(task_id)
        if task_row is not None:
            created_at = float(task_row.created_at)
        if parsed["created_after"] is not None and created_at is not None and created_at < parsed["created_after"]:
            continue
        if parsed["created_before"] is not None and created_at is not None and created_at > parsed["created_before"]:
            continue

        feedback = _latest_feedback(layer.feedback)
        status = _validation_status(feedback)
        confirmed = _outcome_confirmed(feedback)
        comps = _rerank_components(float(bucket["similarity"]), status, outcome_confirmed=confirmed)

        formalized = ""
        if layer.formalized_task is not None:
            formalized = str(layer.formalized_task.text)

        payload = feedback.get("payload") if feedback and isinstance(feedback.get("payload"), Mapping) else {}
        corrections = list(payload.get("corrections") or []) if isinstance(payload, Mapping) else []
        outcome = payload.get("outcome") if isinstance(payload, Mapping) else None

        fingerprints: dict[str, Any] = {}
        # 报告 stats / 生命周期里若有指纹则透传
        for report in layer.reports:
            stats = report.get("stats") if isinstance(report, Mapping) else None
            if isinstance(stats, Mapping):
                for key in ("agent_fingerprint", "model_fingerprint", "procedure_fingerprint"):
                    if key in stats and key not in fingerprints:
                        fingerprints[key] = stats[key]

        cases.append(
            {
                "task_id": task_id,
                "validation_status": status,
                "use_as": _use_as(status),
                "source_ids": list(bucket["source_ids"]),
                "formalized_summary": formalized,
                "summaries": [
                    {
                        "summary_id": row.get("summary_id"),
                        "kind": row.get("kind"),
                        "text": row.get("text"),
                        "status": row.get("status"),
                    }
                    for row in layer.summaries
                ],
                "reports": [
                    {
                        "report_id": row.get("report_id"),
                        "kind": row.get("kind"),
                        "text": row.get("text"),
                        "status": row.get("status"),
                    }
                    for row in layer.reports
                ],
                "feedback_disposition": (
                    _normalize_disposition(feedback.get("disposition")) if feedback else None
                ),
                "corrections": corrections,
                "outcome": outcome,
                "outcome_confirmed": confirmed,
                "similarity": comps["similarity"],
                "rerank_score": comps["rerank_score"],
                "rerank_components": {
                    "similarity": comps["similarity"],
                    "accepted_bonus": comps["accepted_bonus"],
                    "mixed_bonus": comps["mixed_bonus"],
                    "outcome_confirmed_bonus": comps["outcome_confirmed_bonus"],
                    "rejected_penalty": comps["rejected_penalty"],
                },
                "fingerprints": fingerprints,
                "created_at": created_at,
            }
        )

    cases.sort(key=lambda item: (-float(item["rerank_score"]), str(item["task_id"])))
    limited = cases[: int(parsed["limit"])]
    return _success(
        {
            "cases": limited,
            "rerank_formula": RERANK_FORMULA,
            "truncated": len(limited),
        }
    )


def make_past_cases_handler(
    *,
    store: Any | None = None,
    vector_index: Any | None = None,
) -> Handler:
    """绑定 store / VectorIndex 的 handler；运行时可经 provider 更新依赖。"""

    deps = {"store": store, "vector_index": vector_index}

    async def _handler(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
        class _Deps:
            store = deps["store"]
            vector_index = deps["vector_index"]

        # 允许 provider 在装载后通过闭包 deps 热绑定
        return await _past_cases(_Deps(), arguments)

    _handler._past_cases_deps = deps  # type: ignore[attr-defined]
    return _handler


PAST_CASES_HANDLERS: dict[str, Handler] = {
    PAST_CASES_PROCEDURE_ID: make_past_cases_handler(),
}
