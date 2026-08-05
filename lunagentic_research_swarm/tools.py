"""Planner-facing schemas and result helpers for research task controls."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from lunagentic_research_swarm.models import TaskStatus


START_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {"type": "string", "minLength": 1, "description": "要调查的问题、案例或研究主题。"},
        "time_budget_seconds": {"type": "integer", "minimum": 1, "description": "期望报告间隔（秒）。"},
        "effort_level": {"type": "number", "minimum": 0, "default": 1.0, "description": "相对于默认 credits 的倍率。"},
        "planner_context": {"type": "string", "description": "Planner 提供的补充上下文。"},
    },
    "required": ["objective"],
    # ``stream_id`` is injected by the Host execution context, never supplied
    # as a planner argument.  Rejecting extra arguments prevents a model from
    # selecting another stream in the normal SDK validation path.
    "additionalProperties": False,
}

TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"task_id": {"type": "string", "minLength": 1, "description": "调查任务 ID。"}},
    "required": ["task_id"],
    "additionalProperties": False,
}

CONTINUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string", "minLength": 1, "description": "调查任务 ID。"},
        "time_budget_seconds": {"type": "integer", "minimum": 1, "description": "重置后的报告间隔（秒）。"},
        "credit_adjustment": {"type": "number", "default": 0, "description": "本次继续调查的 signed credits 调整。"},
    },
    "required": ["task_id"],
    "additionalProperties": False,
}

STOP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string", "minLength": 1, "description": "调查任务 ID。"},
        "reason": {"type": "string", "description": "停止原因。"},
    },
    "required": ["task_id"],
    "additionalProperties": False,
}

CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string", "minLength": 1, "description": "调查任务 ID。"},
        "information": {"type": "string", "minLength": 1, "description": "追加给调查分支的上下文或要求。"},
    },
    "required": ["task_id", "information"],
    "additionalProperties": False,
}

LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "description": "按任务状态过滤。"},
        "created_after": {"type": "string", "format": "date-time", "description": "创建时间下界（ISO 8601）。"},
        "created_before": {"type": "string", "format": "date-time", "description": "创建时间上界（ISO 8601）。"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20, "description": "最多返回的任务数量。"},
    },
    "additionalProperties": False,
}

PUBLIC_TASK_FIELDS = (
    "task_id", "status", "round_id", "round_number", "generation", "active_leaves",
    "raw_context_released", "created_at", "initial_credits", "effective_time_budget_seconds",
    "effective_credits_or_adjustment", "round",
)


def public_task_dto(result: Mapping[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    """只保留可向 Planner 公开的任务快照字段。"""

    output = {key: result[key] for key in PUBLIC_TASK_FIELDS if key in result}
    if task_id and "task_id" not in output:
        output["task_id"] = task_id
    leaves = output.get("active_leaves")
    if isinstance(leaves, (list, tuple)):
        output["active_leaves"] = [
            {
                key: leaf[key]
                for key in ("branch_id", "credits")
                if isinstance(leaf, Mapping) and key in leaf
            }
            for leaf in leaves
            if isinstance(leaf, Mapping)
        ]
    return output


def success_result(
    result: Mapping[str, Any],
    *,
    effective_time_budget_seconds: int | None = None,
    adjustment: float = 0.0,
    task_id: str | None = None,
) -> dict[str, Any]:
    """将 manager 的公开快照统一为 Planner 稳定 JSON shape。"""

    if result.get("success") is False or "error" in result:
        output = public_task_dto(result, task_id=task_id)
        output["success"] = False
        error = result.get("error")
        if isinstance(error, Mapping):
            output["error"] = {
                "code": str(error.get("code") or "manager_error")[:64],
                "message": str(error.get("message") or "研究任务未能继续")[:256],
            }
        else:
            output["error"] = {"code": "manager_error", "message": "研究任务未能继续"}
        output.setdefault("task_id", task_id)
        output.setdefault("round", output.get("round_id"))
        output.setdefault("status", None)
        output.setdefault("effective_time_budget_seconds", effective_time_budget_seconds)
        output.setdefault("effective_credits_or_adjustment", adjustment)
        return output
    output = public_task_dto(result, task_id=task_id)
    output["success"] = True
    if "round" not in output:
        output["round"] = output.get("round_id")
    output.setdefault("effective_time_budget_seconds", effective_time_budget_seconds)
    output.setdefault("effective_credits_or_adjustment", output.get("initial_credits", adjustment))
    return output


def failure_result(code: str, message: str, *, task_id: str | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {"success": False, "error": {"code": str(code), "message": str(message)}}
    if task_id:
        output["task_id"] = task_id
    return output


def mutation_failure_result(
    code: str,
    message: str,
    *,
    task_id: str | None = None,
    round_value: Any = None,
    status: str | None = None,
    effective_time_budget_seconds: int | None = None,
    adjustment: float | None = None,
) -> dict[str, Any]:
    """构造所有 mutating Tool 共用的失败 envelope。"""

    output = failure_result(code, message, task_id=task_id)
    output["task_id"] = task_id
    output["round"] = round_value
    output["status"] = status
    output["effective_time_budget_seconds"] = effective_time_budget_seconds
    output["effective_credits_or_adjustment"] = adjustment
    return output


def validate_nonblank(value: Any, field: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{field} 不能为空"
    return None


def validate_time_budget(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return "time_budget_seconds 必须为正整数"
    return None


def validate_effort(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)) or value < 0:
        return "effort_level 必须为非负有限数"
    return None


def validate_adjustment(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "credit_adjustment 必须为有限数"
    return None


def validate_iso_timestamp(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return f"{field} 必须为 ISO 8601 字符串"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return f"{field} 必须为 ISO 8601 字符串"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return f"{field} 必须包含时区"
    return None


def parse_iso_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("时间戳必须包含时区")
    return parsed


def validate_status(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in {item.value for item in TaskStatus}:
        return "status 不是受支持的任务状态"
    return None


async def invoke_manager(manager: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """调用 manager，兼容测试替身和同步适配器。"""

    method: Callable[..., Any] = getattr(manager, method_name)
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def manager_error(
    exc: BaseException,
    *,
    task_id: str | None = None,
    mutating: bool = False,
    effective_time_budget_seconds: int | None = None,
    adjustment: float | None = None,
) -> dict[str, Any]:
    if isinstance(exc, PermissionError):
        code, message = "task_access_denied", "无权访问该调查任务"
    elif isinstance(exc, LookupError):
        code, message = "task_not_found", "调查任务不存在"
    elif isinstance(exc, ValueError):
        code, message = "invalid_state", str(exc)[:256]
    elif isinstance(exc, AttributeError):
        code, message = "manager_unavailable", "研究运行时尚未初始化"
    elif isinstance(exc, TypeError):
        code, message = "invalid_argument", "研究任务参数无效"
    else:
        code, message = "manager_error", "研究运行时暂时不可用"
    if mutating:
        return mutation_failure_result(
            code,
            message,
            task_id=task_id,
            effective_time_budget_seconds=effective_time_budget_seconds,
            adjustment=adjustment,
        )
    return failure_result(code, message, task_id=task_id)


__all__ = [
    "CONTEXT_SCHEMA", "CONTINUE_SCHEMA", "LIST_SCHEMA", "START_SCHEMA", "STOP_SCHEMA", "TASK_SCHEMA",
    "failure_result", "invoke_manager", "manager_error", "mutation_failure_result", "parse_iso_timestamp", "public_task_dto",
    "success_result", "validate_adjustment", "validate_effort", "validate_iso_timestamp", "validate_nonblank",
    "validate_status", "validate_time_budget",
]
