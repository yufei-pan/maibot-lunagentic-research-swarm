"""Planner-facing schemas and result helpers for research task controls."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any


START_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {"type": "string", "minLength": 1, "description": "要调查的问题、案例或研究主题。"},
        "time_budget_seconds": {"type": "integer", "minimum": 1, "description": "期望报告间隔（秒）。"},
        "effort_level": {"type": "number", "minimum": 0, "default": 1.0, "description": "相对于默认 credits 的倍率。"},
        "planner_context": {"type": "string", "description": "Planner 提供的补充上下文。"},
    },
    "required": ["objective"],
    "additionalProperties": True,
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


def success_result(result: Mapping[str, Any], *, effective_time_budget_seconds: int | None = None, adjustment: float = 0.0) -> dict[str, Any]:
    """将 manager 的公开快照统一为 Planner 稳定 JSON shape。"""

    output = dict(result)
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
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return f"{field} 必须为 ISO 8601 字符串"
    return None


async def invoke_manager(manager: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """调用 manager，兼容测试替身和同步适配器。"""

    method: Callable[..., Any] = getattr(manager, method_name)
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def manager_error(exc: BaseException, *, task_id: str | None = None) -> dict[str, Any]:
    if isinstance(exc, LookupError):
        code = "task_not_found"
    elif isinstance(exc, ValueError):
        code = "invalid_state"
    elif isinstance(exc, AttributeError):
        code = "manager_unavailable"
    elif isinstance(exc, TypeError):
        code = "invalid_argument"
    else:
        code = "manager_error"
    return failure_result(code, str(exc), task_id=task_id)


__all__ = [
    "CONTEXT_SCHEMA", "CONTINUE_SCHEMA", "LIST_SCHEMA", "START_SCHEMA", "STOP_SCHEMA", "TASK_SCHEMA",
    "failure_result", "invoke_manager", "manager_error", "success_result", "validate_adjustment",
    "validate_effort", "validate_iso_timestamp", "validate_nonblank", "validate_time_budget",
]
