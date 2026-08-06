"""内置旁路承包商 Procedure：定义与运行时依赖 stub（完整循环见后续任务）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult

Handler = Callable[[Any, Mapping[str, Any]], Awaitable[ProcedureResult]]

CONTRACTOR_PROCEDURE_ID = "builtin.contractor"

_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
}


@dataclass
class ContractorDeps:
    """承包商运行时依赖；Task 6 起注入完整实现。"""

    llm: Any = None
    prices: Any = None
    resolve_agent: Callable[[str], Any] | None = None
    invoke_nested_procedure: Callable[..., Awaitable[ProcedureResult]] | None = None


def _failure(code: str, message: str) -> ProcedureResult:
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata={},
    )


def contractor_procedure_definitions() -> list[ProcedureDefinition]:
    """构造 builtin.contractor 定义（硬超时关闭，软预算由参数控制）。"""

    payload = {
        "procedure_id": CONTRACTOR_PROCEDURE_ID,
        "version": "1",
        "display_name": "旁路承包商",
        "description": (
            "将旁路问题交给目录中的智能体作为 outsider 工具作答：新鲜上下文、无子代理扇出，"
            "可选研究额度与软时间预算；通过 return / contractor_return / 末条正文返回。"
        ),
        "arguments_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "minLength": 1},
                "question": {"type": "string", "minLength": 1},
                "temperature": {"type": "number"},
                "personality": {"type": ["string", "null"]},
                "time_budget_seconds": {"type": "number", "minimum": 0, "default": 0},
            },
            "required": ["agent_id", "question"],
            "additionalProperties": False,
        },
        "result_schema": _RESULT_SCHEMA,
        "idempotent": False,
        "timeout_seconds": 0.0,
        "external_cost_kind": "none",
        "enabled": True,
    }
    return [ProcedureDefinition.model_validate(payload)]


def make_contractor_handler(deps: ContractorDeps | None = None) -> Handler:
    """绑定可选 deps；缺失时返回 contractor_runtime_missing（完整循环为后续任务）。"""

    bound = deps

    async def _contractor(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
        del arguments  # Task 6 才校验并消费参数
        if bound is None or bound.llm is None or bound.resolve_agent is None:
            return _failure(
                "contractor_runtime_missing",
                "承包商运行时依赖尚未注入，暂无法执行。",
            )
        return _failure(
            "contractor_runtime_missing",
            "承包商多轮循环尚未实现。",
        )

    return _contractor


CONTRACTOR_HANDLERS: dict[str, Handler] = {
    CONTRACTOR_PROCEDURE_ID: make_contractor_handler(),
}
