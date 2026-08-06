"""内置旁路承包商 Procedure：outsider 新鲜上下文多轮循环。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult
from lunagentic_research_swarm.llm.gateway import GenerationRequest
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.procedures.core import research_credits_for_summarizer_usage

Handler = Callable[..., Awaitable[ProcedureResult]]

CONTRACTOR_PROCEDURE_ID = "builtin.contractor"
_NESTED_REJECT_MESSAGE = "嵌套 Procedure 调用被拒绝（当前承包商不允许执行该 procedure_id）：{procedure_id}"
_MAX_TURNS = 16

_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
}


@dataclass
class ContractorDeps:
    """承包商运行时依赖。"""

    llm: Any = None
    prices: Any = None
    resolve_agent: Callable[[str], Any] | None = None
    invoke_nested_procedure: Callable[..., Awaitable[ProcedureResult]] | None = None
    resolve_procedure_catalog: Callable[[str], Sequence[Mapping[str, Any]]] | None = None


class ContractorTurnEnvelope(BaseModel):
    """承包商 JSON 瘦信封：无 delegations；可选显式 return。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    report: str = ""
    procedures: list[ProcedureRequest] = Field(default_factory=list)
    return_value: str | None = Field(default=None, alias="return")


CONTRACTOR_NATIVE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "call_procedure",
            "description": "请求执行一个允许的研究 Procedure（禁止 checkpoint/terminate/contractor）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure_id": {"type": "string"},
                    "arguments": {"type": "object"},
                    "credits": {"type": "number", "minimum": 0},
                    "call_id": {"type": "string"},
                },
                "required": ["procedure_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contractor_return",
            "description": "以最终答案结束承包商任务并返回给调用方。",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    },
]


def _failure(code: str, message: str, *, metadata: Mapping[str, Any] | None = None) -> ProcedureResult:
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata=dict(metadata or {}),
    )


def _success(
    result_text: str,
    *,
    charged: float,
    termination_reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> ProcedureResult:
    meta = {
        "termination_reason": termination_reason,
        "research_credits_charged": float(charged),
        **dict(metadata or {}),
    }
    return ProcedureResult(
        success=True,
        data={"result": str(result_text)},
        error=None,
        metadata=meta,
        research_credits_charged=max(0.0, float(charged)),
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


def _agent_attr(agent: Any, name: str, default: Any = None) -> Any:
    if agent is None:
        return default
    if isinstance(agent, Mapping):
        return agent.get(name, default)
    return getattr(agent, name, default)


def _build_system_prompt(
    *,
    personality: str,
    procedure_catalog: Sequence[Mapping[str, Any]],
    protocol: str,
) -> str:
    catalog_lines: list[str] = []
    for item in procedure_catalog:
        pid = str(item.get("procedure_id", "") or "")
        if not pid or pid == CONTRACTOR_PROCEDURE_ID:
            continue
        desc = str(item.get("description", "") or item.get("display_name", "") or "")
        catalog_lines.append(f"- {pid}: {desc}" if desc else f"- {pid}")
    catalog_block = "\n".join(catalog_lines) if catalog_lines else "- （本智能体暂无额外 Procedure 目录条目）"
    if protocol == "native_tools":
        return_rules = (
            "协议：native_tools。结束时调用 contractor_return(result=...)。"
            "需要工具时调用 call_procedure；禁止启动子代理、禁止再调用 builtin.contractor / core.checkpoint / core.terminate。"
        )
    else:
        return_rules = (
            "协议：json_envelope。每轮只返回一个 JSON 对象："
            '{"report":"...","procedures":[],"return":"可选最终答案"}。'
            "无 delegations 字段。有明确结论时设置 return；不要启动子代理。"
            "禁止再调用 builtin.contractor / core.checkpoint / core.terminate。"
        )
    return (
        "你是调查 swarm 中的旁路承包商（outsider）。你只有本题的新鲜上下文，"
        "看不到调用方 transcript、正式任务或其它分支历史。\n"
        "规则：不得启动子代理；不得递归承包商；只回答当前问题。\n"
        f"人设：\n{personality}\n"
        f"可用 Procedure 短目录（不含承包商自身）：\n{catalog_block}\n"
        f"{return_rules}"
    )


def _parse_contractor_json(raw: str) -> ContractorTurnEnvelope | None:
    text = (raw or "").strip().lstrip("\ufeff").strip()
    if not text:
        return None
    # 剥一层可选 markdown fence
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        return ContractorTurnEnvelope.model_validate(dict(payload))
    except ValidationError:
        return None


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(loaded, Mapping):
            return dict(loaded)
    return {}


@dataclass
class _TurnOutcome:
    explicit_return: str | None = None
    last_text: str = ""
    procedure_requests: list[ProcedureRequest] = field(default_factory=list)
    native_call_notes: list[str] = field(default_factory=list)


def _interpret_json_turn(response: str) -> _TurnOutcome:
    envelope = _parse_contractor_json(response)
    if envelope is None:
        return _TurnOutcome(last_text=str(response or "").strip())
    last = (envelope.return_value if envelope.return_value is not None else "") or envelope.report or ""
    return _TurnOutcome(
        explicit_return=envelope.return_value,
        last_text=str(last),
        procedure_requests=list(envelope.procedures),
    )


def _interpret_native_turn(response: str, tool_calls: Sequence[Mapping[str, Any]] | None) -> _TurnOutcome:
    outcome = _TurnOutcome(last_text=str(response or "").strip())
    if not tool_calls:
        return outcome
    for call in tool_calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name") or "")
        args = _parse_tool_arguments(function.get("arguments"))
        if name == "contractor_return":
            result = args.get("result")
            if result is None:
                result = args.get("return")
            outcome.explicit_return = str(result if result is not None else "")
            outcome.last_text = outcome.explicit_return
            # 同轮显式 return 优先，忽略 sibling procedures
            outcome.procedure_requests = []
            return outcome
        if name == "call_procedure":
            pid = str(args.get("procedure_id") or "").strip()
            if not pid:
                outcome.native_call_notes.append("call_procedure 缺少 procedure_id")
                continue
            try:
                req = ProcedureRequest.model_validate(
                    {
                        "procedure_id": pid,
                        "arguments": dict(args.get("arguments") or {}),
                        "credits": float(args.get("credits") or 0.0),
                        "call_id": args.get("call_id"),
                    }
                )
            except ValidationError as exc:
                outcome.native_call_notes.append(f"call_procedure 参数无效：{exc}")
                continue
            outcome.procedure_requests.append(req)
        else:
            outcome.native_call_notes.append(f"未知工具调用被忽略：{name}")
    return outcome


async def _reject_nested_procedures(
    requests: Sequence[ProcedureRequest],
    *,
    deps: ContractorDeps,
) -> list[str]:
    """Task 6 stub：全部拒绝；Task 7 再打开 allowlist。"""

    notes: list[str] = []
    for req in requests:
        pid = str(req.procedure_id)
        if deps.invoke_nested_procedure is not None:
            # 预留 Task 7 钩子：即便注入了 invoker，本任务仍统一拒绝。
            pass
        notes.append(_NESTED_REJECT_MESSAGE.format(procedure_id=pid))
    return notes


def _meter_turn(*, deps: ContractorDeps, model_name: str, usage: Any) -> float:
    return research_credits_for_summarizer_usage(
        catalog=deps.prices,
        model_name=str(model_name or ""),
        usage=usage,
    )


def _procedure_catalog_for_agent(deps: ContractorDeps, agent_id: str, agent: Any) -> list[dict[str, Any]]:
    if deps.resolve_procedure_catalog is not None:
        try:
            items = deps.resolve_procedure_catalog(agent_id)
            return [dict(item) for item in items if isinstance(item, Mapping)]
        except Exception:
            pass
    allowed = _agent_attr(agent, "allowed_procedures", ["*"]) or ["*"]
    if allowed == ["*"]:
        return [{"procedure_id": "*", "description": "调用方目录中的全部研究 Procedure（不含承包商递归）"}]
    return [{"procedure_id": str(pid), "description": ""} for pid in allowed if str(pid) != CONTRACTOR_PROCEDURE_ID]


async def run_contractor(
    *,
    arguments: Mapping[str, Any],
    scoped_metadata: Mapping[str, Any],
    deps: ContractorDeps,
) -> ProcedureResult:
    """承包商 outsider 多轮循环：新鲜上下文、调用方协议、计量与返回。"""

    if deps.llm is None or deps.resolve_agent is None:
        return _failure("contractor_runtime_missing", "承包商运行时依赖尚未注入，暂无法执行。")

    agent_id = str(arguments.get("agent_id") or "").strip()
    question = str(arguments.get("question") or "").strip()
    if not agent_id or not question:
        return _failure("invalid_arguments", "agent_id 与 question 均为必填。")

    agent = deps.resolve_agent(agent_id)
    if agent is None:
        return _failure("agent_unavailable", f"智能体不在目录中：{agent_id}")

    protocol = str(scoped_metadata.get("caller_protocol") or "json_envelope").strip() or "json_envelope"
    if protocol not in {"json_envelope", "native_tools"}:
        protocol = "json_envelope"
    credit_budget = float(scoped_metadata.get("credit_budget") or 0.0)
    caller_agent_id = str(scoped_metadata.get("caller_agent_id") or scoped_metadata.get("agent_id") or "")

    personality_arg = arguments.get("personality", None)
    if personality_arg is None:
        personality = str(_agent_attr(agent, "character_prompt", "") or "")
    else:
        personality = str(personality_arg)

    selector = str(_agent_attr(agent, "model_selector", "") or "")
    if not selector:
        return _failure("invalid_arguments", f"智能体缺少 model_selector：{agent_id}")

    temperature = arguments.get("temperature")
    if temperature is None:
        temperature = None
    else:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            return _failure("invalid_arguments", "temperature 必须是数字。")

    time_budget_seconds = float(arguments.get("time_budget_seconds") or 0.0)
    catalog = _procedure_catalog_for_agent(deps, agent_id, agent)
    system = _build_system_prompt(personality=personality, procedure_catalog=catalog, protocol=protocol)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    total_charged = 0.0
    turn_count = 0
    last_text = ""
    tools = CONTRACTOR_NATIVE_TOOLS if protocol == "native_tools" else None
    termination_reason = "returned"

    try:
        while turn_count < _MAX_TURNS:
            turn_count += 1
            request = GenerationRequest(
                selector=selector,
                messages=list(messages),
                tools=tools,
                temperature=temperature,
            )
            result = await deps.llm.generate(request)
            usage = getattr(result, "usage", None)
            model_name = str(getattr(result, "model_name", "") or "")
            turn_charge = _meter_turn(deps=deps, model_name=model_name, usage=usage)
            total_charged += turn_charge

            success = bool(getattr(result, "success", True))
            if not success:
                err = getattr(result, "error", None)
                message = getattr(err, "message", None) if err is not None else None
                last_text = str(message or "LLM 调用失败")
                termination_reason = "returned"
                break

            response_text = str(getattr(result, "response", "") or "")
            tool_calls = getattr(result, "tool_calls", None)
            if protocol == "native_tools":
                outcome = _interpret_native_turn(response_text, tool_calls)
            else:
                outcome = _interpret_json_turn(response_text)

            if outcome.last_text:
                last_text = outcome.last_text
            elif response_text.strip():
                last_text = response_text.strip()

            # 助手轮次写入 transcript（供下一轮与末条返回）
            assistant_content = response_text
            if protocol == "native_tools" and tool_calls:
                assistant_content = json.dumps(
                    {"response": response_text, "tool_calls": list(tool_calls)},
                    ensure_ascii=False,
                )
            messages.append({"role": "assistant", "content": assistant_content})

            if outcome.explicit_return is not None:
                last_text = str(outcome.explicit_return)
                termination_reason = "returned"
                break

            notes: list[str] = list(outcome.native_call_notes)
            if outcome.procedure_requests:
                notes.extend(await _reject_nested_procedures(outcome.procedure_requests, deps=deps))

            if notes:
                messages.append(
                    {
                        "role": "user",
                        "content": "procedure_results:\n" + "\n".join(f"- {note}" for note in notes),
                    }
                )
                continue

            # 无工具 / 无 procedure → 末条正文返回
            termination_reason = "returned"
            break
        else:
            termination_reason = "returned"
    finally:
        charged = float(total_charged)

    return _success(
        last_text or "",
        charged=charged,
        termination_reason=termination_reason,
        metadata={
            "agent_id": agent_id,
            "caller_agent_id": caller_agent_id,
            "caller_protocol": protocol,
            "turn_count": turn_count,
            "time_budget_seconds": time_budget_seconds,
            "budget_hint": credit_budget,
            # Task 7：soft timeout / insufficient_funds 钩子占位
            "credit_budget": credit_budget,
        },
    )


def make_contractor_handler(deps: ContractorDeps | None = None) -> Handler:
    """绑定可选 deps；缺失时返回 contractor_runtime_missing。"""

    bound = deps

    async def _contractor(
        _ctx: Any,
        arguments: Mapping[str, Any],
        *,
        scoped_metadata: Mapping[str, Any] | None = None,
    ) -> ProcedureResult:
        if bound is None or bound.llm is None or bound.resolve_agent is None:
            return _failure(
                "contractor_runtime_missing",
                "承包商运行时依赖尚未注入，暂无法执行。",
            )
        return await run_contractor(
            arguments=arguments,
            scoped_metadata=scoped_metadata or {},
            deps=bound,
        )

    _contractor._contractor_deps = bound  # type: ignore[attr-defined]
    return _contractor


CONTRACTOR_HANDLERS: dict[str, Handler] = {
    CONTRACTOR_PROCEDURE_ID: make_contractor_handler(),
}
