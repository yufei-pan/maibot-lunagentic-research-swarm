"""内置旁路承包商 Procedure：outsider 新鲜上下文多轮循环。"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult
from lunagentic_research_swarm.llm.gateway import GenerationRequest
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.procedures.billing import extract_research_credits_charged
from lunagentic_research_swarm.procedures.core import research_credits_for_summarizer_usage

Handler = Callable[..., Awaitable[ProcedureResult]]

CONTRACTOR_PROCEDURE_ID = "builtin.contractor"
_FORBIDDEN_NESTED_PROCEDURE_IDS = frozenset(
    {
        CONTRACTOR_PROCEDURE_ID,
        "core.checkpoint",
        "core.terminate",
    }
)
_NESTED_REJECT_MESSAGE = "嵌套 Procedure 调用被拒绝（当前承包商不允许执行该 procedure_id）：{procedure_id}"
_NESTED_RUNTIME_MISSING = "嵌套 Procedure 调用失败：运行时未注入 invoke_nested_procedure（{procedure_id}）"
_FORCE_RETURN_INSUFFICIENT = "【终止说明】研究额度不足，承包商强制返回（insufficient_funds）。"
_FORCE_RETURN_TIMEOUT = "【终止说明】软时间预算已耗尽，承包商强制返回（timeout）。"
_MAX_TURNS = 16

_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
}


@dataclass
class ContractorDeps:
    """承包商运行时依赖。

    ``round_snapshot_for_task`` 在有冻结 round 时优先于 live ``resolve_agent`` /
    ``prices`` / ``resolve_procedure_catalog``（按 ``scoped_metadata.task_id`` 查找）。
    """

    llm: Any = None
    prices: Any = None
    resolve_agent: Callable[[str], Any] | None = None
    invoke_nested_procedure: Callable[..., Awaitable[ProcedureResult]] | None = None
    resolve_procedure_catalog: Callable[[str], Sequence[Mapping[str, Any]]] | None = None
    round_snapshot_for_task: Callable[[str], Any | None] | None = None


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


def _failure(
    code: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    charged: float = 0.0,
) -> ProcedureResult:
    charged_value = max(0.0, float(charged))
    meta = {**dict(metadata or {}), "research_credits_charged": charged_value}
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata=meta,
        research_credits_charged=charged_value,
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


def _format_attempted_tools(requests: Sequence[ProcedureRequest]) -> str:
    if not requests:
        return ""
    lines = []
    for req in requests:
        args = dict(req.arguments or {})
        lines.append(f"- {req.procedure_id}: {json.dumps(args, ensure_ascii=False)}")
    return "尝试的工具调用：\n" + "\n".join(lines)


def _compose_force_return_text(
    *,
    last_text: str,
    attempted: Sequence[ProcedureRequest],
    note: str,
) -> str:
    parts: list[str] = []
    body = str(last_text or "").strip()
    if body:
        parts.append(body)
    tools = _format_attempted_tools(attempted)
    if tools:
        parts.append(tools)
    parts.append(note)
    return "\n\n".join(parts)


def _is_forbidden_nested(procedure_id: str) -> bool:
    return str(procedure_id) in _FORBIDDEN_NESTED_PROCEDURE_IDS


async def _run_nested_procedures(
    requests: Sequence[ProcedureRequest],
    *,
    deps: ContractorDeps,
    messages: Sequence[Mapping[str, Any]],
    outsider_task: str,
    task_id: str = "",
) -> tuple[list[str], float]:
    """执行 allowlist 内嵌套调用；禁止项写入错误；返回 (transcript notes, nested_charge_sum)。

    嵌套 ``core.compact`` 时注入承包商 transcript 与 outsider 问题作为总结上下文
    （agent 侧常见空 arguments），使 compact 可成功并申报计费。
    ``task_id`` 传入 nested invoker，以便 compact 计价使用冻结 round ``price_catalog``。
    """

    notes: list[str] = []
    nested_charged = 0.0
    transcript = [dict(item) for item in messages if isinstance(item, Mapping)]
    task_text = str(outsider_task or "").strip() or "承包商旁路问题"
    scoped_task_id = str(task_id or "").strip()
    for req in requests:
        pid = str(req.procedure_id)
        if _is_forbidden_nested(pid):
            notes.append(_NESTED_REJECT_MESSAGE.format(procedure_id=pid))
            continue
        if deps.invoke_nested_procedure is None:
            notes.append(_NESTED_RUNTIME_MISSING.format(procedure_id=pid))
            continue
        scoped: dict[str, Any] = {}
        if scoped_task_id:
            scoped["task_id"] = scoped_task_id
        if pid == "core.compact":
            scoped["formalized_task"] = task_text
            scoped["branch_history"] = list(transcript)
        try:
            result = await deps.invoke_nested_procedure(
                pid,
                dict(req.arguments or {}),
                credits=float(req.credits or 0.0),
                call_id=req.call_id,
                scoped_metadata=scoped,
            )
        except Exception as exc:  # noqa: BLE001 — 嵌套失败写入 transcript，不中断承包商
            notes.append(f"嵌套 Procedure 执行异常（{pid}）：{type(exc).__name__}: {exc}")
            continue
        charge = extract_research_credits_charged(result)
        nested_charged += charge
        if getattr(result, "success", False):
            data = getattr(result, "data", None)
            payload = json.dumps(data, ensure_ascii=False) if data is not None else "null"
            notes.append(f"{pid} → success charge={charge}: {payload}")
        else:
            err = getattr(result, "error", None) or {}
            notes.append(f"{pid} → error charge={charge}: {err}")
    return notes, nested_charged


def _meter_turn(*, prices: Any, model_name: str, usage: Any) -> float:
    return research_credits_for_summarizer_usage(
        catalog=prices,
        model_name=str(model_name or ""),
        usage=usage,
    )


def _lookup_round_snapshot(deps: ContractorDeps, scoped_metadata: Mapping[str, Any]) -> Any | None:
    task_id = str(scoped_metadata.get("task_id") or "").strip()
    if not task_id or deps.round_snapshot_for_task is None:
        return None
    try:
        return deps.round_snapshot_for_task(task_id)
    except Exception:
        return None


def _resolve_agent_from_deps(
    deps: ContractorDeps,
    agent_id: str,
    *,
    snapshot: Any | None,
) -> Any | None:
    if snapshot is not None:
        catalog = getattr(snapshot, "agent_catalog", None)
        if catalog is not None:
            entry = catalog.get(agent_id) if hasattr(catalog, "get") else None
            if entry is None:
                return None
            return getattr(entry, "definition", entry)
        # 有 snapshot 但无 agent_catalog 时不回落 live，避免混用冻结边界
        return None
    if deps.resolve_agent is None:
        return None
    return deps.resolve_agent(agent_id)


def _prices_from_deps(deps: ContractorDeps, *, snapshot: Any | None) -> Any:
    if snapshot is not None:
        catalog = getattr(snapshot, "price_catalog", None)
        if catalog is not None:
            return catalog
    return deps.prices


def _procedure_catalog_for_agent(
    deps: ContractorDeps,
    agent_id: str,
    agent: Any,
    *,
    snapshot: Any | None = None,
) -> list[dict[str, Any]]:
    if snapshot is not None:
        agent_catalog = getattr(snapshot, "agent_catalog", None)
        proc_catalog = getattr(snapshot, "procedure_catalog", None)
        if agent_catalog is not None and proc_catalog is not None and hasattr(agent_catalog, "resolve_allowed_procedures"):
            try:
                allowed = agent_catalog.resolve_allowed_procedures(agent_id, proc_catalog)
            except Exception:
                allowed = ()
            items: list[dict[str, Any]] = []
            for procedure_id in allowed:
                if procedure_id == CONTRACTOR_PROCEDURE_ID:
                    continue
                entry = proc_catalog.get(procedure_id) if hasattr(proc_catalog, "get") else None
                if entry is None:
                    items.append({"procedure_id": str(procedure_id), "description": ""})
                    continue
                definition = getattr(entry, "definition", entry)
                items.append(
                    {
                        "procedure_id": str(procedure_id),
                        "display_name": str(_agent_attr(definition, "display_name", "") or ""),
                        "description": str(_agent_attr(definition, "description", "") or ""),
                    }
                )
            return items
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

    if deps.llm is None or (deps.resolve_agent is None and deps.round_snapshot_for_task is None):
        return _failure("contractor_runtime_missing", "承包商运行时依赖尚未注入，暂无法执行。")

    agent_id = str(arguments.get("agent_id") or "").strip()
    question = str(arguments.get("question") or "").strip()
    if not agent_id or not question:
        return _failure("invalid_arguments", "agent_id 与 question 均为必填。")

    snapshot = _lookup_round_snapshot(deps, scoped_metadata)
    agent = _resolve_agent_from_deps(deps, agent_id, snapshot=snapshot)
    if agent is None:
        return _failure("agent_unavailable", f"智能体不在目录中：{agent_id}")

    prices = _prices_from_deps(deps, snapshot=snapshot)

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
    catalog = _procedure_catalog_for_agent(deps, agent_id, agent, snapshot=snapshot)
    system = _build_system_prompt(personality=personality, procedure_catalog=catalog, protocol=protocol)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    total_charged = 0.0
    internal_balance = float(credit_budget)
    turn_count = 0
    last_text = ""
    tools = CONTRACTOR_NATIVE_TOOLS if protocol == "native_tools" else None
    termination_reason = "returned"
    llm_failure: tuple[str, str] | None = None
    force_return_text: str | None = None
    loop_started = time.monotonic()

    try:
        while turn_count < _MAX_TURNS:
            # Soft timeout：在下一轮模型调用前检查（保证至少能跑完已开始的一轮结算）。
            if (
                turn_count > 0
                and time_budget_seconds > 0
                and (time.monotonic() - loop_started) >= time_budget_seconds
            ):
                termination_reason = "timeout"
                force_return_text = _compose_force_return_text(
                    last_text=last_text,
                    attempted=(),
                    note=_FORCE_RETURN_TIMEOUT,
                )
                break

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
            turn_charge = _meter_turn(prices=prices, model_name=model_name, usage=usage)
            total_charged += turn_charge
            internal_balance -= turn_charge

            success = bool(getattr(result, "success", True))
            if not success:
                err = getattr(result, "error", None)
                code = str(getattr(err, "code", None) or "llm_generation_failed")
                message = str(getattr(err, "message", None) or "LLM 调用失败")
                llm_failure = (code, message)
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

            # 4. 余额不足 → 立即强制返回（即使同轮有 return / procedures，也不执行嵌套）
            if internal_balance < 0:
                termination_reason = "insufficient_funds"
                force_return_text = _compose_force_return_text(
                    last_text=last_text,
                    attempted=outcome.procedure_requests,
                    note=_FORCE_RETURN_INSUFFICIENT,
                )
                break

            # 5. 显式 return
            if outcome.explicit_return is not None:
                last_text = str(outcome.explicit_return)
                termination_reason = "returned"
                break

            notes: list[str] = list(outcome.native_call_notes)

            # 6. 无工具 / 无 procedure → 末条正文返回
            if not outcome.procedure_requests:
                termination_reason = "returned"
                break

            # 7. 嵌套 procedures（allowlist + 计费折入）
            nested_notes, nested_charge = await _run_nested_procedures(
                outcome.procedure_requests,
                deps=deps,
                messages=messages,
                outsider_task=question,
                task_id=str(scoped_metadata.get("task_id") or ""),
            )
            notes.extend(nested_notes)
            total_charged += nested_charge
            internal_balance -= nested_charge

            if notes:
                messages.append(
                    {
                        "role": "user",
                        "content": "procedure_results:\n" + "\n".join(f"- {note}" for note in notes),
                    }
                )

            if internal_balance < 0:
                termination_reason = "insufficient_funds"
                force_return_text = _compose_force_return_text(
                    last_text=last_text,
                    attempted=outcome.procedure_requests,
                    note=_FORCE_RETURN_INSUFFICIENT,
                )
                break

            # 8. Soft timeout：嵌套结算后、进入下一轮前再检查一次
            if time_budget_seconds > 0 and (time.monotonic() - loop_started) >= time_budget_seconds:
                termination_reason = "timeout"
                force_return_text = _compose_force_return_text(
                    last_text=last_text,
                    attempted=outcome.procedure_requests,
                    note=_FORCE_RETURN_TIMEOUT,
                )
                break
        else:
            termination_reason = "returned"
    finally:
        charged = float(total_charged)

    common_meta = {
        "agent_id": agent_id,
        "caller_agent_id": caller_agent_id,
        "caller_protocol": protocol,
        "turn_count": turn_count,
        "time_budget_seconds": time_budget_seconds,
        "budget_hint": credit_budget,
        "credit_budget": credit_budget,
    }
    if llm_failure is not None:
        code, message = llm_failure
        return _failure(
            code,
            message,
            metadata={**common_meta, "termination_reason": "llm_failed"},
            charged=charged,
        )

    result_text = force_return_text if force_return_text is not None else (last_text or "")
    return _success(
        result_text,
        charged=charged,
        termination_reason=termination_reason,
        metadata=common_meta,
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
        if bound is None or bound.llm is None or (
            bound.resolve_agent is None and bound.round_snapshot_for_task is None
        ):
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


def make_nested_procedure_invoker(
    *,
    provider: Any,
    summarizer: Any | None = None,
    price_catalog: Any | None = None,
    round_snapshot_for_task: Callable[[str], Any | None] | None = None,
) -> Callable[..., Awaitable[ProcedureResult]]:
    """构造 handler-local 嵌套 invoker：走 provider / core.compact，不经外层 batch 扣费。

    嵌套 ``core.compact`` 计价优先使用 ``round_snapshot_for_task`` 冻结的
    ``price_catalog``（与 turn metering 同源）；无 snapshot 时才回落 live catalog。
    """

    from lunagentic_research_swarm.procedures.core import CORE_COMPACT_ID, execute_core_procedure

    def _resolve_compact_price_catalog(meta: Mapping[str, Any]) -> Any | None:
        task_id = str(meta.get("task_id") or "").strip()
        if task_id and round_snapshot_for_task is not None:
            try:
                snapshot = round_snapshot_for_task(task_id)
            except Exception:
                snapshot = None
            if snapshot is not None:
                frozen = getattr(snapshot, "price_catalog", None)
                if frozen is not None:
                    return frozen
        return price_catalog

    async def _invoke(
        procedure_id: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        credits: float = 0.0,
        call_id: str | None = None,
        scoped_metadata: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> ProcedureResult:
        del _kwargs
        pid = str(procedure_id)
        args = dict(arguments or {})
        meta = dict(scoped_metadata or {})
        if pid == CORE_COMPACT_ID:
            formalized = args.get("formalized_task", meta.get("formalized_task"))
            history_raw = args.get("branch_history", meta.get("branch_history"))
            history: Sequence[Mapping[str, Any]] = ()
            if isinstance(history_raw, Sequence) and not isinstance(history_raw, (str, bytes, bytearray)):
                history = tuple(item for item in history_raw if isinstance(item, Mapping))
            # 空 arguments 的 agent 请求：回落到 scoped outsider 上下文
            if not formalized:
                formalized = meta.get("formalized_task") or "承包商旁路问题"
            if not history:
                fallback = meta.get("branch_history")
                if isinstance(fallback, Sequence) and not isinstance(fallback, (str, bytes, bytearray)):
                    history = tuple(item for item in fallback if isinstance(item, Mapping))
            return await execute_core_procedure(
                CORE_COMPACT_ID,
                args,
                summarizer=summarizer,
                formalized_task=formalized,
                branch_history=history,
                price_catalog=_resolve_compact_price_catalog(meta),
                bill_research_credits=True,
            )
        if pid.startswith("core.") or pid == CONTRACTOR_PROCEDURE_ID:
            return ProcedureResult(
                success=False,
                data=None,
                error={"code": "procedure_not_allowed", "message": f"承包商嵌套禁止：{pid}"},
                metadata={"research_credits_charged": 0.0},
                research_credits_charged=0.0,
            )
        scoped = dict(meta)
        scoped.setdefault("credit_budget", float(credits or 0.0))
        if call_id:
            scoped.setdefault("call_id", call_id)
        result = await provider.invoke(pid, args, scoped_metadata=scoped)
        if isinstance(result, ProcedureResult):
            return result
        return ProcedureResult.model_validate(result)

    return _invoke


CONTRACTOR_HANDLERS: dict[str, Handler] = {
    CONTRACTOR_PROCEDURE_ID: make_contractor_handler(),
}
