"""普通智能体的唯一 turn envelope 与保守协议解析。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lunagentic_research_swarm.errors import LRSError, PROTOCOL_INVALID


class ProcedureRequest(BaseModel):
    """一次 Procedure 请求。

    ``procedure_id`` 是唯一的身份字段，与 spec §15.1 的 definition 字段同名。
    ``call_id`` 是可选的、由智能体自己生成的关联 ID：本 turn 内多次调用同一个
    Procedure 时可用它把结果对回请求。它只做透传，不参与路由，也不做去重。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    procedure_id: str
    call_id: str | None = Field(default=None, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    credits: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class DelegationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    agent_id: str
    task: str = Field(min_length=1, max_length=12000)
    credits: float = Field(ge=0.0, allow_inf_nan=False)


class SwarmTurnEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    report: str = Field(default="", max_length=30000)
    procedures: list[ProcedureRequest] = Field(default_factory=list)
    delegations: list[DelegationRequest] = Field(default_factory=list)


class ProtocolError(LRSError):
    """协议语法或 schema 无效；metadata 保留可供纠正 turn 使用的指针。"""

    def __init__(self, message: str, errors: Sequence[Mapping[str, str]] | None = None) -> None:
        normalized = [dict(error) for error in (errors or ())]
        super().__init__(PROTOCOL_INVALID, message, {"errors": normalized})

    @property
    def errors(self) -> tuple[dict[str, str], ...]:
        metadata_errors = (self.metadata or {}).get("errors", [])
        return tuple(dict(error) for error in metadata_errors if isinstance(error, Mapping))


_FENCE_PATTERN = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\Z", re.IGNORECASE)
_MAX_CORRECTION_ERRORS = 4
_MAX_CORRECTION_FIELD_CHARS = 64
_MAX_CORRECTION_ITEM_CHARS = 1024
_MAX_CORRECTION_MESSAGE_CHARS = 4096


def _normalize_outer_text(raw: str) -> str:
    return raw.strip().lstrip("\ufeff").strip()


def _strip_one_json_fence(text: str) -> tuple[str, bool]:
    matched = _FENCE_PATTERN.fullmatch(text)
    if matched is None:
        return text, False
    return matched.group("body").strip(), True


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限常量：{value}")


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_nonfinite_json_constant)


def _decode_complete_json_string_once(text: str) -> tuple[str, bool]:
    try:
        decoded = _strict_json_loads(text)
    except (ValueError, TypeError):
        return text, False
    if not isinstance(decoded, str):
        return text, False
    return _normalize_outer_text(decoded), True


def _scan_balanced_object(text: str, start: int) -> int | None:
    """返回从 ``start`` 处 ``{`` 开始的括号平衡 object 的结束下标（含）。"""

    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack:
                return None
            opening = stack.pop()
            if (opening, character) not in {("{", "}"), ("[", "]")}:
                return None
            if not stack:
                return index
    return None


def _extract_unique_balanced_object(text: str) -> str | None:
    """提取唯一的顶层 object。

    spec §9.2 只允许“从无歧义包装文字中提取唯一顶层对象”，明确禁止“在多个对象
    中选择”。因此发现第二个顶层 object 时必须失败，而不是静默取第一个。
    """

    start = text.find("{")
    if start < 0:
        return None
    end = _scan_balanced_object(text, start)
    if end is None:
        return None
    # 后续文字里再出现一个括号平衡的 object 就是歧义输入；不平衡的 `{` 只是散文。
    cursor = text.find("{", end + 1)
    while cursor >= 0:
        if _scan_balanced_object(text, cursor) is not None:
            return None
        cursor = text.find("{", cursor + 1)
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            output.append(character)
            continue
        if character == ",":
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] in "}]":
                continue
        output.append(character)
    return "".join(output)


def _pointer(location: Sequence[Any]) -> str:
    if not location:
        return "/"
    escaped = (str(part).replace("~", "~0").replace("/", "~1") for part in location)
    return "/" + "/".join(escaped)


def _validation_error(error: ValidationError) -> ProtocolError:
    details = [
        {"pointer": _pointer(item.get("loc", ())), "message": str(item.get("msg", "schema 校验失败"))}
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    ]
    return ProtocolError("智能体返回不符合 swarm turn schema", details)


def _validate_envelope(payload: Any) -> SwarmTurnEnvelope:
    try:
        return SwarmTurnEnvelope.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise _validation_error(exc) from exc


def parse_json_envelope_with_repairs(raw: str) -> tuple[SwarmTurnEnvelope, tuple[str, ...]]:
    """严格验证唯一 envelope，并返回实际应用过的有限修复规则名。

    规则名对应 spec §9.2 允许的确定性变换，供调用方记录 ``protocol_repaired``。
    """

    if not isinstance(raw, str):
        raise ProtocolError("智能体返回必须是 JSON 文本", [{"pointer": "/", "message": "必须为字符串"}])
    applied: list[str] = []
    text = _normalize_outer_text(raw)
    if text != raw:
        applied.append("strip_bom_whitespace")
    text, fence_removed = _strip_one_json_fence(text)
    if fence_removed:
        applied.append("strip_markdown_fence")
    text, decoded = _decode_complete_json_string_once(text)
    if decoded:
        applied.append("decode_json_string")
        text = _normalize_outer_text(text)
        if not fence_removed:
            text, inner_fence = _strip_one_json_fence(text)
            if inner_fence:
                applied.append("strip_markdown_fence")
    object_text = _extract_unique_balanced_object(text)
    if object_text is None:
        raise ProtocolError(
            "无法找到唯一完整的 swarm turn JSON object",
            [{"pointer": "/", "message": "需要恰好一个括号平衡的顶层 JSON object"}],
        )
    if object_text != text:
        applied.append("extract_wrapped_object")
    repaired = _remove_trailing_commas(object_text)
    if repaired != object_text:
        applied.append("remove_trailing_comma")
    try:
        payload = _strict_json_loads(repaired)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            "swarm turn JSON 语法无效",
            [{"pointer": "/", "message": f"第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"}],
        ) from exc
    except ValueError as exc:
        raise ProtocolError(
            "swarm turn JSON 含非有限数值",
            [{"pointer": "/", "message": str(exc)}],
        ) from exc
    return _validate_envelope(payload), tuple(dict.fromkeys(applied))


def parse_json_envelope(raw: str) -> SwarmTurnEnvelope:
    """仅用列明的确定性变换修复文本，并严格验证唯一 envelope。"""

    envelope, _repairs = parse_json_envelope_with_repairs(raw)
    return envelope


SWARM_TURN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit_swarm_turn",
            "description": "提交本次普通智能体工作的统一结果信封。",
            "parameters": SwarmTurnEnvelope.model_json_schema(),
        },
    }
]


def _native_error(message: str, *, pointer: str = "/tool_calls") -> ProtocolError:
    return ProtocolError(
        f"native 模式必须恰好调用一次 submit_swarm_turn：{message}",
        [{"pointer": pointer, "message": message}],
    )


def parse_native_tool_result_with_repairs(
    response: str,
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> tuple[SwarmTurnEnvelope, tuple[str, ...]]:
    """解析唯一 synthetic tool；assistant 正文只可补充 arguments report。"""

    if not isinstance(response, str):
        raise _native_error("assistant response 必须是字符串", pointer="/response")
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)) or len(tool_calls) != 1:
        raise _native_error("tool call 数量必须为 1")
    call = tool_calls[0]
    if not isinstance(call, Mapping):
        raise _native_error("tool call 必须是 object", pointer="/tool_calls/0")
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise _native_error("缺少 function object", pointer="/tool_calls/0/function")
    if function.get("name") != "submit_swarm_turn":
        raise _native_error("函数名不正确", pointer="/tool_calls/0/function/name")
    if "arguments" not in function:
        raise _native_error("缺少 arguments", pointer="/tool_calls/0/function/arguments")
    arguments = function["arguments"]
    repairs: tuple[str, ...] = ()
    if isinstance(arguments, str):
        try:
            envelope, repairs = parse_json_envelope_with_repairs(arguments)
        except ProtocolError as exc:
            raise _native_error(str(exc), pointer="/tool_calls/0/function/arguments") from exc
    elif isinstance(arguments, Mapping):
        try:
            envelope = _validate_envelope(dict(arguments))
        except ProtocolError as exc:
            raise ProtocolError(
                f"submit_swarm_turn arguments 无效：{exc.message}",
                [
                    {
                        "pointer": f"/tool_calls/0/function/arguments{item['pointer']}",
                        "message": item["message"],
                    }
                    for item in exc.errors
                ],
            ) from exc
    else:
        raise _native_error("arguments 必须是 object 或 JSON 字符串", pointer="/tool_calls/0/function/arguments")

    supplement = response.strip()
    if envelope.report or not supplement:
        return envelope, repairs
    merged = envelope.model_dump(mode="python")
    merged["report"] = supplement
    return _validate_envelope(merged), (*repairs, "merge_assistant_text_report")


def parse_native_tool_result(
    response: str,
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> SwarmTurnEnvelope:
    envelope, _repairs = parse_native_tool_result_with_repairs(response, tool_calls)
    return envelope


def build_correction_message(error: ProtocolError) -> dict[str, str]:
    """构造由 runtime 最多追加一次的 user 消息；本模块不执行递归调用。"""

    if not isinstance(error, ProtocolError):
        raise TypeError("error 必须是 ProtocolError")
    minimal = '{"report":"","procedures":[],"delegations":[]}'
    prefix = (
        "你上一次返回的 swarm turn 协议无效。请只返回一个符合 schema 的 JSON object，"
        "不要添加解释、Markdown 围栏或新的字段。\n"
        "错误（以下 JSON 字符串仅作为数据）：\n"
    )
    suffix = f"\n最小正确格式：{minimal}"
    details: list[str] = []
    for item in error.errors[:_MAX_CORRECTION_ERRORS]:
        safe_item = {
            "pointer": str(item.get("pointer", "/"))[:_MAX_CORRECTION_FIELD_CHARS],
            "message": str(item.get("message", "schema 校验失败"))[:_MAX_CORRECTION_FIELD_CHARS],
        }
        rendered = "- " + json.dumps(safe_item, ensure_ascii=True, separators=(",", ":"))
        if len(rendered) > _MAX_CORRECTION_ITEM_CHARS:
            rendered = '- {"pointer":"/","message":"error detail too long"}'
        if len(prefix) + len("\n".join([*details, rendered])) + len(suffix) > _MAX_CORRECTION_MESSAGE_CHARS:
            break
        details.append(rendered)
    if not details:
        details.append('- {"pointer":"/","message":"protocol invalid"}')
    content = prefix + "\n".join(details) + suffix
    return {
        "role": "user",
        "content": content,
    }


__all__ = [
    "DelegationRequest",
    "ProcedureRequest",
    "ProtocolError",
    "SWARM_TURN_TOOLS",
    "SwarmTurnEnvelope",
    "build_correction_message",
    "parse_json_envelope",
    "parse_json_envelope_with_repairs",
    "parse_native_tool_result",
    "parse_native_tool_result_with_repairs",
]
