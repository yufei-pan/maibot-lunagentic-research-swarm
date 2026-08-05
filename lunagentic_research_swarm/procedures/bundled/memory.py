"""内置聊天 / 消息 / 人物 / 知识库记忆 Procedures。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult

Handler = Callable[[Any, Mapping[str, Any]], Awaitable[ProcedureResult]]

_OBJECT_SCHEMA: dict[str, Any] = {"type": "object"}

_MEMORY_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "procedure_id": "builtin.chat_streams",
        "display_name": "列出聊天流",
        "description": "按类型列出聊天流（全部 / 群聊 / 私聊）。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["all", "group", "private"]},
                "platform": {"type": "string"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "truncated": {"type": "integer"},
            },
        },
    },
    {
        "procedure_id": "builtin.message_recent",
        "display_name": "最近消息",
        "description": "读取指定聊天流的最近消息，返回可读文本与最小消息元数据。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["stream_id", "limit"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string"},
                "items": {"type": "array"},
                "readable": {"type": "string"},
                "truncated": {"type": "integer"},
            },
        },
    },
    {
        "procedure_id": "builtin.message_by_id",
        "display_name": "按 ID 取消息",
        "description": "按消息 ID 读取单条消息；永不请求二进制附件。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "minLength": 1},
                "stream_id": {"type": "string"},
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string"},
                "items": {"type": "array"},
                "readable": {"type": "string"},
                "truncated": {"type": "integer"},
            },
        },
    },
    {
        "procedure_id": "builtin.message_time_range",
        "display_name": "按时间范围取消息",
        "description": "按时间范围读取聊天流消息，再裁剪到显式 limit。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string", "minLength": 1},
                "start_time": {"type": "string", "minLength": 1},
                "end_time": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["stream_id", "start_time", "end_time", "limit"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string"},
                "items": {"type": "array"},
                "readable": {"type": "string"},
                "truncated": {"type": "integer"},
            },
        },
    },
    {
        "procedure_id": "builtin.person_lookup",
        "display_name": "人物查询",
        "description": "按 id / name / field 模式查询人物公开能力。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["id", "name", "field"]},
                "platform": {"type": "string"},
                "user_id": {"type": "string"},
                "person_name": {"type": "string"},
                "person_id": {"type": "string"},
                "field_name": {"type": "string"},
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "truncated": {"type": "integer"},
            },
        },
    },
    {
        "procedure_id": "builtin.knowledge_search",
        "display_name": "知识库搜索",
        "description": "在 Host 知识库中搜索相关条目。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "items": {"type": "array"},
                "truncated": {"type": "integer"},
            },
        },
    },
)


def memory_procedure_definitions() -> list[ProcedureDefinition]:
    """构造六个记忆类 Procedure 定义。"""

    definitions: list[ProcedureDefinition] = []
    for item in _MEMORY_DEFINITIONS:
        payload = {
            **item,
            "version": "1",
            "idempotent": True,
            "timeout_seconds": 30.0,
            "external_cost_kind": "none",
            "enabled": True,
            "result_schema": item.get("result_schema") or _OBJECT_SCHEMA,
        }
        definitions.append(ProcedureDefinition.model_validate(payload))
    return definitions


def _failure(code: str, message: str) -> ProcedureResult:
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata={},
    )


def _success(data: Mapping[str, Any]) -> ProcedureResult:
    return ProcedureResult(
        success=True,
        data=dict(data),
        error=None,
        metadata={},
    )


def _require_str(arguments: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str | ProcedureResult:
    value = arguments.get(key)
    if not isinstance(value, str):
        return _failure("invalid_arguments", f"{key} 必须为字符串")
    if not allow_empty and not value.strip():
        return _failure("invalid_arguments", f"{key} 不能为空")
    return value


def _require_int_in_range(
    arguments: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int | ProcedureResult:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return _failure("invalid_arguments", f"{key} 必须为整数")
    if value < minimum or value > maximum:
        return _failure("invalid_arguments", f"{key} 必须在 {minimum}..{maximum} 范围内")
    return value


def _message_item(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"message_id": "", "timestamp": ""}
    message_id = raw.get("message_id")
    if message_id is None:
        message_id = raw.get("id", "")
    timestamp = raw.get("timestamp")
    if timestamp is None:
        timestamp = raw.get("time", "")
    return {
        "message_id": "" if message_id is None else str(message_id),
        "timestamp": "" if timestamp is None else str(timestamp),
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


async def _chat_streams(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    kind = arguments.get("kind")
    if kind not in {"all", "group", "private"}:
        return _failure("invalid_arguments", "kind 必须为 all|group|private")
    platform_raw = arguments.get("platform", "qq")
    if not isinstance(platform_raw, str) or not platform_raw.strip():
        return _failure("invalid_arguments", "platform 必须为非空字符串")
    platform = platform_raw
    chat = getattr(ctx, "chat", None)
    if chat is None:
        return _failure("host_capability_failed", "ctx.chat 不可用")
    method_name = {
        "all": "get_all_streams",
        "group": "get_group_streams",
        "private": "get_private_streams",
    }[kind]
    method = getattr(chat, method_name, None)
    if not callable(method):
        return _failure("host_capability_failed", f"ctx.chat.{method_name} 不可用")
    try:
        streams = await method(platform)
    except Exception:
        return _failure("host_capability_failed", "聊天流查询失败")
    items = _as_list(streams)
    return _success({"items": items, "truncated": len(items)})


async def _message_recent(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    stream_id = _require_str(arguments, "stream_id")
    if isinstance(stream_id, ProcedureResult):
        return stream_id
    limit = _require_int_in_range(arguments, "limit", minimum=1, maximum=50)
    if isinstance(limit, ProcedureResult):
        return limit
    message = getattr(ctx, "message", None)
    if message is None:
        return _failure("host_capability_failed", "ctx.message 不可用")
    try:
        raw_messages = await message.get_recent(stream_id, limit)
        messages = _as_list(raw_messages)[:limit]
        readable = await message.build_readable(messages)
    except Exception:
        return _failure("host_capability_failed", "最近消息查询失败")
    items = [_message_item(item) for item in messages]
    return _success(
        {
            "stream_id": stream_id,
            "items": items,
            "readable": "" if readable is None else str(readable),
            "truncated": len(items),
        }
    )


async def _message_by_id(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    message_id = _require_str(arguments, "message_id")
    if isinstance(message_id, ProcedureResult):
        return message_id
    stream_id_raw = arguments.get("stream_id", "")
    if stream_id_raw is None:
        stream_id_raw = ""
    if not isinstance(stream_id_raw, str):
        return _failure("invalid_arguments", "stream_id 必须为字符串")
    stream_id = stream_id_raw
    message = getattr(ctx, "message", None)
    if message is None:
        return _failure("host_capability_failed", "ctx.message 不可用")
    try:
        raw = await message.get_by_id(
            message_id,
            stream_id=stream_id,
            include_binary_data=False,
        )
        messages = _as_list(raw)
        readable = await message.build_readable(messages)
    except Exception:
        return _failure("host_capability_failed", "按 ID 查询消息失败")
    items = [_message_item(item) for item in messages]
    return _success(
        {
            "stream_id": stream_id,
            "items": items,
            "readable": "" if readable is None else str(readable),
            "truncated": len(items),
        }
    )


async def _message_time_range(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    stream_id = _require_str(arguments, "stream_id")
    if isinstance(stream_id, ProcedureResult):
        return stream_id
    start_time = _require_str(arguments, "start_time")
    if isinstance(start_time, ProcedureResult):
        return start_time
    end_time = _require_str(arguments, "end_time")
    if isinstance(end_time, ProcedureResult):
        return end_time
    limit = _require_int_in_range(arguments, "limit", minimum=1, maximum=100)
    if isinstance(limit, ProcedureResult):
        return limit
    message = getattr(ctx, "message", None)
    if message is None:
        return _failure("host_capability_failed", "ctx.message 不可用")
    try:
        raw_messages = await message.get_by_time_in_chat(stream_id, start_time, end_time)
        messages = _as_list(raw_messages)[:limit]
        readable = await message.build_readable(messages)
    except Exception:
        return _failure("host_capability_failed", "按时间范围查询消息失败")
    items = [_message_item(item) for item in messages]
    return _success(
        {
            "stream_id": stream_id,
            "items": items,
            "readable": "" if readable is None else str(readable),
            "truncated": len(items),
        }
    )


async def _person_lookup(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    mode = arguments.get("mode")
    if mode not in {"id", "name", "field"}:
        return _failure("invalid_arguments", "mode 必须为 id|name|field")
    person = getattr(ctx, "person", None)
    if person is None:
        return _failure("host_capability_failed", "ctx.person 不可用")
    try:
        if mode == "id":
            platform = _require_str(arguments, "platform")
            if isinstance(platform, ProcedureResult):
                return platform
            user_id = _require_str(arguments, "user_id")
            if isinstance(user_id, ProcedureResult):
                return user_id
            value = await person.get_id(platform, user_id)
            items = [{"person_id": value, "platform": platform, "user_id": user_id}]
        elif mode == "name":
            person_name = _require_str(arguments, "person_name")
            if isinstance(person_name, ProcedureResult):
                return person_name
            value = await person.get_id_by_name(person_name)
            items = [{"person_id": value, "person_name": person_name}]
        else:
            person_id = _require_str(arguments, "person_id")
            if isinstance(person_id, ProcedureResult):
                return person_id
            field_name = _require_str(arguments, "field_name")
            if isinstance(field_name, ProcedureResult):
                return field_name
            value = await person.get_value(person_id, field_name)
            items = [{"person_id": person_id, "field_name": field_name, "value": value}]
    except Exception:
        return _failure("host_capability_failed", "人物查询失败")
    return _success({"items": items, "truncated": len(items)})


async def _knowledge_search(ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    query = _require_str(arguments, "query")
    if isinstance(query, ProcedureResult):
        return query
    if len(query) > 2000:
        return _failure("invalid_arguments", "query 长度必须在 1..2000 范围内")
    limit = _require_int_in_range(arguments, "limit", minimum=1, maximum=20)
    if isinstance(limit, ProcedureResult):
        return limit
    knowledge = getattr(ctx, "knowledge", None)
    if knowledge is None:
        return _failure("host_capability_failed", "ctx.knowledge 不可用")
    try:
        raw = await knowledge.search(query, limit)
    except Exception:
        return _failure("host_capability_failed", "知识库搜索失败")
    items = _as_list(raw)[:limit]
    return _success({"query": query, "items": items, "truncated": len(items)})


MEMORY_HANDLERS: dict[str, Handler] = {
    "builtin.chat_streams": _chat_streams,
    "builtin.message_recent": _message_recent,
    "builtin.message_by_id": _message_by_id,
    "builtin.message_time_range": _message_time_range,
    "builtin.person_lookup": _person_lookup,
    "builtin.knowledge_search": _knowledge_search,
}
