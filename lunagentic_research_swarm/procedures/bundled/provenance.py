"""URL 标准化、去重与 provenance 整理 Procedures。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult

Handler = Callable[[Any, Mapping[str, Any]], Awaitable[ProcedureResult]]

_DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
}


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


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4 remove_dot_segments。"""

    input_buffer = path
    output: list[str] = []
    while input_buffer:
        if input_buffer.startswith("../"):
            input_buffer = input_buffer[3:]
        elif input_buffer.startswith("./"):
            input_buffer = input_buffer[2:]
        elif input_buffer.startswith("/./"):
            input_buffer = "/" + input_buffer[3:]
        elif input_buffer == "/.":
            input_buffer = "/"
        elif input_buffer.startswith("/../"):
            input_buffer = "/" + input_buffer[4:]
            if output:
                output.pop()
        elif input_buffer == "/..":
            input_buffer = "/"
            if output:
                output.pop()
        elif input_buffer in {".", ".."}:
            input_buffer = ""
        else:
            if input_buffer.startswith("/"):
                slash, rest = "/", input_buffer[1:]
            else:
                slash, rest = "", input_buffer
            if "/" in rest:
                segment, input_buffer = rest.split("/", 1)
                input_buffer = "/" + input_buffer
            else:
                segment, input_buffer = rest, ""
            output.append(slash + segment)
    return "".join(output)


def normalize_url(url: str) -> str:
    """标准化 URL：小写 scheme/host、IDNA、去默认端口、path 去点、去 fragment；保留 query。"""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url 必须为非空字符串")
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.hostname:
        raise ValueError("url 必须包含 scheme 与 host")
    scheme = parts.scheme.lower()
    try:
        host = parts.hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError) as exc:
        raise ValueError("host 未通过 IDNA 校验") from exc
    # urlsplit 对 IPv6 返回无方括号 hostname；重建 netloc 时必须加回括号
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and _DEFAULT_PORTS.get(scheme) == port:
        netloc = host
    elif port is not None:
        netloc = f"{host}:{port}"
    else:
        netloc = host
    if parts.username is not None:
        userinfo = parts.username
        if parts.password is not None:
            userinfo = f"{userinfo}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"
    path = _remove_dot_segments(parts.path or "")
    # 保留原始 query 字节顺序与值；丢弃 fragment
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def normalize_urls(items: Sequence[Mapping[str, Any]]) -> ProcedureResult:
    """按标准化 URL 去重：保留最先出现的 provenance，合并后续 source_id。"""

    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return _failure("invalid_arguments", "items 必须为数组")

    ordered: list[dict[str, Any]] = []
    index_by_url: dict[str, int] = {}
    duplicate_urls: list[str] = []
    seen_duplicates: set[str] = set()

    for raw in items:
        if not isinstance(raw, Mapping):
            return _failure("invalid_arguments", "items 元素必须为对象")
        url_raw = raw.get("url")
        if not isinstance(url_raw, str):
            return _failure("invalid_arguments", "url 必须为字符串")
        try:
            normalized = normalize_url(url_raw)
        except ValueError as exc:
            return _failure("invalid_url", str(exc))
        source_id = raw.get("source_id")
        source_ids: list[str] = []
        if source_id is not None:
            if not isinstance(source_id, str) or not source_id:
                return _failure("invalid_arguments", "source_id 必须为非空字符串")
            source_ids.append(source_id)

        existing = index_by_url.get(normalized)
        if existing is None:
            entry = {
                "url": normalized,
                "source_ids": source_ids,
                "title": "" if raw.get("title") is None else str(raw.get("title")),
                "snippet": "" if raw.get("snippet") is None else str(raw.get("snippet")),
            }
            index_by_url[normalized] = len(ordered)
            ordered.append(entry)
        else:
            if normalized not in seen_duplicates:
                duplicate_urls.append(normalized)
                seen_duplicates.add(normalized)
            if source_ids:
                merged = ordered[existing]["source_ids"]
                for sid in source_ids:
                    if sid not in merged:
                        merged.append(sid)

    return _success({"items": ordered, "duplicate_urls": duplicate_urls})


def organize_provenance(
    claims: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
) -> ProcedureResult:
    """整理 claim→source 映射；不判断真假，不改写 snippet。"""

    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        return _failure("invalid_arguments", "claims 必须为数组")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return _failure("invalid_arguments", "sources 必须为数组")

    source_rows: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    url_to_ids: dict[str, list[str]] = {}

    for raw in sources:
        if not isinstance(raw, Mapping):
            return _failure("invalid_arguments", "sources 元素必须为对象")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            return _failure("invalid_arguments", "source_id 必须为非空字符串")
        url_raw = raw.get("url", "")
        if url_raw is None:
            url_raw = ""
        if not isinstance(url_raw, str):
            return _failure("invalid_arguments", "url 必须为字符串")
        normalized = ""
        if url_raw.strip():
            try:
                normalized = normalize_url(url_raw)
            except ValueError as exc:
                return _failure("invalid_url", str(exc))
        source_type = raw.get("source_type", "")
        timestamp = raw.get("timestamp", "")
        snippet = raw.get("snippet", "")
        row = {
            "source_id": source_id,
            "url": normalized,
            "source_type": "" if source_type is None else str(source_type),
            "timestamp": "" if timestamp is None else str(timestamp),
            "snippet": "" if snippet is None else str(snippet),
        }
        source_rows.append(row)
        known_ids.add(source_id)
        if normalized:
            url_to_ids.setdefault(normalized, []).append(source_id)

    duplicate_urls = [url for url, ids in url_to_ids.items() if len(ids) > 1]

    claim_sources: dict[str, list[str]] = {}
    unbacked: list[str] = []
    seen_claim_ids: set[str] = set()
    for raw in claims:
        if not isinstance(raw, Mapping):
            return _failure("invalid_arguments", "claims 元素必须为对象")
        claim_id = raw.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            return _failure("invalid_arguments", "claim_id 必须为非空字符串")
        if claim_id in seen_claim_ids:
            return _failure("invalid_arguments", f"重复的 claim_id：{claim_id}")
        seen_claim_ids.add(claim_id)
        source_ids_raw = raw.get("source_ids", [])
        if source_ids_raw is None:
            source_ids_raw = []
        if not isinstance(source_ids_raw, list):
            return _failure("invalid_arguments", "source_ids 必须为数组")
        resolved: list[str] = []
        for sid in source_ids_raw:
            if not isinstance(sid, str) or not sid:
                return _failure("invalid_arguments", "source_ids 元素必须为非空字符串")
            if sid in known_ids and sid not in resolved:
                resolved.append(sid)
        claim_sources[claim_id] = resolved
        if not resolved:
            unbacked.append(claim_id)

    return _success(
        {
            "claim_sources": claim_sources,
            "unbacked_claims": unbacked,
            "duplicate_urls": duplicate_urls,
            "sources": source_rows,
        }
    )


_PROVENANCE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "procedure_id": "builtin.normalize_urls",
        "display_name": "URL 标准化去重",
        "description": "标准化 URL 并按规范化结果去重，合并 source_id，保留首次 provenance。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "source_id": {"type": "string"},
                            "title": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                        "required": ["url"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "duplicate_urls": {"type": "array"},
            },
        },
    },
    {
        "procedure_id": "builtin.organize_provenance",
        "display_name": "来源整理",
        "description": "整理 claim 与 source 映射，标出无依据声明与重复 URL；不改写摘要。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "text": {"type": "string"},
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["claim_id"],
                        "additionalProperties": True,
                    },
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "url": {"type": "string"},
                            "source_type": {"type": "string"},
                            "timestamp": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                        "required": ["source_id"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["claims", "sources"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "claim_sources": {"type": "object"},
                "unbacked_claims": {"type": "array"},
                "duplicate_urls": {"type": "array"},
                "sources": {"type": "array"},
            },
        },
    },
)


def provenance_procedure_definitions() -> list[ProcedureDefinition]:
    """构造来源处理 Procedure 定义。"""

    definitions: list[ProcedureDefinition] = []
    for item in _PROVENANCE_DEFINITIONS:
        payload = {
            **item,
            "version": "1",
            "idempotent": True,
            "timeout_seconds": 30.0,
            "external_cost_kind": "none",
            "enabled": True,
        }
        definitions.append(ProcedureDefinition.model_validate(payload))
    return definitions


async def _normalize_urls(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    items = arguments.get("items")
    if not isinstance(items, list):
        return _failure("invalid_arguments", "items 必须为数组")
    return normalize_urls(items)


async def _organize_provenance(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    claims = arguments.get("claims")
    sources = arguments.get("sources")
    if not isinstance(claims, list):
        return _failure("invalid_arguments", "claims 必须为数组")
    if not isinstance(sources, list):
        return _failure("invalid_arguments", "sources 必须为数组")
    return organize_provenance(claims, sources)


PROVENANCE_HANDLERS: dict[str, Handler] = {
    "builtin.normalize_urls": _normalize_urls,
    "builtin.organize_provenance": _organize_provenance,
}
