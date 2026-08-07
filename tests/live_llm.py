"""真实 LLM 调用辅助：读取仓库根 `.debug_api_call_credentials`。"""

from __future__ import annotations

import json
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from lunagentic_research_swarm.llm.gateway import GenerationRequest, GenerationResult
from lunagentic_research_swarm.llm.pricing import TokenUsage

REPO_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = REPO_ROOT / ".debug_api_call_credentials"

_DEEP_SCORE_KEYS = ("relevance", "completeness", "groundedness")
_DEEP_SCORE_MIN = 3


@dataclass(frozen=True, slots=True)
class LiveLLMCredentials:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: float = 180.0
    max_tokens: int | None = None
    e2e_timeout_seconds: float = 180.0
    thorough_timeout_seconds: float = 900.0
    web_search_enabled: bool = False
    web_search: dict[str, Any] = field(default_factory=dict)


def credentials_available() -> bool:
    if not CREDENTIALS_PATH.is_file():
        return False
    try:
        creds = load_live_llm_credentials()
    except Exception:
        return False
    if "YOUR_API_KEY" in creds.api_key or "example" in creds.base_url:
        return False
    return bool(creds.base_url.strip() and creds.api_key.strip() and creds.model.strip())


def live_tools_available() -> bool:
    if not credentials_available():
        return False
    return bool(load_live_llm_credentials().web_search_enabled)


def load_live_llm_credentials(path: Path | None = None) -> LiveLLMCredentials:
    target = path or CREDENTIALS_PATH
    raw = tomllib.loads(target.read_text(encoding="utf-8"))
    max_tokens = raw.get("max_tokens")
    return LiveLLMCredentials(
        base_url=str(raw["base_url"]).rstrip("/"),
        api_key=str(raw["api_key"]),
        model=str(raw["model"]),
        temperature=float(raw.get("temperature", 0.2)),
        timeout_seconds=float(raw.get("timeout_seconds", 180)),
        max_tokens=int(max_tokens) if max_tokens not in (None, "") else None,
        e2e_timeout_seconds=float(raw.get("e2e_timeout_seconds", 180)),
        thorough_timeout_seconds=float(raw.get("thorough_timeout_seconds", 900)),
        web_search_enabled=bool(raw.get("web_search_enabled", False)),
        web_search=dict(raw.get("web_search") or {}),
    )


async def chat_completion(
    credentials: LiveLLMCredentials,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """调用 OpenAI-compatible `/chat/completions`，返回标准化 result dict。"""

    payload: dict[str, Any] = {
        "model": credentials.model,
        "messages": messages,
        "temperature": credentials.temperature,
    }
    if credentials.max_tokens is not None:
        payload["max_tokens"] = credentials.max_tokens
    headers = {
        "Authorization": f"Bearer {credentials.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{credentials.base_url}/chat/completions"
    async with httpx.AsyncClient(timeout=credentials.timeout_seconds) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    usage = body.get("usage") or {}
    return {
        "success": True,
        "response": str(content),
        "model": str(body.get("model") or credentials.model),
        "model_name": str(body.get("model") or credentials.model),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens") or 0),
        "prompt_cache_miss_tokens": int(
            usage.get("prompt_cache_miss_tokens") or usage.get("prompt_tokens") or 0
        ),
        "raw": body,
    }


def _messages_for_chat(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        text = messages.strip()
        return [{"role": "user", "content": text}] if text else []
    if messages is None:
        return []
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
        else:
            normalized.append({"role": "user", "content": str(item)})
    return normalized


class LiveLLMGateway:
    """OpenAI-compatible chat completions → ``GenerationResult`` for TurnWorker live drives."""

    def __init__(self, credentials: LiveLLMCredentials) -> None:
        self._credentials = credentials
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: GenerationRequest | None = None,
        *,
        selector: str | None = None,
        messages: Any = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if request is not None:
            selector = request.selector.raw
            messages = request.messages
            kwargs = {
                "tools": request.tools,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
        call = {"selector": selector or "", "messages": messages, **kwargs}
        self.calls.append(call)

        credentials = self._credentials
        chat_messages = _messages_for_chat(messages)
        payload: dict[str, Any] = {
            "model": credentials.model,
            "messages": chat_messages,
            "temperature": credentials.temperature,
        }
        max_tokens = credentials.max_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        tools = kwargs.get("tools")
        if tools is not None:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{credentials.base_url}/chat/completions"
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=credentials.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        duration = time.perf_counter() - started

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        raw_tool_calls = message.get("tool_calls")
        tool_calls: list[dict[str, Any]] | None = None
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls = [dict(item) for item in raw_tool_calls if isinstance(item, dict)]
            if not tool_calls:
                tool_calls = None

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cache_hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        cache_miss = int(usage.get("prompt_cache_miss_tokens") or usage.get("prompt_tokens") or 0)
        return GenerationResult(
            response=str(content),
            tool_calls=tool_calls,
            model_name=str(body.get("model") or credentials.model),
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_hit_tokens=cache_hit,
                cache_miss_tokens=cache_miss,
                source="actual",
            ),
            success=True,
            error=None,
            duration=duration,
        )


def _extract_first_json_object(text: str) -> str | None:
    """从回复中提取第一个括号平衡的 ``{...}`` 对象（允许前后有包装文字）。"""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
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
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_judge_payload(raw: str) -> dict[str, Any] | None:
    candidate = _extract_first_json_object(raw) or raw.strip()
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def _judge_once(
    credentials: LiveLLMCredentials,
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    result = await chat_completion(credentials, messages)
    response = str((result or {}).get("response") or "")
    return _parse_judge_payload(response)


async def _judge_with_one_parse_retry(
    credentials: LiveLLMCredentials,
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    payload = await _judge_once(credentials, messages)
    if payload is not None:
        return payload
    return await _judge_once(credentials, messages)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "pass"}:
            return True
        if lowered in {"false", "no", "0", "fail"}:
            return False
    return False


async def light_judge(
    credentials: LiveLLMCredentials,
    *,
    objective: str,
    report: str,
) -> dict[str, Any]:
    """轻量判定：FINAL 是否回应形式化目标。仅返回 ``pass`` / ``reason``。"""

    messages = [
        {
            "role": "system",
            "content": (
                "你是严格的研究报告评审。只输出一个 JSON 对象，不要 Markdown、不要代码围栏、不要其它文字。"
                '键必须且仅有：{"pass": bool, "reason": string}。'
                "pass=true 仅当报告实质回应了 objective；否则 pass=false。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"objective:\n{objective}\n\n"
                f"report:\n{report}\n\n"
                '请判定报告是否回应目标，并只输出 JSON：{"pass": bool, "reason": string}'
            ),
        },
    ]
    payload = await _judge_with_one_parse_retry(credentials, messages)
    if payload is None:
        return {"pass": False, "reason": "judge JSON parse failed after one retry"}
    return {
        "pass": _coerce_bool(payload.get("pass")),
        "reason": str(payload.get("reason") or "").strip() or "no reason",
    }


def _normalize_deep_scores(raw_scores: Any) -> dict[str, int]:
    source = raw_scores if isinstance(raw_scores, Mapping) else {}
    scores: dict[str, int] = {}
    for key in _DEEP_SCORE_KEYS:
        value = source.get(key, 0)
        try:
            scores[key] = int(value)
        except (TypeError, ValueError):
            scores[key] = 0
    return scores


async def deep_judge(
    credentials: LiveLLMCredentials,
    *,
    objective: str,
    report: str,
    evidence: str = "",
) -> dict[str, Any]:
    """深度量规：relevance / completeness / groundedness（1–5），Python 强制三项均 ≥ 3 才 pass。"""

    messages = [
        {
            "role": "system",
            "content": (
                "你是严格的研究报告评审。只输出一个 JSON 对象，不要 Markdown、不要代码围栏、不要其它文字。"
                "键必须包含：pass (bool)、reason (string)、"
                'scores 对象 {"relevance": int, "completeness": int, "groundedness": int}，'
                "三项分数均为 1–5 整数。"
                "relevance：是否切题；completeness：是否覆盖目标；groundedness：是否可被 evidence/事实支撑。"
                "仅当三项均 ≥ 3 时才可设 pass=true；否则 pass=false。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"objective:\n{objective}\n\n"
                f"report:\n{report}\n\n"
                f"evidence:\n{evidence}\n\n"
                "请按量规打分并只输出 JSON："
                '{"pass": bool, "reason": string, '
                '"scores": {"relevance": int, "completeness": int, "groundedness": int}}'
            ),
        },
    ]
    payload = await _judge_with_one_parse_retry(credentials, messages)
    if payload is None:
        return {
            "pass": False,
            "reason": "judge JSON parse failed after one retry",
            "scores": {key: 0 for key in _DEEP_SCORE_KEYS},
        }
    scores = _normalize_deep_scores(payload.get("scores"))
    # 深度 pass 仅由分数门槛决定，无视模型自报的 pass（禁止静默放行）。
    passed = all(scores[key] >= _DEEP_SCORE_MIN for key in _DEEP_SCORE_KEYS)
    reason = str(payload.get("reason") or "").strip() or "no reason"
    model_pass = _coerce_bool(payload.get("pass"))
    if model_pass != passed:
        reason = f"{reason} (scores gate: require all >= {_DEEP_SCORE_MIN}, got {scores})"
    return {"pass": passed, "reason": reason, "scores": scores}


__all__ = [
    "CREDENTIALS_PATH",
    "LiveLLMCredentials",
    "LiveLLMGateway",
    "chat_completion",
    "credentials_available",
    "deep_judge",
    "light_judge",
    "live_tools_available",
    "load_live_llm_credentials",
]
