"""真实 LLM 调用辅助：读取仓库根 `.debug_api_call_credentials`。"""

from __future__ import annotations

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


__all__ = [
    "CREDENTIALS_PATH",
    "LiveLLMCredentials",
    "LiveLLMGateway",
    "chat_completion",
    "credentials_available",
    "live_tools_available",
    "load_live_llm_credentials",
]
