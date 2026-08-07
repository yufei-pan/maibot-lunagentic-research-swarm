"""真实 LLM 调用辅助：读取仓库根 `.debug_api_call_credentials`。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

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


__all__ = [
    "CREDENTIALS_PATH",
    "LiveLLMCredentials",
    "chat_completion",
    "credentials_available",
    "live_tools_available",
    "load_live_llm_credentials",
]
