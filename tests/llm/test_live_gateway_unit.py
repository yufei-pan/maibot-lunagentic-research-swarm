# tests/llm/test_live_gateway_unit.py
from __future__ import annotations

from typing import Any

import pytest

from live_llm import LiveLLMCredentials, LiveLLMGateway
from lunagentic_research_swarm.llm.gateway import GenerationRequest


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


@pytest.mark.asyncio
async def test_live_gateway_generate_maps_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = LiveLLMCredentials(
        base_url="http://127.0.0.1:9/v1",
        api_key="sk-test",
        model="local-model",
        temperature=1.0,
    )
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse(
                {
                    "model": "local-model",
                    "choices": [{"message": {"content": '{"report":"ok","procedures":[],"delegations":[]}'}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            )

    monkeypatch.setattr("live_llm.httpx.AsyncClient", _Client)
    gateway = LiveLLMGateway(creds)
    result = await gateway.generate(
        GenerationRequest(selector="task:mid_memory", messages=[{"role": "user", "content": "hi"}])
    )
    assert result.success is True
    assert "report" in result.response
    assert result.model_name == "local-model"
    assert captured["json"]["model"] == "local-model"
    assert captured["json"]["temperature"] == 1.0
    assert gateway.calls[0]["selector"] == "task:mid_memory"
