# tests/llm/test_live_gateway_unit.py
from __future__ import annotations

from typing import Any

import pytest

from live_llm import LiveLLMCredentials, LiveLLMGateway
from lunagentic_research_swarm.llm.gateway import GenerationRequest


class _FakeResponse:
    def __init__(self, body: dict[str, Any], *, status_code: int = 200, text: str = "") -> None:
        self._body = body
        self.status_code = status_code
        self.text = text
        self.request = None

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


@pytest.mark.asyncio
async def test_live_gateway_reports_http_error_as_a_failed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """上游 4xx 必须降级为失败结果并保留响应正文。

    抛异常会让整个 drain 中止，而不是只终结这一条分支；Host 的 LLM capability
    也是返回失败结果而不是抛出。响应正文是服务器唯一说明原因的地方
    （如「context length exceeded」），必须带出来。
    """

    creds = LiveLLMCredentials(base_url="http://127.0.0.1:9/v1", api_key="sk-test", model="local-model")

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
            return _FakeResponse({}, status_code=400, text="context length exceeded: 41000 > 32768")

    monkeypatch.setattr("live_llm.httpx.AsyncClient", _Client)
    gateway = LiveLLMGateway(creds)
    result = await gateway.generate(
        GenerationRequest(selector="task:mid_memory", messages=[{"role": "user", "content": "hi"}])
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "http_400"
    assert "context length exceeded" in result.error.message
    assert gateway.exchanges[-1]["error"]["code"] == "http_400"
