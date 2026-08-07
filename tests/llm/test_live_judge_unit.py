# tests/llm/test_live_judge_unit.py
from __future__ import annotations

import json
from typing import Any

import pytest

from live_llm import LiveLLMCredentials, deep_judge, light_judge

CREDS = LiveLLMCredentials("http://127.0.0.1:9/v1", "sk", "m", temperature=1.0)


@pytest.mark.asyncio
async def test_light_judge_parses_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_chat(_creds: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"success": True, "response": json.dumps({"pass": True, "reason": "addresses objective"}, ensure_ascii=False)}

    monkeypatch.setattr("live_llm.chat_completion", _fake_chat)
    result = await light_judge(CREDS, objective="终结并报告", report="已终结：完成最小自测。")
    assert result["pass"] is True


@pytest.mark.asyncio
async def test_deep_judge_requires_scores_ge_3(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_chat(_creds: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "success": True,
            "response": json.dumps(
                {
                    "pass": True,
                    "reason": "ok",
                    "scores": {"relevance": 4, "completeness": 3, "groundedness": 2},
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr("live_llm.chat_completion", _fake_chat)
    result = await deep_judge(CREDS, objective="查 X", report="...", evidence="TOKEN")
    assert result["pass"] is False  # groundedness 2 < 3 overrides model pass
