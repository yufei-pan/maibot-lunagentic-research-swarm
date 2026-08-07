# tests/integration/test_live_thorough_live_tools.py
from __future__ import annotations

import pytest

from live_llm import deep_judge, live_tools_available, load_live_llm_credentials

pytestmark = [
    pytest.mark.live_llm_live_tools,
    pytest.mark.skipif(not live_tools_available(), reason="未启用 web_search_enabled 或缺少 LLM 凭证"),
]


@pytest.mark.asyncio
async def test_live_thorough_real_web_search_and_deep_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    harness.use_live_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_live_procedures(creds.web_search)
    await harness.start(
        "用真实网页搜索查证一个简单事实性问题，并给出有依据的简短结论。",
        credits=150.0,
        time_budget=900,
    )
    await harness.manager.wait_idle(harness.task_id)
    await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=tmp_path / "live_tools",
    )
    assert harness.live_search_invokes >= 1
    last = harness.reports[-1]
    final_text = str(getattr(last, "text", None) or getattr(last, "body", None) or last)
    verdict = await deep_judge(
        creds,
        objective="真实网页搜索事实核查",
        report=final_text,
        evidence="",
    )
    assert verdict["pass"] is True, verdict
