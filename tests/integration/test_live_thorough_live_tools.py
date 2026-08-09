# tests/integration/test_live_thorough_live_tools.py
from __future__ import annotations

import pytest

from live_llm import deep_judge, live_tools_available, load_live_llm_credentials

from lunagentic_research_swarm.models import ReportKind, TaskStatus

# Concrete, searchable question: forces real web_search before concluding.
LIVE_TOOLS_OBJECTIVE = (
    "调研问题：加州高铁（California High-Speed Rail）Merced–Bakersfield 段"
    "当前规划开通目标年份（或最新官方业务计划中给出的服务日期）是什么？"
    "硬性要求：必须至少一次调用 builtin.web_search；"
    "arguments 必须同时包含 engine 与 query；优先 engine=\"searxng\"（若失败可改 duckduckgo）；"
    "query 须包含子串 \"California High-Speed Rail\" 与 \"Merced\"。"
    "根据检索到的网页结果给出简短结论，并在 report 中点名至少一处来源标题或域名；"
    "若某次搜索失败，必须换引擎或改写 query 再搜，不得在无结果时直接终结；"
    "禁止在未搜索的情况下凭记忆直接作答。"
)

JUDGE_OBJECTIVE = (
    "回答加州高铁 Merced–Bakersfield 段的开通/服务目标年份或官方计划日期，"
    "且结论应体现已使用网页检索（提及来源或检索依据）。"
)

pytestmark = [
    pytest.mark.live_llm_live_tools,
    pytest.mark.skipif(not live_tools_available(), reason="未启用 web_search_enabled 或缺少 LLM 凭证"),
]


def _status_value(status: dict) -> str:
    raw = status.get("status")
    return raw.value if hasattr(raw, "value") else str(raw)


def _final_report_text(harness) -> str:
    assert harness.reports, "expected at least one report on coordinator"
    final = harness.reports[-1]
    kind = getattr(final, "kind", None)
    if kind is not None:
        assert kind is ReportKind.FINAL, f"last report kind={kind!r}"
    text = str(getattr(final, "text", None) or getattr(final, "body", None) or final).strip()
    assert text, "FINAL report must be non-empty"
    return text


@pytest.mark.asyncio
async def test_live_thorough_real_web_search_and_deep_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    harness.use_bundled_agents()
    harness.use_real_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_live_procedures(creds.web_search)
    await harness.start(
        LIVE_TOOLS_OBJECTIVE,
        credits=150.0,
        time_budget=900,
    )
    await harness.manager.wait_idle(harness.task_id)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=tmp_path / "live_tools",
    )
    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }
    assert harness.live_search_invokes >= 1, (
        f"expected builtin.web_search invoke, got {harness.live_search_invokes}"
    )
    final_text = _final_report_text(harness)
    verdict = await deep_judge(
        creds,
        objective=JUDGE_OBJECTIVE,
        report=final_text,
        evidence="",
    )
    assert verdict["pass"] is True, verdict
