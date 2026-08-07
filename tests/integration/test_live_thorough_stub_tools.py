# tests/integration/test_live_thorough_stub_tools.py
from __future__ import annotations

import pytest

from live_llm import credentials_available, deep_judge, load_live_llm_credentials

from lunagentic_research_swarm.models import ReportKind, TaskStatus

STUB_FACT = "LRS_STUB_FACT_CALIFORNIA_RAIL_42"

pytestmark = [
    pytest.mark.live_llm_thorough,
    pytest.mark.skipif(not credentials_available(), reason="未配置可用的 .debug_api_call_credentials"),
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
async def test_live_thorough_stub_web_search_and_deep_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    harness.use_live_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_stub_procedures(
        {
            "california": {
                "results": [
                    {
                        "title": "California HSRA Budget Timeline (Stub)",
                        "url": "https://example.invalid/rail",
                        "snippet": (
                            "2008 Proposition 1A authorized ~$9.95B; 2019 business plan cited "
                            f"~$80B system cost. Budget note {STUB_FACT}."
                        ),
                    }
                ]
            }
        }
    )
    await harness.start(
        (
            "调查加州高铁（California High-Speed Rail）预算时间线。"
            "硬性要求：必须先调用 builtin.web_search（query 含 california），"
            "再下结论；最终报告必须原样引用检索 snippet 中的关键事实标记字符串。"
        ),
        credits=120.0,
        time_budget=600,
    )
    await harness.manager.wait_idle(harness.task_id)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=tmp_path / "thorough",
    )
    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }
    assert harness.stub_search_invokes >= 1, (
        f"expected builtin.web_search stub invoke, got {harness.stub_search_invokes}"
    )
    final_text = _final_report_text(harness)
    verdict = await deep_judge(
        creds,
        objective="加州高铁预算时间线（须使用搜索）",
        report=final_text,
        evidence=STUB_FACT,
    )
    assert verdict["pass"] is True, verdict
