"""Live MaiBot-like E2E: plugin tool → real LLM → real web_search → report.

This is the production-shaped offline drain (not FakeLLM scripting). Requires
``.debug_api_call_credentials`` with ``web_search_enabled = true``.
"""

from __future__ import annotations

import pytest
from live_llm import deep_judge, live_tools_available, load_live_llm_credentials

from lunagentic_research_swarm.models import ReportKind, TaskStatus

OBJECTIVE = (
    "调研问题：加州高铁（California High-Speed Rail）Merced–Bakersfield 段"
    "当前规划开通目标年份（或最新官方业务计划中给出的服务日期）是什么？"
    "硬性要求："
    "1) 必须至少一次调用 builtin.web_search（arguments 同时含 engine 与 query；"
    "优先 engine=\"searxng\"，失败可改 ddgs；query 须含 "
    "\"California High-Speed Rail\" 与 \"Merced\"）；"
    "2) 必须至少一次把子任务委派给目录中的另一个专职智能体（非空 delegations）；"
    "3) 最终结论要点名至少一处来源标题或域名；"
    "禁止未搜索就凭记忆作答。"
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
async def test_live_e2e_plugin_tool_real_llm_tools_to_report(
    runtime_harness, plugin_module, tmp_path
) -> None:
    """As close to MaiBot as the offline harness gets: tool entry + live model + live search."""

    creds = load_live_llm_credentials()
    harness = runtime_harness
    await harness.open()
    harness.use_bundled_agents()
    harness.use_real_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_live_procedures(creds.web_search)

    plugin = plugin_module.create_plugin()
    plugin._manager = harness.manager

    started = await plugin.start_deep_research(
        objective=OBJECTIVE,
        time_budget_seconds=600,
        effort_level=1.5,
        stream_id=harness.stream_id,
    )
    assert started.get("success") is True, started
    harness.task_id = str(started["task_id"])
    harness.round_id = str((await harness.manager.status(harness.task_id))["round_id"])
    harness.coordinator = harness.manager.report_coordinators.get(harness.task_id)

    # Live summarizer formalizes without FakeSummarizer gate.
    await harness.manager.wait_idle(harness.task_id)
    harness.coordinator = harness.manager.report_coordinators[harness.task_id]
    harness.formalized_task = (await harness.store.load_task(harness.task_id)).formalized_task
    assert harness.formalized_task is not None

    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=tmp_path / "live_plugin_tool_e2e",
    )
    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }, status

    assert harness.llm.calls, "expected live agent LLM calls"
    assert harness.live_search_invokes >= 1, (
        f"expected builtin.web_search, got {harness.live_search_invokes}"
    )

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None and layer.reports, "must persist a report"
    final_text = _final_report_text(harness)

    # Multi-branch / retire_parent path: at least one BRANCH_FINAL beyond a trivial single leaf
    # is best-effort (model-dependent); always require a FINAL report body.
    verdict = await deep_judge(
        creds,
        objective=JUDGE_OBJECTIVE,
        report=final_text,
        evidence="",
    )
    assert verdict["pass"] is True, verdict
