# tests/integration/test_live_e2e_ab.py
from __future__ import annotations

import pytest
from fakes import RuntimeHarness
from live_llm import credentials_available, light_judge, load_live_llm_credentials

from lunagentic_research_swarm.models import ReportKind, TaskStatus

pytestmark = [
    pytest.mark.live_llm_e2e,
    pytest.mark.skipif(not credentials_available(), reason="未配置可用的 .debug_api_call_credentials"),
]


def _status_value(status: dict) -> str:
    raw = status.get("status")
    return raw.value if hasattr(raw, "value") else str(raw)


def _final_report_text(harness) -> str:
    """Resolve FINAL report body via ``harness.reports`` (ReportRecord.text)."""

    assert harness.reports, "expected at least one report on coordinator"
    final = harness.reports[-1]
    assert final.kind is ReportKind.FINAL, f"last report kind={getattr(final, 'kind', None)!r}"
    text = str(getattr(final, "text", "") or "").strip()
    assert text, "FINAL report must be non-empty"
    return text


@pytest.mark.asyncio
async def test_live_e2e_a_thin_terminate_and_light_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    # Seeded formalize guides the live agent turn; light_judge scores the FINAL
    # body (not the raw JSON envelope), so keep the objective outcome-oriented.
    formalized = (
        "# 正式任务\n完成一次最小自测：在一个 turn 内调用 core.terminate 结束，"
        "并在 report 中用一两句明确说明「已完成最小自测」。不要写长文。\n"
    )
    await harness.start("最小协议自测", credits=50.0, time_budget=120)
    await harness.formalize(formalized)  # seeded via FakeSummarizer
    harness.use_live_llm(creds)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.e2e_timeout_seconds,
        artifact_dir=tmp_path / "a",
    )
    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }
    final_text = _final_report_text(harness)
    assert harness.llm.calls, "expected at least one live LLM call"
    verdict = await light_judge(creds, objective=formalized, report=final_text)
    assert verdict["pass"] is True, verdict


@pytest.mark.asyncio
async def test_live_e2e_b_root_delegates_child_then_finishes(tmp_path) -> None:
    creds = load_live_llm_credentials()
    # Agent seed (protocol) vs judge objective (FINAL outcome) — FINAL never embeds agent_id.
    # Include a concrete JSON example: protocol_prompt demos builtin.researcher, but the
    # integration catalog only exposes builtin.quick_thinker.
    formalized = (
        "# 正式任务\n"
        "【硬性要求】第一次回复必须包含非空 delegations，禁止只调用 core.terminate。\n"
        "示例（字段名必须一致）：\n"
        '{"report":"已委派子分支","procedures":[],'
        '"delegations":[{"agent_id":"builtin.quick_thinker",'
        '"task":"写一句含「子分支工作完成」的 report 后 terminate","credits":20}]}\n'
        "子分支 report 必须包含短语「子分支工作完成」，然后 terminate；"
        "根分支在委派后可 terminate。\n"
    )
    judge_objective = (
        "多分支协作已正常结束，最终说明中应体现子分支完成了工作"
        "（例如出现「子分支工作完成」或等价陈述）。"
    )
    timeout = max(float(creds.e2e_timeout_seconds), 300.0)
    # One extra attempt for local-model flake where root terminates without delegating.
    last_detail = ""
    for attempt in range(2):
        harness = RuntimeHarness(tmp_path / f"b-{attempt}")
        await harness.open()
        try:
            await harness.start("多分支自测", credits=80.0, time_budget=180)
            await harness.formalize(formalized)
            harness.use_live_llm(creds)
            status = await harness.drive_live_until_terminal(
                timeout_seconds=timeout,
                artifact_dir=tmp_path / f"b-{attempt}-artifacts",
            )
            assert _status_value(status) in {
                TaskStatus.COMPLETED.value,
                TaskStatus.COMPLETED_WITH_ERRORS.value,
            }
            child_rows = await harness.store_count_child_branches()
            if child_rows < 1:
                last_detail = f"attempt={attempt} children=0 calls={len(harness.llm.calls)}"
                continue
            final_text = _final_report_text(harness)
            assert harness.llm.calls, "expected at least one live LLM call"
            verdict = await light_judge(creds, objective=judge_objective, report=final_text)
            assert verdict["pass"] is True, verdict
            return
        finally:
            await harness.close()
    pytest.fail(f"B 未物化子分支（local model flake）：{last_detail}")
