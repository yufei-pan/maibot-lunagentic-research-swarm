# tests/integration/test_live_drive_offline.py
from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeLLMResponse
from live_harness import drive_until_terminal

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID


@pytest.mark.asyncio
async def test_drive_until_terminal_with_scripted_terminate(runtime_harness) -> None:
    harness = runtime_harness
    await harness.start("离线驱动自测", credits=40.0, time_budget=60)
    await harness.formalize(
        "正式任务：用一次 turn 调用 core.terminate 并给出简短 report。"
    )
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "离线完成",
                "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}, "credits": 0}],
                "delegations": [],
            }
        )
    )
    status = await drive_until_terminal(harness, timeout_seconds=30)
    raw = status.get("status")
    value = raw.value if hasattr(raw, "value") else str(raw)
    assert value in {TaskStatus.COMPLETED.value, TaskStatus.COMPLETED_WITH_ERRORS.value}

    # Empty-queue FakeLLM soft-default also reaches COMPLETED via reason=no_further_work;
    # require terminate wiring evidence so that path fails this test.
    reasons = list(getattr(harness, "live_drain_branch_summary_reasons", []) or [])
    assert reasons, "drain 应记录 PerformBranchSummary.reason"
    assert reasons[-1] == "terminate", f"expected finalize reason terminate, got {reasons!r}"

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None and layer.summaries
    assert layer.summaries[-1]["kind"] == "BRANCH_FINAL"
    summary_text = str(layer.summaries[-1].get("text") or "")
    assert "离线完成" in summary_text
    assert "fake response" not in summary_text
    assert harness.summarizer.branch_requests, "terminate 路径必须调用 finalize_branch"


@pytest.mark.asyncio
async def test_drive_until_terminal_timeout_dumps_artifacts(runtime_harness, tmp_path: Path) -> None:
    """Blocked FakeLLM + short timeout must dump artifacts then raise TimeoutError."""

    harness = runtime_harness
    await harness.start("超时制品自测", credits=40.0, time_budget=60)
    await harness.formalize("正式任务：阻塞 LLM 以触发 drain 超时。")
    harness.llm.block()
    artifact_dir = tmp_path / "timeout_artifacts"
    with pytest.raises(TimeoutError, match="drive_until_terminal 超时"):
        await drive_until_terminal(harness, timeout_seconds=0.2, artifact_dir=artifact_dir)
    assert artifact_dir.is_dir()
    assert (artifact_dir / "final_status.json").is_file()
    assert (artifact_dir / "scheduler_pending.txt").is_file()
    harness.llm.release()
