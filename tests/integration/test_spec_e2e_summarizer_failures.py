"""§23.2 / §13.4 summarizer failure E2E (FakeSummarizer + RuntimeHarness).

Pins durable FAILED / unavailable branch records and COMPLETED_WITH_ERRORS on
task-finalizer failure. Does not duplicate manager-level
``test_failed_final_synthesis_completes_with_errors_without_prompt_leak``.
"""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.llm.gateway import GenerationError
from lunagentic_research_swarm.models import BranchLifecycle, ReportKind, SummaryKind, TaskStatus


def _branch_finals(layer) -> dict[str, object]:
    return {
        str(row["branch_id"]): row
        for row in layer.summaries
        if str(row["kind"]) == SummaryKind.BRANCH_FINAL.value
    }


@pytest.mark.asyncio
async def test_spec_23_2_branch_finalize_fail_durable_unavailable_sibling_continues(
    runtime_harness,
) -> None:
    """§23.2 — failed finalize_branch → durable FAILED BRANCH_FINAL; sibling continues."""

    harness = runtime_harness
    await harness.start("分支总结失败", credits=100.0, time_budget=60)
    await harness.formalize("正式分支总结失败任务")
    await harness.root_delegates({"A": 50.0, "B": 50.0})

    # Selectively fail A via history marker (request has no branch_id).
    harness.summarizer.fail_branch_history_markers.add("A evidence")

    await harness.finalize_all()

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None
    finals = _branch_finals(layer)
    assert set(finals) == {"A", "B"}

    failed = finals["A"]
    assert failed["status"] == "FAILED"
    assert failed["text"] in (None, "")
    assert failed["error_code"] == "summary_unavailable"
    # No fabricated branch prose from the failed summarizer call.
    assert not (failed["text"] or "").strip()

    ok = finals["B"]
    assert ok["status"] == "SUCCEEDED"
    assert ok["text"] and "branch-summary:" in str(ok["text"])
    assert "A evidence" not in str(ok["text"])

    assert harness.coordinator is not None
    assert harness.coordinator.branches["A"].lifecycle is BranchLifecycle.FINALIZED
    assert harness.coordinator.branches["B"].lifecycle is BranchLifecycle.FINALIZED

    # Single available coverage → synthesis uses B's body (no invent from A).
    assert harness.reports[-1].kind is ReportKind.FINAL
    assert harness.reports[-1].status == "SUCCEEDED"
    assert "branch-summary:B evidence" in harness.reports[-1].text
    assert harness.task_status is TaskStatus.COMPLETED
    assert len(harness.summarizer.branch_requests) == 2


@pytest.mark.asyncio
async def test_spec_13_4_23_2_task_finalize_fail_completed_with_errors(runtime_harness) -> None:
    """§13.4 / §23.2 — finalize_task fail → COMPLETED_WITH_ERRORS; coverage kept, no invent."""

    harness = runtime_harness
    formalized = "正式任务级总结失败调查"
    await harness.start("任务总结失败", credits=100.0, time_budget=60)
    await harness.formalize(formalized)
    await harness.root_delegates({"A": 50.0, "B": 50.0})

    harness.summarizer.task_failure = GenerationError(
        "provider_unavailable", "summary provider unavailable"
    )

    await harness.finalize_all()

    assert harness.task_status is TaskStatus.COMPLETED_WITH_ERRORS
    assert harness.reports[-1].kind is ReportKind.FINAL
    assert harness.reports[-1].status == "FAILED"
    assert "最终报告总结器不可用" in harness.reports[-1].text
    # Deterministic unavailable body — not the FakeSummarizer success string.
    assert "task-summary" not in harness.reports[-1].text
    assert formalized not in (harness.reports[-1].error_message or "")

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None
    finals = _branch_finals(layer)
    assert len(finals) == 2
    assert all(row["status"] == "SUCCEEDED" and row["text"] for row in finals.values())
    # Prior branch coverage remains in the report's frozen coverage set.
    coverage_texts = [item.text for item in harness.reports[-1].coverage.items]
    assert len(coverage_texts) == 2
    assert all(text and "branch-summary:" in text for text in coverage_texts)

    durable_report = layer.reports[-1]
    assert durable_report["kind"] == "FINAL"
    assert durable_report["status"] == "FAILED"
    assert "最终报告总结器不可用" in str(durable_report["text"])
    assert len(harness.summarizer.task_requests) == 1
