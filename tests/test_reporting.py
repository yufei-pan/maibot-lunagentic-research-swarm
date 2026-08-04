from __future__ import annotations

from lunagentic_research_swarm.models import ReportKind, SummaryKind
from lunagentic_research_swarm.reporting import CoverageSummary, build_coverage, freeze_report_kind, render_report


def test_coverage_keeps_terminal_and_latest_active_checkpoint_in_stable_order() -> None:
    coverage = build_coverage(
        [
            CoverageSummary("old", "A", SummaryKind.CHECKPOINT, 1, "old checkpoint", 1.0),
            CoverageSummary("final-b", "B", SummaryKind.BRANCH_FINAL, 1, "B final", 4.0),
            CoverageSummary("new", "A", SummaryKind.CHECKPOINT, 2, "A latest", 3.0),
            CoverageSummary("failed", "C", SummaryKind.CHECKPOINT, 1, None, 2.0, status="FAILED"),
        ],
        active_branch_ids={"A", "C"},
    )

    assert [item.summary_id for item in coverage.items] == ["new", "final-b"]
    assert coverage.unavailable_count == 1


def test_checkpoint_input_freezes_intermediate_even_if_no_branches_remain() -> None:
    assert freeze_report_kind(active_branch_count=0, coverage_has_checkpoint=True) is ReportKind.INTERMEDIATE
    assert freeze_report_kind(active_branch_count=0, coverage_has_checkpoint=False) is ReportKind.FINAL


def test_report_header_marks_intermediate_with_required_runtime_statistics() -> None:
    text = render_report(
        kind=ReportKind.INTERMEDIATE,
        body="目前证据",
        task_id="task-1",
        round_id="round-1",
        epoch=3,
        running_branch_count=2,
        queued_branch_count=1,
        unavailable_count=4,
        elapsed_seconds=17,
        next_interval_seconds=120,
        credit_balance=9.5,
        credit_pool=-1.0,
        pending_work=("核实来源",),
    )

    assert "中间报告" in text
    assert "task-1/round-1/3" in text
    assert "仍运行/排队分支：2/1" in text
    assert "coverage 不可用：4" in text
    assert "目前证据" in text
