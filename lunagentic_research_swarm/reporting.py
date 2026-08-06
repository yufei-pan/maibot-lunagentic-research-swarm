"""Deterministic report coverage selection and presentation.

This module deliberately contains no scheduler, store, or model call.  A
report epoch freezes its inputs here before the coordinator asks a task
finalizer to write prose, so a report can never turn from intermediate into a
final report while that model call is in flight.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from lunagentic_research_swarm.models import ReportKind, SummaryKind


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """A durable branch summary considered by task-level synthesis."""

    summary_id: str
    branch_id: str | None
    kind: SummaryKind
    epoch: int | None
    text: str | None
    created_at: float
    status: str = "SUCCEEDED"

    @property
    def terminal(self) -> bool:
        return self.kind is SummaryKind.BRANCH_FINAL

    @property
    def available(self) -> bool:
        return self.status == "SUCCEEDED" and bool(self.text and self.text.strip())


@dataclass(frozen=True, slots=True)
class CoverageSet:
    """The fixed, ordered coverage set used for one synthesis call."""

    items: tuple[CoverageSummary, ...]
    unavailable_count: int = 0

    @property
    def has_checkpoint(self) -> bool:
        return any(item.kind is SummaryKind.CHECKPOINT for item in self.items)


def build_coverage(
    summaries: Iterable[CoverageSummary],
    *,
    active_branch_ids: set[str] | frozenset[str] | Sequence[str],
) -> CoverageSet:
    """Keep every successful terminal and each active branch's newest checkpoint.

    Superseded checkpoints remain durable audit data, but are intentionally not
    sent to the task finalizer.  Failed/missing active checkpoint summaries are
    counted so an intermediate header accurately reports incomplete coverage.
    """

    active = frozenset(str(branch_id) for branch_id in active_branch_ids)
    ordered = sorted(summaries, key=lambda item: (item.created_at, item.summary_id))
    terminal = [item for item in ordered if item.terminal and item.available]
    latest: dict[str, CoverageSummary] = {}
    for item in ordered:
        if item.branch_id not in active or item.kind is not SummaryKind.CHECKPOINT:
            continue
        if item.available:
            previous = latest.get(item.branch_id)
            if previous is None or (item.created_at, item.summary_id) >= (previous.created_at, previous.summary_id):
                latest[item.branch_id] = item
    # An active branch with no usable latest checkpoint is unavailable exactly
    # once, regardless of whether its attempted checkpoint failed.
    unavailable = sum(1 for branch_id in active if branch_id not in latest)
    selected = sorted((*terminal, *latest.values()), key=lambda item: (item.created_at, item.summary_id))
    return CoverageSet(tuple(selected), unavailable)


def freeze_report_kind(*, active_branch_count: int, coverage_has_checkpoint: bool) -> ReportKind:
    """Select the kind once, before a task finalizer can be started."""

    if active_branch_count > 0 or coverage_has_checkpoint:
        return ReportKind.INTERMEDIATE
    return ReportKind.FINAL


def coverage_messages(coverage: CoverageSet) -> tuple[dict[str, str], ...]:
    """Encode each coverage item as a separate assistant message."""

    return tuple(
        {
            "role": "assistant",
            "content": item.text or "",
            "name": f"branch:{item.branch_id or 'task'}:{item.kind.value}:epoch:{item.epoch}",
        }
        for item in coverage.items
    )


def render_report(
    *,
    kind: ReportKind,
    body: str,
    task_id: str,
    round_id: str,
    epoch: int,
    running_branch_count: int,
    queued_branch_count: int,
    unavailable_count: int,
    elapsed_seconds: float,
    next_interval_seconds: int,
    credit_balance: float,
    credit_pool: float,
    pending_work: Sequence[str] = (),
    stats: dict[str, Any] | None = None,
    report_id: str = "",
) -> str:
    """Add the deterministic user-facing header around a synthesized body.

    ``report_id`` is embedded in the header when provided so Maisaka consumers can
    dedupe duplicate appends after a crash (design §17.2).
    """

    rendered_stats = "；".join(f"{key}={value}" for key, value in sorted((stats or {}).items()))
    report_id_line = f"report_id：{report_id}\n" if report_id else ""
    if kind is ReportKind.INTERMEDIATE:
        # design §13.3: an intermediate report also carries token / cache
        # hit-miss / credits / failure figures, not just branch counts.
        header = (
            "中间报告\n"
            f"{report_id_line}"
            f"task/round/epoch：{task_id}/{round_id}/{epoch}\n"
            f"仍运行/排队分支：{max(0, running_branch_count)}/{max(0, queued_branch_count)}\n"
            f"coverage 不可用：{max(0, unavailable_count)}\n"
            f"已用时间：{max(0, int(elapsed_seconds))}s；下一报告间隔：{max(0, next_interval_seconds)}s\n"
            f"当前余额/pool：{credit_balance:g}/{credit_pool:g}\n"
            f"统计：{rendered_stats or '无'}\n"
            f"主要未决工作：{'；'.join(pending_work) if pending_work else '无'}"
        )
    else:
        header = (
            "最终结论\n"
            f"{report_id_line}"
            f"task/round/epoch：{task_id}/{round_id}/{epoch}\n"
            f"统计：{rendered_stats or '无'}"
        )
    return f"{header}\n\n{body.strip()}" if body.strip() else header


__all__ = [
    "CoverageSet",
    "CoverageSummary",
    "build_coverage",
    "coverage_messages",
    "freeze_report_kind",
    "render_report",
]
