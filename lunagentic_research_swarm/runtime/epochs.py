"""Branch-aware checkpoint, grace, and report-epoch coordination.

`ReportCoordinator` owns only ephemeral branch execution state.  Every
summary/report becomes visible to branches or starts an external effect only
after its corresponding SQLite transaction has completed successfully.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lunagentic_research_swarm.llm.summarizer import BranchFinalizationRequest, TaskFinalizationRequest
from lunagentic_research_swarm.models import BranchLifecycle, BranchRuntime, FormalizedTask, ReportKind, SummaryKind
from lunagentic_research_swarm.reporting import (
    CoverageSet,
    CoverageSummary,
    build_coverage,
    coverage_messages,
    freeze_report_kind,
    render_report,
)
from lunagentic_research_swarm.storage.sqlite import StoreCommand


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _history_copy(messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(message) for message in messages)


@dataclass(slots=True)
class FrontierEntry:
    """One branch captured by an epoch's frontier, never a live branch alias."""

    branch_id: str
    stable_history: tuple[dict[str, Any], ...]
    checkpoint_requested: bool = False
    checkpoint_summary_id: str | None = None
    terminal_summary_id: str | None = None
    failed: bool = False

    @property
    def ready(self) -> bool:
        return bool(self.checkpoint_summary_id or self.terminal_summary_id or self.failed)


@dataclass(slots=True)
class ReportEpoch:
    epoch: int
    frozen_at: float
    frontier: dict[str, FrontierEntry]
    kind: ReportKind = ReportKind.INTERMEDIATE
    grace_deadline_at: float | None = None
    synthesis_started: bool = False
    synthesis_finished: bool = False
    # `CoverageSet` and its entries are frozen dataclasses.  Keeping the
    # exact object captured at synthesis start prevents a later terminal
    # summary from replacing this epoch's checkpoint coverage.
    frozen_coverage: CoverageSet | None = None


@dataclass(frozen=True, slots=True)
class ReportRecord:
    report_id: str
    epoch: int
    kind: ReportKind
    text: str
    status: str
    created_at: float
    coverage: CoverageSet


# Public name from the runtime contract.  The immutable implementation lives
# with report formatting so it can also be used by non-runtime consumers.
ReportCoverage = CoverageSet


class ReportCoordinator:
    """Coordinates report epochs without pausing the investigation graph."""

    def __init__(
        self,
        *,
        task_id: str,
        round_id: str,
        formalized_task: FormalizedTask,
        branches: Mapping[str, BranchRuntime],
        store: Any,
        summarizer: Any,
        clock: Callable[[], float],
        launch_delegation: Callable[[str, Mapping[str, Any]], Awaitable[None] | None] | None = None,
        broadcast_summary: Callable[[CoverageSummary], Awaitable[None] | None] | None = None,
        on_synthesis_complete: Callable[[ReportRecord], Awaitable[None] | None] | None = None,
        time_budget_seconds: int = 120,
        grace_period_seconds: int = 60,
        credit_pool: float = 0.0,
        started_at: float | None = None,
    ) -> None:
        if time_budget_seconds <= 0 or grace_period_seconds < 0:
            raise ValueError("报告时间预算必须为正且 grace 不得为负")
        self.task_id, self.round_id, self.formalized_task = task_id, round_id, formalized_task
        self.branches = dict(branches)
        self.store, self.summarizer, self.clock = store, summarizer, clock
        self.launch_delegation, self.broadcast_summary = launch_delegation, broadcast_summary
        self.on_synthesis_complete = on_synthesis_complete
        self.time_budget_seconds, self.grace_period_seconds = time_budget_seconds, grace_period_seconds
        self.credit_pool = float(credit_pool)
        self.started_at = float(clock() if started_at is None else started_at)
        self.deadline_at = self.started_at + time_budget_seconds
        self.current_epoch: ReportEpoch | None = None
        self.epochs: list[ReportEpoch] = []
        self.reports: list[ReportRecord] = []
        self._summaries: list[CoverageSummary] = []
        self._held: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self._seen: dict[str, set[str]] = {branch_id: set() for branch_id in self.branches}
        self._synthesis_tasks: set[asyncio.Task[None]] = set()

    def active_branch_ids(self) -> set[str]:
        return {
            branch_id for branch_id, branch in self.branches.items()
            if branch.lifecycle is not BranchLifecycle.FINALIZED
        }

    def pending_summary_messages(self, branch_id: str) -> tuple[dict[str, str], ...]:
        """Return committed broadcasts once per branch, for its next LLM call."""

        seen = self._seen.setdefault(branch_id, set())
        pending = [item for item in self._summaries if item.summary_id not in seen and item.available]
        seen.update(item.summary_id for item in pending)
        return coverage_messages(CoverageSet(tuple(pending)))

    async def open_epoch(self, *, epoch: int | None = None) -> ReportEpoch:
        """Freeze a frontier at deadline (or an all-checkpoint early boundary)."""

        if self.current_epoch is not None and not self.current_epoch.synthesis_finished:
            return self.current_epoch
        epoch_number = epoch if epoch is not None else (self.epochs[-1].epoch + 1 if self.epochs else 1)
        frontier: dict[str, FrontierEntry] = {}
        for branch_id in sorted(self.active_branch_ids()):
            branch = self.branches[branch_id]
            entry = FrontierEntry(branch_id, _history_copy(branch.messages))
            if branch.latest_checkpoint_id:
                entry.checkpoint_requested = True
                entry.checkpoint_summary_id = branch.latest_checkpoint_id
            frontier[branch_id] = entry
        now = float(self.clock())
        report_epoch = ReportEpoch(epoch_number, now, frontier, grace_deadline_at=now + self.grace_period_seconds)
        self.current_epoch = report_epoch
        self.epochs.append(report_epoch)
        # A held manual checkpoint is released exactly when an epoch opens;
        # its durable checkpoint is already part of this epoch's coverage.
        await self._release_held()
        await self._maybe_start_synthesis(report_epoch)
        return report_epoch

    async def on_branch_safe_point(
        self,
        branch_id: str,
        *,
        checkpoint: bool = False,
        terminal: bool = False,
        delegations: Sequence[Mapping[str, Any]] = (),
    ) -> ReportEpoch | None:
        """Handle a completed turn after normal accounting/procedures are durable.

        Manual checkpoint holds its children until the next epoch.  A branch
        returning during an existing epoch grace window is checkpointed from a
        stable clone while its new children immediately belong to the next
        epoch.
        """

        branch = self.branches[branch_id]
        if terminal:
            summary_id = await self._summarize_branch(
                branch_id, checkpoint=False, history=_history_copy(branch.messages)
            )
            branch.lifecycle = BranchLifecycle.FINALIZED
            if self.current_epoch and branch_id in self.current_epoch.frontier:
                entry = self.current_epoch.frontier[branch_id]
                entry.terminal_summary_id = summary_id
                entry.failed = summary_id is None
                if not self.current_epoch.synthesis_finished:
                    await self._maybe_start_synthesis(self.current_epoch)
                    return self.current_epoch
                # This branch reached a terminal safe point after the frozen
                # intermediate report was already delivered.  It cannot
                # alter that record, but an empty frontier now warrants the
                # next FINAL epoch immediately.
                if not self.active_branch_ids():
                    return await self.open_epoch()
                return self.current_epoch
            # A terminal-only investigation has no deadline frontier to
            # create its final report.  Open an empty frontier after the
            # durable terminal summary so `_maybe_start_synthesis` can freeze
            # FINAL coverage and deliver it immediately.
            if self.current_epoch is None:
                return await self.open_epoch()
            return self.current_epoch

        active_epoch = self.current_epoch
        in_frontier = (
            active_epoch is not None
            and branch_id in active_epoch.frontier
            and not active_epoch.synthesis_finished
        )
        if checkpoint:
            summary_id = await self._summarize_branch(
                branch_id, checkpoint=True, history=_history_copy(branch.messages)
            )
            if summary_id is not None:
                branch.latest_checkpoint_id = summary_id
            branch.lifecycle = BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT
            self._held[branch_id] = tuple(dict(item) for item in delegations)
            if in_frontier:
                entry = active_epoch.frontier[branch_id]
                entry.checkpoint_requested = True
                entry.checkpoint_summary_id = summary_id
                entry.failed = summary_id is None
                await self._maybe_start_synthesis(active_epoch)
                return active_epoch
            if self._all_active_checkpointed():
                return await self.open_epoch()
            return None

        # A grace return must snapshot the fully committed pre-return history
        # before launching children: child creation may mutate branch runtime
        # bookkeeping, but that work belongs to the next epoch.
        stable_history = _history_copy(branch.messages)
        if in_frontier:
            await self._launch(delegations, branch_id, report_epoch=active_epoch.epoch + 1)
        else:
            await self._launch(delegations, branch_id)
        if in_frontier:
            entry = active_epoch.frontier[branch_id]
            entry.checkpoint_requested = True
            summary_id = await self._summarize_branch(
                branch_id, checkpoint=True, history=stable_history
            )
            entry.checkpoint_summary_id = summary_id
            entry.failed = summary_id is None
            if summary_id is not None:
                branch.latest_checkpoint_id = summary_id
            await self._maybe_start_synthesis(active_epoch)
            return active_epoch
        return None

    async def on_grace_expired(self, epoch: int | None = None) -> ReportEpoch | None:
        """Clone each still-in-flight frontier's pre-call stable context."""

        active_epoch = self.current_epoch
        if active_epoch is None or (epoch is not None and active_epoch.epoch != epoch):
            return None
        for entry in active_epoch.frontier.values():
            if entry.ready:
                continue
            entry.checkpoint_requested = True
            summary_id = await self._summarize_branch(entry.branch_id, checkpoint=True, history=entry.stable_history)
            entry.checkpoint_summary_id = summary_id
            entry.failed = summary_id is None
            if summary_id is not None and entry.branch_id in self.branches:
                self.branches[entry.branch_id].latest_checkpoint_id = summary_id
        await self._maybe_start_synthesis(active_epoch)
        return active_epoch

    async def synthesize(self, epoch: ReportEpoch | None = None) -> ReportRecord | None:
        """Persist an immutable kind/coverage snapshot, then invoke finalizer."""

        report_epoch = epoch or self.current_epoch
        if report_epoch is None or report_epoch.synthesis_finished:
            return None
        active = self.active_branch_ids()
        coverage = report_epoch.frozen_coverage
        if coverage is None:
            # Direct callers remain safe; normal scheduling records this
            # immutable snapshot in `_maybe_start_synthesis` before creating
            # the coroutine.
            coverage = build_coverage(self._summaries, active_branch_ids=active)
            report_epoch.frozen_coverage = coverage
            report_epoch.kind = freeze_report_kind(
                active_branch_count=len(active), coverage_has_checkpoint=coverage.has_checkpoint
            )
        # The epoch could begin with a checkpoint even if its branch finalizes
        # before the finalizer returns.  Freeze this decision before awaiting.
        frozen_kind = report_epoch.kind
        report_epoch.synthesis_started = True
        running = len(active)
        stats = self._stats(report_epoch, coverage, running)
        if not coverage.items:
            body = "当前尚无可用分支摘要；调查仍在进行。" if frozen_kind is ReportKind.INTERMEDIATE else "没有可用的终结摘要。"
            status = "DEGRADED"
        elif len(coverage.items) == 1:
            body = coverage.items[0].text or ""
            status = "SUCCEEDED"
        else:
            request = TaskFinalizationRequest(
                formalized_task=self.formalized_task,
                coverage_summaries=tuple(item.text or "" for item in coverage.items),
                report_kind=frozen_kind,
                statistics=stats,
                running_branch_count=running,
            )
            result = await self.summarizer.finalize_task(request)
            if result.success and result.text.strip():
                body, status = result.text, "SUCCEEDED"
            else:
                body, status = "报告总结器不可用；以下为已提交 coverage。", "FAILED"
                if frozen_kind is ReportKind.FINAL:
                    body = "最终报告总结器不可用；以下为已提交终结 coverage。"
        created_at = float(self.clock())
        report_id = _new_id("rpt")
        text = render_report(
            kind=frozen_kind, body=body, task_id=self.task_id, round_id=self.round_id, epoch=report_epoch.epoch,
            running_branch_count=running,
            queued_branch_count=len(self._held),
            unavailable_count=coverage.unavailable_count,
            elapsed_seconds=created_at - self.started_at, next_interval_seconds=self.time_budget_seconds,
            credit_balance=sum(
                branch.credits
                for branch in self.branches.values()
                if branch.lifecycle is not BranchLifecycle.FINALIZED
            ),
            credit_pool=self.credit_pool, pending_work=tuple(self._held), stats=stats,
        )
        commands = (
            StoreCommand("insert_report", {
                "report_id": report_id, "task_id": self.task_id, "round_id": self.round_id,
                "epoch": report_epoch.epoch, "kind": frozen_kind.value, "text": text, "status": status,
                "running_branch_count": running, "stats_json": json.dumps(stats, ensure_ascii=False, sort_keys=True),
                "created_at": created_at,
            }),
            StoreCommand("insert_outbox", {
                "outbox_id": _new_id("out"), "task_id": self.task_id, "round_id": self.round_id,
                "report_id": report_id,
                "delivery_kind": frozen_kind.value,
                "idempotency_key": f"{self.round_id}:{report_epoch.epoch}",
                "payload_json": json.dumps({"text": text}, ensure_ascii=False), "status": "PENDING",
                "next_attempt_at": created_at, "created_at": created_at,
            }),
        )
        await self.store.transact(commands)
        record = ReportRecord(report_id, report_epoch.epoch, frozen_kind, text, status, created_at, coverage)
        self.reports.append(record)
        report_epoch.synthesis_finished = True
        self.deadline_at = created_at + self.time_budget_seconds
        if self.on_synthesis_complete is not None:
            completed = self.on_synthesis_complete(record)
            if hasattr(completed, "__await__"):
                await completed
        # A synthesis that was necessarily intermediate at its frozen start
        # may finish after every branch has become terminal.  Do not mutate
        # that immutable record; open a distinct final epoch instead.
        if frozen_kind is ReportKind.INTERMEDIATE and not self.active_branch_ids():
            await self.open_epoch()
        return record

    async def wait_for_synthesis(self) -> None:
        while self._synthesis_tasks:
            await asyncio.gather(*tuple(self._synthesis_tasks))

    async def _maybe_start_synthesis(self, epoch: ReportEpoch) -> None:
        if epoch.synthesis_started or not all(entry.ready for entry in epoch.frontier.values()):
            return
        coverage = build_coverage(self._summaries, active_branch_ids=self.active_branch_ids())
        # Mark the epoch started and freeze its kind synchronously, before
        # yielding to the task that calls the finalizer.  A terminal event may
        # arrive in the small scheduling gap, but it must not upgrade this
        # already-created checkpoint report to FINAL.
        epoch.synthesis_started = True
        epoch.kind = freeze_report_kind(
            active_branch_count=len(self.active_branch_ids()),
            coverage_has_checkpoint=coverage.has_checkpoint,
        )
        epoch.frozen_coverage = coverage
        task = asyncio.create_task(self.synthesize(epoch))
        self._synthesis_tasks.add(task)
        task.add_done_callback(self._synthesis_tasks.discard)

    async def _summarize_branch(
        self, branch_id: str, *, checkpoint: bool, history: Sequence[Mapping[str, Any]]
    ) -> str | None:
        branch = self.branches[branch_id]
        kind = SummaryKind.CHECKPOINT if checkpoint else SummaryKind.BRANCH_FINAL
        summary_id = _new_id("sum")
        result = await self.summarizer.finalize_branch(
            BranchFinalizationRequest(self.formalized_task, _history_copy(history), checkpoint=checkpoint)
        )
        created_at = float(self.clock())
        success = bool(result.success and result.text.strip())
        summary = CoverageSummary(
            summary_id, branch_id, kind, self.current_epoch.epoch if self.current_epoch else None,
            result.text if success else None, created_at, "SUCCEEDED" if success else "FAILED",
        )
        await self.store.transact((
            StoreCommand("insert_summary", {
                "summary_id": summary_id, "task_id": self.task_id, "round_id": self.round_id,
                "branch_id": branch_id, "kind": kind.value, "report_epoch": summary.epoch,
                "text": summary.text, "status": summary.status,
                "error_code": None if success else getattr(getattr(result, "error", None), "code", "summary_failed"),
                "created_at": created_at,
            }),
        ))
        self._summaries.append(summary)
        if success:
            await self._broadcast(summary)
            return summary_id
        return None

    async def _broadcast(self, summary: CoverageSummary) -> None:
        for branch_id in self.active_branch_ids():
            if branch_id != summary.branch_id:
                # Do not append to in-flight prompt history.  The value is
                # queued in seen state and consumed by pending_summary_messages.
                self._seen.setdefault(branch_id, set())
        if self.broadcast_summary is not None:
            returned = self.broadcast_summary(summary)
            if hasattr(returned, "__await__"):
                await returned

    async def _release_held(self) -> None:
        held, self._held = self._held, {}
        for branch_id, delegations in held.items():
            branch = self.branches.get(branch_id)
            if branch is not None and branch.lifecycle is BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT:
                branch.lifecycle = BranchLifecycle.READY
            await self._launch(delegations, branch_id, report_epoch=self.current_epoch.epoch + 1)

    async def _launch(
        self,
        delegations: Sequence[Mapping[str, Any]],
        parent_branch_id: str,
        *,
        report_epoch: int | None = None,
    ) -> None:
        if self.launch_delegation is None:
            return
        for raw in delegations:
            child = dict(raw)
            if report_epoch is not None:
                child.setdefault("report_epoch", report_epoch)
            returned = self.launch_delegation(parent_branch_id, child)
            if hasattr(returned, "__await__"):
                await returned

    def _all_active_checkpointed(self) -> bool:
        active = self.active_branch_ids()
        return bool(active) and all(self.branches[branch_id].latest_checkpoint_id for branch_id in active)

    def _stats(self, epoch: ReportEpoch, coverage: CoverageSet, running: int) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "round_id": self.round_id,
            "epoch": epoch.epoch,
            "running_branches": running,
            "coverage": len(coverage.items),
            "coverage_unavailable": coverage.unavailable_count,
            "branches": len(self.branches),
            "max_depth": max((branch.depth for branch in self.branches.values()), default=0),
            "credits_pool": self.credit_pool,
        }


__all__ = ["FrontierEntry", "ReportCoordinator", "ReportCoverage", "ReportEpoch", "ReportRecord"]
