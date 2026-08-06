"""Branch-aware checkpoint, grace, and report-epoch coordination.

`ReportCoordinator` owns only ephemeral branch execution state.  Every
summary/report becomes visible to branches or starts an external effect only
after its corresponding SQLite transaction has completed successfully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lunagentic_research_swarm.llm.summarizer import BranchFinalizationRequest, SummaryResult, TaskFinalizationRequest
from lunagentic_research_swarm.models import (
    BranchLifecycle,
    BranchRuntime,
    FormalizedTask,
    ReportKind,
    SummaryKind,
    new_outbox_id,
    new_report_id,
    new_summary_id,
)
from lunagentic_research_swarm.reporting import (
    CoverageSet,
    CoverageSummary,
    build_coverage,
    coverage_messages,
    freeze_report_kind,
    render_report,
)
from lunagentic_research_swarm.runtime.credits import meter_summarizer_usage
from lunagentic_research_swarm.statistics import StatisticsService, compute_task_stats
from lunagentic_research_swarm.storage.sqlite import StoreCommand

_LOG = logging.getLogger(__name__)


def _history_copy(messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(message) for message in messages)


def _safe_synthesis_error(
    error: Any,
    *,
    formalized_task: FormalizedTask,
    coverage: CoverageSet,
) -> tuple[str, str]:
    """Extract bounded lifecycle metadata without echoing report inputs."""

    raw_code = str(getattr(error, "code", "") or "")
    code = raw_code if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", raw_code) else "report_synthesis_failed"
    raw_message = str(getattr(error, "message", "") or "")
    sensitive = (formalized_task.text, *(item.text or "" for item in coverage.items))
    if not raw_message or any(token and token in raw_message for token in sensitive):
        return code, "报告总结器不可用。"
    return code, raw_message[:512]


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
    # Error details are kept separate from the rendered report body.  The body
    # may contain user/task evidence, while these fields are safe lifecycle
    # metadata for the controller callback.
    error_code: str | None = None
    error_message: str | None = None


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
        release_held_delegations: (
            Callable[[str, Sequence[Mapping[str, Any]]], Awaitable[None] | None] | None
        ) = None,
        on_synthesis_complete: Callable[[ReportRecord], Awaitable[None] | None] | None = None,
        time_budget_seconds: int = 120,
        grace_period_seconds: int = 60,
        credit_pool: float = 0.0,
        started_at: float | None = None,
        statistics: StatisticsService | None = None,
        price_catalog: Any | None = None,
        summarizer_selector: str = "task:mid_memory",
        max_report_chars: int = 60000,
        max_stats_chars: int = 12000,
        deliver_intermediate: bool = True,
        deliver_final: bool = True,
    ) -> None:
        if time_budget_seconds <= 0 or grace_period_seconds < 0:
            raise ValueError("报告时间预算必须为正且 grace 不得为负")
        self.task_id, self.round_id, self.formalized_task = task_id, round_id, formalized_task
        self.branches = dict(branches)
        self.store, self.summarizer, self.clock = store, summarizer, clock
        self.launch_delegation = launch_delegation
        self.broadcast_summary = broadcast_summary
        self.release_held_delegations = release_held_delegations
        self.on_synthesis_complete = on_synthesis_complete
        self.time_budget_seconds, self.grace_period_seconds = time_budget_seconds, grace_period_seconds
        self.credit_pool = float(credit_pool)
        self.started_at = float(clock() if started_at is None else started_at)
        self.deadline_at = self.started_at + time_budget_seconds
        self.statistics = statistics
        self.price_catalog = price_catalog
        self.summarizer_selector = str(summarizer_selector or "task:mid_memory")
        self.max_report_chars = max(1000, int(max_report_chars))
        self.max_stats_chars = max(1000, int(max_stats_chars))
        self.deliver_intermediate = bool(deliver_intermediate)
        self.deliver_final = bool(deliver_final)
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
        """Return committed broadcasts once per branch, for its next LLM call.

        Design §13.5 broadcasts a summary to the *other* active branches; a branch
        already carries its own history, so its own checkpoint is never echoed
        back to it.
        """

        seen = self._seen.setdefault(branch_id, set())
        pending = [
            item
            for item in self._summaries
            if item.summary_id not in seen and item.available and item.branch_id != branch_id
        ]
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

        Manual checkpoint before a report epoch holds children until the next
        ``open_epoch`` (grace start).  A checkpoint requested while already in
        an open frontier/grace releases held children as soon as the branch
        summary finishes (design §13.1).  A non-checkpoint return during grace
        is cloned for coverage while new children belong to the next epoch.
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
            # Next-epoch / grace children are outside the frozen frontier.
            # Once the prior epoch has fully committed and no live branches
            # remain, open FINAL (same outcome as the in-frontier empty path).
            if self.current_epoch.synthesis_finished and not self.active_branch_ids():
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
                # §13.1: checkpoint during grace → continue after branch summary.
                await self._release_held_branch(branch_id)
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
        prompt_stats = self._stats(report_epoch, coverage, running)
        if not coverage.items:
            body = "当前尚无可用分支摘要；调查仍在进行。" if frozen_kind is ReportKind.INTERMEDIATE else "没有可用的终结摘要。"
            status = "DEGRADED"
            error_code = "report_coverage_unavailable"
            error_message = "最终报告没有可用的终结摘要。" if frozen_kind is ReportKind.FINAL else "报告 coverage 暂不可用。"
        elif len(coverage.items) == 1:
            body = coverage.items[0].text or ""
            status = "SUCCEEDED"
            error_code = error_message = None
        else:
            request = TaskFinalizationRequest(
                formalized_task=self.formalized_task,
                coverage_summaries=tuple(item.text or "" for item in coverage.items),
                report_kind=frozen_kind,
                statistics=prompt_stats,
                running_branch_count=running,
            )
            result = await self.summarizer.finalize_task(request)
            await self._meter_summarizer(role="task_summarizer", branch_id=None, result=result)
            if result.success and result.text.strip():
                body, status = result.text, "SUCCEEDED"
                error_code = error_message = None
            else:
                body, status = "报告总结器不可用；以下为已提交 coverage。", "FAILED"
                error = getattr(result, "error", None)
                error_code, error_message = _safe_synthesis_error(
                    error, formalized_task=self.formalized_task, coverage=coverage,
                )
                if frozen_kind is ReportKind.FINAL:
                    body = "最终报告总结器不可用；以下为已提交终结 coverage。"
        created_at = float(self.clock())
        report_id = new_report_id()
        credit_balance = sum(
            branch.credits
            for branch in self.branches.values()
            if branch.lifecycle is not BranchLifecycle.FINALIZED
        )
        # 同 transaction snapshot：在写入报告的同一 BEGIN IMMEDIATE 内重算 stats。
        _ledger_stats, text = await self._persist_report_same_tx(
            report_id=report_id,
            frozen_kind=frozen_kind,
            body=body,
            status=status,
            report_epoch=report_epoch,
            running=running,
            coverage=coverage,
            created_at=created_at,
            credit_balance=credit_balance,
        )
        record = ReportRecord(
            report_id, report_epoch.epoch, frozen_kind, text, status, created_at, coverage,
            error_code, error_message,
        )
        self.reports.append(record)
        self.deadline_at = created_at + self.time_budget_seconds
        if self.on_synthesis_complete is not None:
            completed = self.on_synthesis_complete(record)
            if hasattr(completed, "__await__"):
                await completed
        # Mark finished only after the completion callback so concurrent
        # terminals cannot open the next FINAL epoch while status is still
        # mid-REPORTING (A3).
        report_epoch.synthesis_finished = True
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
        if branch_id not in self.branches:
            raise KeyError(branch_id)
        kind = SummaryKind.CHECKPOINT if checkpoint else SummaryKind.BRANCH_FINAL
        summary_id = new_summary_id()
        result = await self.summarizer.finalize_branch(
            BranchFinalizationRequest(self.formalized_task, _history_copy(history), checkpoint=checkpoint)
        )
        await self._meter_summarizer(
            role="checkpoint_summarizer" if checkpoint else "branch_summarizer",
            branch_id=branch_id,
            result=result,
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

    async def _meter_summarizer(
        self,
        *,
        role: str,
        branch_id: str | None,
        result: SummaryResult,
    ) -> None:
        """尽力写入总结器 telemetry；失败只记日志，绝不中断报告/摘要路径。"""

        usage = getattr(result, "usage", None)
        if usage is None:
            return
        try:
            commands = meter_summarizer_usage(
                role=role,
                task_id=self.task_id,
                round_id=self.round_id,
                branch_id=branch_id,
                selector=self.summarizer_selector,
                catalog=self.price_catalog,
                model_name=str(getattr(result, "model_name", "") or ""),
                usage=usage,
                created_at=float(self.clock()),
            )
            if commands:
                await self.store.transact(commands)
        except Exception:
            _LOG.warning(
                "总结器计量失败（已跳过）role=%s task_id=%s branch_id=%s",
                role,
                self.task_id,
                branch_id,
                exc_info=True,
            )

    async def _persist_report_same_tx(
        self,
        *,
        report_id: str,
        frozen_kind: ReportKind,
        body: str,
        status: str,
        report_epoch: ReportEpoch,
        running: int,
        coverage: CoverageSet,
        created_at: float,
        credit_balance: float,
    ) -> tuple[dict[str, Any], str]:
        """在同一 store transaction 内重算 stats 并写入 report/outbox。"""

        def _clip(text: str, limit: int) -> str:
            return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"

        def _build_text(ledger_stats: dict[str, Any]) -> str:
            # `[reporting] max_stats_chars` / `max_report_chars` bound what a single
            # report may put into Maisaka context.
            bounded_stats = dict(ledger_stats)
            rendered = json.dumps(bounded_stats, ensure_ascii=False, sort_keys=True)
            if len(rendered) > self.max_stats_chars:
                bounded_stats = {"task_id": self.task_id, "stats_truncated": True}
            return _clip(
                render_report(
                    kind=frozen_kind,
                    body=body,
                    task_id=self.task_id,
                    round_id=self.round_id,
                    epoch=report_epoch.epoch,
                    running_branch_count=running,
                    queued_branch_count=len(self._held),
                    unavailable_count=coverage.unavailable_count,
                    elapsed_seconds=created_at - self.started_at,
                    next_interval_seconds=self.time_budget_seconds,
                    credit_balance=credit_balance,
                    credit_pool=self.credit_pool,
                    pending_work=tuple(self._held),
                    stats=bounded_stats,
                ),
                self.max_report_chars,
            )

        def _commands(ledger_stats: dict[str, Any], text: str) -> tuple[StoreCommand, ...]:
            # `[reporting] deliver_intermediate` / `deliver_final` only gate Maisaka
            # delivery.  The report itself is always durable, so `/swarm status`,
            # statistics and past-case retrieval keep working.
            deliver = (
                self.deliver_final if frozen_kind is ReportKind.FINAL else self.deliver_intermediate
            )
            report_command = StoreCommand(
                "insert_report",
                {
                    "report_id": report_id,
                    "task_id": self.task_id,
                    "round_id": self.round_id,
                    "epoch": report_epoch.epoch,
                    "kind": frozen_kind.value,
                    "text": text,
                    "status": status,
                    "running_branch_count": running,
                    "stats_json": json.dumps(ledger_stats, ensure_ascii=False, sort_keys=True),
                    "created_at": created_at,
                },
            )
            if not deliver:
                return (report_command,)
            return (
                report_command,
                StoreCommand(
                    "insert_outbox",
                    {
                        "outbox_id": new_outbox_id(),
                        "task_id": self.task_id,
                        "round_id": self.round_id,
                        "report_id": report_id,
                        "delivery_kind": "append_report",
                        "idempotency_key": f"lrs:{self.task_id}:{self.round_id}:{report_id}:append",
                        "payload_json": json.dumps(
                            {
                                "text": text,
                                "kind": frozen_kind.value,
                                "round_number": report_epoch.epoch,
                                "running_branch_count": running,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "status": "PENDING",
                        "next_attempt_at": created_at,
                        "created_at": created_at,
                    },
                ),
            )

        def _commit(connection: Any) -> tuple[dict[str, Any], str]:
            ledger_stats = compute_task_stats(connection, self.task_id)
            text = _build_text(ledger_stats)
            apply = getattr(self.store, "apply_commands", None)
            if not callable(apply):
                raise RuntimeError("store 缺少 apply_commands，无法同事务写入报告")
            apply(connection, _commands(ledger_stats, text))
            return ledger_stats, text

        run_transaction = getattr(self.store, "run_transaction", None)
        if callable(run_transaction):
            return await run_transaction(_commit)

        # 测试假 store：无 SQLite 连接时退化为独立 snapshot + transact。
        if self.statistics is not None:
            ledger_stats = await self.statistics.task(self.task_id)
        else:
            ledger_stats = {"task_id": self.task_id}
        text = _build_text(ledger_stats)
        await self.store.transact(_commands(ledger_stats, text))
        return ledger_stats, text

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
            await self._launch_held(branch_id, delegations)

    async def _release_held_branch(self, branch_id: str) -> None:
        """Release one branch's held children after an in-grace checkpoint summary."""

        if branch_id not in self._held:
            branch = self.branches.get(branch_id)
            if branch is not None and branch.lifecycle is BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT:
                branch.lifecycle = BranchLifecycle.READY
            return
        delegations = self._held.pop(branch_id)
        await self._launch_held(branch_id, delegations)

    async def _launch_held(self, branch_id: str, delegations: Sequence[Mapping[str, Any]]) -> None:
        branch = self.branches.get(branch_id)
        if branch is not None and branch.lifecycle is BranchLifecycle.WAITING_REPORT_WITH_CHECKPOINT:
            branch.lifecycle = BranchLifecycle.READY
        next_epoch = (self.current_epoch.epoch + 1) if self.current_epoch is not None else None
        if self.release_held_delegations is not None:
            returned = self.release_held_delegations(branch_id, delegations)
            if hasattr(returned, "__await__"):
                await returned
        else:
            await self._launch(delegations, branch_id, report_epoch=next_epoch)

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
