"""确定性统计：从权威账本重算，与报告 snapshot 一致，且不读 raw debug。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import pytest

from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import BranchLifecycle, BranchRuntime, FormalizedTask, ReportKind
from lunagentic_research_swarm.runtime.credits import meter_summarizer_usage, reconcile_usage, reserve_input
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator
from lunagentic_research_swarm.statistics import StatisticsService, compute_task_stats
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


async def _seed_task_graph(store: SQLiteStateStore, *, task_id: str = "lrs_stats") -> None:
    await store.transact(
        [
            StoreCommand(
                "insert_task",
                {
                    "task_id": task_id,
                    "stream_id": "s",
                    "formalized_text": "formal",
                    "formalized_sha256": "abc",
                    "created_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "rnd_1",
                    "task_id": task_id,
                    "round_number": 1,
                    "generation": 1,
                    "status": "COMPLETED",
                    "time_budget_seconds": 120,
                    "credit_pool": -2.5,
                    "started_at": 1.0,
                    "ended_at": 40.0,
                },
            ),
            StoreCommand(
                "insert_branch",
                {
                    "branch_id": "br_root",
                    "round_id": "rnd_1",
                    "parent_branch_id": None,
                    "agent_id": "builtin.quick_thinker",
                    "lifecycle": "FINALIZED",
                    "depth": 0,
                    "credit_balance": 0.0,
                    "generation": 1,
                    "created_at": 1.0,
                    "finalized_at": 30.0,
                },
            ),
            StoreCommand(
                "insert_branch",
                {
                    "branch_id": "br_child",
                    "round_id": "rnd_1",
                    "parent_branch_id": "br_root",
                    "agent_id": "builtin.researcher",
                    "lifecycle": "IN_FLIGHT",
                    "depth": 2,
                    "credit_balance": 1.0,
                    "generation": 1,
                    "created_at": 2.0,
                },
            ),
        ]
    )


async def _open_seeded(tmp_path: Path) -> SQLiteStateStore:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_task_graph(store)
    catalog = PriceCatalog.from_sources({}, {"m1": PriceProfile(price_in=10.0, price_out=20.0)}, {})
    # 生产形状：agent reserve + reconcile；总结器 meter；纠错 call_id 后缀；unreconciled。
    agent_reservation = reserve_input(
        1.5,
        task_id="lrs_stats",
        round_id="rnd_1",
        branch_id="br_root",
        call_id="c_agent",
        usage_id="u_agent",
        role="agent",
        selector="task:utils",
        estimated_model_name="m1",
        prompt_tokens=100,
        completion_tokens=0,
        cache_hit_tokens=40,
        cache_miss_tokens=60,
        created_at=3.0,
    )
    agent_reconcile = reconcile_usage(
        agent_reservation,
        actual_charge=1.2,
        actual_model_name="m1-phys",
        usage=TokenUsage(100, 20, 40, 60, source="actual"),
        created_at=3.1,
    )
    summarizer_commands = meter_summarizer_usage(
        role="summarizer",
        task_id="lrs_stats",
        round_id="rnd_1",
        branch_id="br_root",
        call_id="c_sum",
        selector="task:mid_memory",
        catalog=catalog,
        model_name="m1",
        usage=TokenUsage(50, 10, 0, 50, source="actual"),
        created_at=4.0,
    )
    unrec = reserve_input(
        0.4,
        task_id="lrs_stats",
        round_id="rnd_1",
        branch_id="br_child",
        call_id="c_unrec",
        usage_id="u_unrec",
        role="agent",
        selector="task:utils",
        estimated_model_name="m1",
        prompt_tokens=10,
        cache_miss_tokens=10,
        created_at=5.0,
    )
    unrec_done = reconcile_usage(
        unrec,
        success=False,
        usage=None,
        created_at=5.1,
    )
    correction = reserve_input(
        0.1,
        task_id="lrs_stats",
        round_id="rnd_1",
        branch_id="br_root",
        call_id="c_agent:correction",
        usage_id="u_corr",
        role="agent",
        selector="model:m1-phys",
        estimated_model_name="m1-phys",
        prompt_tokens=0,
        metadata={"correction_count": 1},
        created_at=5.5,
    )
    await store.transact(
        [
            *agent_reservation.commands,
            *agent_reconcile.commands,
            *summarizer_commands,
            *unrec.commands,
            *unrec_done.commands,
            *correction.commands,
            StoreCommand(
                "insert_procedure_call",
                {
                    "request_id": "p1",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "turn_id": "c_agent",
                    "agent_id": "builtin.quick_thinker",
                    "procedure_id": "builtin.web_search",
                    "provider_plugin_id": "builtin",
                    "status": "SUCCEEDED",
                    "duration_ms": 12,
                    "error_code": None,
                    "provenance_json": "{}",
                    "external_cost_json": json.dumps({"credits": 0.25}, ensure_ascii=False),
                    "created_at": 6.0,
                },
            ),
            StoreCommand(
                "insert_procedure_call",
                {
                    "request_id": "p2",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_root",
                    "turn_id": "c_agent",
                    "agent_id": "builtin.quick_thinker",
                    "procedure_id": "core.compact",
                    "provider_plugin_id": "core",
                    "status": "SUCCEEDED",
                    "duration_ms": 5,
                    "error_code": None,
                    "provenance_json": "{}",
                    "external_cost_json": None,
                    "created_at": 7.0,
                },
            ),
            StoreCommand(
                "insert_procedure_call",
                {
                    "request_id": "p3",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "branch_id": "br_child",
                    "turn_id": "c_unrec",
                    "agent_id": "builtin.researcher",
                    "procedure_id": "core.checkpoint",
                    "provider_plugin_id": "core",
                    "status": "ERROR",
                    "duration_ms": 3,
                    "error_code": "checkpoint_failed",
                    "provenance_json": "{}",
                    "external_cost_json": None,
                    "created_at": 8.0,
                },
            ),
            StoreCommand(
                "insert_lifecycle_event",
                {
                    "event_id": "e_cont",
                    "task_id": "lrs_stats",
                    "round_id": "rnd_1",
                    "event_type": "ContinueRequested",
                    "from_status": "PAUSED",
                    "to_status": "RUNNING",
                    "metadata_json": "{}",
                    "created_at": 10.0,
                },
            ),
        ]
    )
    return store


@pytest.mark.asyncio
async def test_reserve_reconcile_pair_counts_as_single_agent_call(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_task_graph(store, task_id="lrs_once")
    reservation = reserve_input(
        1.0,
        task_id="lrs_once",
        round_id="rnd_1",
        branch_id="br_root",
        call_id="u1",
        usage_id="u1",
        role="agent",
        selector="task:utils",
        estimated_model_name="m1",
        prompt_tokens=100,
        cache_hit_tokens=40,
        cache_miss_tokens=60,
    )
    reconciliation = reconcile_usage(
        reservation,
        actual_charge=0.9,
        actual_model_name="m1",
        usage=TokenUsage(100, 20, 40, 60, source="actual"),
    )
    await store.transact([*reservation.commands, *reconciliation.commands])
    rows = await store.run_locked(
        lambda connection: connection.execute(
            "SELECT usage_id, reconciliation_status FROM llm_usage WHERE task_id = ? ORDER BY usage_id",
            ("lrs_once",),
        ).fetchall()
    )
    assert [str(row["reconciliation_status"]) for row in rows] == ["reserved", "actual"]
    stats = await StatisticsService(store).task("lrs_once")
    assert stats["agent_calls"] == 1
    assert stats["prompt_tokens"] == 100
    assert stats["completion_tokens"] == 20
    assert stats["cache_hit_tokens"] == 40
    assert stats["cache_miss_tokens"] == 60
    assert stats["estimated_credits"] == pytest.approx(1.0)
    assert stats["actual_credits"] == pytest.approx(0.9)
    plugin = await StatisticsService(store).plugin()
    assert plugin["models"]["m1"]["calls"] == 1
    assert plugin["models"]["m1"]["actual_credits"] == pytest.approx(0.9)
    await store.close()


@pytest.mark.asyncio
async def test_orphan_reservation_visible_via_credit_ledger(tmp_path: Path) -> None:
    """未核销的 input_reservation 必须出现在 estimated/unreconciled，不可静默归零。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_task_graph(store, task_id="lrs_orphan")
    reservation = reserve_input(
        1.5,
        task_id="lrs_orphan",
        round_id="rnd_1",
        branch_id="br_root",
        call_id="c_orphan",
        usage_id="u_orphan",
        role="agent",
        selector="task:utils",
        estimated_model_name="m1",
        prompt_tokens=77,
        cache_miss_tokens=77,
        created_at=9.0,
    )
    await store.transact(list(reservation.commands))
    stats = await StatisticsService(store).task("lrs_orphan")
    assert stats["agent_calls"] == 1
    assert stats["prompt_tokens"] == 77
    assert stats["estimated_credits"] == pytest.approx(1.5)
    assert stats["actual_credits"] == pytest.approx(0.0)
    assert stats["unreconciled_credits"] == pytest.approx(1.5)
    await store.close()


@pytest.mark.asyncio
async def test_task_stats_recomputed_from_ledger_equal_report_snapshot(tmp_path: Path) -> None:
    store = await _open_seeded(tmp_path)
    service = StatisticsService(store)
    stats = await service.task("lrs_stats")
    # agent + unreconciled + orphan correction reserved；tokens 仍单计核销行。
    assert stats["agent_calls"] == 3
    assert stats["summarizer_calls"] == 1
    assert stats["prompt_tokens"] == 160
    assert stats["completion_tokens"] == 30
    assert stats["cache_hit_tokens"] == 40
    assert stats["cache_miss_tokens"] == 120  # 60 agent + 50 summarizer + 10 unrec
    # credit_ledger：1.5+0.4+0.1 reservation；actual 仅 c_agent 净 1.2；unrec=0.4+0.1
    assert stats["estimated_credits"] == pytest.approx(2.0)
    assert stats["actual_credits"] == pytest.approx(1.2)
    assert stats["unreconciled_credits"] == pytest.approx(0.5)
    assert stats["cost_equivalent_credits"] > 0.0
    assert stats["credit_pool"] == pytest.approx(-2.5)
    assert stats["credit_debt"] == pytest.approx(2.5)
    assert stats["branches_total"] == 2
    assert stats["branches_active"] == 1
    assert stats["branches_finalized"] == 1
    assert stats["max_branch_depth"] == 2
    assert stats["compact_count"] == 1
    assert stats["checkpoint_count"] == 1
    assert stats["protocol_correction_count"] == 1
    assert stats["continue_count"] == 1
    assert stats["procedures_success"] == 2
    assert stats["procedures_error"] == 1
    assert stats["external_cost_credits"] == pytest.approx(0.25)
    assert stats["error_count"] >= 1

    # 生产路径：ReportCoordinator 写入 stats_json，测试再从账本独立重算并 equality。
    task = FormalizedTask.create("formal")
    branch = BranchRuntime(
        branch_id="br_root",
        task=task,
        catalog_fingerprint="fp",
        generation=1,
        messages=[{"role": "assistant", "content": "done"}],
        credits=0.0,
        depth=0,
        lifecycle=BranchLifecycle.FINALIZED,
    )

    class _Summarizer:
        async def finalize_task(self, request):
            return SummaryResult(True, "合成结论", "m1-phys", None, None)

        async def finalize_branch(self, request):
            return SummaryResult(True, "分支", "m1-phys", None, None)

    coordinator = ReportCoordinator(
        task_id="lrs_stats",
        round_id="rnd_1",
        formalized_task=task,
        branches={"br_root": branch},
        store=store,
        summarizer=_Summarizer(),
        clock=lambda: 41.0,
        statistics=service,
        time_budget_seconds=120,
        grace_period_seconds=0,
        started_at=1.0,
    )
    from lunagentic_research_swarm.models import SummaryKind
    from lunagentic_research_swarm.reporting import CoverageSet, CoverageSummary
    from lunagentic_research_swarm.runtime.epochs import ReportEpoch

    coverage = CoverageSet(
        (
            CoverageSummary(
                "sum_1", "br_root", SummaryKind.BRANCH_FINAL, 1, "唯一覆盖", 40.0, "SUCCEEDED"
            ),
        )
    )
    epoch = ReportEpoch(1, 40.0, {}, kind=ReportKind.FINAL, synthesis_started=False, frozen_coverage=coverage)
    record = await coordinator.synthesize(epoch)
    assert record is not None
    layer = await store.load_summary_layer("lrs_stats")
    assert layer is not None
    saved = next(item["stats"] for item in layer.reports if item["report_id"] == record.report_id)
    recomputed = await service.task("lrs_stats")
    assert dict(saved) == recomputed
    cache_a = await service.cache("lrs_stats")
    cache_b = await service.cache("lrs_stats")
    assert cache_a == cache_b
    await store.close()


@pytest.mark.asyncio
async def test_synthesize_stats_computed_inside_write_transaction(tmp_path: Path) -> None:
    """stats 必须在写入 report 的同一 BEGIN 内计算，避免与并发 reconcile 分叉。"""

    store = await _open_seeded(tmp_path)
    service = StatisticsService(store)
    observed: dict[str, object] = {}
    real_compute = compute_task_stats

    def tracking_compute(connection, task_id: str):
        observed["in_transaction"] = bool(connection.in_transaction)
        observed["stats"] = real_compute(connection, task_id)
        return observed["stats"]

    import lunagentic_research_swarm.runtime.epochs as epochs_mod

    monkey_target = epochs_mod.compute_task_stats
    epochs_mod.compute_task_stats = tracking_compute  # type: ignore[assignment]
    try:
        task = FormalizedTask.create("formal")
        branch = BranchRuntime(
            branch_id="br_root",
            task=task,
            catalog_fingerprint="fp",
            generation=1,
            messages=[{"role": "assistant", "content": "done"}],
            credits=0.0,
            depth=0,
            lifecycle=BranchLifecycle.FINALIZED,
        )

        class _Summarizer:
            async def finalize_task(self, request):
                return SummaryResult(True, "合成结论", "m1", None, None)

            async def finalize_branch(self, request):
                return SummaryResult(True, "分支", "m1", None, None)

        from lunagentic_research_swarm.models import SummaryKind
        from lunagentic_research_swarm.reporting import CoverageSet, CoverageSummary
        from lunagentic_research_swarm.runtime.epochs import ReportEpoch

        coordinator = ReportCoordinator(
            task_id="lrs_stats",
            round_id="rnd_1",
            formalized_task=task,
            branches={"br_root": branch},
            store=store,
            summarizer=_Summarizer(),
            clock=lambda: 41.0,
            statistics=service,
            time_budget_seconds=120,
            grace_period_seconds=0,
            started_at=1.0,
        )
        coverage = CoverageSet(
            (
                CoverageSummary(
                    "sum_1", "br_root", SummaryKind.BRANCH_FINAL, 1, "唯一覆盖", 40.0, "SUCCEEDED"
                ),
            )
        )
        epoch = ReportEpoch(1, 40.0, {}, kind=ReportKind.FINAL, synthesis_started=False, frozen_coverage=coverage)
        record = await coordinator.synthesize(epoch)
    finally:
        epochs_mod.compute_task_stats = monkey_target  # type: ignore[assignment]

    assert record is not None
    assert observed.get("in_transaction") is True
    layer = await store.load_summary_layer("lrs_stats")
    saved = next(item["stats"] for item in layer.reports if item["report_id"] == record.report_id)
    assert dict(saved) == observed["stats"]
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_reconcile_cannot_split_report_stats_snapshot(tmp_path: Path) -> None:
    """并发核销在报告事务持锁期间排队；提交后 stats_json 仍等于同事务快照。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_task_graph(store, task_id="lrs_race")
    service = StatisticsService(store)
    inside = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_compute = compute_task_stats
    injected_stats: dict[str, object] = {}

    def blocking_compute(connection, task_id: str):
        stats = real_compute(connection, task_id)
        injected_stats["before"] = dict(stats)
        assert connection.in_transaction
        loop.call_soon_threadsafe(inside.set)
        # 给 inject 一点时间卡在 store 锁上；本回调不 await inject，避免死锁。
        time.sleep(0.05)
        return stats

    import lunagentic_research_swarm.runtime.epochs as epochs_mod

    previous = epochs_mod.compute_task_stats
    epochs_mod.compute_task_stats = blocking_compute  # type: ignore[assignment]

    task = FormalizedTask.create("formal")
    branch = BranchRuntime(
        branch_id="br_root",
        task=task,
        catalog_fingerprint="fp",
        generation=1,
        messages=[{"role": "assistant", "content": "done"}],
        credits=0.0,
        depth=0,
        lifecycle=BranchLifecycle.FINALIZED,
    )

    class _Summarizer:
        async def finalize_task(self, request):
            return SummaryResult(True, "合成结论", "m1", None, None)

        async def finalize_branch(self, request):
            return SummaryResult(True, "分支", "m1", None, None)

    from lunagentic_research_swarm.models import SummaryKind
    from lunagentic_research_swarm.reporting import CoverageSet, CoverageSummary
    from lunagentic_research_swarm.runtime.epochs import ReportEpoch

    coordinator = ReportCoordinator(
        task_id="lrs_race",
        round_id="rnd_1",
        formalized_task=task,
        branches={"br_root": branch},
        store=store,
        summarizer=_Summarizer(),
        clock=lambda: 50.0,
        statistics=service,
        time_budget_seconds=120,
        grace_period_seconds=0,
        started_at=1.0,
    )
    coverage = CoverageSet(
        (
            CoverageSummary(
                "sum_1", "br_root", SummaryKind.BRANCH_FINAL, 1, "覆盖", 40.0, "SUCCEEDED"
            ),
        )
    )
    epoch = ReportEpoch(1, 40.0, {}, kind=ReportKind.FINAL, synthesis_started=False, frozen_coverage=coverage)

    async def inject_after_begin() -> None:
        await inside.wait()
        reservation = reserve_input(
            0.7,
            task_id="lrs_race",
            round_id="rnd_1",
            branch_id="br_root",
            call_id="c_race",
            usage_id="u_race",
            role="agent",
            selector="task:utils",
            estimated_model_name="m1",
            prompt_tokens=777,
            cache_miss_tokens=777,
            created_at=49.0,
        )
        await store.transact(list(reservation.commands))

    try:
        injector = asyncio.create_task(inject_after_begin())
        record = await coordinator.synthesize(epoch)
        await injector
    finally:
        epochs_mod.compute_task_stats = previous  # type: ignore[assignment]

    assert record is not None
    layer = await store.load_summary_layer("lrs_race")
    saved = next(item["stats"] for item in layer.reports if item["report_id"] == record.report_id)
    # 并发写入发生在报告事务之后，故保存的 snapshot 不含 777 tokens。
    assert saved["prompt_tokens"] == injected_stats["before"]["prompt_tokens"]
    assert int(saved["prompt_tokens"]) != 777
    after = await service.task("lrs_race")
    assert after["prompt_tokens"] == saved["prompt_tokens"] + 777
    await store.close()


@pytest.mark.asyncio
async def test_synthesize_meters_summarizer_into_saved_stats(tmp_path: Path) -> None:
    """删除生产路径 ``_meter_summarizer`` 调用会使本测试失败。"""

    store = await _open_seeded(tmp_path)
    catalog = PriceCatalog.from_sources({}, {"sum-model": PriceProfile(price_in=10.0)}, {})
    service = StatisticsService(store)
    before = await service.task("lrs_stats")

    class _Summarizer:
        async def finalize_task(self, request):
            return SummaryResult(
                True,
                "合成结论",
                "sum-model",
                TokenUsage(100_000, 0, 0, 100_000, source="actual"),
                None,
            )

        async def finalize_branch(self, request):
            return SummaryResult(True, "分支", "sum-model", None, None)

    from lunagentic_research_swarm.models import SummaryKind
    from lunagentic_research_swarm.reporting import CoverageSet, CoverageSummary
    from lunagentic_research_swarm.runtime.epochs import ReportEpoch

    task = FormalizedTask.create("formal")
    branch = BranchRuntime(
        branch_id="br_root",
        task=task,
        catalog_fingerprint="fp",
        generation=1,
        messages=[{"role": "assistant", "content": "done"}],
        credits=0.0,
        depth=0,
        lifecycle=BranchLifecycle.FINALIZED,
    )
    coordinator = ReportCoordinator(
        task_id="lrs_stats",
        round_id="rnd_1",
        formalized_task=task,
        branches={"br_root": branch},
        store=store,
        summarizer=_Summarizer(),
        clock=lambda: 41.0,
        statistics=service,
        price_catalog=catalog,
        time_budget_seconds=120,
        grace_period_seconds=0,
        started_at=1.0,
    )
    # 两条 coverage 才会走 finalize_task → _meter_summarizer。
    coverage = CoverageSet(
        (
            CoverageSummary(
                "sum_1", "br_root", SummaryKind.BRANCH_FINAL, 1, "覆盖甲", 40.0, "SUCCEEDED"
            ),
            CoverageSummary(
                "sum_2", "br_child", SummaryKind.BRANCH_FINAL, 1, "覆盖乙", 40.5, "SUCCEEDED"
            ),
        )
    )
    epoch = ReportEpoch(1, 40.0, {}, kind=ReportKind.FINAL, synthesis_started=False, frozen_coverage=coverage)
    record = await coordinator.synthesize(epoch)
    assert record is not None
    layer = await store.load_summary_layer("lrs_stats")
    saved = next(item["stats"] for item in layer.reports if item["report_id"] == record.report_id)
    assert saved["summarizer_calls"] == before["summarizer_calls"] + 1
    assert saved["cost_equivalent_credits"] == pytest.approx(before["cost_equivalent_credits"] + 100.0)
    await store.close()


@pytest.mark.asyncio
async def test_synthesize_survives_metering_without_price_catalog(tmp_path: Path) -> None:
    store = await _open_seeded(tmp_path)
    service = StatisticsService(store)

    class _Summarizer:
        async def finalize_task(self, request):
            return SummaryResult(
                True,
                "合成结论",
                "sum-model",
                TokenUsage(10, 1, 0, 10, source="actual"),
                None,
            )

        async def finalize_branch(self, request):
            return SummaryResult(True, "分支", "sum-model", None, None)

    from lunagentic_research_swarm.models import SummaryKind
    from lunagentic_research_swarm.reporting import CoverageSet, CoverageSummary
    from lunagentic_research_swarm.runtime.epochs import ReportEpoch

    task = FormalizedTask.create("formal")
    branch = BranchRuntime(
        branch_id="br_root",
        task=task,
        catalog_fingerprint="fp",
        generation=1,
        messages=[{"role": "assistant", "content": "done"}],
        credits=0.0,
        depth=0,
        lifecycle=BranchLifecycle.FINALIZED,
    )
    coordinator = ReportCoordinator(
        task_id="lrs_stats",
        round_id="rnd_1",
        formalized_task=task,
        branches={"br_root": branch},
        store=store,
        summarizer=_Summarizer(),
        clock=lambda: 41.0,
        statistics=service,
        price_catalog=None,
        time_budget_seconds=120,
        grace_period_seconds=0,
        started_at=1.0,
    )
    coverage = CoverageSet(
        (
            CoverageSummary(
                "sum_1", "br_root", SummaryKind.BRANCH_FINAL, 1, "覆盖甲", 40.0, "SUCCEEDED"
            ),
            CoverageSummary(
                "sum_2", "br_child", SummaryKind.BRANCH_FINAL, 1, "覆盖乙", 40.5, "SUCCEEDED"
            ),
        )
    )
    epoch = ReportEpoch(1, 40.0, {}, kind=ReportKind.FINAL, synthesis_started=False, frozen_coverage=coverage)
    record = await coordinator.synthesize(epoch)
    assert record is not None
    assert record.status == "SUCCEEDED"
    assert "合成结论" in record.text
    await store.close()


@pytest.mark.asyncio
async def test_summarizer_metering_produces_cost_equivalent(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_task_graph(store, task_id="lrs_sum")
    catalog = PriceCatalog.from_sources({}, {"sum-model": PriceProfile(price_in=10.0)}, {})
    commands = meter_summarizer_usage(
        role="task_summarizer",
        task_id="lrs_sum",
        round_id="rnd_1",
        call_id="sum_1",
        catalog=catalog,
        model_name="sum-model",
        usage=TokenUsage(100_000, 0, 0, 100_000, source="actual"),
    )
    assert commands
    assert all(cmd.kind == "insert_llm_usage" for cmd in commands)
    await store.transact(list(commands))
    stats = await StatisticsService(store).task("lrs_sum")
    assert stats["summarizer_calls"] == 1
    assert stats["agent_calls"] == 0
    assert stats["cost_equivalent_credits"] == pytest.approx(100.0)
    assert stats["actual_credits"] == 0.0
    await store.close()


@pytest.mark.asyncio
async def test_cache_hit_rate_null_when_denominator_zero(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "lrs_empty", "stream_id": "s", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "rnd_e",
                    "task_id": "lrs_empty",
                    "round_number": 1,
                    "generation": 1,
                    "status": "RUNNING",
                    "time_budget_seconds": 10,
                    "credit_pool": 0.0,
                    "started_at": 1.0,
                },
            ),
        ]
    )
    service = StatisticsService(store)
    stats = await service.task("lrs_empty")
    assert stats["cache_hit_rate"] is None
    assert (await service.cache("lrs_empty"))["hit_rate"] is None
    await store.close()


@pytest.mark.asyncio
async def test_plugin_stats_aggregate_without_reading_raw(tmp_path: Path) -> None:
    store = await _open_seeded(tmp_path)
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    with sqlite3.connect(debug_dir / "raw.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE agent_transcripts (id INTEGER PRIMARY KEY, messages TEXT)"
        )
        connection.execute(
            "INSERT INTO agent_transcripts(messages) VALUES (?)",
            ("raw-agent-secret",),
        )
    service = StatisticsService(store)
    plugin = await service.plugin()
    assert "models" in plugin
    assert "agents" in plugin
    assert "builtin.quick_thinker" in plugin["agents"]
    assert "procedures" in plugin
    assert "builtin.web_search" in plugin["procedures"]
    assert "tasks" in plugin
    assert "lrs_stats" in plugin["tasks"]
    encoded = json.dumps(plugin, ensure_ascii=False)
    assert "raw-agent-secret" not in encoded
    await store.close()


@pytest.mark.asyncio
async def test_statistics_ignore_missing_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    service = StatisticsService(store)
    stats = await service.task("missing")
    assert stats["agent_calls"] == 0
    assert stats["cache_hit_rate"] is None
    await store.close()
