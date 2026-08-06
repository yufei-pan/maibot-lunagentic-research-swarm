"""Design §11.2 / §13.3 / §17.2 — pricing cache=false, report header, outbox report_id."""

from __future__ import annotations

import json

import pytest

from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage, charge
from lunagentic_research_swarm.models import ReportKind
from lunagentic_research_swarm.reporting import render_report
from lunagentic_research_swarm.runtime.credits import reconcile_usage, reserve_input
from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from fakes import FakeMaisaka


def test_spec_11_2_cache_false_bills_all_prompt_tokens_at_price_in() -> None:
    """§11.2 — cache=false: entire prompt priced at price_in (hits are not discounted)."""

    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=100,
        cache_hit_tokens=600,
        cache_miss_tokens=400,
        source="actual",
    )
    no_cache = PriceProfile(price_in=2.0, cache=False, cache_price_in=0.5, price_out=3.0)
    with_cache = PriceProfile(price_in=2.0, cache=True, cache_price_in=0.5, price_out=3.0)

    # cache=false ignores hit/miss split for pricing and charges all prompt at price_in.
    no_cache_credits = charge(no_cache, usage)
    assert no_cache_credits == pytest.approx(((1000 * 2.0 + 100 * 3.0) / 1_000_000) * 100)

    cached_credits = charge(with_cache, usage)
    assert cached_credits == pytest.approx(((400 * 2.0 + 600 * 0.5 + 100 * 3.0) / 1_000_000) * 100)
    assert no_cache_credits > cached_credits

    catalog = PriceCatalog.from_sources({}, {"m": no_cache}, {"utils": ["m"]})
    charged = catalog.charge_actual(
        actual_model_name="m",
        prompt_tokens=1000,
        completion_tokens=100,
        cache_hit_tokens=600,
        cache_miss_tokens=400,
    )
    assert charged.credits == pytest.approx(no_cache_credits)
    assert charged.price.profile.cache is False


def test_spec_11_2_missing_actual_model_name_settles_as_estimated() -> None:
    """§11.2 — no actual model_name → charge via estimated model, status flagged estimated."""

    catalog = PriceCatalog.from_sources(
        {},
        {"first": PriceProfile(price_in=1.0, cache=False, cache_price_in=0.0, price_out=2.0)},
        {"utils": ["first"]},
    )
    reservation = reserve_input(
        estimated_charge=0.5,
        task_id="task-1",
        round_id="round-1",
        branch_id="branch-1",
        call_id="call-1",
        usage_id="usage-1",
        ledger_id="ledger-1",
        role="agent",
        selector="task:utils",
        estimated_model_name="first",
        price_source="host_config",
        price_fingerprint=catalog.fingerprint,
        prompt_tokens=100,
        cache_miss_tokens=100,
    )

    result = reconcile_usage(
        reservation,
        catalog=catalog,
        actual_model_name="",
        usage=TokenUsage(1_000_000, 0, 0, 1_000_000, source="actual"),
        success=True,
    )

    assert result.status == "estimated"
    # Falls back to estimated model "first" for charging (1M tokens × price_in 1.0 → 100 credits).
    assert result.actual_charge == pytest.approx(100.0)
    assert result.adjustment == pytest.approx(0.5 - 100.0)
    assert result.usage_command.values["reconciliation_status"] == "estimated"
    assert result.usage_command.values["actual_model_name"] == ""
    assert result.usage_command.values["estimated_model_name"] == "first"
    assert result.ledger_command is not None
    assert result.ledger_command.values["entry_kind"] == "input_reconciliation"


def test_spec_13_3_intermediate_report_header_stats_separate_from_body() -> None:
    """§13.3 — intermediate header carries plugin stats; body stays after the blank line."""

    body = "UNIQUE_BODY_EVIDENCE_MARKER"
    text = render_report(
        kind=ReportKind.INTERMEDIATE,
        body=body,
        task_id="task-42",
        round_id="round-7",
        epoch=3,
        running_branch_count=2,
        queued_branch_count=1,
        unavailable_count=4,
        elapsed_seconds=17.8,
        next_interval_seconds=120,
        credit_balance=9.5,
        credit_pool=-1.0,
        pending_work=("核实来源", "补证据"),
        stats={
            "prompt_tokens": 100,
            "cache_hit_tokens": 40,
            "cache_miss_tokens": 60,
            "error_count": 1,
            "research_credits": 3.25,
        },
    )

    assert "\n\n" in text
    header, _, rendered_body = text.partition("\n\n")
    assert rendered_body.strip() == body
    assert body not in header

    assert header.startswith("中间报告")
    assert "task/round/epoch：task-42/round-7/3" in header
    assert "仍运行/排队分支：2/1" in header
    assert "coverage 不可用：4" in header
    assert "已用时间：17s；下一报告间隔：120s" in header
    assert "当前余额/pool：9.5/-1" in header
    assert "prompt_tokens=100" in header
    assert "cache_hit_tokens=40" in header
    assert "cache_miss_tokens=60" in header
    assert "error_count=1" in header
    assert "research_credits=3.25" in header
    assert "主要未决工作：核实来源；补证据" in header


@pytest.mark.asyncio
async def test_spec_17_2_outbox_delivery_carries_stable_report_id(tmp_path) -> None:
    """§17.2 — Maisaka append/trigger payloads expose the same report_id for consumer dedupe."""

    report_id = "rpt_stable_dedupe_001"
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "task-1", "stream_id": "stream-1", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "round-1",
                    "task_id": "task-1",
                    "round_number": 1,
                    "generation": 0,
                    "status": "RUNNING",
                    "time_budget_seconds": 10,
                    "credit_pool": 1.0,
                    "started_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-append-1",
                    "task_id": "task-1",
                    "round_id": "round-1",
                    "report_id": report_id,
                    "delivery_kind": "append_report",
                    "idempotency_key": f"lrs:task-1:round-1:{report_id}:append",
                    "payload_json": json.dumps(
                        {
                            "text": f"中间报告正文\nreport_id={report_id}",
                            "kind": "INTERMEDIATE",
                            "running_branch_count": 2,
                        }
                    ),
                    "status": "PENDING",
                    "next_attempt_at": 0.0,
                    "created_at": 1.0,
                },
            ),
        ]
    )

    maisaka = FakeMaisaka()
    outbox = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)

    assert await outbox.deliver_once() == 1
    assert len(maisaka.append_calls) == 1
    append = maisaka.append_calls[0]
    assert append["message_id"] == f"lrs-report:{report_id}"
    assert append["source_kind"] == "lrs_report"
    assert report_id in append["visible_text"]

    # Append completion inserts the paired trigger row; deliver it next.
    assert await outbox.deliver_once() == 1
    assert len(maisaka.trigger_calls) == 1
    trigger = maisaka.trigger_calls[0]
    metadata = trigger["metadata"]
    assert metadata["report_id"] == report_id
    assert metadata["idempotency_key"] == f"lrs:task-1:round-1:{report_id}:trigger"
    assert metadata["outbox_id"]
    # Same stable id on both delivery phases — consumer dedupe key.
    assert append["message_id"].removeprefix("lrs-report:") == metadata["report_id"] == report_id

    await store.close()
