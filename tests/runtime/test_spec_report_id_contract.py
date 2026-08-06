"""Design §17.2 — report_id contract pins (Maisaka delivery + body dedupe).

Current production contract: append ``message_id`` and trigger ``metadata["report_id"]``
share the same stable report id. Spec also wants that id consumer-visible in the
report body for crash/dedupe; ``render_report`` does not yet embed it.
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeMaisaka
from lunagentic_research_swarm.models import ReportKind
from lunagentic_research_swarm.reporting import render_report
from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


async def _seed_append_outbox(
    store: SQLiteStateStore,
    *,
    report_id: str,
    text: str = "中间报告正文（无嵌入 report_id）",
) -> None:
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
                            "text": text,
                            "kind": "INTERMEDIATE",
                            "running_branch_count": 1,
                        }
                    ),
                    "status": "PENDING",
                    "next_attempt_at": 0.0,
                    "created_at": 1.0,
                },
            ),
        ]
    )


@pytest.mark.asyncio
async def test_spec_17_2_maisaka_append_message_id_matches_trigger_report_id(tmp_path) -> None:
    """§17.2 — append message_id and trigger metadata["report_id"] are equal and stable."""

    report_id = "rpt_contract_stable_001"
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_append_outbox(store, report_id=report_id)

    maisaka = FakeMaisaka()
    outbox = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)

    assert await outbox.deliver_once() == 1
    assert len(maisaka.append_calls) == 1
    append = maisaka.append_calls[0]
    assert append["message_id"] == f"lrs-report:{report_id}"
    assert append["source_kind"] == "lrs_report"

    assert await outbox.deliver_once() == 1
    assert len(maisaka.trigger_calls) == 1
    trigger_meta = maisaka.trigger_calls[0]["metadata"]
    assert trigger_meta["report_id"] == report_id
    assert trigger_meta["idempotency_key"] == f"lrs:task-1:round-1:{report_id}:trigger"
    assert trigger_meta["outbox_id"]

    # Same stable id on both Maisaka delivery phases — current production contract.
    assert append["message_id"].removeprefix("lrs-report:") == trigger_meta["report_id"] == report_id

    await store.close()


@pytest.mark.asyncio
async def test_spec_17_2_harness_delivery_surfaces_share_persisted_report_id(runtime_harness) -> None:
    """§17.2 — real report→outbox path: Maisaka surfaces equal the durable report_id."""

    harness = runtime_harness
    await harness.start("report_id 契约", credits=50.0, time_budget=30)
    await harness.formalize("正式 report_id 契约")
    await harness.root_delegates({"A": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(30)
    await harness.run_until_idle()

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None and layer.reports
    report_id = str(layer.reports[0]["report_id"])
    assert report_id.startswith("rpt")

    delivered = await harness.deliver_outbox()
    assert delivered >= 2
    assert len(harness.maisaka.append_calls) >= 1
    assert len(harness.maisaka.trigger_calls) >= 1

    append = harness.maisaka.append_calls[0]
    trigger_meta = harness.maisaka.trigger_calls[0]["metadata"]
    assert append["message_id"] == f"lrs-report:{report_id}"
    assert trigger_meta["report_id"] == report_id
    assert append["message_id"].removeprefix("lrs-report:") == trigger_meta["report_id"]


@pytest.mark.xfail(
    strict=True,
    reason="spec §17.2: report body should carry stable report_id",
)
def test_spec_17_2_render_report_body_carries_stable_report_id_for_dedupe() -> None:
    """§17.2 — consumer-visible body should embed report_id for crash/dedupe.

    Production ``render_report`` has no ``report_id`` parameter and does not
    write the id into header/body; Maisaka metadata carries it instead.
    This xfail documents the spec mismatch without changing production.
    """

    report_id = "rpt_body_dedupe_should_appear"
    # Production render_report has no report_id parameter; body/header omit the id.
    text = render_report(
        kind=ReportKind.INTERMEDIATE,
        body="证据摘要",
        task_id="task-42",
        round_id="round-7",
        epoch=1,
        running_branch_count=1,
        queued_branch_count=0,
        unavailable_count=0,
        elapsed_seconds=10,
        next_interval_seconds=60,
        credit_balance=5.0,
        credit_pool=0.0,
        pending_work=(),
    )

    assert report_id in text


@pytest.mark.xfail(
    strict=True,
    reason="spec §17.2: report body should carry stable report_id",
)
@pytest.mark.asyncio
async def test_spec_17_2_persisted_report_text_embeds_report_id(runtime_harness) -> None:
    """§17.2 — durable report text (Maisaka visible_text) should include report_id."""

    harness = runtime_harness
    await harness.start("正文去重", credits=50.0, time_budget=30)
    await harness.formalize("正式正文去重")
    await harness.root_delegates({"A": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(30)
    await harness.run_until_idle()

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None and layer.reports
    report = layer.reports[0]
    report_id = str(report["report_id"])
    text = str(report["text"])

    assert report_id in text
