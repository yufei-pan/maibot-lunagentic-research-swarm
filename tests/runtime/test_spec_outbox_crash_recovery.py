"""Design §18.4 / §17.2 — crash mid-outbox recovery (append done, trigger pending).

Append and trigger are independent durable phases: after append is marked delivered
and the paired trigger row exists, a process restart must not re-append the report
body; the trigger must still fire once with the same ``report_id``.
"""

from __future__ import annotations

import json

import pytest

from fakes import FakeMaisaka
from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


REPORT_ID = "rpt_crash_mid_outbox_001"
VISIBLE_TEXT = (
    "【中间报告】\n"
    f"report_id：{REPORT_ID}\n"
    "证据摘要：崩溃恢复交付测试"
)


async def _seed_task_round(store: SQLiteStateStore) -> None:
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
        ]
    )


async def _seed_pending_append(store: SQLiteStateStore, *, report_id: str = REPORT_ID) -> None:
    await _seed_task_round(store)
    await store.transact(
        [
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
                            "text": VISIBLE_TEXT,
                            "kind": "INTERMEDIATE",
                            "running_branch_count": 1,
                        },
                        ensure_ascii=False,
                    ),
                    "status": "PENDING",
                    "next_attempt_at": 0.0,
                    "created_at": 1.0,
                },
            ),
        ]
    )


async def _outbox_rows(store: SQLiteStateStore) -> list[dict[str, object]]:
    def _query(connection: object) -> list[dict[str, object]]:
        rows = connection.execute(  # type: ignore[attr-defined]
            "SELECT outbox_id, report_id, delivery_kind, status, idempotency_key "
            "FROM maisaka_outbox ORDER BY created_at, outbox_id"
        ).fetchall()
        return [dict(row) for row in rows]

    return await store.run_locked(_query)


@pytest.mark.asyncio
async def test_spec_18_4_crash_after_append_before_trigger_does_not_reappend(tmp_path) -> None:
    """§18.4 / §17.2 — deliver append, crash outbox, restart: one append, one trigger."""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_pending_append(store)

    maisaka = FakeMaisaka()
    outbox_v1 = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)

    # Phase 1: append succeeds; complete_outbox_append inserts paired trigger.
    assert await outbox_v1.deliver_once() == 1
    assert len(maisaka.append_calls) == 1
    assert len(maisaka.trigger_calls) == 0
    append = maisaka.append_calls[0]
    assert append["message_id"] == f"lrs-report:{REPORT_ID}"
    assert f"report_id：{REPORT_ID}" in append["visible_text"]

    rows_after_append = await _outbox_rows(store)
    append_row = next(r for r in rows_after_append if str(r["delivery_kind"]).startswith("append"))
    trigger_row = next(r for r in rows_after_append if "trigger" in str(r["delivery_kind"]).lower())
    assert str(append_row["status"]).lower() == "delivered"
    assert str(trigger_row["status"]).lower() == "pending"
    assert str(trigger_row["report_id"]) == REPORT_ID

    # Crash: drop the worker without delivering the trigger.
    await outbox_v1.close()

    # Restart on the same durable store + Maisaka recorder.
    outbox_v2 = MaisakaOutbox(store, maisaka, clock=lambda: 200.0)
    assert await outbox_v2.deliver_once() == 1

    assert len(maisaka.append_calls) == 1, "append must not be duplicated after restart"
    assert len(maisaka.trigger_calls) == 1
    trigger_meta = maisaka.trigger_calls[0]["metadata"]
    assert trigger_meta["report_id"] == REPORT_ID
    assert trigger_meta["idempotency_key"] == f"lrs:task-1:round-1:{REPORT_ID}:trigger"
    assert append["message_id"].removeprefix("lrs-report:") == trigger_meta["report_id"] == REPORT_ID

    # Idle: nothing left due.
    assert await outbox_v2.deliver_once() == 0
    assert len(maisaka.append_calls) == 1
    assert len(maisaka.trigger_calls) == 1

    await outbox_v2.close()
    await store.close()


@pytest.mark.asyncio
async def test_spec_18_4_restart_from_durable_append_delivered_trigger_pending(tmp_path) -> None:
    """§18.4 — seed post-crash durable state (append delivered + trigger pending), resume once."""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await _seed_task_round(store)

    # Durable crash window: append already delivered; trigger never sent.
    await store.transact(
        [
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-append-done",
                    "task_id": "task-1",
                    "round_id": "round-1",
                    "report_id": REPORT_ID,
                    "delivery_kind": "append_report",
                    "idempotency_key": f"lrs:task-1:round-1:{REPORT_ID}:append",
                    "payload_json": json.dumps(
                        {
                            "text": VISIBLE_TEXT,
                            "kind": "INTERMEDIATE",
                            "running_branch_count": 1,
                        },
                        ensure_ascii=False,
                    ),
                    "status": "delivered",
                    "next_attempt_at": 0.0,
                    "created_at": 1.0,
                    "delivered_at": 50.0,
                },
            ),
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-trigger-pending",
                    "task_id": "task-1",
                    "round_id": "round-1",
                    "report_id": REPORT_ID,
                    "delivery_kind": "trigger_report_review",
                    "idempotency_key": f"lrs:task-1:round-1:{REPORT_ID}:trigger",
                    "payload_json": json.dumps(
                        {
                            "intent": "review_intermediate_report",
                            "reason": "LRS intermediate report ready",
                            "metadata": {
                                "task_id": "task-1",
                                "report_id": REPORT_ID,
                                "round_id": "round-1",
                                "report_kind": "INTERMEDIATE",
                                "running_branch_count": 1,
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "status": "PENDING",
                    "next_attempt_at": 0.0,
                    "created_at": 1.1,
                },
            ),
        ]
    )

    maisaka = FakeMaisaka()
    # Simulate that append already hit Maisaka before the crash (consumer dedupe via message_id).
    maisaka.append_calls.append(
        {
            "stream_id": "stream-1",
            "segments": [{"type": "text", "content": VISIBLE_TEXT}],
            "visible_text": VISIBLE_TEXT,
            "source_kind": "lrs_report",
            "message_id": f"lrs-report:{REPORT_ID}",
        }
    )

    outbox = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)
    assert await outbox.deliver_once() == 1

    assert len(maisaka.append_calls) == 1, "restart must not re-append a delivered append row"
    assert len(maisaka.trigger_calls) == 1
    trigger_meta = maisaka.trigger_calls[0]["metadata"]
    assert trigger_meta["report_id"] == REPORT_ID
    assert trigger_meta["outbox_id"] == "out-trigger-pending"
    assert trigger_meta["idempotency_key"] == f"lrs:task-1:round-1:{REPORT_ID}:trigger"
    assert f"report_id：{REPORT_ID}" in maisaka.append_calls[0]["visible_text"]

    assert await outbox.deliver_once() == 0
    assert len(maisaka.trigger_calls) == 1

    await outbox.close()
    await store.close()
