from __future__ import annotations

import asyncio
import json

import pytest

from lunagentic_research_swarm.storage.outbox import MaisakaOutbox
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


class FakeMaisaka:
    def __init__(self) -> None:
        self.append_calls = 0
        self.trigger_calls = 0
        self.fail_trigger = 0
        self.trigger_kwargs = []

    async def append(self, stream_id, segments, **kwargs):
        self.append_calls += 1

    async def trigger(self, stream_id, intent, **kwargs):
        self.trigger_calls += 1
        self.trigger_kwargs.append(kwargs)
        if self.fail_trigger:
            self.fail_trigger -= 1
            raise RuntimeError("transient trigger")


async def _row(store, outbox_id: str):
    def read():
        conn = store._require_connection()  # noqa: SLF001 - test probe
        return conn.execute("SELECT * FROM maisaka_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()

    return await store._call(read)  # noqa: SLF001 - test probe


@pytest.mark.asyncio
async def test_trigger_failure_does_not_repeat_completed_append(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "task-1", "stream_id": "stream-1", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "round-1", "task_id": "task-1", "round_number": 1, "generation": 0,
                    "status": "RUNNING", "time_budget_seconds": 10, "credit_pool": 1.0, "started_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-append", "task_id": "task-1", "round_id": "round-1", "report_id": "report-1",
                    "delivery_kind": "append_report", "idempotency_key": "lrs:task-1:1:report-1:append",
                    "payload_json": json.dumps({"text": "hello", "kind": "INTERMEDIATE", "running_branch_count": 1}),
                    "status": "PENDING", "next_attempt_at": 0.0, "created_at": 1.0,
                },
            ),
        ]
    )
    maisaka = FakeMaisaka()
    maisaka.fail_trigger = 1
    now = [100.0]
    outbox = MaisakaOutbox(store, maisaka, poll_interval_seconds=0.001, clock=lambda: now[0])
    await outbox.start()
    for _ in range(30):
        await outbox.deliver_once()
        row = await _row(store, "out-append")
        if row["status"].lower() == "delivered" and maisaka.trigger_calls >= 2:
            break
        now[0] += 5
        await asyncio.sleep(0)
    row = await _row(store, "out-append")
    assert maisaka.append_calls == 1
    assert maisaka.trigger_calls == 2
    assert row["status"].lower() == "delivered"
    await outbox.close()
    await store.close()


@pytest.mark.asyncio
async def test_trigger_includes_stable_outbox_metadata(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "task-1", "stream_id": "stream-1", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "round-1", "task_id": "task-1", "round_number": 1, "generation": 0,
                    "status": "RUNNING", "time_budget_seconds": 10, "credit_pool": 1.0, "started_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-trigger", "task_id": "task-1", "round_id": "round-1", "report_id": "report-1",
                    "delivery_kind": "trigger_report_review", "idempotency_key": "lrs:task-1:1:report-1:trigger",
                    "payload_json": json.dumps(
                        {"intent": "review_intermediate_report", "metadata": {"report_kind": "INTERMEDIATE"}}
                    ),
                    "status": "PENDING", "next_attempt_at": 0.0, "created_at": 1.0,
                },
            ),
        ]
    )
    maisaka = FakeMaisaka()
    outbox = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)
    await outbox.deliver_once()
    metadata = maisaka.trigger_kwargs[0]["metadata"]
    assert metadata["idempotency_key"] == "lrs:task-1:1:report-1:trigger"
    assert metadata["outbox_id"] == "out-trigger"
    assert metadata["report_id"] == "report-1"
    await store.close()


@pytest.mark.asyncio
async def test_delivery_failure_persists_only_safe_error_code(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "task-1", "stream_id": "stream-1", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "round-1", "task_id": "task-1", "round_number": 1, "generation": 0,
                    "status": "RUNNING", "time_budget_seconds": 10, "credit_pool": 1.0, "started_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-trigger", "task_id": "task-1", "round_id": "round-1", "report_id": "report-1",
                    "delivery_kind": "trigger_report_review", "idempotency_key": "lrs:task-1:1:report-1:trigger",
                    "payload_json": json.dumps({"intent": "review_intermediate_report"}),
                    "status": "PENDING", "next_attempt_at": 0.0, "created_at": 1.0,
                },
            ),
        ]
    )
    maisaka = FakeMaisaka()
    maisaka.fail_trigger = 1
    outbox = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)
    await outbox.deliver_once()
    row = await _row(store, "out-trigger")
    assert row["status"].lower() == "pending"
    assert row["last_error"] == "delivery_failed:RuntimeError"
    assert "transient trigger" not in row["last_error"]
    await store.close()


@pytest.mark.asyncio
async def test_deliver_once_releases_lock_before_maisaka_io(tmp_path) -> None:
    """C-I2: claim under lock; network I/O must not hold ``_lock``."""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand("insert_task", {"task_id": "task-1", "stream_id": "stream-1", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "round-1", "task_id": "task-1", "round_number": 1, "generation": 0,
                    "status": "RUNNING", "time_budget_seconds": 10, "credit_pool": 1.0, "started_at": 1.0,
                },
            ),
            StoreCommand(
                "insert_outbox",
                {
                    "outbox_id": "out-trigger", "task_id": "task-1", "round_id": "round-1", "report_id": "report-1",
                    "delivery_kind": "trigger_report_review", "idempotency_key": "lrs:task-1:1:report-1:trigger",
                    "payload_json": json.dumps({"intent": "review_intermediate_report"}),
                    "status": "PENDING", "next_attempt_at": 0.0, "created_at": 1.0,
                },
            ),
        ]
    )

    gate = asyncio.Event()
    entered = asyncio.Event()

    class BlockingMaisaka(FakeMaisaka):
        async def trigger(self, stream_id, intent, **kwargs):
            entered.set()
            await gate.wait()
            return await super().trigger(stream_id, intent, **kwargs)

    maisaka = BlockingMaisaka()
    outbox = MaisakaOutbox(store, maisaka, clock=lambda: 100.0)
    task = asyncio.create_task(outbox.deliver_once())
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    # While Maisaka I/O is blocked, the outbox lock must be free.
    acquired = False
    try:
        await asyncio.wait_for(outbox._lock.acquire(), timeout=0.2)
        acquired = True
    finally:
        if acquired:
            outbox._lock.release()
    assert acquired, "MaisakaOutbox._lock was held across network I/O"
    gate.set()
    assert await task == 1
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_claim_due_outbox_is_exclusive_between_workers(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    first, second = SQLiteStateStore(path), SQLiteStateStore(path)
    await first.open()
    await first.transact(
        [
            StoreCommand("insert_task", {"task_id": "task-1", "stream_id": "stream-1", "created_at": 1.0}),
            StoreCommand(
                "insert_round",
                {
                    "round_id": "round-1", "task_id": "task-1", "round_number": 1, "generation": 0,
                    "status": "RUNNING", "time_budget_seconds": 10, "credit_pool": 1.0, "started_at": 1.0,
                },
            ),
            *[
                StoreCommand(
                    "insert_outbox",
                    {
                        "outbox_id": f"out-{index}", "task_id": "task-1", "round_id": "round-1",
                        "report_id": f"report-{index}",
                        "delivery_kind": "trigger_report_review", "idempotency_key": f"key-{index}",
                        "payload_json": json.dumps({"intent": "review_intermediate_report"}),
                        "status": "PENDING", "next_attempt_at": 0.0, "created_at": float(index),
                    },
                )
                for index in range(2)
            ],
        ]
    )
    await second.open()
    claimed_a, claimed_b = await asyncio.gather(
        first.claim_due_outbox(100.0, lease_seconds=30.0, limit=1),
        second.claim_due_outbox(100.0, lease_seconds=30.0, limit=1),
    )
    assert {claimed_a[0]["outbox_id"], claimed_b[0]["outbox_id"]} == {"out-0", "out-1"}
    await first.close()
    await second.close()
