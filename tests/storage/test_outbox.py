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

    async def append(self, stream_id, segments, **kwargs):
        self.append_calls += 1

    async def trigger(self, stream_id, intent, **kwargs):
        self.trigger_calls += 1
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
