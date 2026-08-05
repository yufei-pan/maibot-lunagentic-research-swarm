"""Feedback 不可变事件、lesson 与 schema 校验。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.feedback import FeedbackService
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from lunagentic_research_swarm.storage.vectors import SOURCE_KIND_FEEDBACK_LESSON


@dataclass
class _FeedbackRow:
    feedback_id: str
    disposition: str
    supersedes_feedback_id: str | None
    payload: dict[str, Any]
    lesson: str
    round_id: str


@dataclass
class FakeVectorIndex:
    enqueued: list[tuple[str, str]]

    def __init__(self) -> None:
        self.enqueued = []

    async def enqueue(self, *, source_kind: str, source_id: str) -> Any:
        self.enqueued.append((source_kind, source_id))
        return type("Ok", (), {"success": True})()


class FeedbackHarness:
    def __init__(
        self,
        store: SQLiteStateStore,
        service: FeedbackService,
        index: FakeVectorIndex,
        clock: list[float],
    ) -> None:
        self.store = store
        self.service = service
        self.index = index
        self._clock = clock
        self.task_id = "lrs_1"
        self.round_id = "rnd_1"

    @classmethod
    async def create(cls, tmp_path: Path) -> FeedbackHarness:
        store = SQLiteStateStore(tmp_path / "feedback.sqlite3")
        await store.open()
        await store.transact(
            [
                StoreCommand(
                    "insert_task",
                    {
                        "task_id": "lrs_1",
                        "stream_id": "stream-1",
                        "current_round_number": 1,
                        "created_at": 1.0,
                        "updated_at": 1.0,
                    },
                ),
                StoreCommand(
                    "insert_round",
                    {
                        "round_id": "rnd_1",
                        "task_id": "lrs_1",
                        "round_number": 1,
                        "generation": 0,
                        "status": "COMPLETED",
                        "time_budget_seconds": 60,
                        "credit_pool": 10.0,
                        "started_at": 1.0,
                        "ended_at": 10.0,
                    },
                ),
            ]
        )
        index = FakeVectorIndex()
        clock = [100.0]

        def _now() -> float:
            return float(clock[0])

        service = FeedbackService(
            store,
            vector_index=index,
            clock=_now,
            feedback_wait_seconds=600,
            reminders_enabled=True,
            index_lessons=True,
            max_lesson_chars=8000,
        )
        return cls(store, service, index, clock)

    async def close(self) -> None:
        await self.store.close()

    async def submit(self, **kwargs: Any) -> Any:
        self._clock[0] += 1.0
        return await self.service.submit(**kwargs)

    async def events(self) -> list[_FeedbackRow]:
        layer = await self.store.load_summary_layer(self.task_id)
        assert layer is not None
        rows: list[_FeedbackRow] = []
        for item in layer.feedback:
            raw = item["payload"]
            payload = dict(raw) if isinstance(raw, Mapping) else {}
            rows.append(
                _FeedbackRow(
                    feedback_id=str(item["feedback_id"]),
                    disposition=str(item["disposition"]),
                    supersedes_feedback_id=(
                        str(payload["supersedes_feedback_id"])
                        if payload.get("supersedes_feedback_id")
                        else None
                    ),
                    payload=payload,
                    lesson=str(payload.get("lesson") or ""),
                    round_id=str(item["round_id"]),
                )
            )
        return rows


@pytest.fixture
async def feedback_harness(tmp_path: Path) -> AsyncIterator[FeedbackHarness]:
    harness = await FeedbackHarness.create(tmp_path)
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_feedback_is_immutable_and_supersession_is_explicit(feedback_harness: FeedbackHarness) -> None:
    first = await feedback_harness.submit(task_id="lrs_1", disposition="mixed", corrections=["A应为B"])
    second = await feedback_harness.submit(
        task_id="lrs_1",
        disposition="superseded",
        supersedes_feedback_id=first.feedback_id,
        corrections=["最终应为C"],
    )
    rows = await feedback_harness.events()
    assert len(rows) == 2
    assert rows[0].feedback_id == first.feedback_id
    assert rows[1].supersedes_feedback_id == first.feedback_id
    assert second.feedback_id != first.feedback_id


@pytest.mark.asyncio
async def test_feedback_renders_deterministic_lesson_and_enqueues_vector(
    feedback_harness: FeedbackHarness,
) -> None:
    result = await feedback_harness.submit(
        task_id="lrs_1",
        disposition="accepted",
        useful_findings=["对照实验有效"],
        corrections=["剂量单位应为 mg"],
        missing_information=["缺少随访窗口"],
        outcome="方案已采纳",
        notes="额外备注不应主导 lesson",
    )
    rows = await feedback_harness.events()
    assert len(rows) == 1
    lesson = rows[0].lesson
    assert "lrs_1" in lesson
    assert "accepted" in lesson
    assert "source_feedback_id" in lesson
    assert result.feedback_id in lesson
    assert "对照实验有效" in lesson
    assert "剂量单位应为 mg" in lesson
    assert "缺少随访窗口" in lesson
    assert "方案已采纳" in lesson
    assert len(lesson) <= 8000
    assert feedback_harness.index.enqueued == [(SOURCE_KIND_FEEDBACK_LESSON, result.feedback_id)]
    assert result.lesson_id == result.feedback_id
    assert result.disposition == "accepted"
    assert result.lesson_indexing == "indexed"
    assert result.lesson_index_error is None


@pytest.mark.asyncio
async def test_post_commit_enqueue_exception_does_not_fail_submit(
    feedback_harness: FeedbackHarness,
) -> None:
    class BoomIndex(FakeVectorIndex):
        async def enqueue(self, *, source_kind: str, source_id: str) -> Any:
            self.enqueued.append((source_kind, source_id))
            raise RuntimeError("lancedb unavailable")

    boom = BoomIndex()
    feedback_harness.index = boom
    feedback_harness.service.vector_index = boom

    result = await feedback_harness.submit(task_id="lrs_1", disposition="accepted", notes="仍应成功")
    rows = await feedback_harness.events()
    assert len(rows) == 1
    assert result.feedback_id
    assert result.lesson_indexing == "failed"
    assert result.lesson_index_error is not None
    assert "lancedb" in result.lesson_index_error


@pytest.mark.asyncio
async def test_post_commit_enqueue_vector_op_failure_is_reported(
    feedback_harness: FeedbackHarness,
) -> None:
    from lunagentic_research_swarm.errors import LRSError, VECTOR_INDEX_REBUILDING
    from lunagentic_research_swarm.storage.vectors import VectorOpResult

    class FailIndex(FakeVectorIndex):
        async def enqueue(self, *, source_kind: str, source_id: str) -> Any:
            self.enqueued.append((source_kind, source_id))
            return VectorOpResult.fail(LRSError(VECTOR_INDEX_REBUILDING, "向量索引正在重建"))

    failing = FailIndex()
    feedback_harness.index = failing
    feedback_harness.service.vector_index = failing

    result = await feedback_harness.submit(task_id="lrs_1", disposition="mixed", corrections=["x"])
    assert result.feedback_id
    assert result.lesson_indexing == "pending"
    assert result.lesson_index_error == VECTOR_INDEX_REBUILDING
    rows = await feedback_harness.events()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_superseded_requires_same_task_parent_and_others_forbid_supersedes(
    feedback_harness: FeedbackHarness,
) -> None:
    first = await feedback_harness.submit(task_id="lrs_1", disposition="rejected", notes="初版")
    with pytest.raises(ValueError, match="supersedes"):
        await feedback_harness.submit(task_id="lrs_1", disposition="accepted", supersedes_feedback_id=first.feedback_id)
    with pytest.raises(ValueError, match="supersedes"):
        await feedback_harness.submit(task_id="lrs_1", disposition="superseded")
    with pytest.raises(ValueError, match="supersedes"):
        await feedback_harness.submit(
            task_id="lrs_1",
            disposition="superseded",
            supersedes_feedback_id="missing_fb",
        )


@pytest.mark.asyncio
async def test_submit_cancels_pending_reminder_for_round(feedback_harness: FeedbackHarness) -> None:
    await feedback_harness.service.schedule(task_id="lrs_1", round_id="rnd_1", ended_at=10.0)
    await feedback_harness.submit(task_id="lrs_1", disposition="mixed", corrections=["更正"])

    def _read(connection: Any) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT status FROM feedback_reminders WHERE round_id = ?",
            ("rnd_1",),
        ).fetchall()
        return [dict(row) for row in rows]

    reminders = await feedback_harness.store.run_locked(_read)
    assert reminders
    assert all(row["status"] == "cancelled" for row in reminders)


@pytest.mark.asyncio
async def test_duplicate_reminder_insert_is_noop(feedback_harness: FeedbackHarness) -> None:
    """UNIQUE(round_id) conflict must not abort; second insert is discarded."""

    await feedback_harness.store.transact(
        [
            feedback_harness.service.schedule_command(task_id="lrs_1", round_id="rnd_1", ended_at=10.0),
        ]
    )
    await feedback_harness.store.transact(
        [
            feedback_harness.service.schedule_command(task_id="lrs_1", round_id="rnd_1", ended_at=20.0),
        ]
    )

    def _read(connection: Any) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT due_at, status FROM feedback_reminders WHERE round_id = ?",
            ("rnd_1",),
        ).fetchall()
        return [dict(row) for row in rows]

    rows = await feedback_harness.store.run_locked(_read)
    assert len(rows) == 1
    assert rows[0]["due_at"] == 610.0
    assert rows[0]["status"] == "pending"
