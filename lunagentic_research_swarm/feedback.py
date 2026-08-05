"""研究反馈事件、确定性 lesson 与 600 秒提醒。

Feedback 只做透明检索/排序/统计/lesson，绝不静默改写 prompts、selectors 或路由。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.storage.sqlite import StoreCommand
from lunagentic_research_swarm.storage.vectors import SOURCE_KIND_FEEDBACK_LESSON

_LOG = logging.getLogger(__name__)

DISPOSITIONS = frozenset({"accepted", "mixed", "rejected", "superseded"})
REMINDER_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
        TaskStatus.STOPPED.value,
    }
)
_NO_REMINDER_TERMINALS = frozenset(
    {
        TaskStatus.EXPIRED.value,
        TaskStatus.INTERRUPTED.value,
        TaskStatus.FAILED.value,
    }
)
_REMINDER_PENDING = "pending"
_REMINDER_TRIGGERED = "triggered"
_REMINDER_CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    feedback_id: str
    lesson_id: str
    disposition: str
    round_id: str
    lesson: str
    # 提交已持久化后的 lesson 索引状态；不得因索引失败而让 submit 看起来失败。
    # indexed | pending | failed | skipped | degraded
    lesson_indexing: str = "skipped"
    lesson_index_error: str | None = None


def render_feedback_lesson(
    *,
    feedback_id: str,
    task_id: str,
    disposition: str,
    round_number: int | None,
    useful_findings: Sequence[str],
    corrections: Sequence[str],
    missing_information: Sequence[str],
    outcome: str | None,
    max_chars: int = 8000,
) -> str:
    """确定性 lesson：不调用第五种总结器，只拼接非空短段。"""

    lines = [
        f"Task ID: {task_id}",
        f"Round: {round_number if round_number is not None else 'n/a'}",
        f"Disposition: {disposition}",
        f"source_feedback_id: {feedback_id}",
    ]
    if useful_findings:
        lines.append("Useful findings:")
        lines.extend(f"- {item}" for item in useful_findings if str(item).strip())
    if corrections:
        lines.append("Corrections:")
        lines.extend(f"- {item}" for item in corrections if str(item).strip())
    if missing_information:
        lines.append("Missing information:")
        lines.extend(f"- {item}" for item in missing_information if str(item).strip())
    if isinstance(outcome, str) and outcome.strip():
        lines.append("Outcome:")
        lines.append(outcome.strip())
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[: max(0, max_chars - 1)] + "…"
    return text


class FeedbackService:
    """append-only feedback 事件、lesson 索引与终态提醒。"""

    def __init__(
        self,
        store: Any,
        *,
        vector_index: Any | None = None,
        outbox: Any | None = None,
        clock: Callable[[], float] = time.time,
        feedback_wait_seconds: int = 600,
        reminders_enabled: bool = True,
        index_lessons: bool = True,
        max_lesson_chars: int = 8000,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.vector_index = vector_index
        self.outbox = outbox
        self.clock = clock
        self.feedback_wait_seconds = max(1, int(feedback_wait_seconds))
        self.reminders_enabled = bool(reminders_enabled)
        self.index_lessons = bool(index_lessons)
        self.max_lesson_chars = max(1, int(max_lesson_chars))
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._worker: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._closed = False

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._closed = False
        self._worker = asyncio.create_task(self._run(), name="lrs-feedback-reminders")

    async def close(self) -> None:
        self._closed = True
        self._wake.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    def wake(self) -> None:
        self._wake.set()

    def schedule_command(self, *, task_id: str, round_id: str, ended_at: float) -> StoreCommand | None:
        """与终态 status 同一 transaction 插入 unique(round_id) reminder。"""

        if not self.reminders_enabled:
            return None
        return StoreCommand(
            "insert_feedback_reminder",
            {
                "reminder_id": f"fbr_{uuid.uuid4().hex}",
                "task_id": str(task_id),
                "round_id": str(round_id),
                "due_at": float(ended_at) + float(self.feedback_wait_seconds),
                "status": _REMINDER_PENDING,
                "triggered_at": None,
            },
        )

    def cancel_command(self, *, round_id: str) -> StoreCommand:
        return StoreCommand(
            "cancel_pending_feedback_reminders",
            {"round_id": str(round_id)},
        )

    def commands_for_status_transition(
        self,
        *,
        task_id: str,
        round_id: str,
        new_status: str,
        ended_at: float,
        cancel_round_id: str | None = None,
    ) -> tuple[StoreCommand, ...]:
        commands: list[StoreCommand] = []
        if cancel_round_id:
            commands.append(self.cancel_command(round_id=cancel_round_id))
        status = str(new_status)
        if status in REMINDER_TERMINAL_STATUSES:
            scheduled = self.schedule_command(task_id=task_id, round_id=round_id, ended_at=ended_at)
            if scheduled is not None:
                commands.append(scheduled)
        elif status in _NO_REMINDER_TERMINALS:
            pass
        return tuple(commands)

    async def schedule(self, *, task_id: str, round_id: str, ended_at: float) -> None:
        command = self.schedule_command(task_id=task_id, round_id=round_id, ended_at=ended_at)
        if command is None:
            return
        await self.store.transact([command])
        self.wake()

    async def cancel_due_to_continue(self, *, task_id: str, round_id: str) -> None:
        del task_id  # 取消按 round 作用域；保留参数以匹配服务接口。
        await self.store.transact([self.cancel_command(round_id=round_id)])

    async def submit(
        self,
        *,
        task_id: str,
        disposition: str,
        round_number: int | None = None,
        rating: int | None = None,
        useful_findings: Sequence[str] | None = None,
        incorrect_findings: Sequence[str] | None = None,
        missing_information: Sequence[str] | None = None,
        decision: str | None = None,
        outcome: str | None = None,
        corrections: Sequence[str] | None = None,
        notes: str | None = None,
        supersedes_feedback_id: str | None = None,
    ) -> FeedbackResult:
        task_id = str(task_id).strip()
        if not task_id:
            raise ValueError("task_id 不能为空")
        disposition = str(disposition).strip().lower()
        if disposition not in DISPOSITIONS:
            raise ValueError(f"disposition 必须为 {sorted(DISPOSITIONS)} 之一")
        if rating is not None:
            if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
                raise ValueError("rating 必须为 1..5 的整数")
        useful = _string_list(useful_findings, "useful_findings")
        incorrect = _string_list(incorrect_findings, "incorrect_findings")
        missing = _string_list(missing_information, "missing_information")
        corrections_list = _string_list(corrections, "corrections")
        decision_text = _optional_str(decision, "decision")
        outcome_text = _optional_str(outcome, "outcome")
        notes_text = _optional_str(notes, "notes")
        supersedes = _optional_str(supersedes_feedback_id, "supersedes_feedback_id")

        if disposition == "superseded":
            if not supersedes:
                raise ValueError("superseded 必须提供 supersedes_feedback_id")
        elif supersedes:
            raise ValueError("仅 disposition=superseded 可提供 supersedes_feedback_id")

        feedback_id = f"fb_{uuid.uuid4().hex}"
        now = float(self.clock())

        def _commit(connection: Any) -> FeedbackResult:
            task = connection.execute(
                "SELECT task_id, current_round_number FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise LookupError(f"调查任务不存在：{task_id}")
            if round_number is None:
                target_number = int(task["current_round_number"])
            else:
                if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
                    raise ValueError("round_number 必须为正整数")
                target_number = int(round_number)
            round_row = connection.execute(
                """
                SELECT round_id, round_number FROM investigation_rounds
                WHERE task_id = ? AND round_number = ?
                """,
                (task_id, target_number),
            ).fetchone()
            if round_row is None:
                raise LookupError(f"round_number={target_number} 不存在")
            round_id = str(round_row["round_id"])
            if supersedes:
                parent = connection.execute(
                    """
                    SELECT feedback_id FROM feedback_events
                    WHERE feedback_id = ? AND task_id = ?
                    """,
                    (supersedes, task_id),
                ).fetchone()
                if parent is None:
                    raise ValueError("supersedes_feedback_id 必须引用同 Task 已有 event")

            lesson = render_feedback_lesson(
                feedback_id=feedback_id,
                task_id=task_id,
                disposition=disposition,
                round_number=int(round_row["round_number"]),
                useful_findings=useful,
                corrections=corrections_list,
                missing_information=missing,
                outcome=outcome_text,
                max_chars=self.max_lesson_chars,
            )
            payload = {
                "task_id": task_id,
                "round_number": int(round_row["round_number"]),
                "disposition": disposition,
                "rating": rating,
                "useful_findings": useful,
                "incorrect_findings": incorrect,
                "missing_information": missing,
                "decision": decision_text,
                "outcome": outcome_text,
                "corrections": corrections_list,
                "notes": notes_text,
                "supersedes_feedback_id": supersedes,
                "lesson": lesson,
                "source_feedback_id": feedback_id,
            }
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO feedback_events(
                        feedback_id, task_id, round_id, disposition, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        task_id,
                        round_id,
                        disposition,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE feedback_reminders
                    SET status = ?
                    WHERE round_id = ? AND status = ?
                    """,
                    (_REMINDER_CANCELLED, round_id, _REMINDER_PENDING),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return FeedbackResult(
                feedback_id=feedback_id,
                lesson_id=feedback_id,
                disposition=disposition,
                round_id=round_id,
                lesson=lesson,
            )

        result = await self.store.run_locked(_commit)
        return await self._index_lesson_after_commit(result)

    async def _index_lesson_after_commit(self, result: FeedbackResult) -> FeedbackResult:
        """提交已提交后尽力入队；失败只降级报告，绝不回滚或冒充 submit 失败。"""

        if not self.index_lessons:
            return result
        if self.vector_index is None:
            return FeedbackResult(
                feedback_id=result.feedback_id,
                lesson_id=result.lesson_id,
                disposition=result.disposition,
                round_id=result.round_id,
                lesson=result.lesson,
                lesson_indexing="degraded",
                lesson_index_error="vector_index_unavailable",
            )
        enqueue = getattr(self.vector_index, "enqueue", None)
        if not callable(enqueue):
            return FeedbackResult(
                feedback_id=result.feedback_id,
                lesson_id=result.lesson_id,
                disposition=result.disposition,
                round_id=result.round_id,
                lesson=result.lesson,
                lesson_indexing="degraded",
                lesson_index_error="vector_enqueue_unavailable",
            )
        try:
            op = await enqueue(source_kind=SOURCE_KIND_FEEDBACK_LESSON, source_id=result.feedback_id)
        except Exception as exc:
            _LOG.warning(
                "feedback lesson enqueue failed after commit feedback_id=%s: %s",
                result.feedback_id,
                exc,
                exc_info=True,
            )
            return FeedbackResult(
                feedback_id=result.feedback_id,
                lesson_id=result.lesson_id,
                disposition=result.disposition,
                round_id=result.round_id,
                lesson=result.lesson,
                lesson_indexing="failed",
                lesson_index_error=str(exc)[:256] or type(exc).__name__,
            )
        success = bool(getattr(op, "success", False))
        code = getattr(op, "code", None)
        error = getattr(op, "error", None)
        error_code = getattr(error, "code", None) if error is not None else None
        if success:
            indexing = "indexed" if code in (None, "indexed") else str(code)
            return FeedbackResult(
                feedback_id=result.feedback_id,
                lesson_id=result.lesson_id,
                disposition=result.disposition,
                round_id=result.round_id,
                lesson=result.lesson,
                lesson_indexing=indexing,
                lesson_index_error=None,
            )
        # 重建中等可恢复状态报告为 pending，其余为 failed；均不抛出。
        status = "pending" if error_code == "vector_index_rebuilding" or code == "vector_index_rebuilding" else "failed"
        detail = str(error_code or code or "vector_enqueue_failed")
        _LOG.warning(
            "feedback lesson enqueue returned %s feedback_id=%s code=%s",
            status,
            result.feedback_id,
            detail,
        )
        return FeedbackResult(
            feedback_id=result.feedback_id,
            lesson_id=result.lesson_id,
            disposition=result.disposition,
            round_id=result.round_id,
            lesson=result.lesson,
            lesson_indexing=status,
            lesson_index_error=detail,
        )

    async def process_due(self) -> int:
        """处理到期 reminder：无该 round feedback/新 round 时写 outbox 并标 triggered。"""

        now = float(self.clock())

        def _claim(connection: Any) -> list[dict[str, Any]]:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT reminder_id, task_id, round_id, due_at, status
                    FROM feedback_reminders
                    WHERE status = ? AND due_at <= ?
                    ORDER BY due_at ASC, reminder_id ASC
                    LIMIT 50
                    """,
                    (_REMINDER_PENDING, now),
                ).fetchall()
                claimed: list[dict[str, Any]] = []
                for row in rows:
                    item = dict(row)
                    task_id = str(item["task_id"])
                    round_id = str(item["round_id"])
                    has_feedback = connection.execute(
                        "SELECT 1 FROM feedback_events WHERE round_id = ? LIMIT 1",
                        (round_id,),
                    ).fetchone()
                    round_meta = connection.execute(
                        """
                        SELECT round_number FROM investigation_rounds
                        WHERE round_id = ? AND task_id = ?
                        """,
                        (round_id, task_id),
                    ).fetchone()
                    task = connection.execute(
                        "SELECT current_round_number, stream_id FROM tasks WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    newer_round = False
                    if round_meta is not None and task is not None:
                        newer_round = int(task["current_round_number"]) > int(round_meta["round_number"])
                    if has_feedback is not None or newer_round:
                        connection.execute(
                            """
                            UPDATE feedback_reminders
                            SET status = ?
                            WHERE reminder_id = ? AND status = ?
                            """,
                            (_REMINDER_CANCELLED, item["reminder_id"], _REMINDER_PENDING),
                        )
                        continue
                    stream_id = str(task["stream_id"]) if task is not None else task_id
                    outbox_id = f"out_{uuid.uuid4().hex}"
                    payload = {
                        "intent": (
                            "请检查该深度调查任务的报告与结论，并携带当前 task_id "
                            f"调用 submit_research_feedback（task_id={task_id}）。"
                        ),
                        "reason": f"LRS feedback reminder due (task_id={task_id}, round_id={round_id})",
                        "priority": "normal",
                        "metadata": {
                            "task_id": task_id,
                            "round_id": round_id,
                            "reminder_id": str(item["reminder_id"]),
                            "intent_tool": "submit_research_feedback",
                        },
                    }
                    connection.execute(
                        """
                        INSERT INTO maisaka_outbox(
                            outbox_id, task_id, round_id, report_id, delivery_kind,
                            idempotency_key, payload_json, status, attempt_count,
                            next_attempt_at, last_error, created_at, delivered_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            outbox_id,
                            task_id,
                            round_id,
                            None,
                            "trigger",
                            f"lrs:feedback-reminder:{round_id}",
                            json.dumps(payload, ensure_ascii=False, sort_keys=True),
                            "PENDING",
                            0,
                            now,
                            None,
                            now,
                            None,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE feedback_reminders
                        SET status = ?, triggered_at = ?
                        WHERE reminder_id = ? AND status = ?
                        """,
                        (_REMINDER_TRIGGERED, now, item["reminder_id"], _REMINDER_PENDING),
                    )
                    item["stream_id"] = stream_id
                    item["outbox_id"] = outbox_id
                    claimed.append(item)
                connection.execute("COMMIT")
                return claimed
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        claimed = await self.store.run_locked(_claim)
        if claimed and self.outbox is not None:
            wake = getattr(self.outbox, "wake", None)
            if callable(wake):
                wake()
        return len(claimed)

    async def _run(self) -> None:
        while not self._closed:
            try:
                await self.process_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("feedback reminder process_due failed; will retry on next poll")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()


def _string_list(value: Sequence[str] | None, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} 必须为字符串数组")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} 必须为字符串数组")
        text = item.strip()
        if text:
            items.append(text)
    return items


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须为字符串")
    text = value.strip()
    return text or None


def validate_feedback_arguments(payload: Mapping[str, Any]) -> str | None:
    """供 Planner tool 使用的轻量参数校验；返回中文错误信息。"""

    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return "task_id 不能为空"
    disposition = payload.get("disposition")
    if not isinstance(disposition, str) or disposition.strip().lower() not in DISPOSITIONS:
        return "disposition 必须为 accepted|mixed|rejected|superseded"
    rating = payload.get("rating")
    if rating is not None and (isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5):
        return "rating 必须为 1..5 的整数"
    for field in (
        "useful_findings",
        "incorrect_findings",
        "missing_information",
        "corrections",
    ):
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return f"{field} 必须为字符串数组"
        if any(not isinstance(item, str) for item in value):
            return f"{field} 必须为字符串数组"
    for field in ("decision", "outcome", "notes", "supersedes_feedback_id"):
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            return f"{field} 必须为字符串"
    round_number = payload.get("round_number")
    if round_number is not None and (
        isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1
    ):
        return "round_number 必须为正整数"
    return None


__all__ = [
    "DISPOSITIONS",
    "FeedbackResult",
    "FeedbackService",
    "REMINDER_TERMINAL_STATUSES",
    "render_feedback_lesson",
    "validate_feedback_arguments",
]
