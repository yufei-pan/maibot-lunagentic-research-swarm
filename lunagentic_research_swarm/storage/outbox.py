"""Crash-safe, at-least-once delivery of report notifications through Maisaka."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any


class MaisakaOutbox:
    """Deliver durable append/trigger intents in two independent phases.

    ``append_report`` rows are marked delivered and their corresponding trigger
    row is inserted in one SQLite transaction only after the append capability
    succeeds.  Consequently trigger retries never repeat a completed append.
    """

    def __init__(
        self,
        store: Any,
        maisaka: Any,
        *,
        poll_interval_seconds: float = 2.0,
        poll_interval: float | None = None,
        lease_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.maisaka = maisaka
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds if poll_interval is None else poll_interval))
        self.lease_seconds = max(0.001, float(lease_seconds))
        self.clock = clock
        self._wake_event = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._closed = False
        self._worker = asyncio.create_task(self._run(), name="lrs-maisaka-outbox")

    async def close(self) -> None:
        self._closed = True
        self._wake_event.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    def wake(self) -> None:
        self._wake_event.set()

    async def deliver_once(self) -> int:
        """Attempt currently due rows and return the number attempted."""

        async with self._lock:
            now = float(self.clock())
            claim = getattr(self.store, "claim_due_outbox", None)
            if callable(claim):
                try:
                    rows = await claim(now, lease_seconds=self.lease_seconds, limit=100)
                except (AttributeError, TypeError):
                    # Lightweight fake stores used by plugin tests may expose
                    # only list_due_outbox and therefore cannot claim leases.
                    rows = await self.store.list_due_outbox(now)
            else:
                rows = await self.store.list_due_outbox(now)
            attempted = 0
            for row in rows:
                attempted += 1
                try:
                    await self._deliver_row(row)
                except Exception as exc:
                    await self._record_failure(row, exc)
            return attempted

    async def _run(self) -> None:
        while not self._closed:
            try:
                await self.deliver_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Delivery errors are persisted per row; a worker failure must
                # not prevent subsequent due rows from being retried.
                pass
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                pass
            self._wake_event.clear()

    async def _deliver_row(self, row: Mapping[str, Any]) -> None:
        payload = json.loads(str(row["payload_json"]))
        # SQLite queries join the task stream; lightweight test/store adapters
        # may expose only the outbox columns, in which case task_id is the
        # safest stable routing fallback.
        stream_id = str(row.get("stream_id") or row.get("task_id"))
        kind = str(row["delivery_kind"]).lower()
        if kind in {"append", "append_report"}:
            await self._append(stream_id, payload, row)
            trigger = self._trigger_row(row, payload)
            await self.store.complete_outbox_append(
                str(row["outbox_id"]), trigger, delivered_at=float(self.clock()),
                **self._lease_kwargs(row),
            )
            return
        if kind in {"trigger", "trigger_report_review"}:
            await self._trigger(stream_id, payload, row)
            await self.store.mark_outbox_delivered(
                str(row["outbox_id"]), delivered_at=float(self.clock()), **self._lease_kwargs(row)
            )
            return
        raise ValueError(f"unknown Maisaka outbox delivery kind: {kind}")

    async def _append(self, stream_id: str, payload: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        text = str(payload.get("text", ""))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            segments = [{"type": "text", "content": text}]
        capability = getattr(getattr(self.maisaka, "context", None), "append", None)
        if capability is None:
            capability = getattr(self.maisaka, "append")
        await capability(
            stream_id,
            segments,
            visible_text=text,
            source_kind="lrs_report",
            message_id=f"lrs-report:{row.get('report_id') or row['outbox_id']}",
        )

    async def _trigger(self, stream_id: str, payload: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        else:
            metadata = dict(metadata)
        metadata["outbox_id"] = str(row["outbox_id"])
        if row.get("report_id") is not None:
            metadata["report_id"] = str(row["report_id"])
        if row.get("idempotency_key") is not None:
            metadata["idempotency_key"] = str(row["idempotency_key"])
        intent = str(payload.get("intent") or "review_lrs_report")
        reason = str(payload.get("reason") or "")
        capability = getattr(getattr(self.maisaka, "proactive", None), "trigger", None)
        if capability is None:
            capability = getattr(self.maisaka, "trigger")
        await capability(
            stream_id,
            intent,
            reason=reason,
            priority=str(payload.get("priority") or "normal"),
            metadata=metadata,
        )

    def _trigger_row(self, row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
        task_id, round_id, report_id = str(row["task_id"]), str(row["round_id"]), row.get("report_id")
        kind = str(payload.get("kind") or row.get("delivery_kind") or "INTERMEDIATE").upper()
        running = int(payload.get("running_branch_count", 0))
        append_key = str(row.get("idempotency_key") or f"lrs:{task_id}:{round_id}:{report_id}:append")
        trigger_key = append_key[:-len(":append")] + ":trigger" if append_key.endswith(":append") else f"{append_key}:trigger"
        return {
            "outbox_id": f"out_{uuid.uuid4().hex}",
            "task_id": task_id,
            "round_id": round_id,
            "report_id": report_id,
            "delivery_kind": "trigger_report_review",
            "idempotency_key": trigger_key,
            "payload_json": json.dumps(
                {
                    "intent": f"review_{kind.lower()}_report",
                    "reason": f"LRS {kind.lower()} report ready (task_id={task_id}, running_branch_count={running})",
                    "metadata": {
                        "task_id": task_id,
                        "report_id": report_id,
                        "round_id": round_id,
                        "report_kind": kind,
                        "running_branch_count": running,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "next_attempt_at": float(self.clock()),
            "created_at": float(self.clock()),
        }

    async def _record_failure(self, row: Mapping[str, Any], exc: Exception) -> None:
        attempt = int(row.get("attempt_count", 0)) + 1
        delay = min(300.0, float(2 ** min(attempt, 8)))
        await self.store.mark_outbox_failed(
            str(row["outbox_id"]), attempt_count=attempt,
            next_attempt_at=float(self.clock()) + delay,
            error=f"{type(exc).__name__}: {exc}"[:1000],
            **self._lease_kwargs(row),
        )

    @staticmethod
    def _lease_kwargs(row: Mapping[str, Any]) -> dict[str, float]:
        lease_until = row.get("_lease_until")
        return {"lease_until": float(lease_until)} if lease_until is not None else {}


__all__ = ["MaisakaOutbox"]
