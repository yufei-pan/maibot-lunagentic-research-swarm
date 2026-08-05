"""可选 raw debug 存储：独立 DB，默认关闭，永不进入权威 summary / LanceDB。"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_transcripts (
  transcript_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  messages_json TEXT NOT NULL,
  envelope_json TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS procedure_raw_payloads (
  payload_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  procedure_id TEXT NOT NULL,
  request_json TEXT NOT NULL,
  arguments_json TEXT NOT NULL,
  raw_result_json TEXT,
  created_at REAL NOT NULL
);
"""


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class DebugStore:
    """独立 ``data_dir/debug/raw.sqlite3``；两开关全关时不创建目录。"""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        store_agent_transcripts: bool = False,
        store_raw_procedure_payloads: bool = False,
        authority_store: Any | None = None,
        on_storage_failed: Callable[[Mapping[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.store_agent_transcripts = bool(store_agent_transcripts)
        self.store_raw_procedure_payloads = bool(store_raw_procedure_payloads)
        self.authority_store = authority_store
        self.on_storage_failed = on_storage_failed
        self.clock = clock
        self._path = self.data_dir / "debug" / "raw.sqlite3"
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor | None = None

    @property
    def enabled(self) -> bool:
        return self.store_agent_transcripts or self.store_raw_procedure_payloads

    @property
    def path(self) -> Path:
        return self._path

    async def _call(self, function: Callable[..., Any], *args: Any) -> Any:
        async with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lrs-debug")
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, partial(function, *args))

    def _open_sync(self) -> None:
        if not self.enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.executescript(_SCHEMA)
        connection.commit()
        self._connection = connection

    def _close_sync(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    async def open(self) -> None:
        await self._call(self._open_sync)

    async def close(self) -> None:
        await self._call(self._close_sync)
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("DebugStore 尚未 open")
        return self._connection

    async def save_transcript(
        self,
        *,
        task_id: str,
        round_id: str,
        branch_id: str,
        turn_id: str,
        messages: Sequence[Mapping[str, Any]] | Sequence[Any] = (),
        envelope: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.store_agent_transcripts:
            return
        try:
            await self._call(
                self._save_transcript_sync,
                str(task_id),
                str(round_id),
                str(branch_id),
                str(turn_id),
                tuple(dict(item) if isinstance(item, Mapping) else item for item in messages),
                dict(envelope) if isinstance(envelope, Mapping) else envelope,
            )
        except Exception as exc:
            await self._record_failure(kind="transcript", error=exc, task_id=task_id, round_id=round_id)

    def _save_transcript_sync(
        self,
        task_id: str,
        round_id: str,
        branch_id: str,
        turn_id: str,
        messages: Sequence[Any],
        envelope: Any,
    ) -> None:
        connection = self._require_connection()
        connection.execute(
            """
            INSERT INTO agent_transcripts(
                transcript_id, task_id, round_id, branch_id, turn_id,
                messages_json, envelope_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"tr_{uuid.uuid4().hex}",
                task_id,
                round_id,
                branch_id,
                turn_id,
                _json_dump(messages),
                _json_dump(envelope) if envelope is not None else None,
                float(self.clock()),
            ),
        )
        connection.commit()

    async def save_payload(
        self,
        *,
        task_id: str,
        round_id: str,
        branch_id: str,
        turn_id: str,
        request_id: str,
        procedure_id: str,
        request: Mapping[str, Any] | Any = None,
        arguments: Mapping[str, Any] | None = None,
        raw_result: Any = None,
    ) -> None:
        if not self.store_raw_procedure_payloads:
            return
        try:
            await self._call(
                self._save_payload_sync,
                str(task_id),
                str(round_id),
                str(branch_id),
                str(turn_id),
                str(request_id),
                str(procedure_id),
                request,
                dict(arguments or {}),
                raw_result,
            )
        except Exception as exc:
            await self._record_failure(kind="payload", error=exc, task_id=task_id, round_id=round_id)

    def _save_payload_sync(
        self,
        task_id: str,
        round_id: str,
        branch_id: str,
        turn_id: str,
        request_id: str,
        procedure_id: str,
        request: Any,
        arguments: Mapping[str, Any],
        raw_result: Any,
    ) -> None:
        connection = self._require_connection()
        connection.execute(
            """
            INSERT INTO procedure_raw_payloads(
                payload_id, task_id, round_id, branch_id, turn_id, request_id,
                procedure_id, request_json, arguments_json, raw_result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"pl_{uuid.uuid4().hex}",
                task_id,
                round_id,
                branch_id,
                turn_id,
                request_id,
                procedure_id,
                _json_dump(request if request is not None else {"procedure_id": procedure_id}),
                _json_dump(arguments),
                _json_dump(raw_result) if raw_result is not None else None,
                float(self.clock()),
            ),
        )
        connection.commit()

    async def _record_failure(
        self,
        *,
        kind: str,
        error: BaseException,
        task_id: str = "",
        round_id: str = "",
    ) -> None:
        payload = {
            "code": "debug_storage_failed",
            "kind": kind,
            "message": f"debug {kind} 写入失败：{type(error).__name__}",
            "task_id": str(task_id or ""),
            "round_id": str(round_id or ""),
        }
        _LOG.error("debug 存储失败（不影响权威事务）: %s", payload["message"])
        if self.on_storage_failed is not None:
            try:
                self.on_storage_failed(payload)
            except Exception:
                _LOG.exception("debug_storage_failed 回调异常")
        store = self.authority_store
        if store is None:
            return
        try:
            from lunagentic_research_swarm.storage.sqlite import StoreCommand

            await store.transact(
                [
                    StoreCommand(
                        "insert_extension_fingerprint",
                        {
                            "event_id": f"dbg_{uuid.uuid4().hex}",
                            "provider_plugin_id": "lrs.debug",
                            "extension_kind": "debug_storage",
                            "fingerprint": "debug_storage_failed",
                            "availability": "error",
                            "error_json": _json_dump(payload),
                            "created_at": float(self.clock()),
                        },
                    )
                ]
            )
        except Exception:
            _LOG.exception("写入 debug_storage_failed 权威事件失败")


__all__ = ["DebugStore"]
