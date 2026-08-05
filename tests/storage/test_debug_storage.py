"""可选 raw debug 存储：两开关独立、不进 LanceDB、失败不回滚权威事务。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.llm.gateway import GenerationResult
from lunagentic_research_swarm.llm.pricing import TokenUsage
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall, PerformProcedureBatch
from lunagentic_research_swarm.runtime.turns import TurnWorker
from lunagentic_research_swarm.storage.debug import DebugStore
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition


class _FakeLLM:
    async def generate(self, request: Any) -> GenerationResult:
        return GenerationResult(
            json.dumps(
                {
                    "report": "ok",
                    "delegations": [],
                    "procedures": [],
                },
                ensure_ascii=False,
            ),
            None,
            "gpt-test",
            TokenUsage(prompt_tokens=4, completion_tokens=2, cache_hit_tokens=1, cache_miss_tokens=1),
            True,
            None,
            0.0,
        )


class _FakeAPI:
    async def call(self, name: str, *, version: str = "1", **kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "data": {"answer": "raw-procedure-secret", "echo": kwargs.get("arguments")},
            "error": None,
            "metadata": {},
        }


class _FakeVector:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def contains(self, needle: str) -> bool:
        return any(needle in text for text in self.texts)

    async def enqueue(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        text = kwargs.get("text") or (args[1] if len(args) > 1 else "")
        self.texts.append(str(text))
        return SimpleNamespace(success=True, status="indexed")


def _catalog() -> ProcedureCatalogSnapshot:
    definition = ProcedureDefinition.model_validate(
        {
            "procedure_id": "builtin.debug_probe",
            "version": "1",
            "display_name": "debug",
            "description": "debug probe",
            "arguments_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "idempotent": True,
            "timeout_seconds": 5.0,
        }
    )
    return ProcedureCatalogSnapshot(
        [
            ProcedureCatalogEntry(
                definition=definition,
                provider_plugin_id="builtin",
                api_name="builtin.invoke_procedure",
                api_version="1",
                fingerprint="fp-debug",
            )
        ]
    )


class DebugHarness:
    """驱动一轮 agent call + procedure，并断言 debug / vector / authority 边界。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.store = SQLiteStateStore(data_dir / "state.sqlite3")
        self.vector = _FakeVector()
        self.debug: DebugStore | None = None
        self._authority_errors: list[dict[str, Any]] = []

    async def open(self) -> None:
        await self.store.open()
        await self.store.transact(
            [
                StoreCommand(
                    "insert_task",
                    {"task_id": "lrs_debug", "stream_id": "s", "created_at": 1.0},
                ),
                StoreCommand(
                    "insert_round",
                    {
                        "round_id": "rnd_debug",
                        "task_id": "lrs_debug",
                        "round_number": 1,
                        "generation": 1,
                        "status": "RUNNING",
                        "time_budget_seconds": 60,
                        "credit_pool": 10.0,
                        "started_at": 1.0,
                    },
                ),
            ]
        )

    async def close(self) -> None:
        if self.debug is not None:
            await self.debug.close()
        await self.store.close()

    def debug_dir(self) -> Path:
        return self.data_dir / "debug"

    def has_transcript_rows(self) -> bool:
        path = self.debug_dir() / "raw.sqlite3"
        if not path.exists():
            return False
        with sqlite3.connect(path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM agent_transcripts").fetchone()
        return int(row[0]) > 0

    def has_payload_rows(self) -> bool:
        path = self.debug_dir() / "raw.sqlite3"
        if not path.exists():
            return False
        with sqlite3.connect(path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM procedure_raw_payloads").fetchone()
        return int(row[0]) > 0

    def vector_text_contains(self, needle: str) -> bool:
        return self.vector.contains(needle)

    async def run_one_turn(self, *, transcripts: bool, payloads: bool) -> None:
        self.debug = DebugStore(
            self.data_dir,
            store_agent_transcripts=transcripts,
            store_raw_procedure_payloads=payloads,
            authority_store=self.store,
            on_storage_failed=self._record_failure,
        )
        await self.debug.open()

        procedures = ProcedureExecutor(
            _catalog(),
            api=_FakeAPI(),
            debug_store=self.debug,
        )
        worker = TurnWorker(_FakeLLM(), procedures, debug_store=self.debug)
        effect = PerformAgentCall(
            "lrs_debug",
            "rnd_debug",
            1,
            payload={
                "branch_id": "br_root",
                "call_id": "call_1",
                "selector": "task:utils",
                "protocol": "json_envelope",
                "messages": (
                    {"role": "user", "content": "raw-agent-secret"},
                    {"role": "assistant", "content": "thinking"},
                ),
                "estimated_charge": 0.1,
                "credits_after_reservation": 9.9,
                "completed_event_id": "evt_turn:completed",
                "result_id": "call_1:result",
            },
        )
        completed = await worker.perform_agent_call(effect)
        # 模拟权威 summary 成功后立即清空 runtime messages；debug 已落盘也不再回读。
        list(getattr(completed, "messages", ()) or ())

        batch = PerformProcedureBatch(
            "lrs_debug",
            "rnd_debug",
            1,
            payload={
                "branch_id": "br_root",
                "turn_id": "call_1",
                "agent_id": "builtin.quick_thinker",
                "call_id": "call_1",
                "requests": [
                    ProcedureRequest(
                        procedure_id="builtin.debug_probe",
                        arguments={"secret": "raw-procedure-secret"},
                    )
                ],
            },
        )
        await worker.perform_procedure_batch(batch)
        layer = await self.store.load_summary_layer("lrs_debug")
        assert layer is not None
        encoded = json.dumps(
            {
                "formalized": layer.formalized_task.text if layer.formalized_task else "",
                "summaries": [dict(item) for item in layer.summaries],
                "reports": [dict(item) for item in layer.reports],
            },
            ensure_ascii=False,
            default=str,
        )
        assert "raw-agent-secret" not in encoded
        assert "raw-procedure-secret" not in encoded
        await self.vector.enqueue("formalized_task", "lrs_debug", text="formalized only")

    def _record_failure(self, payload: dict[str, Any]) -> None:
        self._authority_errors.append(dict(payload))


@pytest.fixture
async def debug_harness(tmp_path: Path):
    harness = DebugHarness(tmp_path)
    await harness.open()
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.parametrize(
    ("transcripts", "payloads", "expect_transcript", "expect_payload"),
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    ],
)
@pytest.mark.asyncio
async def test_raw_storage_toggles_are_independent(
    debug_harness: DebugHarness,
    transcripts: bool,
    payloads: bool,
    expect_transcript: bool,
    expect_payload: bool,
) -> None:
    await debug_harness.run_one_turn(transcripts=transcripts, payloads=payloads)
    assert debug_harness.has_transcript_rows() is expect_transcript
    assert debug_harness.has_payload_rows() is expect_payload
    assert not debug_harness.vector_text_contains("raw-agent-secret")
    assert not debug_harness.vector_text_contains("raw-procedure-secret")
    if not transcripts and not payloads:
        assert not debug_harness.debug_dir().exists()
    else:
        assert (debug_harness.debug_dir() / "raw.sqlite3").exists()
        with sqlite3.connect(debug_harness.debug_dir() / "raw.sqlite3") as connection:
            columns = {
                row[1]
                for table in ("agent_transcripts", "procedure_raw_payloads")
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
        assert "reasoning" not in {name.lower() for name in columns}


@pytest.mark.asyncio
async def test_debug_write_failure_does_not_block_authority_and_records_minimal_error(
    tmp_path: Path,
) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    failures: list[dict[str, Any]] = []

    class BrokenDebug(DebugStore):
        async def save_transcript(self, **kwargs: Any) -> None:
            raise OSError("disk full")

        async def save_payload(self, **kwargs: Any) -> None:
            raise OSError("disk full")

    debug = BrokenDebug(
        tmp_path,
        store_agent_transcripts=True,
        store_raw_procedure_payloads=True,
        authority_store=store,
        on_storage_failed=lambda payload: failures.append(dict(payload)),
    )
    await debug.open()
    worker = TurnWorker(
        _FakeLLM(),
        ProcedureExecutor(_catalog(), api=_FakeAPI(), debug_store=debug),
        debug_store=debug,
    )
    effect = PerformAgentCall(
        "lrs_fail",
        "rnd_fail",
        1,
        payload={
            "branch_id": "br",
            "call_id": "c1",
            "selector": "task:utils",
            "messages": ({"role": "user", "content": "raw-agent-secret"},),
            "estimated_charge": 0.0,
            "credits_after_reservation": 0.0,
        },
    )
    completed = await worker.perform_agent_call(effect)
    assert completed is not None
    assert failures
    assert all(item.get("code") == "debug_storage_failed" for item in failures)
    assert all("raw-agent-secret" not in json.dumps(item, ensure_ascii=False) for item in failures)
    await debug.close()
    await store.close()
