from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import FrozenInstanceError

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.storage import sqlite as sqlite_module
from lunagentic_research_swarm.storage.migrations import MIGRATIONS, Migration
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


def _task(task_id: str, created_at: float = 1.0, **values: object) -> StoreCommand:
    return StoreCommand(
        "insert_task",
        {"task_id": task_id, "stream_id": f"stream_{task_id}", "created_at": created_at, **values},
    )


def _round(
    round_id: str,
    task_id: str,
    status: TaskStatus | str,
    *,
    round_number: int = 1,
    started_at: float = 1.0,
) -> StoreCommand:
    return StoreCommand(
        "insert_round",
        {
            "round_id": round_id,
            "task_id": task_id,
            "round_number": round_number,
            "status": status.value if isinstance(status, TaskStatus) else status,
            "generation": round_number,
            "time_budget_seconds": 120,
            "credit_pool": 10.0,
            "started_at": started_at,
        },
    )


@pytest.mark.asyncio
async def test_commands_commit_atomically_and_load_current_round(tmp_path) -> None:
    """若 task 与 round 未在同一 transition 落盘，读取将出现半状态。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact([_task("lrs_a"), _round("rnd_a", "lrs_a", TaskStatus.FORMALIZING)])

    loaded = await store.load_task("lrs_a")

    assert loaded is not None
    assert loaded.task_id == "lrs_a"
    assert loaded.current_round is not None
    assert loaded.current_round.status is TaskStatus.FORMALIZING
    await store.close()


@pytest.mark.asyncio
async def test_failed_command_rolls_back_whole_transition(tmp_path) -> None:
    """若第二条命令失败后首条仍存在，reducer 会在重启后看到假成功。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    with pytest.raises(sqlite3.IntegrityError):
        await store.transact([_task("lrs_b"), _task("lrs_b")])

    assert await store.load_task("lrs_b") is None
    await store.close()


@pytest.mark.asyncio
async def test_unknown_command_rolls_back_prior_commands(tmp_path) -> None:
    """若未知命令被忽略或在 transaction 外失败，前序写入会泄漏。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    with pytest.raises(ValueError, match="未知存储命令"):
        await store.transact([_task("lrs_unknown"), StoreCommand("drop_everything", {})])

    assert await store.load_task("lrs_unknown") is None
    await store.close()


@pytest.mark.asyncio
async def test_missing_task_formalization_target_raises_and_rolls_back_prior_mutation(tmp_path) -> None:
    """若 formalization UPDATE 静默 no-op，reducer 会提交一个不存在的正式任务。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    try:
        with pytest.raises(LookupError, match="不存在"):
            await store.transact(
                [
                    _task("lrs_prior_formalization"),
                    StoreCommand(
                        "update_task_formalization",
                        {
                            "task_id": "lrs_missing",
                            "formalized_text": "不存在的任务",
                            "formalized_sha256": "missing_hash",
                            "updated_at": 2.0,
                        },
                    ),
                ]
            )
        assert await store.load_task("lrs_prior_formalization") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_missing_current_round_target_raises_and_rolls_back_prior_mutation(tmp_path) -> None:
    """若 current round UPDATE 静默 no-op，后续读取会继续选中旧 round。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    try:
        with pytest.raises(LookupError, match="不存在"):
            await store.transact(
                [
                    _task("lrs_prior_current_round"),
                    StoreCommand(
                        "set_task_current_round",
                        {
                            "task_id": "lrs_missing",
                            "current_round_number": 2,
                            "updated_at": 2.0,
                        },
                    ),
                ]
            )
        assert await store.load_task("lrs_prior_current_round") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_missing_round_status_target_raises_and_rolls_back_prior_mutation(tmp_path) -> None:
    """若 round status UPDATE 静默 no-op，内存状态会领先于 SQLite 权威状态。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    try:
        with pytest.raises(LookupError, match="不存在"):
            await store.transact(
                [
                    _task("lrs_prior_round_status"),
                    StoreCommand(
                        "update_round_status",
                        {
                            "round_id": "rnd_missing",
                            "status": TaskStatus.RUNNING.value,
                        },
                    ),
                ]
            )
        assert await store.load_task("lrs_prior_round_status") is None
    finally:
        await store.close()


def test_store_command_copies_and_freezes_values() -> None:
    """若调用方能在排队后改写 values，transaction 内容将取决于竞态。"""

    values = {"task_id": "lrs_frozen", "metadata": {"tags": ["initial"]}}
    command = StoreCommand("insert_task", values)
    values["task_id"] = "lrs_mutated"
    values["metadata"]["tags"].append("mutated")  # type: ignore[index, union-attr]

    assert command.values["task_id"] == "lrs_frozen"
    assert command.values["metadata"]["tags"] == ("initial",)
    with pytest.raises(TypeError):
        command.values["task_id"] = "lrs_other"  # type: ignore[index]
    with pytest.raises(TypeError):
        command.values["metadata"]["other"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        command.kind = "other"  # type: ignore[misc]


def test_store_command_snapshots_mutable_buffer_values() -> None:
    """若 command 共享 bytearray/memoryview，排队后的 buffer 修改会改变落盘值。"""

    payload = bytearray(b"payload")
    viewed_payload = bytearray(b"viewed")
    command = StoreCommand(
        "buffer_probe",
        {
            "bytearray": payload,
            "writable_view": memoryview(viewed_payload),
            "readonly_view": memoryview(b"readonly"),
        },
    )
    payload[0] = ord("X")
    viewed_payload[0] = ord("Y")

    assert command.values["bytearray"] == b"payload"
    assert command.values["writable_view"] == b"viewed"
    assert command.values["readonly_view"] == b"readonly"
    assert all(isinstance(value, bytes) for value in command.values.values())


@pytest.mark.asyncio
async def test_cancelled_call_holds_lock_until_worker_finishes(tmp_path) -> None:
    """若取消 await 会提前释放锁，单连接可能被两个 worker 同时访问。"""

    entered = threading.Event()
    release = threading.Event()

    class BlockingStore(SQLiteStateStore):
        def _insert_task(self, connection, values) -> None:  # type: ignore[no-untyped-def, override]
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("测试未及时释放阻塞命令")
            super()._insert_task(connection, values)

    store = BlockingStore(tmp_path / "state.sqlite3")
    await store.open()
    first = asyncio.create_task(store.transact([_task("lrs_cancelled")]))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    first.cancel()
    await asyncio.sleep(0)
    second = asyncio.create_task(store.load_task("lrs_cancelled"))
    await asyncio.sleep(0.01)

    assert not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second is not None
    await store.close()


@pytest.mark.asyncio
async def test_cancelled_call_surfaces_worker_storage_error(tmp_path) -> None:
    """若取消掩盖 worker 的 SQLite 错误，调用方会误把持久化失败当作普通取消。"""

    entered = threading.Event()
    release = threading.Event()

    class FailingStore(SQLiteStateStore):
        def _insert_task(self, connection, values) -> None:  # type: ignore[no-untyped-def, override]
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("测试未及时释放失败命令")
            raise sqlite3.IntegrityError("字面存储失败")

    store = FailingStore(tmp_path / "state.sqlite3")
    await store.open()
    transaction = asyncio.create_task(store.transact([_task("lrs_cancel_error")]))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    transaction.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(sqlite3.IntegrityError, match="字面存储失败"):
        await transaction
    await store.close()


@pytest.mark.asyncio
async def test_migration_is_recorded_once_and_reopen_is_idempotent(tmp_path) -> None:
    """若 migration 重开时重复执行，已有权威数据会阻止插件启动。"""

    path = tmp_path / "state.sqlite3"
    first = SQLiteStateStore(path)
    await first.open()
    await first.transact([_task("lrs_reopen")])
    await first.close()

    second = SQLiteStateStore(path)
    await second.open()
    loaded = await second.load_task("lrs_reopen")
    await second.close()

    with sqlite3.connect(path) as connection:
        migrations = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert loaded is not None
    assert migrations == [(1, "authoritative_state"), (2, "vector_generations")]


@pytest.mark.asyncio
async def test_failing_migration_rolls_back_its_schema_and_record(tmp_path, monkeypatch) -> None:
    """若 migration 中途失败仍保留 DDL 或版本记录，重启将无法可靠修复。"""

    path = tmp_path / "state.sqlite3"
    failing = Migration(
        version=3,
        name="deliberate_failure",
        statements=(
            "CREATE TABLE migration_atomicity_probe (probe_id TEXT PRIMARY KEY)",
            "CREATE TABLE tasks (duplicate_name TEXT)",
        ),
    )
    monkeypatch.setattr(sqlite_module, "MIGRATIONS", (*MIGRATIONS, failing))
    store = SQLiteStateStore(path)

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        await store.open()

    with sqlite3.connect(path) as connection:
        probe = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'migration_atomicity_probe'"
        ).fetchone()
        versions = connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    assert probe is None
    assert versions == [(1,), (2,)]


@pytest.mark.asyncio
async def test_store_enforces_foreign_keys_and_required_pragmas(tmp_path) -> None:
    """若连接未启用 FK/WAL/FULL，孤儿状态或非持久提交可能成为权威状态。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    settings = await store.pragma_settings()

    assert settings == {"foreign_keys": 1, "journal_mode": "wal", "synchronous": 2}
    with pytest.raises(sqlite3.IntegrityError):
        await store.transact([_round("rnd_orphan", "lrs_missing", TaskStatus.RUNNING)])
    await store.close()


@pytest.mark.asyncio
async def test_load_task_preserves_formalized_task_and_selected_round(tmp_path) -> None:
    """若重开时正式任务或 current_round_number 关联错误，新 round 会丢失 User 1。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            _task(
                "lrs_task",
                formalized_text="正式任务原文\n",
                formalized_sha256="sha256_literal",
                current_round_number=2,
            ),
            _round("rnd_old", "lrs_task", TaskStatus.COMPLETED, round_number=1),
            _round("rnd_current", "lrs_task", TaskStatus.PAUSED, round_number=2),
        ]
    )

    loaded = await store.load_task("lrs_task")

    assert loaded is not None
    assert loaded.formalized_task is not None
    assert loaded.formalized_task.text == "正式任务原文\n"
    assert loaded.formalized_task.sha256 == "sha256_literal"
    assert loaded.current_round is not None
    assert loaded.current_round.round_id == "rnd_current"
    assert loaded.current_round.status is TaskStatus.PAUSED
    await store.close()


@pytest.mark.asyncio
async def test_list_active_rounds_includes_exactly_the_six_nonterminal_statuses(tmp_path) -> None:
    """若 terminal round 被恢复或 FINALIZING 被漏掉，crash recovery 会错误续跑。"""

    active = (
        TaskStatus.FORMALIZING,
        TaskStatus.RUNNING,
        TaskStatus.REPORTING,
        TaskStatus.PAUSING,
        TaskStatus.PAUSED,
        TaskStatus.FINALIZING,
    )
    terminal = tuple(status for status in TaskStatus if status not in active)
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    commands: list[StoreCommand] = []
    for index, status in enumerate((*active, *terminal), start=1):
        task_id = f"lrs_{index}"
        commands.extend([_task(task_id, float(index)), _round(f"rnd_{index}", task_id, status)])
    await store.transact(commands)

    rounds = await store.list_active_rounds()

    assert {round_.status for round_ in rounds} == set(active)
    assert len(rounds) == 6
    await store.close()


@pytest.mark.asyncio
async def test_summary_layer_reads_only_durable_summary_inputs(tmp_path) -> None:
    """若 restart summary layer 漏读 ContextSupplied 或读取运行期原文，续研输入会越界。"""

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            _task(
                "lrs_summary",
                formalized_text="研究目标",
                formalized_sha256="formalized_hash",
            ),
            _round("rnd_summary", "lrs_summary", TaskStatus.PAUSED),
            StoreCommand(
                "insert_lifecycle_event",
                {
                    "event_id": "evt_context",
                    "task_id": "lrs_summary",
                    "round_id": "rnd_summary",
                    "event_type": "ContextSupplied",
                    "metadata_json": json.dumps({"context": "用户新增材料"}, ensure_ascii=False),
                    "created_at": 2.0,
                },
            ),
            StoreCommand(
                "insert_summary",
                {
                    "summary_id": "sum_1",
                    "task_id": "lrs_summary",
                    "round_id": "rnd_summary",
                    "kind": "CHECKPOINT",
                    "text": "检查点总结",
                    "status": "COMPLETED",
                    "created_at": 3.0,
                },
            ),
            StoreCommand(
                "insert_report",
                {
                    "report_id": "rpt_1",
                    "task_id": "lrs_summary",
                    "round_id": "rnd_summary",
                    "epoch": 1,
                    "kind": "INTERMEDIATE",
                    "text": "中间报告",
                    "status": "COMPLETED",
                    "running_branch_count": 1,
                    "stats_json": "{}",
                    "created_at": 4.0,
                },
            ),
            StoreCommand(
                "insert_feedback_event",
                {
                    "feedback_id": "fb_1",
                    "task_id": "lrs_summary",
                    "round_id": "rnd_summary",
                    "disposition": "ACCEPTED",
                    "payload_json": json.dumps({"feedback": "补充对照组"}, ensure_ascii=False),
                    "created_at": 5.0,
                },
            ),
        ]
    )

    layer = await store.load_summary_layer("lrs_summary")

    assert layer is not None
    assert layer.formalized_task is not None
    assert layer.formalized_task.text == "研究目标"
    assert [item["text"] for item in layer.summaries] == ["检查点总结"]
    assert [item["text"] for item in layer.reports] == ["中间报告"]
    assert [item["payload"]["feedback"] for item in layer.feedback] == ["补充对照组"]
    assert layer.supplied_context == ("用户新增材料",)
    await store.close()


@pytest.mark.asyncio
async def test_crash_interruption_updates_only_active_rounds_and_preserves_usage(tmp_path) -> None:
    """若 crash recovery 改写 terminal round 或 reservation，对账与历史状态会被破坏。"""

    path = tmp_path / "state.sqlite3"
    store = SQLiteStateStore(path)
    await store.open()
    await store.transact(
        [
            _task("lrs_active"),
            _round("rnd_active", "lrs_active", TaskStatus.FINALIZING),
            _task("lrs_terminal"),
            _round("rnd_terminal", "lrs_terminal", TaskStatus.COMPLETED),
            StoreCommand(
                "insert_llm_usage",
                {
                    "usage_id": "usage_1",
                    "task_id": "lrs_active",
                    "round_id": "rnd_active",
                    "call_id": "call_1",
                    "role": "worker",
                    "selector": "default",
                    "price_source": "estimate",
                    "price_fingerprint": "prices_1",
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 10,
                    "estimated_charge": 1.25,
                    "reconciliation_status": "estimated_unreconciled",
                    "duration_ms": 0,
                    "created_at": 2.0,
                },
            ),
        ]
    )

    interrupted_count = await store.mark_active_rounds_interrupted(99.0)
    active = await store.load_task("lrs_active")
    terminal = await store.load_task("lrs_terminal")
    await store.close()

    with sqlite3.connect(path) as connection:
        usage = connection.execute(
            "SELECT reconciliation_status, estimated_charge FROM llm_usage WHERE usage_id = ?",
            ("usage_1",),
        ).fetchone()
    assert interrupted_count == 1
    assert active is not None and active.current_round is not None
    assert active.current_round.status is TaskStatus.INTERRUPTED
    assert active.current_round.ended_at == 99.0
    assert terminal is not None and terminal.current_round is not None
    assert terminal.current_round.status is TaskStatus.COMPLETED
    assert terminal.current_round.ended_at is None
    assert usage == ("estimated_unreconciled", 1.25)
