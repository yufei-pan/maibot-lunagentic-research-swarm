"""LanceDB generation 索引：维度/selector/model/schema mismatch 与检索不可用。"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.config import EmbeddingSection
from lunagentic_research_swarm.errors import EMBEDDING_GENERATION_MISMATCH, VECTOR_INDEX_REBUILDING
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from lunagentic_research_swarm.storage.vectors import (
    EmbeddingGenerationMismatch,
    INDEXABLE_CONTENT_STATUSES,
    VectorIndex,
    VECTOR_SCHEMA_VERSION,
    _load_indexable_sources,
)


@dataclass
class FakeEmbedder:
    """可控的 Host llm.embed 替身。"""

    model_name: str = "fake-embed-v1"
    queued: list[list[list[float]]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def return_vectors(self, vectors: Sequence[Sequence[float]] | Sequence[Sequence[Any]]) -> None:
        self.queued.append([list(vector) for vector in vectors])

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    async def embed(
        self,
        text: str | None = None,
        *,
        texts: list[str] | None = None,
        task_name: str = "embedding",
        model: str = "",
        model_name: str = "",
        max_concurrent: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        batch = list(texts) if texts is not None else ([text] if text is not None else [])
        self.calls.append(
            {
                "texts": batch,
                "task_name": task_name,
                "model": model,
                "model_name": model_name,
                "max_concurrent": max_concurrent,
                **kwargs,
            }
        )
        if self.queued:
            vectors = self.queued.pop(0)
        else:
            vectors = [[0.1, 0.2, 0.3] for _ in batch]
        return {
            "success": True,
            "results": [
                {"embedding": list(vector), "model_name": self.model_name} for vector in vectors
            ],
        }


@dataclass
class VectorHarness:
    tmp_path: Path
    store: SQLiteStateStore
    embedder: FakeEmbedder
    index: VectorIndex
    _source_seq: int = 0

    @classmethod
    async def create(
        cls,
        tmp_path: Path,
        *,
        selector: str = "task:embedding",
        auto_rebuild: bool = True,
    ) -> VectorHarness:
        store = SQLiteStateStore(tmp_path / "state.sqlite3")
        await store.open()
        embedder = FakeEmbedder()
        embedding = EmbeddingSection(
            selector=selector,
            batch_size=8,
            max_concurrent=2,
            auto_rebuild=auto_rebuild,
            retired_generation_retention_seconds=86400,
        )
        index = VectorIndex(
            store,
            embedder,
            embedding,
            tmp_path / "vectors" / "lancedb",
        )
        await index.start()
        return cls(tmp_path=tmp_path, store=store, embedder=embedder, index=index)

    async def close(self) -> None:
        await self.index.close()
        await self.store.close()

    async def status(self) -> Any:
        return await self.index.status()

    async def rebuild(self, *, force: bool = False) -> Any:
        return await self.index.rebuild(force=force)

    async def search(self, query: str, *, limit: int = 5) -> Any:
        return await self.index.search(query, limit=limit)

    async def seed_formalized(self, source_id: str, text: str) -> None:
        now = 1.0
        await self.store.transact(
            [
                StoreCommand(
                    "insert_task",
                    {
                        "task_id": source_id,
                        "stream_id": f"stream_{source_id}",
                        "formalized_text": text,
                        "formalized_sha256": f"sha_{source_id}",
                        "created_at": now,
                    },
                )
            ]
        )

    async def _ensure_task_round(self, task_id: str, round_id: str, now: float) -> list[StoreCommand]:
        return [
            StoreCommand(
                "insert_task",
                {
                    "task_id": task_id,
                    "stream_id": f"stream_{task_id}",
                    "formalized_text": f"任务 {task_id}",
                    "formalized_sha256": f"sha_{task_id}",
                    "created_at": now,
                },
            ),
            StoreCommand(
                "insert_round",
                {
                    "round_id": round_id,
                    "task_id": task_id,
                    "round_number": 1,
                    "status": "COMPLETED",
                    "generation": 1,
                    "time_budget_seconds": 120,
                    "credit_pool": 10.0,
                    "started_at": now,
                },
            ),
        ]

    async def seed_summary(
        self,
        source_id: str,
        *,
        kind: str = "CHECKPOINT",
        text: str = "摘要",
        status: str = "SUCCEEDED",
    ) -> None:
        """status 默认与 runtime/epochs.py 写入值对齐（SUCCEEDED/FAILED/DEGRADED）。"""

        self._source_seq += 1
        task_id = f"task_for_{source_id}"
        round_id = f"rnd_for_{source_id}"
        now = float(self._source_seq)
        commands = await self._ensure_task_round(task_id, round_id, now)
        commands.append(
            StoreCommand(
                "insert_summary",
                {
                    "summary_id": source_id,
                    "task_id": task_id,
                    "round_id": round_id,
                    "branch_id": None,
                    "kind": kind,
                    "report_epoch": None,
                    "text": text,
                    "status": status,
                    "error_code": None,
                    "created_at": now,
                },
            )
        )
        await self.store.transact(commands)

    async def seed_report(
        self,
        source_id: str,
        *,
        kind: str = "FINAL",
        text: str = "最终报告",
        status: str = "SUCCEEDED",
    ) -> None:
        self._source_seq += 1
        task_id = f"task_for_{source_id}"
        round_id = f"rnd_for_{source_id}"
        now = float(self._source_seq)
        commands = await self._ensure_task_round(task_id, round_id, now)
        commands.append(
            StoreCommand(
                "insert_report",
                {
                    "report_id": source_id,
                    "task_id": task_id,
                    "round_id": round_id,
                    "epoch": 1,
                    "kind": kind,
                    "text": text,
                    "status": status,
                    "running_branch_count": 0,
                    "stats_json": "{}",
                    "created_at": now,
                },
            )
        )
        await self.store.transact(commands)

    async def seed_feedback_lesson(
        self,
        source_id: str,
        *,
        lesson: str = "可复用教训",
        disposition: str = "accept",
    ) -> None:
        self._source_seq += 1
        task_id = f"task_for_{source_id}"
        round_id = f"rnd_for_{source_id}"
        now = float(self._source_seq)
        commands = await self._ensure_task_round(task_id, round_id, now)
        commands.append(
            StoreCommand(
                "insert_feedback_event",
                {
                    "feedback_id": source_id,
                    "task_id": task_id,
                    "round_id": round_id,
                    "disposition": disposition,
                    "payload_json": json.dumps({"lesson": lesson, "raw_note": "不应入库"}, ensure_ascii=False),
                    "created_at": now,
                },
            )
        )
        await self.store.transact(commands)

    async def build_with_vectors(self, vectors: Sequence[Sequence[float]]) -> None:
        await self.seed_formalized("task-seed", "正式任务种子")
        self.embedder.return_vectors(list(vectors))
        result = await self.index.rebuild(force=True)
        assert result.success, getattr(result, "error", result)

    async def index_new_source(self, source_id: str) -> Any:
        await self.seed_summary(source_id, text=f"新摘要 {source_id}")
        return await self.index.enqueue(
            source_kind="checkpoint_summary",
            source_id=source_id,
        )


@pytest.fixture
async def vector_harness(tmp_path):
    harness = await VectorHarness.create(tmp_path)
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_dimension_change_starts_new_generation_without_padding(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()
    # enqueue 探测 2 维 → mismatch；立即返回 rebuilding，后台全量重嵌
    vector_harness.embedder.return_vectors([[1.0, 2.0]])
    vector_harness.embedder.return_vectors([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = await vector_harness.index_new_source("summary-2")
    assert not result.success
    assert result.error.code == VECTOR_INDEX_REBUILDING
    await vector_harness.index.wait_rebuild()
    status = await vector_harness.status()
    assert not status.rebuilding
    assert not status.candidate_active
    assert status.dimension == 2
    assert old.dimension == 3
    assert old.active_generation in status.retired_generations
    jobs = await vector_harness.index.list_jobs(status="done")
    assert any(job["source_id"] == "summary-2" for job in jobs)


@pytest.mark.asyncio
async def test_batch_dimension_inconsistency_aborts_generation(vector_harness) -> None:
    await vector_harness.seed_formalized("t1", "正式任务一")
    await vector_harness.seed_formalized("t2", "正式任务二")
    vector_harness.embedder.return_vectors([[1.0, 2.0], [1.0, 2.0, 3.0]])
    with pytest.raises(EmbeddingGenerationMismatch):
        await vector_harness.rebuild(force=True)
    status = await vector_harness.status()
    assert not status.candidate_active
    assert status.active_generation is None
    assert status.idle


@pytest.mark.asyncio
async def test_selector_change_triggers_mismatch_rebuild(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.index.set_selector("task:other-embedding")
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = await vector_harness.index_new_source("summary-selector")
    assert not result.success
    assert result.error.code == VECTOR_INDEX_REBUILDING
    await vector_harness.index.wait_rebuild()
    status = await vector_harness.status()
    assert not status.rebuilding
    assert status.selector == "task:other-embedding"


@pytest.mark.asyncio
async def test_actual_model_change_triggers_mismatch(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.set_model_name("fake-embed-v2")
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = await vector_harness.index_new_source("summary-model")
    assert not result.success
    assert result.error.code == VECTOR_INDEX_REBUILDING
    await vector_harness.index.wait_rebuild()
    status = await vector_harness.status()
    assert status.actual_model_name == "fake-embed-v2"
    assert not status.rebuilding


@pytest.mark.asyncio
async def test_schema_version_bump_forces_rebuild(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.index.override_schema_version(VECTOR_SCHEMA_VERSION + 1)
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = await vector_harness.index_new_source("summary-schema")
    assert not result.success
    assert result.error.code == VECTOR_INDEX_REBUILDING
    await vector_harness.index.wait_rebuild()
    status = await vector_harness.status()
    assert status.schema_version == VECTOR_SCHEMA_VERSION + 1
    assert not status.rebuilding


@pytest.mark.asyncio
async def test_concurrent_enqueue_fails_fast_while_auto_rebuild_in_progress(vector_harness) -> None:
    """第二路 enqueue 在全量 rebuild 进行中必须快速返回，不得排队等待整库 re-embed。"""

    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])

    entered = asyncio.Event()
    release = asyncio.Event()
    real_rebuild = vector_harness.index._run_full_rebuild

    async def blocked_rebuild(*, selector):  # type: ignore[no-untyped-def]
        entered.set()
        await release.wait()
        return await real_rebuild(selector=selector)

    vector_harness.index._run_full_rebuild = blocked_rebuild  # type: ignore[method-assign]
    vector_harness.embedder.return_vectors([[1.0, 2.0]])  # mismatch probe

    first = await vector_harness.index_new_source("summary-a")
    assert not first.success
    assert first.error.code == VECTOR_INDEX_REBUILDING
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    assert vector_harness.index._rebuilding

    await vector_harness.seed_summary("summary-b", text="并发摘要")
    second = await asyncio.wait_for(
        vector_harness.index.enqueue(source_kind="checkpoint_summary", source_id="summary-b"),
        timeout=1.0,
    )
    assert not second.success
    assert second.error.code == VECTOR_INDEX_REBUILDING
    assert not release.is_set(), "second enqueue must not wait for full rebuild"

    # 放行后台重建（corpus: task-seed + task_for_summary-a + summary-a + task_for_summary-b + summary-b）
    vector_harness.embedder.return_vectors(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    )
    release.set()
    await vector_harness.index.wait_rebuild(timeout=5.0)
    status = await vector_harness.status()
    assert not status.rebuilding
    assert status.dimension == 2
    done = await vector_harness.index.list_jobs(status="done")
    assert any(job["source_id"] == "summary-a" for job in done)


@pytest.mark.asyncio
async def test_search_while_rebuilding_returns_explicit_unavailable(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.index._rebuilding = True
    search = await vector_harness.search("查询")
    assert not search.success
    assert search.error.code == VECTOR_INDEX_REBUILDING


@pytest.mark.asyncio
async def test_search_rejects_same_dimension_model_swap(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.set_model_name("fake-embed-v2")
    vector_harness.embedder.return_vectors([[0.9, 0.1, 0.0]])
    search = await vector_harness.search("正式任务种子")
    assert not search.success
    assert search.error.code == EMBEDDING_GENERATION_MISMATCH


@pytest.mark.asyncio
async def test_model_selector_is_explicitly_unsupported(tmp_path) -> None:
    harness = await VectorHarness.create(tmp_path, selector="model:text-embedding-3-small")
    try:
        await harness.seed_formalized("t-model", "正式任务")
        harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
        result = await harness.rebuild(force=True)
        assert not result.success
        assert result.error.code == "physical_embedding_selector_unsupported"
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_empty_index_is_idle_until_first_source(vector_harness) -> None:
    status = await vector_harness.status()
    assert status.idle
    assert status.uninitialized
    assert status.active_generation is None
    assert status.dimension is None
    assert status.failed_candidate is None


@pytest.mark.asyncio
async def test_empty_force_rebuild_does_not_create_failed_candidate(vector_harness) -> None:
    result = await vector_harness.rebuild(force=True)
    assert result.success
    assert result.code == "empty"
    status = await vector_harness.status()
    assert status.idle
    assert status.failed_candidate is None


@pytest.mark.asyncio
async def test_vectors_are_finite_and_nonempty_required(vector_harness) -> None:
    await vector_harness.seed_formalized("bad", "正式任务")
    vector_harness.embedder.return_vectors([[math.nan, 1.0, 2.0]])
    with pytest.raises(EmbeddingGenerationMismatch):
        await vector_harness.rebuild(force=True)


@pytest.mark.asyncio
async def test_non_numeric_embedding_fails_job_not_pending(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    await vector_harness.seed_summary("sum-bad", text="坏向量源")
    vector_harness.embedder.return_vectors([["not-a-number", 1.0, 2.0]])
    result = await vector_harness.index.enqueue(
        source_kind="checkpoint_summary",
        source_id="sum-bad",
    )
    assert not result.success
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH
    jobs = await vector_harness.index.list_jobs(status="PENDING")
    assert jobs == []
    failed = await vector_harness.index.list_jobs(status="failed")
    assert any(job["source_id"] == "sum-bad" for job in failed)


@pytest.mark.asyncio
async def test_bool_embedding_elements_rejected(vector_harness) -> None:
    await vector_harness.seed_formalized("bool-bad", "正式任务")
    vector_harness.embedder.return_vectors([[True, False, True]])
    with pytest.raises(EmbeddingGenerationMismatch):
        await vector_harness.rebuild(force=True)


@pytest.mark.asyncio
async def test_task_selector_uses_task_name_not_model_pin(vector_harness) -> None:
    await vector_harness.seed_formalized("t-sel", "正式任务")
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.rebuild(force=True)
    assert result.success
    assert vector_harness.embedder.calls
    call = vector_harness.embedder.calls[0]
    assert call["task_name"] == "embedding"
    assert call["model"] == ""
    assert call["model_name"] == ""


@pytest.mark.asyncio
async def test_runtime_statuses_make_summaries_and_reports_indexable(vector_harness) -> None:
    assert INDEXABLE_CONTENT_STATUSES == frozenset({"SUCCEEDED"})
    assert "READY" not in INDEXABLE_CONTENT_STATUSES
    await vector_harness.seed_summary("s-ok", text="检查点摘要", status="SUCCEEDED")
    await vector_harness.seed_summary("s-branch", kind="BRANCH_FINAL", text="分支终态", status="SUCCEEDED")
    await vector_harness.seed_summary("s-fail", kind="CHECKPOINT", text="失败摘要不应入库", status="FAILED")
    await vector_harness.seed_report("r-final", kind="FINAL", text="终报", status="SUCCEEDED")
    await vector_harness.seed_report(
        "r-deg",
        kind="INTERMEDIATE",
        text="当前尚无可用分支摘要；调查仍在进行。",
        status="DEGRADED",
    )
    await vector_harness.seed_report(
        "r-fail",
        kind="FINAL",
        text="报告总结器不可用；以下为已提交 coverage。",
        status="FAILED",
    )
    await vector_harness.seed_feedback_lesson("fb-1", lesson="只索引 lesson")

    sources = await vector_harness.store.run_locked(_load_indexable_sources)
    kinds = {(s.source_kind, s.source_id) for s in sources}
    assert ("checkpoint_summary", "s-ok") in kinds
    assert ("branch_final_summary", "s-branch") in kinds
    assert ("checkpoint_summary", "s-fail") not in kinds
    assert ("final_report", "r-final") in kinds
    assert ("intermediate_report", "r-deg") not in kinds
    assert ("final_report", "r-fail") not in kinds
    assert ("feedback_lesson", "fb-1") in kinds
    lesson = next(s for s in sources if s.source_id == "fb-1")
    assert lesson.text == "只索引 lesson"
    assert "不应入库" not in lesson.text
    apology_texts = {s.text for s in sources}
    assert "当前尚无可用分支摘要；调查仍在进行。" not in apology_texts
    assert "报告总结器不可用；以下为已提交 coverage。" not in apology_texts
    assert "失败摘要不应入库" not in apology_texts


@pytest.mark.asyncio
async def test_transient_index_unavailable_does_not_auto_rebuild(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()
    await vector_harness.seed_summary("sum-io", text="瞬态 IO 源")
    calls_before = len(vector_harness.embedder.calls)

    original_detect = vector_harness.index._detect_mismatch

    def _io_fail(*args, **kwargs):
        from lunagentic_research_swarm.storage.vectors import VectorIndexUnavailable

        return VectorIndexUnavailable("simulated open_table IO", {"table_name": old.table_name})

    vector_harness.index._detect_mismatch = _io_fail  # type: ignore[method-assign]
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    try:
        result = await vector_harness.index.enqueue(
            source_kind="checkpoint_summary",
            source_id="sum-io",
        )
    finally:
        vector_harness.index._detect_mismatch = original_detect  # type: ignore[method-assign]

    assert not result.success
    assert result.error is not None
    assert result.error.code == "vector_index_unavailable"
    status = await vector_harness.status()
    assert status.active_generation == old.active_generation
    assert status.dimension == old.dimension
    # 仅探测 embedding，无全量 rebuild
    assert len(vector_harness.embedder.calls) == calls_before + 1


@pytest.mark.asyncio
async def test_start_reconciles_stranded_building(tmp_path) -> None:
    import time

    from lunagentic_research_swarm.storage.vectors import _insert_generation, _table_name_for

    harness = await VectorHarness.create(tmp_path)
    try:
        await harness.build_with_vectors([[1.0, 2.0, 3.0]])
        old = await harness.status()

        def _strand(connection) -> None:
            _insert_generation(
                connection,
                generation=55,
                selector=str(old.selector),
                actual_model_name=None,
                model_fingerprint="pending",
                dimension=None,
                table_name=_table_name_for(55),
                schema_version=1,
                status="building",
                created_at=time.time(),
            )

        await harness.store.run_locked(_strand)
        lance_dir = harness.tmp_path / "vectors" / "lancedb"
        await harness.index.close()

        restarted = VectorIndex(
            harness.store,
            harness.embedder,
            EmbeddingSection(
                selector="task:embedding",
                batch_size=8,
                max_concurrent=2,
                auto_rebuild=True,
                retired_generation_retention_seconds=86400,
            ),
            lance_dir,
        )
        await restarted.start()
        try:
            status = await restarted.status()
            assert not status.candidate_active
            assert not status.rebuilding
            assert status.active_generation == old.active_generation
            search = await restarted.search("正式任务种子")
            assert search.success
        finally:
            await restarted.close()
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_empty_index_enqueue_mismatch_returns_result_not_raise(vector_harness) -> None:
    await vector_harness.seed_formalized("first", "首条正式任务")
    vector_harness.embedder.return_vectors([[1.0, 2.0], [1.0, 2.0, 3.0]])
    result = await vector_harness.index.enqueue(source_kind="formalized_task", source_id="first")
    assert not result.success
    assert result.error is not None
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH
    jobs = await vector_harness.index.list_jobs(status="failed")
    assert any(job["source_id"] == "first" for job in jobs)


@pytest.mark.asyncio
async def test_non_whitelist_summary_kinds_never_indexable(vector_harness) -> None:
    await vector_harness.seed_summary("s-form", kind="FORMALIZATION", text="形式化摘要机密")
    await vector_harness.seed_summary("s-final", kind="TASK_FINAL", text="任务终态机密")
    sources = await vector_harness.store.run_locked(_load_indexable_sources)
    texts = {s.text for s in sources}
    assert "形式化摘要机密" not in texts
    assert "任务终态机密" not in texts
    assert all(s.source_kind != "formalized_task" or "机密" not in s.text for s in sources)


@pytest.mark.asyncio
async def test_reindex_same_source_upserts_not_duplicates(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    await vector_harness.seed_summary("dup-1", text="第一版摘要")
    vector_harness.embedder.return_vectors([[1.0, 0.0, 0.0]])
    first = await vector_harness.index.enqueue(source_kind="checkpoint_summary", source_id="dup-1")
    assert first.success
    vector_harness.embedder.return_vectors([[0.0, 1.0, 0.0]])
    second = await vector_harness.index.enqueue(source_kind="checkpoint_summary", source_id="dup-1")
    assert second.success

    status = await vector_harness.status()
    table_name = status.table_name
    assert table_name is not None

    def _count() -> int:
        assert vector_harness.index._db is not None
        table = vector_harness.index._db.open_table(table_name)
        return int(table.count_rows("source_id = 'dup-1'"))

    import asyncio

    assert await asyncio.to_thread(_count) == 1
