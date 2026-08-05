"""LanceDB generation 索引：维度/selector/model/schema mismatch 与检索不可用。"""

from __future__ import annotations

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
    VectorIndex,
    VECTOR_SCHEMA_VERSION,
)


@dataclass
class FakeEmbedder:
    """可控的 Host llm.embed 替身。"""

    model_name: str = "fake-embed-v1"
    queued: list[list[list[float]]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def return_vectors(self, vectors: Sequence[Sequence[float]]) -> None:
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
        if len(vectors) != len(batch):
            # 测试可故意投喂错误长度；实现侧必须校验
            pass
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
    async def create(cls, tmp_path: Path, *, selector: str = "task:embedding") -> VectorHarness:
        store = SQLiteStateStore(tmp_path / "state.sqlite3")
        await store.open()
        embedder = FakeEmbedder()
        embedding = EmbeddingSection(
            selector=selector,
            batch_size=8,
            max_concurrent=2,
            auto_rebuild=True,
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

    async def seed_summary(self, source_id: str, *, kind: str = "CHECKPOINT", text: str = "摘要") -> None:
        self._source_seq += 1
        task_id = f"task_for_{source_id}"
        round_id = f"rnd_for_{source_id}"
        now = float(self._source_seq)
        await self.store.transact(
            [
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
                        "status": "READY",
                        "error_code": None,
                        "created_at": now,
                    },
                ),
            ]
        )

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
    vector_harness.embedder.return_vectors([[1.0, 2.0]])
    result = await vector_harness.index_new_source("summary-2")
    assert not result.success
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH
    assert (await vector_harness.status()).rebuilding
    assert old.dimension == 3


@pytest.mark.asyncio
async def test_batch_dimension_inconsistency_aborts_generation(vector_harness) -> None:
    await vector_harness.seed_formalized("t1", "正式任务一")
    await vector_harness.seed_formalized("t2", "正式任务二")
    vector_harness.embedder.return_vectors([[1.0, 2.0], [1.0, 2.0, 3.0]])
    with pytest.raises(EmbeddingGenerationMismatch):
        await vector_harness.rebuild(force=True)
    status = await vector_harness.status()
    assert not status.candidate_active
    assert status.active_generation is None or status.idle


@pytest.mark.asyncio
async def test_selector_change_triggers_mismatch_rebuild(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.index.set_selector("task:other-embedding")
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.index_new_source("summary-selector")
    assert not result.success
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH
    assert (await vector_harness.status()).rebuilding


@pytest.mark.asyncio
async def test_actual_model_change_triggers_mismatch(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.set_model_name("fake-embed-v2")
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.index_new_source("summary-model")
    assert not result.success
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH


@pytest.mark.asyncio
async def test_schema_version_bump_forces_rebuild(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.index.override_schema_version(VECTOR_SCHEMA_VERSION + 1)
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.index_new_source("summary-schema")
    assert not result.success
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH


@pytest.mark.asyncio
async def test_search_while_rebuilding_returns_explicit_unavailable(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.return_vectors([[4.0, 5.0]])
    await vector_harness.index_new_source("summary-rebuild-search")
    status = await vector_harness.status()
    assert status.rebuilding
    search = await vector_harness.search("查询")
    assert not search.success
    assert search.error.code == VECTOR_INDEX_REBUILDING


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
    assert status.idle or status.uninitialized
    assert status.active_generation is None
    assert status.dimension is None


@pytest.mark.asyncio
async def test_vectors_are_finite_and_nonempty_required(vector_harness) -> None:
    await vector_harness.seed_formalized("bad", "正式任务")
    vector_harness.embedder.return_vectors([[math.nan, 1.0, 2.0]])
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
