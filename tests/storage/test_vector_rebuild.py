"""LanceDB generation 重建：原子切换、失败候选保留、强制重建。"""

from __future__ import annotations

import time

import pytest

from lunagentic_research_swarm.errors import EMBEDDING_GENERATION_MISMATCH
from lunagentic_research_swarm.storage.vectors import EmbeddingGenerationMismatch, _insert_generation, _table_name_for

from test_vectors import VectorHarness


@pytest.fixture
async def vector_harness(tmp_path):
    harness = await VectorHarness.create(tmp_path)
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_failed_candidate_preserves_old_active(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()
    assert old.active_generation is not None
    old_generation = old.active_generation
    old_fingerprint = old.model_fingerprint

    await vector_harness.seed_formalized("extra-a", "额外正式任务甲")
    await vector_harness.seed_formalized("extra-b", "额外正式任务乙")
    # 主动 generation 仍是 3 维；force rebuild 时一批内维度不一致 → 候选失败
    vector_harness.embedder.return_vectors([[1.0, 2.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(EmbeddingGenerationMismatch):
        await vector_harness.rebuild(force=True)

    status = await vector_harness.status()
    assert status.active_generation == old_generation
    assert status.model_fingerprint == old_fingerprint
    assert not status.candidate_active
    assert not status.rebuilding
    assert status.failed_candidate is not None


@pytest.mark.asyncio
async def test_atomic_switch_activates_candidate_and_retires_old(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()
    old_generation = old.active_generation

    await vector_harness.seed_formalized("next", "下一正式任务")
    # force rebuild：两源同维，应原子切换
    vector_harness.embedder.return_vectors([[9.0, 8.0, 7.0], [1.0, 1.0, 1.0]])
    result = await vector_harness.rebuild(force=True)
    assert result.success

    status = await vector_harness.status()
    assert status.active_generation != old_generation
    assert status.active_generation is not None
    assert not status.rebuilding
    assert not status.candidate_active
    assert old_generation in status.retired_generations


@pytest.mark.asyncio
async def test_rebuild_force_false_returns_already_current_when_fingerprint_matches(
    vector_harness,
) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.rebuild(force=False)
    assert result.success
    assert result.code == "already_current"
    status = await vector_harness.status()
    assert status.active_generation == 1


@pytest.mark.asyncio
async def test_rebuild_force_false_clears_stranded_building_candidate(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()

    def _strand(connection) -> None:
        _insert_generation(
            connection,
            generation=99,
            selector=str(old.selector),
            actual_model_name=None,
            model_fingerprint="pending",
            dimension=None,
            table_name=_table_name_for(99),
            schema_version=1,
            status="building",
            created_at=time.time(),
        )

    await vector_harness.store.run_locked(_strand)
    vector_harness.index._rebuilding = True
    stranded = await vector_harness.status()
    assert stranded.candidate_active
    assert stranded.rebuilding

    result = await vector_harness.rebuild(force=False)
    assert result.success
    assert result.code == "already_current"
    status = await vector_harness.status()
    assert not status.candidate_active
    assert not status.rebuilding
    assert status.active_generation == old.active_generation
    search = await vector_harness.search("正式任务种子")
    assert search.success


@pytest.mark.asyncio
async def test_rebuild_force_creates_new_generation_even_when_current(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.rebuild(force=True)
    assert result.success
    status = await vector_harness.status()
    assert status.active_generation != old.active_generation


@pytest.mark.asyncio
async def test_mismatch_job_marked_and_authoritative_sqlite_untouched(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    vector_harness.embedder.return_vectors([[1.0, 2.0]])
    vector_harness.embedder.return_vectors([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    result = await vector_harness.index_new_source("summary-keep-sqlite")
    assert not result.success
    assert result.error.code == EMBEDDING_GENERATION_MISMATCH

    layer = await vector_harness.store.load_summary_layer("task_for_summary-keep-sqlite")
    assert layer is not None
    assert any(item["summary_id"] == "summary-keep-sqlite" for item in layer.summaries)

    jobs = await vector_harness.index.list_jobs(status="failed")
    assert any(job["error_code"] == EMBEDDING_GENERATION_MISMATCH for job in jobs)


@pytest.mark.asyncio
async def test_successful_index_search_returns_hits(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 0.0, 0.0]])
    vector_harness.embedder.return_vectors([[0.9, 0.1, 0.0]])
    search = await vector_harness.search("正式任务种子", limit=3)
    assert search.success
    assert search.data["hits"]
    hit = search.data["hits"][0]
    assert hit["source_kind"] == "formalized_task"
    assert hit["source_id"] == "task-seed"
    assert "vector" not in hit
    assert isinstance(hit.get("text"), str)


@pytest.mark.asyncio
async def test_fingerprint_includes_selector_model_dimension_schema(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    status = await vector_harness.status()
    assert status.model_fingerprint
    assert status.selector == "task:embedding"
    assert status.actual_model_name == "fake-embed-v1"
    assert status.dimension == 3
    assert status.schema_version == 1


@pytest.mark.asyncio
async def test_ensure_ready_rebuilds_stranded_candidate(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])

    def _strand(connection) -> None:
        _insert_generation(
            connection,
            generation=77,
            selector="task:embedding",
            actual_model_name=None,
            model_fingerprint="pending",
            dimension=None,
            table_name=_table_name_for(77),
            schema_version=1,
            status="building",
            created_at=time.time(),
        )

    await vector_harness.store.run_locked(_strand)
    vector_harness.embedder.return_vectors([[1.0, 2.0, 3.0]])
    result = await vector_harness.index.ensure_ready()
    assert result.success
    status = await vector_harness.status()
    assert not status.candidate_active
    assert not status.rebuilding
