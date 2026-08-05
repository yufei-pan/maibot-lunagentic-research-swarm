"""可重建的 LanceDB generation 索引（SQLite 权威，向量为派生层）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunagentic_research_swarm.config import EmbeddingSection
from lunagentic_research_swarm.errors import (
    EMBEDDING_GENERATION_MISMATCH,
    VECTOR_INDEX_REBUILDING,
    VECTOR_INDEX_UNAVAILABLE,
    VECTOR_REBUILD_FAILED,
    LRSError,
)
from lunagentic_research_swarm.llm.gateway import ModelSelector
from lunagentic_research_swarm.models import ReportKind, SummaryKind

VECTOR_SCHEMA_VERSION = 1

# 仅索引有可用正文的内容态。runtime/epochs.py 对 summaries/reports：
# SUCCEEDED 才带真实文本；FAILED/DEGRADED 是道歉占位（text NULL 或固定话术），不当历史案例。
INDEXABLE_CONTENT_STATUSES = frozenset({"SUCCEEDED"})
INDEXABLE_SUMMARY_KINDS = frozenset({SummaryKind.CHECKPOINT.value, SummaryKind.BRANCH_FINAL.value})
INDEXABLE_REPORT_KINDS = frozenset({ReportKind.INTERMEDIATE.value, ReportKind.FINAL.value})

SOURCE_KIND_FORMALIZED = "formalized_task"
SOURCE_KIND_CHECKPOINT = "checkpoint_summary"
SOURCE_KIND_BRANCH_FINAL = "branch_final_summary"
SOURCE_KIND_INTERMEDIATE_REPORT = "intermediate_report"
SOURCE_KIND_FINAL_REPORT = "final_report"
SOURCE_KIND_FEEDBACK_LESSON = "feedback_lesson"

_SUMMARY_KIND_TO_SOURCE = {
    SummaryKind.CHECKPOINT.value: SOURCE_KIND_CHECKPOINT,
    SummaryKind.BRANCH_FINAL.value: SOURCE_KIND_BRANCH_FINAL,
}
_REPORT_KIND_TO_SOURCE = {
    ReportKind.INTERMEDIATE.value: SOURCE_KIND_INTERMEDIATE_REPORT,
    ReportKind.FINAL.value: SOURCE_KIND_FINAL_REPORT,
}

PHYSICAL_EMBEDDING_SELECTOR_UNSUPPORTED = "physical_embedding_selector_unsupported"


class EmbeddingGenerationMismatch(LRSError):
    """embedding selector/模型/维度/schema 与当前 generation 不一致。"""

    def __init__(self, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(EMBEDDING_GENERATION_MISMATCH, message, metadata)


class VectorRebuildFailed(LRSError):
    """重建过程中的磁盘 / SQLite / LanceDB 等基础设施失败（非 fingerprint mismatch）。"""

    def __init__(self, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(VECTOR_REBUILD_FAILED, message, metadata)


class VectorIndexUnavailable(LRSError):
    """向量索引暂时不可读（如 LanceDB IO），不等于模型 fingerprint 变更。"""

    def __init__(self, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(VECTOR_INDEX_UNAVAILABLE, message, metadata)


@dataclass(frozen=True, slots=True)
class VectorOpResult:
    success: bool
    error: LRSError | None = None
    code: str | None = None
    data: Mapping[str, Any] | None = None

    @classmethod
    def ok(cls, *, code: str | None = None, data: Mapping[str, Any] | None = None) -> VectorOpResult:
        return cls(success=True, code=code, data=data)

    @classmethod
    def fail(cls, error: LRSError) -> VectorOpResult:
        return cls(success=False, error=error, code=error.code)


@dataclass(frozen=True, slots=True)
class VectorIndexStatus:
    idle: bool
    uninitialized: bool
    rebuilding: bool
    candidate_active: bool
    active_generation: int | None
    failed_candidate: int | None
    retired_generations: tuple[int, ...]
    selector: str | None
    actual_model_name: str | None
    model_fingerprint: str | None
    dimension: int | None
    schema_version: int | None
    table_name: str | None


@dataclass(frozen=True, slots=True)
class IndexableSource:
    source_kind: str
    source_id: str
    text: str
    task_id: str | None
    feedback_disposition: str | None = None


def compute_model_fingerprint(
    selector: str,
    actual_model_name: str,
    dimension: int,
    schema_version: int,
) -> str:
    payload = f"{selector}\0{actual_model_name}\0{dimension}\0{schema_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _table_name_for(generation: int) -> str:
    return f"lrs_vectors_g{generation}"


def _is_finite_vector(vector: Sequence[float]) -> bool:
    if not vector:
        return False
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def _validate_embedding_results(
    payload: Mapping[str, Any],
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> tuple[list[list[float]], str]:
    if not payload.get("success", False):
        raise EmbeddingGenerationMismatch(
            "embedding 调用失败",
            {"host_error": payload.get("error")},
        )
    results = payload.get("results")
    if not isinstance(results, list):
        single = payload.get("embedding")
        model_name = str(payload.get("model_name") or "").strip()
        if single is None:
            raise EmbeddingGenerationMismatch("embedding 响应缺少 results/embedding")
        results = [{"embedding": single, "model_name": model_name}]

    if len(results) != expected_count:
        raise EmbeddingGenerationMismatch(
            "embedding 批量结果数量与输入不一致",
            {"expected": expected_count, "actual": len(results)},
        )

    vectors: list[list[float]] = []
    model_names: list[str] = []
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise EmbeddingGenerationMismatch(f"embedding 结果[{index}] 非法")
        raw_vector = item.get("embedding")
        model_name = str(item.get("model_name") or "").strip()
        if not model_name:
            raise EmbeddingGenerationMismatch(f"embedding 结果[{index}] 缺少 model_name")
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
            raise EmbeddingGenerationMismatch(f"embedding 结果[{index}] 向量非法")
        # 先校验元素类型再强制转换，避免 ValueError 泄漏 / bool 被当作 int
        if not _is_finite_vector(raw_vector):
            raise EmbeddingGenerationMismatch(
                f"embedding 结果[{index}] 向量必须为非空有限浮点",
                {"index": index},
            )
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingGenerationMismatch(
                f"embedding 结果[{index}] 向量元素无法转为浮点",
                {"index": index},
            ) from exc
        if expected_dimension is not None and len(vector) != expected_dimension:
            raise EmbeddingGenerationMismatch(
                "embedding 维度与期望不一致",
                {"expected": expected_dimension, "actual": len(vector), "index": index},
            )
        vectors.append(vector)
        model_names.append(model_name)

    if len({len(vector) for vector in vectors}) != 1:
        raise EmbeddingGenerationMismatch(
            "同一批次 embedding 维度不一致",
            {"dimensions": [len(vector) for vector in vectors]},
        )
    if len(set(model_names)) != 1:
        raise EmbeddingGenerationMismatch(
            "同一批次 actual model 不一致",
            {"models": model_names},
        )
    return vectors, model_names[0]


def _schema_vector_dimension(schema: Any) -> int | None:
    try:
        field = schema.field("vector")
    except Exception:
        return None
    vector_type = field.type
    list_size = getattr(vector_type, "list_size", None)
    if isinstance(list_size, int) and list_size > 0:
        return list_size
    return None


def _list_lancedb_tables(db: Any) -> list[str]:
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        tables = getattr(response, "tables", None)
        if tables is not None:
            return list(tables)
    if hasattr(db, "table_names"):
        return list(db.table_names())
    return []


class VectorIndex:
    """LanceDB 派生索引：按 generation 原子切换，失败保留旧 active。"""

    def __init__(
        self,
        store: Any,
        embedder: Any,
        config: EmbeddingSection,
        lance_dir: str | Path,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config.model_copy(deep=True) if hasattr(config, "model_copy") else config
        self._lance_dir = Path(lance_dir)
        self._schema_version = VECTOR_SCHEMA_VERSION
        self._db: Any | None = None
        self._started = False
        self._rebuilding = False
        self._rebuild_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def set_selector(self, selector: str) -> None:
        """测试/热更新：切换 embedding selector。"""

        if hasattr(self._config, "model_copy"):
            self._config = self._config.model_copy(update={"selector": selector})
        else:
            object.__setattr__(self._config, "selector", selector)

    def override_schema_version(self, version: int) -> None:
        """测试：模拟 schema 升级。"""

        self._schema_version = int(version)

    async def start(self) -> None:
        if self._started:
            return
        self._lance_dir.mkdir(parents=True, exist_ok=True)

        def _connect() -> Any:
            import lancedb

            return lancedb.connect(str(self._lance_dir))

        self._db = await asyncio.to_thread(_connect)
        self._started = True
        # 进程崩溃若停在 building，会让 search 永久返回 vector_index_rebuilding。
        # 启动时清掉上一代残留 candidate 及其孤儿 Lance table。
        await self._fail_stranded_building()
        await self._maybe_cleanup_retired()

    async def close(self) -> None:
        task = self._rebuild_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._rebuild_task = None
        self._db = None
        self._started = False
        self._rebuilding = False

    async def status(self) -> VectorIndexStatus:
        rows = await self._store.run_locked(_load_generation_rows)
        active = next((row for row in rows if row["status"] == "active"), None)
        building = next((row for row in rows if row["status"] == "building"), None)
        failed = next((row for row in reversed(rows) if row["status"] == "failed"), None)
        retired = tuple(int(row["generation"]) for row in rows if row["status"] == "retired")
        idle = active is None and building is None
        return VectorIndexStatus(
            idle=idle,
            uninitialized=idle,
            rebuilding=self._rebuilding or building is not None,
            candidate_active=building is not None,
            active_generation=int(active["generation"]) if active else None,
            failed_candidate=int(failed["generation"]) if failed else None,
            retired_generations=retired,
            selector=str(active["selector"]) if active else None,
            actual_model_name=active["actual_model_name"] if active else None,
            model_fingerprint=str(active["model_fingerprint"]) if active else None,
            dimension=int(active["dimension"]) if active and active["dimension"] is not None else None,
            schema_version=int(active["schema_version"]) if active else None,
            table_name=str(active["table_name"]) if active else None,
        )

    async def list_jobs(self, *, status: str | None = None) -> list[dict[str, Any]]:
        def _list(connection: Any) -> list[dict[str, Any]]:
            if status is None:
                rows = connection.execute(
                    "SELECT job_id, source_kind, source_id, generation, status, error_code, error_json, "
                    "attempt_count, created_at, updated_at FROM vector_jobs ORDER BY created_at"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT job_id, source_kind, source_id, generation, status, error_code, error_json, "
                    "attempt_count, created_at, updated_at FROM vector_jobs WHERE status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
            return [dict(row) for row in rows]

        return await self._store.run_locked(_list)

    async def enqueue(self, *, source_kind: str, source_id: str) -> VectorOpResult:
        # 探测/写入在锁内；mismatch auto-rebuild 调度为后台任务（不占 `_lock` 做全量
        # re-embed）。调度前先置 `_rebuilding`，并发 enqueue 观察后快速失败；触发方
        # 立即返回 `vector_index_rebuilding`，不 await 整库重建。
        async with self._lock:
            result, pending = await self._enqueue_unlocked(source_kind=source_kind, source_id=source_id)
            if pending is None:
                return result
            selector, job_id, pending_kind, pending_id = pending
            self._schedule_auto_rebuild(
                selector,
                job_id=job_id,
                source_kind=pending_kind,
                source_id=pending_id,
            )
            return VectorOpResult.fail(
                LRSError(
                    VECTOR_INDEX_REBUILDING,
                    "向量索引正在重建",
                    {"queued": True, "job_id": job_id, "source_id": pending_id},
                )
            )

    async def rebuild(self, *, force: bool = False) -> VectorOpResult:
        await self._await_background_rebuild()
        async with self._lock:
            try:
                return await self._rebuild_unlocked(force=force)
            except VectorRebuildFailed as exc:
                return VectorOpResult.fail(exc)

    async def search(self, query: str, *, limit: int = 10) -> VectorOpResult:
        status = await self.status()
        if status.rebuilding:
            return VectorOpResult.fail(
                LRSError(VECTOR_INDEX_REBUILDING, "向量索引正在重建，历史案例暂不可用")
            )
        if status.active_generation is None or status.dimension is None or status.table_name is None:
            return VectorOpResult.fail(
                LRSError(VECTOR_INDEX_REBUILDING, "向量索引尚未初始化")
            )

        try:
            selector = self._require_task_selector()
        except LRSError as exc:
            return VectorOpResult.fail(exc)

        try:
            vectors, actual_model = await self._embed_texts([query], selector=selector)
        except EmbeddingGenerationMismatch as exc:
            return VectorOpResult.fail(exc)

        query_vector = vectors[0]
        fingerprint = compute_model_fingerprint(
            selector.raw, actual_model, len(query_vector), self._schema_version
        )
        mismatch = self._detect_mismatch(
            status,
            selector_raw=selector.raw,
            actual_model=actual_model,
            dimension=len(query_vector),
            fingerprint=fingerprint,
        )
        if mismatch is not None:
            return VectorOpResult.fail(mismatch)

        table_name = status.table_name
        top_k = max(1, int(limit))

        def _search() -> list[dict[str, Any]]:
            assert self._db is not None
            table = self._db.open_table(table_name)
            rows = table.search(query_vector).limit(top_k).to_list()
            hits: list[dict[str, Any]] = []
            for row in rows:
                hits.append(
                    {
                        "source_kind": row.get("source_kind"),
                        "source_id": row.get("source_id"),
                        "text": row.get("text"),
                        "feedback_disposition": row.get("feedback_disposition") or None,
                        "task_id": row.get("task_id") or None,
                        "_distance": row.get("_distance"),
                    }
                )
            return hits

        try:
            hits = await asyncio.to_thread(_search)
        except Exception as exc:
            return VectorOpResult.fail(
                VectorIndexUnavailable(f"LanceDB 检索失败：{exc}", {"table_name": table_name})
            )
        return VectorOpResult.ok(data={"hits": hits})

    async def ensure_ready(self) -> VectorOpResult:
        """推进 stranded / 未初始化索引到可用态；供 Task 6 与维护路径复用。"""

        await self._await_background_rebuild()
        async with self._lock:
            try:
                status = await self.status()
                if status.candidate_active or self._rebuilding:
                    # 指纹仍匹配时 force=False 即可清 stranded，避免无谓全量重嵌
                    soft = await self._rebuild_unlocked(force=False)
                    if soft.success and not (await self.status()).candidate_active:
                        return soft
                    return await self._rebuild_unlocked(force=True)
                if status.active_generation is None:
                    sources = await self._list_indexable_sources()
                    if not sources:
                        return VectorOpResult.ok(code="empty")
                    return await self._rebuild_unlocked(force=True)
                return await self._rebuild_unlocked(force=False)
            except (EmbeddingGenerationMismatch, VectorRebuildFailed) as exc:
                return VectorOpResult.fail(exc)
            except LRSError as exc:
                return VectorOpResult.fail(exc)

    async def wait_rebuild(self, timeout: float | None = None) -> None:
        """测试/运维：等待后台 auto-rebuild 结束（若有）。"""

        task = self._rebuild_task
        if task is None or task.done():
            return
        if timeout is None:
            await task
        else:
            await asyncio.wait_for(task, timeout=timeout)

    def _schedule_auto_rebuild(
        self,
        selector: ModelSelector,
        *,
        job_id: str,
        source_kind: str,
        source_id: str,
    ) -> None:
        """置 `_rebuilding` 并调度后台全量重建（调用方须已持有 `_lock`）。"""

        self._rebuilding = True
        if self._rebuild_task is not None and not self._rebuild_task.done():
            return
        self._rebuild_task = asyncio.create_task(
            self._auto_rebuild_task(
                selector,
                job_id=job_id,
                source_kind=source_kind,
                source_id=source_id,
            ),
            name="lrs-vector-auto-rebuild",
        )

    async def _await_background_rebuild(self) -> None:
        task = self._rebuild_task
        if task is None or task.done():
            return
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _auto_rebuild_task(
        self,
        selector: ModelSelector,
        *,
        job_id: str,
        source_kind: str,
        source_id: str,
    ) -> None:
        """在锁外执行全量重建；成功后补完触发该次 rebuild 的 job。"""

        try:
            try:
                await self._run_full_rebuild(selector=selector)
            except (EmbeddingGenerationMismatch, VectorRebuildFailed):
                return
            if await self._source_in_active_generation(source_kind, source_id):
                generation = int((await self.status()).active_generation or 0)
                await self._complete_job(job_id, generation=generation)
        finally:
            status_after = await self.status()
            self._rebuilding = bool(status_after.candidate_active)

    async def _enqueue_unlocked(
        self, *, source_kind: str, source_id: str
    ) -> tuple[VectorOpResult, tuple[ModelSelector, str, str, str] | None]:
        """返回 (结果, 待 auto-rebuild 上下文)。仅 fingerprint/维度/schema mismatch 才请求重建。"""

        now = time.time()
        job_id = f"vec_{uuid.uuid4().hex}"
        await self._insert_job(
            job_id=job_id,
            source_kind=source_kind,
            source_id=source_id,
            status="PENDING",
            created_at=now,
        )

        source = await self._resolve_source(source_kind, source_id)
        if source is None:
            await self._fail_job(job_id, "source_not_found", "索引源不存在或不在白名单")
            return VectorOpResult.fail(LRSError("source_not_found", "索引源不存在或不在白名单")), None

        try:
            selector = self._require_task_selector()
        except LRSError as exc:
            await self._fail_job(job_id, exc.code, exc.message)
            return VectorOpResult.fail(exc), None

        status = await self.status()
        if status.rebuilding:
            await self._fail_job(job_id, VECTOR_INDEX_REBUILDING, "向量索引正在重建")
            return (
                VectorOpResult.fail(LRSError(VECTOR_INDEX_REBUILDING, "向量索引正在重建")),
                None,
            )

        if status.active_generation is None:
            # 空库：第一条白名单源触发 generation（失败返回结果，不 raise）
            try:
                rebuild = await self._rebuild_unlocked(force=True)
            except EmbeddingGenerationMismatch as exc:
                await self._fail_job(job_id, exc.code, exc.message, metadata=exc.metadata)
                return VectorOpResult.fail(exc), None
            except VectorRebuildFailed as exc:
                await self._fail_job(job_id, exc.code, exc.message, metadata=exc.metadata)
                return VectorOpResult.fail(exc), None
            if not rebuild.success:
                await self._fail_job(
                    job_id,
                    rebuild.error.code if rebuild.error else "rebuild_failed",
                    rebuild.error.message if rebuild.error else "重建失败",
                )
                return rebuild, None
            await self._complete_job(job_id, generation=int((await self.status()).active_generation or 0))
            return VectorOpResult.ok(code="indexed"), None

        try:
            vectors, actual_model = await self._embed_texts([source.text], selector=selector)
        except EmbeddingGenerationMismatch as exc:
            # 单次 embedding 响应非法 ≠ generation fingerprint 变更；不触发全量重建
            await self._fail_job(job_id, exc.code, exc.message, metadata=exc.metadata)
            return VectorOpResult.fail(exc), None

        vector = vectors[0]
        fingerprint = compute_model_fingerprint(
            selector.raw, actual_model, len(vector), self._schema_version
        )
        mismatch = self._detect_mismatch(
            status,
            selector_raw=selector.raw,
            actual_model=actual_model,
            dimension=len(vector),
            fingerprint=fingerprint,
        )
        if mismatch is not None:
            await self._fail_job(job_id, mismatch.code, mismatch.message, metadata=mismatch.metadata)
            # 仅真正的 fingerprint/维度/schema mismatch 才 auto-rebuild；
            # VectorIndexUnavailable（瞬态 IO）只失败 job，不动 active generation。
            if self._config.auto_rebuild and isinstance(mismatch, EmbeddingGenerationMismatch):
                return VectorOpResult.fail(mismatch), (selector, job_id, source_kind, source_id)
            return VectorOpResult.fail(mismatch), None

        try:
            await self._append_to_active(
                source=source,
                vector=vector,
                actual_model=actual_model,
                fingerprint=fingerprint,
                generation=int(status.active_generation),
                table_name=str(status.table_name),
                dimension=int(status.dimension or len(vector)),
            )
        except EmbeddingGenerationMismatch as exc:
            await self._fail_job(job_id, exc.code, exc.message, metadata=exc.metadata)
            pending = (selector, job_id, source_kind, source_id) if self._config.auto_rebuild else None
            return VectorOpResult.fail(exc), pending
        except VectorIndexUnavailable as exc:
            await self._fail_job(job_id, exc.code, exc.message, metadata=exc.metadata)
            return VectorOpResult.fail(exc), None

        await self._complete_job(job_id, generation=int(status.active_generation))
        return VectorOpResult.ok(code="indexed"), None

    async def _rebuild_unlocked(self, *, force: bool) -> VectorOpResult:
        try:
            selector = self._require_task_selector()
        except LRSError as exc:
            return VectorOpResult.fail(exc)

        status = await self.status()
        if not force and status.active_generation is not None and status.model_fingerprint:
            # 无 mismatch 时 already_current；先用当前 config selector + 一次探测 embedding 验证
            sources = await self._list_indexable_sources()
            if not sources:
                if status.candidate_active:
                    await self._fail_stranded_building()
                    self._rebuilding = False
                return VectorOpResult.ok(code="already_current")
            try:
                probe_vectors, probe_model = await self._embed_texts([sources[0].text], selector=selector)
            except EmbeddingGenerationMismatch as exc:
                return VectorOpResult.fail(exc)
            probe_fp = compute_model_fingerprint(
                selector.raw, probe_model, len(probe_vectors[0]), self._schema_version
            )
            if (
                status.selector == selector.raw
                and status.model_fingerprint == probe_fp
                and status.dimension == len(probe_vectors[0])
                and status.schema_version == self._schema_version
            ):
                # fingerprint 仍匹配时清掉 stranded building，避免索引永久不可用
                if status.candidate_active:
                    await self._fail_stranded_building()
                    self._rebuilding = False
                return VectorOpResult.ok(code="already_current")

        self._rebuilding = True
        try:
            return await self._run_full_rebuild(selector=selector)
        finally:
            # 若仍留有 building 占位（失败路径会标 failed），清除内存态
            status_after = await self.status()
            self._rebuilding = bool(status_after.candidate_active)

    async def _fail_stranded_building(self) -> None:
        def _fail(connection: Any) -> list[str]:
            rows = connection.execute(
                "SELECT table_name FROM vector_generations WHERE status = 'building'"
            ).fetchall()
            connection.execute(
                "UPDATE vector_generations SET status = 'failed' WHERE status = 'building'"
            )
            return [str(row["table_name"]) for row in rows]

        table_names = await self._store.run_locked(_fail)
        for table_name in table_names:
            await self._drop_lance_table(table_name)

    async def _drop_lance_table(self, table_name: str) -> None:
        def _drop() -> None:
            if self._db is None:
                return
            if table_name in _list_lancedb_tables(self._db):
                self._db.drop_table(table_name)

        try:
            await asyncio.to_thread(_drop)
        except Exception:
            pass

    async def _run_full_rebuild(self, *, selector: ModelSelector) -> VectorOpResult:
        sources = await self._list_indexable_sources()
        if not sources:
            # 空库不分配 generation，避免幽灵 failed candidate
            await self._fail_stranded_building()
            self._rebuilding = False
            return VectorOpResult.ok(code="empty")

        # 复用 helper：失败旧 building 行并丢弃孤儿 Lance table（勿用裸 UPDATE）
        await self._fail_stranded_building()

        now = time.time()

        def _prepare(connection: Any) -> tuple[int, str]:
            generation = int(
                connection.execute(
                    "SELECT COALESCE(MAX(generation), 0) + 1 FROM vector_generations"
                ).fetchone()[0]
            )
            table_name = _table_name_for(generation)
            _insert_generation(
                connection,
                generation=generation,
                selector=selector.raw,
                actual_model_name=None,
                model_fingerprint="pending",
                dimension=None,
                table_name=table_name,
                schema_version=self._schema_version,
                status="building",
                created_at=now,
            )
            return generation, table_name

        generation, table_name = await self._store.run_locked(_prepare)

        batch_size = max(1, int(self._config.batch_size))
        prepared_rows: list[dict[str, Any]] = []
        actual_model: str | None = None
        dimension: int | None = None
        fingerprint: str | None = None

        try:
            for offset in range(0, len(sources), batch_size):
                batch = sources[offset : offset + batch_size]
                texts = [item.text for item in batch]
                vectors, batch_model = await self._embed_texts(
                    texts,
                    selector=selector,
                    expected_dimension=dimension,
                )
                if actual_model is None:
                    actual_model = batch_model
                    dimension = len(vectors[0])
                    fingerprint = compute_model_fingerprint(
                        selector.raw, actual_model, dimension, self._schema_version
                    )
                elif batch_model != actual_model:
                    raise EmbeddingGenerationMismatch(
                        "重建过程中 actual model 发生变化",
                        {"expected": actual_model, "actual": batch_model},
                    )
                for source, vector in zip(batch, vectors, strict=True):
                    prepared_rows.append(
                        {
                            "id": f"{source.source_kind}:{source.source_id}",
                            "source_kind": source.source_kind,
                            "source_id": source.source_id,
                            "task_id": source.task_id or "",
                            "text": source.text,
                            "feedback_disposition": source.feedback_disposition or "",
                            "vector": vector,
                        }
                    )

            assert actual_model is not None and dimension is not None and fingerprint is not None

            def _write_table() -> None:
                assert self._db is not None
                if table_name in _list_lancedb_tables(self._db):
                    self._db.drop_table(table_name)
                table = self._db.create_table(table_name, prepared_rows)
                schema_dim = _schema_vector_dimension(table.schema)
                if schema_dim != dimension:
                    raise EmbeddingGenerationMismatch(
                        "LanceDB table schema 维度与 generation 不一致",
                        {"schema_dimension": schema_dim, "generation_dimension": dimension},
                    )
                if table.count_rows() != len(prepared_rows):
                    raise EmbeddingGenerationMismatch(
                        "LanceDB 写入行数与源数量不一致",
                        {"rows": table.count_rows(), "expected": len(prepared_rows)},
                    )

            await asyncio.to_thread(_write_table)

            activated_at = time.time()

            def _activate(connection: Any) -> None:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        """
                        UPDATE vector_generations
                        SET status = 'retired', retired_at = ?
                        WHERE status = 'active'
                        """,
                        (activated_at,),
                    )
                    cursor = connection.execute(
                        """
                        UPDATE vector_generations
                        SET status = 'active',
                            activated_at = ?,
                            actual_model_name = ?,
                            model_fingerprint = ?,
                            dimension = ?
                        WHERE generation = ? AND status = 'building'
                        """,
                        (activated_at, actual_model, fingerprint, dimension, generation),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("候选 generation 激活失败")
                    for row in prepared_rows:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO vector_documents(
                                source_kind, source_id, generation, actual_model_name,
                                model_fingerprint, dimension, indexed_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                row["source_kind"],
                                row["source_id"],
                                generation,
                                actual_model,
                                fingerprint,
                                dimension,
                                activated_at,
                            ),
                        )
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

            await self._store.run_locked(_activate)
            self._rebuilding = False
            await self._maybe_cleanup_retired()
            return VectorOpResult.ok(
                code="rebuilt",
                data={"generation": generation, "dimension": dimension, "count": len(prepared_rows)},
            )
        except EmbeddingGenerationMismatch:
            await self._store.run_locked(
                lambda connection: _set_generation_status(connection, generation, "failed", now=time.time())
            )
            await self._drop_lance_table(table_name)
            raise
        except Exception as exc:
            await self._store.run_locked(
                lambda connection: _set_generation_status(connection, generation, "failed", now=time.time())
            )
            await self._drop_lance_table(table_name)
            raise VectorRebuildFailed(f"重建失败：{exc}") from exc

    def _require_task_selector(self) -> ModelSelector:
        selector = ModelSelector.parse(str(self._config.selector))
        if selector.scheme != "task":
            raise LRSError(
                PHYSICAL_EMBEDDING_SELECTOR_UNSUPPORTED,
                "首发 embedding 不支持 model: 物理 pinning，请使用 task: selector",
                {"selector": selector.raw},
            )
        return selector

    async def _embed_texts(
        self,
        texts: Sequence[str],
        *,
        selector: ModelSelector,
        expected_dimension: int | None = None,
    ) -> tuple[list[list[float]], str]:
        payload = await self._embedder.embed(
            texts=list(texts),
            task_name=selector.task_name,
            max_concurrent=int(self._config.max_concurrent),
        )
        if not isinstance(payload, Mapping):
            raise EmbeddingGenerationMismatch("embedding 响应非法")
        return _validate_embedding_results(
            payload,
            expected_count=len(texts),
            expected_dimension=expected_dimension,
        )

    def _detect_mismatch(
        self,
        status: VectorIndexStatus,
        *,
        selector_raw: str,
        actual_model: str,
        dimension: int,
        fingerprint: str,
    ) -> LRSError | None:
        if status.selector is not None and status.selector != selector_raw:
            return EmbeddingGenerationMismatch(
                "embedding selector 与 active generation 不一致",
                {"active": status.selector, "current": selector_raw},
            )
        if status.schema_version is not None and status.schema_version != self._schema_version:
            return EmbeddingGenerationMismatch(
                "向量 schema_version 与 active generation 不一致",
                {"active": status.schema_version, "current": self._schema_version},
            )
        if status.dimension is not None and status.dimension != dimension:
            return EmbeddingGenerationMismatch(
                "embedding 维度与 active generation 不一致",
                {"active": status.dimension, "current": dimension},
            )
        if status.model_fingerprint is not None and status.model_fingerprint != fingerprint:
            return EmbeddingGenerationMismatch(
                "embedding 模型 fingerprint 与 active generation 不一致",
                {"active": status.model_fingerprint, "current": fingerprint},
            )
        if status.table_name:
            try:
                assert self._db is not None
                table = self._db.open_table(status.table_name)
                schema_dim = _schema_vector_dimension(table.schema)
                if schema_dim is not None and status.dimension is not None and schema_dim != status.dimension:
                    return EmbeddingGenerationMismatch(
                        "LanceDB table schema 维度与 generation metadata 不一致",
                        {"schema_dimension": schema_dim, "generation_dimension": status.dimension},
                    )
            except LRSError:
                raise
            except Exception as exc:
                return VectorIndexUnavailable(
                    f"无法校验 LanceDB schema：{exc}",
                    {"table_name": status.table_name},
                )
        return None

    async def _append_to_active(
        self,
        *,
        source: IndexableSource,
        vector: list[float],
        actual_model: str,
        fingerprint: str,
        generation: int,
        table_name: str,
        dimension: int,
    ) -> None:
        row = {
            "id": f"{source.source_kind}:{source.source_id}",
            "source_kind": source.source_kind,
            "source_id": source.source_id,
            "task_id": source.task_id or "",
            "text": source.text,
            "feedback_disposition": source.feedback_disposition or "",
            "vector": vector,
        }

        def _upsert() -> None:
            assert self._db is not None
            table = self._db.open_table(table_name)
            schema_dim = _schema_vector_dimension(table.schema)
            if schema_dim is not None and schema_dim != dimension:
                raise EmbeddingGenerationMismatch(
                    "LanceDB table schema 维度与写入向量不一致",
                    {"schema_dimension": schema_dim, "vector_dimension": dimension},
                )
            if len(vector) != dimension:
                raise EmbeddingGenerationMismatch(
                    "禁止截断或 padding 向量以适配旧 table",
                    {"table_dimension": dimension, "vector_dimension": len(vector)},
                )
            # merge_insert 原子 upsert，避免 delete-then-add 在 add 失败时丢行
            (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

        try:
            await asyncio.to_thread(_upsert)
        except EmbeddingGenerationMismatch:
            raise
        except Exception as exc:
            raise VectorIndexUnavailable(f"LanceDB 写入失败：{exc}", {"table_name": table_name}) from exc
        indexed_at = time.time()

        def _doc(connection: Any) -> None:
            connection.execute(
                """
                INSERT OR REPLACE INTO vector_documents(
                    source_kind, source_id, generation, actual_model_name,
                    model_fingerprint, dimension, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_kind,
                    source.source_id,
                    generation,
                    actual_model,
                    fingerprint,
                    dimension,
                    indexed_at,
                ),
            )

        await self._store.run_locked(_doc)

    async def _source_in_active_generation(self, source_kind: str, source_id: str) -> bool:
        status = await self.status()
        if status.active_generation is None:
            return False
        generation = int(status.active_generation)

        def _check(connection: Any) -> bool:
            row = connection.execute(
                """
                SELECT 1 FROM vector_documents
                WHERE source_kind = ? AND source_id = ? AND generation = ?
                LIMIT 1
                """,
                (source_kind, source_id, generation),
            ).fetchone()
            return row is not None

        return await self._store.run_locked(_check)

    async def _list_indexable_sources(self) -> list[IndexableSource]:
        return await self._store.run_locked(_load_indexable_sources)

    async def _resolve_source(self, source_kind: str, source_id: str) -> IndexableSource | None:
        return await self._store.run_locked(
            lambda connection: _load_single_source(connection, source_kind, source_id)
        )

    async def _insert_job(
        self,
        *,
        job_id: str,
        source_kind: str,
        source_id: str,
        status: str,
        created_at: float,
        generation: int | None = None,
        error_code: str | None = None,
        error_json: str | None = None,
    ) -> None:
        from lunagentic_research_swarm.storage.sqlite import StoreCommand

        await self._store.transact(
            [
                StoreCommand(
                    "insert_vector_job",
                    {
                        "job_id": job_id,
                        "source_kind": source_kind,
                        "source_id": source_id,
                        "generation": generation,
                        "status": status,
                        "error_code": error_code,
                        "error_json": error_json,
                        "created_at": created_at,
                        "updated_at": created_at,
                    },
                )
            ]
        )

    async def _fail_job(
        self,
        job_id: str,
        error_code: str,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.time()
        payload = json.dumps({"message": message, "metadata": dict(metadata or {})}, ensure_ascii=False)

        def _update(connection: Any) -> None:
            connection.execute(
                """
                UPDATE vector_jobs
                SET status = 'failed', error_code = ?, error_json = ?, updated_at = ?,
                    attempt_count = attempt_count + 1
                WHERE job_id = ?
                """,
                (error_code, payload, now, job_id),
            )

        await self._store.run_locked(_update)

    async def _complete_job(self, job_id: str, *, generation: int) -> None:
        now = time.time()

        def _update(connection: Any) -> None:
            connection.execute(
                """
                UPDATE vector_jobs
                SET status = 'done', generation = ?, updated_at = ?, error_code = NULL, error_json = NULL
                WHERE job_id = ?
                """,
                (generation, now, job_id),
            )

        await self._store.run_locked(_update)

    async def _maybe_cleanup_retired(self) -> None:
        retention = float(getattr(self._config, "retired_generation_retention_seconds", 86400))
        now = time.time()

        def _select(connection: Any) -> list[dict[str, Any]]:
            rows = connection.execute(
                """
                SELECT generation, table_name, retired_at
                FROM vector_generations
                WHERE status = 'retired'
                ORDER BY retired_at ASC, generation ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

        retired = await self._store.run_locked(_select)
        if len(retired) <= 1:
            return
        # 至少保留一个 retired generation
        droppable = retired[:-1]
        for row in droppable:
            retired_at = row.get("retired_at")
            if retired_at is None or (now - float(retired_at)) < retention:
                continue
            table_name = str(row["table_name"])
            generation = int(row["generation"])

            def _drop(name: str = table_name) -> None:
                assert self._db is not None
                if name in _list_lancedb_tables(self._db):
                    self._db.drop_table(name)

            try:
                await asyncio.to_thread(_drop)
            except Exception:
                continue

            def _mark(connection: Any, gen: int = generation) -> None:
                connection.execute(
                    "UPDATE vector_generations SET status = 'purged' WHERE generation = ?",
                    (gen,),
                )

            await self._store.run_locked(_mark)


def _load_generation_rows(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT generation, selector, actual_model_name, model_fingerprint, dimension,
               table_name, schema_version, status, created_at, activated_at, retired_at
        FROM vector_generations
        ORDER BY generation ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _insert_generation(
    connection: Any,
    *,
    generation: int,
    selector: str,
    actual_model_name: str | None,
    model_fingerprint: str,
    dimension: int | None,
    table_name: str,
    schema_version: int,
    status: str,
    created_at: float,
) -> None:
    connection.execute(
        """
        INSERT INTO vector_generations(
            generation, selector, actual_model_name, model_fingerprint, dimension,
            table_name, schema_version, status, created_at, activated_at, retired_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (
            generation,
            selector,
            actual_model_name,
            model_fingerprint,
            dimension,
            table_name,
            schema_version,
            status,
            created_at,
        ),
    )


def _set_generation_status(connection: Any, generation: int, status: str, *, now: float) -> None:
    if status == "failed":
        connection.execute(
            "UPDATE vector_generations SET status = ? WHERE generation = ?",
            (status, generation),
        )
    elif status == "retired":
        connection.execute(
            "UPDATE vector_generations SET status = ?, retired_at = ? WHERE generation = ?",
            (status, now, generation),
        )
    else:
        connection.execute(
            "UPDATE vector_generations SET status = ? WHERE generation = ?",
            (status, generation),
        )


def _load_indexable_sources(connection: Any) -> list[IndexableSource]:
    sources: list[IndexableSource] = []
    status_list = sorted(INDEXABLE_CONTENT_STATUSES)
    status_placeholders = ",".join("?" * len(status_list))

    for row in connection.execute(
        """
        SELECT task_id, formalized_text
        FROM tasks
        WHERE formalized_text IS NOT NULL AND TRIM(formalized_text) != ''
        ORDER BY created_at ASC, task_id ASC
        """
    ).fetchall():
        sources.append(
            IndexableSource(
                source_kind=SOURCE_KIND_FORMALIZED,
                source_id=str(row["task_id"]),
                text=str(row["formalized_text"]),
                task_id=str(row["task_id"]),
            )
        )

    summary_kinds = sorted(INDEXABLE_SUMMARY_KINDS)
    summary_placeholders = ",".join("?" * len(summary_kinds))
    for row in connection.execute(
        f"""
        SELECT summary_id, task_id, kind, text
        FROM summaries
        WHERE status IN ({status_placeholders})
          AND text IS NOT NULL AND TRIM(text) != ''
          AND kind IN ({summary_placeholders})
        ORDER BY created_at ASC, summary_id ASC
        """,
        (*status_list, *summary_kinds),
    ).fetchall():
        kind = str(row["kind"])
        source_kind = _SUMMARY_KIND_TO_SOURCE.get(kind)
        if source_kind is None:
            continue
        sources.append(
            IndexableSource(
                source_kind=source_kind,
                source_id=str(row["summary_id"]),
                text=str(row["text"]),
                task_id=str(row["task_id"]),
            )
        )

    report_kinds = sorted(INDEXABLE_REPORT_KINDS)
    report_placeholders = ",".join("?" * len(report_kinds))
    for row in connection.execute(
        f"""
        SELECT report_id, task_id, kind, text
        FROM reports
        WHERE status IN ({status_placeholders})
          AND text IS NOT NULL AND TRIM(text) != ''
          AND kind IN ({report_placeholders})
        ORDER BY created_at ASC, report_id ASC
        """,
        (*status_list, *report_kinds),
    ).fetchall():
        kind = str(row["kind"])
        source_kind = _REPORT_KIND_TO_SOURCE.get(kind)
        if source_kind is None:
            continue
        sources.append(
            IndexableSource(
                source_kind=source_kind,
                source_id=str(row["report_id"]),
                text=str(row["text"]),
                task_id=str(row["task_id"]),
            )
        )

    for row in connection.execute(
        """
        SELECT feedback_id, task_id, disposition, payload_json
        FROM feedback_events
        ORDER BY created_at ASC, feedback_id ASC
        """
    ).fetchall():
        lesson = _extract_feedback_lesson(row["payload_json"])
        if not lesson:
            continue
        sources.append(
            IndexableSource(
                source_kind=SOURCE_KIND_FEEDBACK_LESSON,
                source_id=str(row["feedback_id"]),
                text=lesson,
                task_id=str(row["task_id"]),
                feedback_disposition=str(row["disposition"]),
            )
        )

    return sources


def _load_single_source(connection: Any, source_kind: str, source_id: str) -> IndexableSource | None:
    for source in _load_indexable_sources(connection):
        if source.source_kind == source_kind and source.source_id == source_id:
            return source
    return None


def _extract_feedback_lesson(payload_json: Any) -> str | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(str(payload_json))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    lesson = payload.get("lesson")
    if isinstance(lesson, str) and lesson.strip():
        return lesson.strip()
    return None
