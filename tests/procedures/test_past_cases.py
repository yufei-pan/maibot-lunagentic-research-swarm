"""历史案例检索：feedback 真值状态、透明 rerank、向量索引失败码。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.errors import (
    VECTOR_INDEX_REBUILDING,
    VECTOR_INDEX_UNAVAILABLE,
    VECTOR_REBUILD_FAILED,
    LRSError,
)
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from lunagentic_research_swarm.storage.vectors import VectorOpResult


@dataclass
class _QueuedHit:
    task_id: str
    source_id: str
    text: str
    similarity: float
    source_kind: str = "formalized_task"
    feedback_disposition: str | None = None


@dataclass
class FakeVectorIndex:
    """可控 VectorIndex.search 替身：用 similarity 注入排序，失败码可预设。"""

    hits: list[_QueuedHit] = field(default_factory=list)
    fail_error: LRSError | None = None
    last_limit: int | None = None

    async def search(self, query: str, *, limit: int = 10) -> VectorOpResult:
        self.last_limit = int(limit)
        if self.fail_error is not None:
            return VectorOpResult.fail(self.fail_error)
        ordered = sorted(self.hits, key=lambda item: item.similarity, reverse=True)
        rows: list[dict[str, Any]] = []
        for item in ordered[: max(1, int(limit))]:
            distance = (1.0 / item.similarity) - 1.0 if item.similarity > 0 else 1e9
            rows.append(
                {
                    "source_kind": item.source_kind,
                    "source_id": item.source_id,
                    "text": item.text,
                    "task_id": item.task_id,
                    "feedback_disposition": item.feedback_disposition,
                    "_distance": distance,
                }
            )
        return VectorOpResult.ok(data={"hits": rows})


@dataclass
class _PendingSeed:
    commands: list[StoreCommand]
    task_id: str
    disposition: str | None
    score: float
    formalized_text: str


@dataclass
class PastCaseHarness:
    store: SQLiteStateStore
    index: FakeVectorIndex
    provider: BundledProcedureProvider
    _seq: int = 0
    _pending: list[_PendingSeed] = field(default_factory=list)

    @classmethod
    async def create(cls, tmp_path: Path) -> PastCaseHarness:
        store = SQLiteStateStore(tmp_path / "state.sqlite3")
        await store.open()
        index = FakeVectorIndex()
        provider = BundledProcedureProvider(
            object(),
            store=store,
            vector_index=index,
        )
        return cls(store=store, index=index, provider=provider)

    async def close(self) -> None:
        await self.store.close()

    def seed(
        self,
        disposition: str | None,
        *,
        score: float,
        formalized_text: str = "formalized task",
        outcome_confirmed: bool = False,
        corrections: list[str] | None = None,
        outcome: str | None = None,
        created_at: float | None = None,
    ) -> str:
        """同步排队一则案例（brief API）；在 search / flush 时写入 SQLite 与假索引。"""

        self._seq += 1
        task_id = f"task_{self._seq}"
        round_id = f"rnd_{self._seq}"
        now = float(created_at if created_at is not None else self._seq)
        commands = [
            StoreCommand(
                "insert_task",
                {
                    "task_id": task_id,
                    "stream_id": f"stream_{task_id}",
                    "formalized_text": formalized_text,
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
                    "summary_id": f"sum_{self._seq}",
                    "task_id": task_id,
                    "round_id": round_id,
                    "branch_id": None,
                    "kind": "CHECKPOINT",
                    "report_epoch": None,
                    "text": f"summary for {task_id}",
                    "status": "SUCCEEDED",
                    "error_code": None,
                    "created_at": now,
                },
            ),
            StoreCommand(
                "insert_report",
                {
                    "report_id": f"rep_{self._seq}",
                    "task_id": task_id,
                    "round_id": round_id,
                    "epoch": 1,
                    "kind": "FINAL",
                    "text": f"report for {task_id}",
                    "status": "SUCCEEDED",
                    "running_branch_count": 0,
                    "stats_json": "{}",
                    "created_at": now,
                },
            ),
        ]
        if disposition is not None:
            payload: dict[str, Any] = {
                "lesson": f"lesson {disposition}",
                "corrections": list(corrections or []),
            }
            if outcome is not None:
                payload["outcome"] = outcome
            if outcome_confirmed:
                payload["outcome_confirmed"] = True
            commands.append(
                StoreCommand(
                    "insert_feedback_event",
                    {
                        "feedback_id": f"fb_{self._seq}",
                        "task_id": task_id,
                        "round_id": round_id,
                        "disposition": disposition,
                        "payload_json": json.dumps(payload, ensure_ascii=False),
                        "created_at": now + 0.1,
                    },
                )
            )
        self._pending.append(
            _PendingSeed(
                commands=commands,
                task_id=task_id,
                disposition=disposition,
                score=score,
                formalized_text=formalized_text,
            )
        )
        return task_id

    async def flush(self) -> None:
        while self._pending:
            pending = self._pending.pop(0)
            await self.store.transact(pending.commands)
            self.index.hits.append(
                _QueuedHit(
                    task_id=pending.task_id,
                    source_id=pending.task_id,
                    text=pending.formalized_text,
                    similarity=pending.score,
                    feedback_disposition=pending.disposition,
                )
            )

    async def seed_async(
        self,
        disposition: str | None,
        *,
        score: float,
        formalized_text: str = "formalized task",
        outcome_confirmed: bool = False,
        corrections: list[str] | None = None,
        outcome: str | None = None,
        created_at: float | None = None,
    ) -> str:
        task_id = self.seed(
            disposition,
            score=score,
            formalized_text=formalized_text,
            outcome_confirmed=outcome_confirmed,
            corrections=corrections,
            outcome=outcome,
            created_at=created_at,
        )
        await self.flush()
        return task_id

    async def search(self, query: str, **kwargs: Any):
        await self.flush()
        arguments = {"query": query, "limit": int(kwargs.pop("limit", 10)), **kwargs}
        return await self.provider.invoke("builtin.past_cases", arguments)


@pytest.fixture
async def past_case_harness(tmp_path: Path) -> AsyncIterator[PastCaseHarness]:
    harness = await PastCaseHarness.create(tmp_path)
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_past_cases_labels_feedback_truth_status(past_case_harness: PastCaseHarness) -> None:
    past_case_harness.seed("accepted", score=0.7)
    past_case_harness.seed("rejected", score=0.9)
    past_case_harness.seed(None, score=0.8)
    result = await past_case_harness.search("formalized task")
    assert result.success
    assert [item["validation_status"] for item in result.data["cases"]] == [
        "accepted",
        "unreviewed",
        "rejected",
    ]
    assert result.data["cases"][-1]["use_as"] == "anti_pattern"


@pytest.mark.asyncio
async def test_past_cases_rerank_components_are_transparent(past_case_harness: PastCaseHarness) -> None:
    await past_case_harness.seed_async("accepted", score=0.7, outcome_confirmed=True)
    result = await past_case_harness.search("formalized task", limit=5)
    assert result.success
    case = result.data["cases"][0]
    comps = case["rerank_components"]
    assert comps["similarity"] == pytest.approx(0.7)
    assert comps["accepted_bonus"] == pytest.approx(0.15)
    assert comps["outcome_confirmed_bonus"] == pytest.approx(0.10)
    assert case["rerank_score"] == pytest.approx(0.7 + 0.15 + 0.10)
    assert "similarity" in result.data["rerank_formula"]


@pytest.mark.asyncio
async def test_past_cases_empty_matches_are_success(past_case_harness: PastCaseHarness) -> None:
    result = await past_case_harness.search("nothing matches")
    assert result.success
    assert result.data["cases"] == []


@pytest.mark.asyncio
async def test_past_cases_vector_rebuilding_is_structured(past_case_harness: PastCaseHarness) -> None:
    past_case_harness.index.fail_error = LRSError(VECTOR_INDEX_REBUILDING, "向量索引正在重建")
    result = await past_case_harness.search("q")
    assert not result.success
    assert result.error["code"] == VECTOR_INDEX_REBUILDING
    assert result.data is None or "cases" not in (result.data or {})


@pytest.mark.asyncio
async def test_past_cases_vector_unavailable_and_failed_codes(past_case_harness: PastCaseHarness) -> None:
    past_case_harness.index.fail_error = LRSError(VECTOR_INDEX_UNAVAILABLE, "LanceDB 不可用")
    unavailable = await past_case_harness.search("q")
    assert not unavailable.success
    assert unavailable.error["code"] == VECTOR_INDEX_UNAVAILABLE

    past_case_harness.index.fail_error = LRSError(VECTOR_REBUILD_FAILED, "重建失败")
    failed = await past_case_harness.search("q")
    assert not failed.success
    assert failed.error["code"] == VECTOR_REBUILD_FAILED


@pytest.mark.asyncio
async def test_past_cases_excludes_current_task_and_aggregates_sources(
    past_case_harness: PastCaseHarness,
) -> None:
    task_a = await past_case_harness.seed_async("mixed", score=0.6)
    past_case_harness.index.hits.append(
        _QueuedHit(
            task_id=task_a,
            source_id=f"sum_extra_{task_a}",
            text="extra summary",
            similarity=0.55,
            source_kind="checkpoint_summary",
            feedback_disposition="mixed",
        )
    )
    other = await past_case_harness.seed_async(None, score=0.5)
    result = await past_case_harness.search(
        "formalized task",
        limit=10,
        exclude_task_id=other,
    )
    assert result.success
    cases = result.data["cases"]
    assert len(cases) == 1
    assert cases[0]["task_id"] == task_a
    assert cases[0]["validation_status"] == "mixed"
    assert cases[0]["use_as"] == "risk_reminder"
    assert set(cases[0]["source_ids"]) >= {task_a, f"sum_extra_{task_a}"}
    assert past_case_harness.index.last_limit == min(50, 10 * 5)


@pytest.mark.asyncio
async def test_past_cases_provider_registers_definition() -> None:
    provider = BundledProcedureProvider(object())
    ids = {item["procedure_id"] for item in provider.describe()}
    assert "builtin.past_cases" in ids


@pytest.mark.asyncio
async def test_past_cases_empty_real_vector_index_is_success(tmp_path: Path) -> None:
    """真实 VectorIndex 空语料：past_cases 须 success + cases=[]，非 vector_index_rebuilding。"""

    from lunagentic_research_swarm.config import EmbeddingSection
    from lunagentic_research_swarm.storage.vectors import VectorIndex

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()

    class _Embedder:
        async def embed(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("空索引 search 不应触发 embed")

    index = VectorIndex(
        store,
        _Embedder(),
        EmbeddingSection(selector="task:embedding"),
        tmp_path / "vectors" / "lancedb",
    )
    await index.start()
    try:
        ready = await index.ensure_ready()
        assert ready.success
        assert ready.code == "empty"
        provider = BundledProcedureProvider(object(), store=store, vector_index=index)
        result = await provider.invoke("builtin.past_cases", {"query": "anything"})
        assert result.success
        assert result.data is not None
        assert result.data["cases"] == []
        assert result.error is None
    finally:
        await index.close()
        await store.close()


@pytest.mark.asyncio
async def test_past_cases_excludes_scoped_task_id_via_invoker(
    past_case_harness: PastCaseHarness,
) -> None:
    """bundled_procedure_invoker 须转发 scoped_metadata.task_id 并默认排除当前任务。"""

    from lunagentic_research_swarm.procedures.executor import bundled_procedure_invoker

    current = await past_case_harness.seed_async("accepted", score=0.9)
    other = await past_case_harness.seed_async(None, score=0.8)
    invoke = bundled_procedure_invoker(past_case_harness.provider)
    payload = await invoke(
        procedure_id="builtin.past_cases",
        arguments={"query": "formalized task", "limit": 10},
        scoped_metadata={"task_id": current, "round_id": "r1"},
    )
    assert payload["success"] is True
    case_ids = [item["task_id"] for item in payload["data"]["cases"]]
    assert current not in case_ids
    assert other in case_ids


@pytest.mark.asyncio
async def test_past_cases_accepted_contradicted_outcome_not_success_pattern(
    past_case_harness: PastCaseHarness,
) -> None:
    """accepted + 证伪 outcome：outcome_correction，不得靠 +0.10 压过 rejected anti_pattern。"""

    contradicted = await past_case_harness.seed_async(
        "accepted",
        score=0.5,
        outcome="上线后被证明结论错误",
        corrections=["结论应反过来"],
    )
    rejected = await past_case_harness.seed_async("rejected", score=0.9)
    result = await past_case_harness.search("formalized task")
    assert result.success
    by_id = {item["task_id"]: item for item in result.data["cases"]}
    bad = by_id[contradicted]
    anti = by_id[rejected]
    assert bad["validation_status"] == "outcome_correction"
    assert bad["use_as"] == "outcome_correction"
    assert bad["outcome_confirmed"] is False
    assert bad["rerank_components"]["accepted_bonus"] == pytest.approx(0.0)
    assert bad["rerank_components"]["outcome_confirmed_bonus"] == pytest.approx(0.0)
    assert anti["validation_status"] == "rejected"
    assert anti["use_as"] == "anti_pattern"
    assert anti["rerank_score"] > bad["rerank_score"]
    statuses = [item["validation_status"] for item in result.data["cases"]]
    assert statuses.index("rejected") < statuses.index("outcome_correction")
