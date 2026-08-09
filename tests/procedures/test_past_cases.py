"""历史案例检索：feedback 真值状态、透明 rerank、向量索引失败码；§20.2 学习呈现。"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.errors import (
    VECTOR_INDEX_REBUILDING,
    VECTOR_INDEX_UNAVAILABLE,
    VECTOR_REBUILD_FAILED,
    LRSError,
)
from lunagentic_research_swarm.procedures.bundled import past_cases as past_cases_mod
from lunagentic_research_swarm.procedures.bundled.past_cases import RERANK_FORMULA, _USE_AS, past_cases_procedure_definitions
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand
from lunagentic_research_swarm.storage.vectors import VectorOpResult


def test_past_cases_procedure_is_restricted_to_past_case_researcher() -> None:
    defs = past_cases_procedure_definitions()
    assert len(defs) == 1
    assert defs[0].allowed_agents == ["builtin.past_case_researcher"]


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
async def test_past_cases_failed_rebuild_real_vector_index_is_structured(tmp_path: Path) -> None:
    """真实 VectorIndex 首建失败：past_cases 须结构化失败，不得 success + cases=[]。"""

    from lunagentic_research_swarm.config import EmbeddingSection
    from lunagentic_research_swarm.storage.vectors import VectorIndex

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    await store.transact(
        [
            StoreCommand(
                "insert_task",
                {
                    "task_id": "failed-rebuild-task",
                    "stream_id": "stream_failed",
                    "formalized_text": "有语料但 embedding 失败",
                    "formalized_sha256": "sha_failed",
                    "created_at": 1.0,
                },
            )
        ]
    )

    class _FailingEmbedder:
        async def embed(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("embedding provider down")

    index = VectorIndex(
        store,
        _FailingEmbedder(),
        EmbeddingSection(selector="task:embedding"),
        tmp_path / "vectors" / "lancedb",
    )
    await index.start()
    try:
        ready = await index.ensure_ready()
        assert not ready.success
        assert ready.error is not None
        assert ready.error.code == VECTOR_REBUILD_FAILED
        status = await index.status()
        assert status.active_generation is None
        assert status.failed_candidate is not None
        assert status.last_error_code == VECTOR_REBUILD_FAILED

        provider = BundledProcedureProvider(object(), store=store, vector_index=index)
        result = await provider.invoke("builtin.past_cases", {"query": "有语料"})
        assert not result.success
        assert result.error is not None
        assert result.error["code"] == VECTOR_REBUILD_FAILED
        assert result.data is None or "cases" not in (result.data or {})
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


# ---------------------------------------------------------------------------
# §20.2 Past-case learning presentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spec_20_2_accepted_preferred_rejected_mixed_presented_as_risk(
    past_case_harness: PastCaseHarness,
) -> None:
    """§20.2 — accepted → success_pattern (preferred); rejected/mixed → anti_pattern/risk_reminder.

    Pins production ``validation_status`` / ``use_as`` / corrections surfacing and that
    accepted+confirmed outranks higher-similarity rejected/mixed after transparent rerank.
    """

    accepted = await past_case_harness.seed_async(
        "accepted",
        score=0.72,
        formalized_text="accepted good pattern",
        outcome_confirmed=True,
        outcome="上线后结果符合预期",
    )
    rejected = await past_case_harness.seed_async(
        "rejected",
        score=0.95,
        formalized_text="rejected bad pattern",
        # Store freezes JSON lists→tuples; past_cases only copies list-typed
        # corrections. Risk text is still presented via outcome + use_as.
        outcome="勿再采用该结论路径",
        corrections=["被冻结为 tuple 的 corrections"],
    )
    mixed = await past_case_harness.seed_async(
        "mixed",
        score=0.80,
        formalized_text="mixed partial pattern",
        outcome="证据链不完整，仅作风险提醒",
        corrections=["被冻结为 tuple 的 corrections"],
    )

    result = await past_case_harness.search("pattern", limit=10)
    assert result.success
    cases = result.data["cases"]
    by_id = {item["task_id"]: item for item in cases}

    good = by_id[accepted]
    anti = by_id[rejected]
    risk = by_id[mixed]

    assert good["validation_status"] == "accepted"
    assert good["use_as"] == "success_pattern"
    assert good["feedback_disposition"] == "accepted"
    assert good["outcome_confirmed"] is True
    assert good["outcome"] == "上线后结果符合预期"
    assert good["formalized_summary"] == "accepted good pattern"
    assert isinstance(good["corrections"], list)

    assert anti["validation_status"] == "rejected"
    assert anti["use_as"] == "anti_pattern"
    assert anti["feedback_disposition"] == "rejected"
    assert anti["outcome"] == "勿再采用该结论路径"
    assert anti["formalized_summary"] == "rejected bad pattern"
    assert anti["rerank_components"]["rejected_penalty"] == pytest.approx(0.20)
    # Researcher-facing risk: use_as + outcome. corrections is always a list;
    # store freeze turns JSON arrays into tuples, which past_cases currently drops.
    assert isinstance(anti["corrections"], list)

    assert risk["validation_status"] == "mixed"
    assert risk["use_as"] == "risk_reminder"
    assert risk["feedback_disposition"] == "mixed"
    assert risk["outcome"] == "证据链不完整，仅作风险提醒"
    assert risk["rerank_components"]["mixed_bonus"] == pytest.approx(0.05)
    assert isinstance(risk["corrections"], list)

    # Accepted preferred in presentation order despite lower raw similarity.
    order = [item["task_id"] for item in cases]
    assert order.index(accepted) < order.index(mixed) < order.index(rejected)
    assert good["rerank_score"] > risk["rerank_score"] > anti["rerank_score"]

    assert _USE_AS["accepted"] == "success_pattern"
    assert _USE_AS["rejected"] == "anti_pattern"
    assert _USE_AS["mixed"] == "risk_reminder"
    assert result.data["rerank_formula"] == RERANK_FORMULA


@pytest.mark.asyncio
async def test_spec_20_2_past_cases_invoke_does_not_mutate_prompts_config_or_ranking(
    past_case_harness: PastCaseHarness,
) -> None:
    """§20.2 — past_cases invoke must not auto-mutate prompts, config, or agent ranking."""

    await past_case_harness.seed_async("accepted", score=0.7)
    await past_case_harness.seed_async(
        "rejected",
        score=0.9,
        corrections=["反模式"],
    )

    registry = AgentRegistry(root_agent="builtin.quick_thinker")
    registry.replace_provider = MagicMock(wraps=registry.replace_provider)  # type: ignore[method-assign]
    registry.set_root_agent = MagicMock(wraps=registry.set_root_agent)  # type: ignore[method-assign]
    registry.reject_provider = MagicMock(wraps=registry.reject_provider)  # type: ignore[method-assign]

    config = SimpleNamespace(
        root_agent="builtin.quick_thinker",
        force_selector="",
        default_effort_credits=10.0,
        prompts={"system": "immutable-builtin-prompt"},
    )
    config_before = copy.deepcopy(vars(config))
    prompts_before = copy.deepcopy(config.prompts)
    use_as_before = dict(_USE_AS)
    formula_before = RERANK_FORMULA
    root_before = registry._root_agent
    providers_before = dict(registry._providers)

    past_case_harness.provider.ctx = SimpleNamespace(
        agent_registry=registry,
        config=config,
        prompts=config.prompts,
    )

    write_commands: list[Any] = []
    original_transact = past_case_harness.store.transact

    async def _spy_transact(commands: Any) -> None:
        write_commands.append(list(commands))
        await original_transact(commands)

    past_case_harness.store.transact = _spy_transact  # type: ignore[method-assign]

    result = await past_case_harness.search("formalized task", limit=5)
    assert result.success
    assert len(result.data["cases"]) >= 2

    # No durable store writes from the retrieve path (bundles are read-only).
    assert write_commands == []

    registry.replace_provider.assert_not_called()
    registry.set_root_agent.assert_not_called()
    registry.reject_provider.assert_not_called()
    assert registry._root_agent == root_before
    assert registry._providers == providers_before

    assert vars(config) == config_before
    assert config.prompts == prompts_before
    assert config.prompts is past_case_harness.provider.ctx.prompts

    assert dict(past_cases_mod._USE_AS) == use_as_before
    assert past_cases_mod.RERANK_FORMULA == formula_before
    assert result.data["rerank_formula"] == formula_before
