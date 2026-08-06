from __future__ import annotations

import pytest

from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.procedures.core import (
    CORE_CHECKPOINT_ID,
    CORE_COMPACT_ID,
    CORE_TERMINATE_ID,
    CoreProcedureDecision,
    execute_core_procedure,
    split_procedure_requests,
)
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.runtime.events import (
    ProcedureBatchCompleted,
    event_from_json,
    event_to_json,
)

# price_in=10, price_out=20 → (100*10 + 50*20) / 1e6 * 100 = 0.2 credits
COMPACT_USAGE = TokenUsage(100, 50, 0, 100, source="actual")
COMPACT_CHARGE = 0.2
COMPACT_CATALOG = PriceCatalog.from_sources({}, {"model:test": PriceProfile(price_in=10.0, price_out=20.0)}, {})


def test_terminate_ignores_checkpoint_but_keeps_compact() -> None:
    """spec §10 步骤 6→7：compact 先执行，分支最终总结运行在压缩后的历史上。"""

    ordinary, controls = split_procedure_requests(
        [
            ProcedureRequest(procedure_id=CORE_COMPACT_ID),
            ProcedureRequest(procedure_id=CORE_CHECKPOINT_ID),
            ProcedureRequest(procedure_id=CORE_TERMINATE_ID, arguments={"reason": "done"}),
        ]
    )

    assert ordinary == []
    assert controls.terminate
    assert controls.compact
    assert not controls.checkpoint
    assert controls.ignored_controls == [CORE_CHECKPOINT_ID]


class FakeSummarizer:
    def __init__(self, *, usage: TokenUsage | None = None) -> None:
        self.calls: list[str] = []
        self.usage = usage

    async def compact_branch(self, request):
        self.calls.append("compact")
        return SummaryResult(True, "压缩摘要", "model:test", self.usage, None)

    async def finalize_branch(self, request):
        self.calls.append("checkpoint")
        return SummaryResult(True, "检查点摘要", "model:test", self.usage, None)


@pytest.mark.asyncio
async def test_core_compact_and_checkpoint_use_summarizer_without_api_or_credits() -> None:
    summarizer = FakeSummarizer()
    compact = await execute_core_procedure(
        CORE_COMPACT_ID,
        {"formalized_task": "正式任务", "branch_history": [{"role": "assistant", "content": "证据"}]},
        summarizer=summarizer,
    )
    checkpoint = await execute_core_procedure(
        "core.checkpoint",
        {"formalized_task": "正式任务", "branch_history": [{"role": "assistant", "content": "证据"}]},
        summarizer=summarizer,
    )

    assert compact.success and compact.data["compacted"]
    assert checkpoint.success and checkpoint.data["checkpoint"]
    assert summarizer.calls == ["compact", "checkpoint"]
    assert compact.metadata["research_credits_charged"] == 0.0
    assert compact.research_credits_charged == 0.0


@pytest.mark.asyncio
async def test_agent_requested_compact_sets_research_credits_from_summarizer_usage() -> None:
    """有 usage + price_catalog 时，core.compact 设置 research_credits_charged（不写 ledger）。"""

    summarizer = FakeSummarizer(usage=COMPACT_USAGE)
    compact = await execute_core_procedure(
        CORE_COMPACT_ID,
        {"formalized_task": "正式任务", "branch_history": [{"role": "assistant", "content": "证据"}]},
        summarizer=summarizer,
        price_catalog=COMPACT_CATALOG,
    )

    assert compact.success
    assert compact.research_credits_charged == pytest.approx(COMPACT_CHARGE)
    assert compact.metadata["research_credits_charged"] == pytest.approx(COMPACT_CHARGE)


@pytest.mark.asyncio
async def test_compact_bill_research_false_forces_zero_charge() -> None:
    """自动 compact 路径传入 bill_research_credits=False 时强制 charge=0。"""

    summarizer = FakeSummarizer(usage=COMPACT_USAGE)
    compact = await execute_core_procedure(
        CORE_COMPACT_ID,
        {"formalized_task": "正式任务", "branch_history": [{"role": "assistant", "content": "证据"}]},
        summarizer=summarizer,
        price_catalog=COMPACT_CATALOG,
        bill_research_credits=False,
    )

    assert compact.success
    assert compact.research_credits_charged == 0.0
    assert compact.metadata["research_credits_charged"] == 0.0


def test_compact_and_checkpoint_can_coexist_without_ordinary_reordering() -> None:
    requests = [
        ProcedureRequest(procedure_id="builtin.one"),
        ProcedureRequest(procedure_id=CORE_COMPACT_ID),
        ProcedureRequest(procedure_id="builtin.two"),
        ProcedureRequest(procedure_id=CORE_CHECKPOINT_ID),
    ]

    ordinary, controls = split_procedure_requests(requests)

    assert [item.procedure_id for item in ordinary] == ["builtin.one", "builtin.two"]
    assert controls == CoreProcedureDecision(compact=True, checkpoint=True)


def test_core_ids_are_not_treated_as_external_procedures() -> None:
    ordinary, controls = split_procedure_requests(
        [ProcedureRequest(procedure_id=CORE_TERMINATE_ID), ProcedureRequest(procedure_id="builtin.search")]
    )

    assert [item.procedure_id for item in ordinary] == ["builtin.search"]
    assert controls.terminate


def test_core_control_requests_are_frozen_and_survive_event_round_trip() -> None:
    _, controls = split_procedure_requests(
        [ProcedureRequest(procedure_id=CORE_COMPACT_ID, arguments={"reason": {"nested": "value"}})]
    )
    event = ProcedureBatchCompleted(
        event_id="event-control",
        task_id="task-1",
        round_id="round-1",
        generation=1,
        controls=controls,
    )

    with pytest.raises(TypeError):
        controls.control_requests[0].arguments["reason"] = "mutated"  # type: ignore[index]

    decoded = event_from_json(event_to_json(event))

    assert decoded.controls.compact
    assert decoded.controls.control_requests[0].procedure_id == CORE_COMPACT_ID
    assert decoded.controls.control_requests[0].arguments["reason"]["nested"] == "value"
    with pytest.raises(TypeError):
        decoded.controls.control_requests[0].arguments["reason"]["nested"] = "mutated"  # type: ignore[index]


def test_core_control_arguments_do_not_persist_raw_payload_or_transcripts() -> None:
    _, controls = split_procedure_requests(
        [
            ProcedureRequest(
                procedure_id=CORE_COMPACT_ID,
                arguments={
                    "reason": "保留控制参数",
                    "raw_payload": "SECRET_CONTROL_RAW",
                    "reasoning": "SECRET_CONTROL_REASONING",
                    "raw_result": "SECRET_CONTROL_RESULT",
                    "raw_arguments": "SECRET_CONTROL_ARGUMENTS",
                    "transcript": "SECRET_CONTROL_TRANSCRIPT",
                    "messages": "SECRET_CONTROL_MESSAGES",
                },
            )
        ]
    )
    event = ProcedureBatchCompleted(
        event_id="event-control-sensitive",
        task_id="task-1",
        round_id="round-1",
        generation=1,
        controls=controls,
    )

    encoded = event_to_json(event)

    assert "保留控制参数" in encoded
    for secret in (
        "SECRET_CONTROL_RAW",
        "SECRET_CONTROL_REASONING",
        "SECRET_CONTROL_RESULT",
        "SECRET_CONTROL_ARGUMENTS",
        "SECRET_CONTROL_TRANSCRIPT",
        "SECRET_CONTROL_MESSAGES",
    ):
        assert secret not in encoded
