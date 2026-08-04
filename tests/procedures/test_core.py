from __future__ import annotations

import pytest

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.procedures.core import (
    CORE_CHECKPOINT_ID,
    CORE_COMPACT_ID,
    CORE_TERMINATE_ID,
    CoreProcedureDecision,
    split_procedure_requests,
)
from lunagentic_research_swarm.llm.protocol import ProcedureRequest


def test_terminate_dominates_other_control_procedures() -> None:
    ordinary, controls = split_procedure_requests(
        [
            ProcedureRequest(procedure_id=CORE_COMPACT_ID),
            ProcedureRequest(procedure_id=CORE_TERMINATE_ID, arguments={"reason": "done"}),
        ]
    )

    assert ordinary == []
    assert controls.terminate
    assert not controls.compact
    assert controls.ignored_controls == [CORE_COMPACT_ID]


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def compact_branch(self, request):
        self.calls.append("compact")
        return SummaryResult(True, "压缩摘要", "model:test", None, None)

    async def finalize_branch(self, request):
        self.calls.append("checkpoint")
        return SummaryResult(True, "检查点摘要", "model:test", None, None)


@pytest.mark.asyncio
async def test_core_compact_and_checkpoint_use_summarizer_without_api_or_credits() -> None:
    from lunagentic_research_swarm.procedures.core import execute_core_procedure

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
