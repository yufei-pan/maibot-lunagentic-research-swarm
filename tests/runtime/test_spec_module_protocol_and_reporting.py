"""Module-level pins for reporting / protocol / turn-order specs (§9, §10, §13.2)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.llm.protocol import (
    ProcedureRequest,
    ProtocolError,
    build_correction_message,
    parse_json_envelope,
)
from lunagentic_research_swarm.models import ReportKind, SummaryKind, TaskStatus
from lunagentic_research_swarm.procedures.core import (
    CORE_CHECKPOINT_ID,
    CORE_COMPACT_ID,
    CORE_TERMINATE_ID,
    CoreProcedureDecision,
    split_procedure_requests,
)
from lunagentic_research_swarm.reporting import CoverageSummary, build_coverage, freeze_report_kind
from lunagentic_research_swarm.runtime.events import AgentCallCompleted, ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    PerformBranchSummary,
    PerformProcedureBatch,
    RuntimeState,
    reduce_event,
)


NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_spec_13_2_freeze_kind_checkpoint_or_live_leaves_never_final() -> None:
    assert freeze_report_kind(active_branch_count=1, coverage_has_checkpoint=False) is ReportKind.INTERMEDIATE
    assert freeze_report_kind(active_branch_count=0, coverage_has_checkpoint=True) is ReportKind.INTERMEDIATE
    assert freeze_report_kind(active_branch_count=0, coverage_has_checkpoint=False) is ReportKind.FINAL


def test_spec_13_2_build_coverage_prefers_latest_per_branch() -> None:
    items = (
        CoverageSummary("s1", "A", SummaryKind.CHECKPOINT, 1, "old", 1.0),
        CoverageSummary("s2", "A", SummaryKind.BRANCH_FINAL, 1, "new", 2.0),
        CoverageSummary("s3", "B", SummaryKind.CHECKPOINT, 1, "b", 1.5),
    )
    coverage = build_coverage(items, active_branch_ids={"B"})
    assert [item.summary_id for item in coverage.items] == ["s3", "s2"]
    assert coverage.has_checkpoint is True


def test_spec_10_1_terminate_drops_checkpoint_keeps_compact() -> None:
    ordinary, controls = split_procedure_requests(
        [
            ProcedureRequest(procedure_id=CORE_COMPACT_ID),
            ProcedureRequest(procedure_id=CORE_CHECKPOINT_ID),
            ProcedureRequest(procedure_id=CORE_TERMINATE_ID),
            ProcedureRequest(procedure_id="builtin.echo", arguments={"x": 1}),
        ]
    )
    assert [item.procedure_id for item in ordinary] == ["builtin.echo"]
    assert controls.terminate and controls.compact and not controls.checkpoint
    assert CORE_CHECKPOINT_ID in controls.ignored_controls


def test_spec_9_2_single_correction_message_is_deterministic() -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_json_envelope('{"report":1,"procedures":[],"delegations":[]}')
    message = build_correction_message(caught.value)
    assert message["role"] == "user"
    assert "report" in message["content"]


def test_spec_10_agent_call_completed_with_valid_envelope_queues_procedures() -> None:
    state = RuntimeState(
        "t",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="r1",
        active_leaves={"a": 5.0},
    )
    transition = reduce_event(
        state,
        AgentCallCompleted(
            "e1",
            "t",
            "r1",
            0,
            occurred_at=NOW,
            branch_id="a",
            call_id="c1",
            result_id="res1",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "cache_hit_tokens": 0, "cache_miss_tokens": 1},
            actual_model_name="m",
            actual_charge=0.1,
            estimated_charge=0.1,
            balance_before_reconciliation=5.0,
            protocol_result={
                "report": "ok",
                "procedures": [{"procedure_id": "builtin.echo", "arguments": {}}],
                "delegations": [],
            },
        ),
    )
    assert any(isinstance(effect, PerformProcedureBatch) for effect in transition.effects)


def test_spec_10_terminate_control_skips_delegations_and_summarizes() -> None:
    state = RuntimeState(
        "t",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="r1",
        active_leaves={"a": 4.0},
    )
    transition = reduce_event(
        state,
        ProcedureBatchCompleted(
            "e2",
            "t",
            "r1",
            0,
            occurred_at=NOW,
            branch_id="a",
            call_id="c1",
            result_id="res1",
            credits_after=3.5,
            controls=CoreProcedureDecision(terminate=True),
            delegations=({"agent_id": "child", "task": "nope", "credits": 1.0},),
            parent_messages=({"role": "assistant", "content": "a"},),
            parent_depth=0,
            live_agent_ids=("child",),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=1,
        ),
    )
    assert len(transition.effects) == 1
    assert isinstance(transition.effects[0], PerformBranchSummary)
    assert transition.effects[0].payload["reason"] == "terminate"
