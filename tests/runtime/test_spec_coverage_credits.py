"""Design §11 / §25.3 credit scenarios not already pinned as dedicated cases."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.credits import (
    allocate_children,
    meter_summarizer_usage,
    redistribute_pool,
    settle_branch,
)
from lunagentic_research_swarm.procedures.core import CoreProcedureDecision
from lunagentic_research_swarm.runtime.events import (
    AgentCallCompleted,
    BranchFinalized,
    ChildMaterialized,
    ProcedureBatchCompleted,
)
from lunagentic_research_swarm.runtime.reducer import PerformBranchSummary, RuntimeState, reduce_event

NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_spec_25_3_root_exact_allocation_scenario() -> None:
    """§25.3#1 — root 100 → A/B/C = 50/25/25."""

    result = allocate_children(100.0, [("A", 50.0), ("B", 25.0), ("C", 25.0)])
    assert result.allocations == {"A": 50.0, "B": 25.0, "C": 25.0}
    assert result.returned_to_pool == 0.0
    assert math.fsum(result.allocations.values()) + result.returned_to_pool == pytest.approx(100.0)


def test_spec_25_3_oversubscription_scales_requests() -> None:
    """§25.3#2 — parent 2, requests 2/1/1 → actual 1/0.5/0.5."""

    result = allocate_children(2.0, [("A", 2.0), ("B", 1.0), ("C", 1.0)])
    assert result.allocations == {"A": 1.0, "B": 0.5, "C": 0.5}
    assert result.returned_to_pool == pytest.approx(0.0)
    assert math.fsum(result.allocations.values()) == pytest.approx(2.0)


def test_spec_25_3_zero_parent_still_launches_zero_credit_children() -> None:
    """§25.3#3 — parent 0, positive requests → all children 0 but launch_allowed."""

    result = allocate_children(0.0, [("A", 9.0), ("B", 1.0), ("C", 3.0)])
    assert result.allocations == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert result.launch_allowed is True


def test_spec_25_3_continue_zero_leaves_splits_signed_pool_evenly() -> None:
    """§25.3#6 — all-zero active leaves; +/− adjustment both split evenly."""

    positive = redistribute_pool(6.0, -3.0, {"A": 0.0, "B": 0.0})
    assert positive.balances == {"A": 1.5, "B": 1.5}
    negative = redistribute_pool(-6.0, 0.0, {"A": 0.0, "B": 0.0})
    assert negative.balances == {"A": -3.0, "B": -3.0}


def test_spec_25_3_no_leaves_restart_funds_gate() -> None:
    """§25.3#7 — empty leaves: non-negative pool starts root; negative → insufficient."""

    ok = redistribute_pool(4.0, 1.0, {})
    assert ok.can_start_root is True
    assert ok.restart_balance == pytest.approx(5.0)

    blocked = redistribute_pool(-5.0, 2.0, {})
    assert blocked.can_start_root is False
    assert blocked.finish_reason == "task_finished_insufficient_funds"


def test_spec_25_3_summarizer_meter_writes_no_research_ledger() -> None:
    """§25.3#8 — summarizer usage is metered but never research-ledger debit."""

    catalog = PriceCatalog.from_sources({}, {"sum-model": PriceProfile(price_in=10.0)}, {})
    commands = meter_summarizer_usage(
        role="branch_summarizer",
        task_id="task-1",
        round_id="round-1",
        branch_id="A",
        call_id="sum-1",
        catalog=catalog,
        model_name="sum-model",
        usage=TokenUsage(100_000, 0, 0, 100_000, source="actual"),
        created_at=NOW.timestamp(),
    )
    assert commands
    assert all(cmd.kind == "insert_llm_usage" for cmd in commands)
    assert not any(cmd.kind == "insert_credit_ledger" for cmd in commands)
    assert all(cmd.values.get("role") == "branch_summarizer" for cmd in commands)


def test_spec_25_3_negative_child_settles_into_pool_without_touching_siblings() -> None:
    """§25.3#5 / §11.5 — negative settle grows dormant pool; live siblings stay put."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"neg": -4.0, "ok": 9.0},
        credit_pool=0.5,
    )
    # settle_branch folds leaf into pool (signed).
    pool_after = settle_branch(balance=-4.0, pool=0.5)
    assert pool_after == pytest.approx(-3.5)

    transition = reduce_event(
        state,
        BranchFinalized(
            "evt-neg",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="neg",
            summary_id="sum-neg",
        ),
    )
    assert "neg" not in transition.next_state.active_leaves
    assert transition.next_state.active_leaves["ok"] == 9.0
    assert transition.next_state.credit_pool == pytest.approx(0.5 + (-4.0))


def test_spec_25_3_zero_credit_child_goes_negative_then_no_grandchildren() -> None:
    """§25.3#4 — zero-credit child incurs cost → negative → procedures then finalize, no fan-out."""

    # Materialize a zero-credit child while keeping the parent alive (grace-style).
    parent_state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"parent": 0.0},
        credit_pool=0.0,
        agent_calls_started=1,
    )
    materialized = reduce_event(
        parent_state,
        ChildMaterialized(
            "evt-child",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent:1",
            parent_branch_id="parent",
            agent_id="child",
            credits=0.0,
            depth=1,
            retire_parent=False,
            parent_credits_after=0.0,
        ),
    )
    assert materialized.next_state.active_leaves["parent:1"] == 0.0

    # Nonzero actual charge against a zero post-reservation balance → negative.
    completed = reduce_event(
        materialized.next_state,
        AgentCallCompleted(
            "evt-done",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent:1",
            call_id="call-1",
            result_id="result-1",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "cache_hit_tokens": 0, "cache_miss_tokens": 10},
            actual_model_name="model-a",
            actual_charge=1.5,
            estimated_charge=0.0,
            balance_before_reconciliation=0.0,
            protocol_result={"report": "did work", "procedures": (), "delegations": ()},
        ),
    )
    assert completed.next_state.active_leaves["parent:1"] == pytest.approx(-1.5)

    # Procedure batch with negative credits_after finalizes for credits, ignoring delegations.
    batch = reduce_event(
        completed.next_state,
        ProcedureBatchCompleted(
            "evt-batch",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="parent:1",
            call_id="call-1",
            result_id="result-1",
            credits_after=-1.5,
            controls=CoreProcedureDecision(),
            delegations=(
                {"agent_id": "grandchild", "task": "must not launch", "credits": 1.0},
            ),
            parent_messages=({"role": "assistant", "content": "parent:1"},),
            parent_depth=1,
            live_agent_ids=("grandchild",),
            max_delegations_per_turn=8,
            max_branch_depth=32,
            max_agent_calls_per_task=256,
            agent_calls_started=2,
        ),
    )
    assert len(batch.effects) == 1
    assert isinstance(batch.effects[0], PerformBranchSummary)
    assert batch.effects[0].payload["reason"] == "negative_credit"
    assert not any(
        getattr(effect, "payload", {}).get("action") == "materialize_child" for effect in batch.effects
    )


def test_spec_protocol_invalid_with_negative_balance_skips_correction() -> None:
    """§23.1 / §9.2 — negative balance + protocol error → finalize, no correction turn."""

    state = RuntimeState(
        "task-1",
        TaskStatus.RUNNING,
        generation=0,
        active_round_id="round-1",
        active_leaves={"branch-1": 0.5},
    )
    transition = reduce_event(
        state,
        AgentCallCompleted(
            "evt-bad",
            "task-1",
            "round-1",
            0,
            occurred_at=NOW,
            branch_id="branch-1",
            call_id="call-1",
            result_id="result-1",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "cache_hit_tokens": 0, "cache_miss_tokens": 10},
            actual_model_name="physical-model-v1",
            actual_charge=2.0,
            estimated_charge=0.5,
            balance_before_reconciliation=0.5,
            protocol_error={"message": "invalid", "errors": [{"pointer": "/report", "message": "required"}]},
            correction_count=0,
            max_correction_turns=1,
            pinning_supported=True,
        ),
    )
    assert transition.next_state.active_leaves["branch-1"] < 0
    assert len(transition.effects) == 1
    assert isinstance(transition.effects[0], PerformBranchSummary)
    assert transition.effects[0].payload["reason"] == "negative_credit"
