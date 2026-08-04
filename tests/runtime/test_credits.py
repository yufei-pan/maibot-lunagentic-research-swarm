from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
from lunagentic_research_swarm.models import CreditBalance, CreditLedgerEntry
from lunagentic_research_swarm.runtime.credits import (
    allocate_children,
    reconcile_usage,
    redistribute_pool,
    reserve_input,
    settle_branch,
)


def test_root_exact_allocation() -> None:
    result = allocate_children(100.0, [("A", 50.0), ("B", 25.0), ("C", 25.0)])

    assert result.allocations == {"A": 50.0, "B": 25.0, "C": 25.0}
    assert result.returned_to_pool == 0.0


def test_oversubscription_scales_proportionally() -> None:
    result = allocate_children(2.0, [("A", 2.0), ("B", 1.0), ("C", 1.0)])

    assert result.allocations == {"A": 1.0, "B": 0.5, "C": 0.5}


def test_zero_balance_still_launches_zero_credit_children() -> None:
    result = allocate_children(0.0, [("A", 9.0), ("B", 1.0)])

    assert result.allocations == {"A": 0.0, "B": 0.0}
    assert result.launch_allowed


def test_negative_balance_cannot_launch_children() -> None:
    result = allocate_children(-0.01, [("A", 0.0)])

    assert not result.launch_allowed
    assert result.allocations == {}


def test_negative_pool_is_dormant_until_continue() -> None:
    assert settle_branch(balance=-40.0, pool=0.2) == pytest.approx(-39.8)


def test_continue_distributes_signed_pool_proportionally() -> None:
    result = redistribute_pool(pool=-12.0, adjustment=0.0, leaves={"A": 9.0, "B": 3.0})

    assert result.balances == {"A": 0.0, "B": 0.0}
    assert result.pool_after == 0.0


def test_continue_distributes_evenly_when_all_active_leaves_are_zero() -> None:
    assert redistribute_pool(6.0, -3.0, {"A": 0.0, "B": 0.0}).balances == {"A": 1.5, "B": 1.5}
    assert redistribute_pool(-6.0, 0.0, {"A": 0.0, "B": 0.0}).balances == {"A": -3.0, "B": -3.0}


def test_no_active_leaf_restart_funds_can_be_negative() -> None:
    result = redistribute_pool(-5.0, 2.0, {})

    assert result.restart_balance == -3.0
    assert result.pool_after == -3.0
    assert not result.can_start_root
    assert result.finish_reason == "task_finished_insufficient_funds"


@pytest.mark.parametrize("credit_request", [("A", -0.01), ("A", math.inf), ("A", math.nan)])
def test_child_request_credits_must_be_nonnegative_and_finite(credit_request: tuple[str, float]) -> None:
    with pytest.raises(ValueError, match="非负有限数"):
        allocate_children(10.0, [credit_request])


def test_allocation_results_are_immutable() -> None:
    result = allocate_children(1.0, [("A", 1.0)])

    with pytest.raises(TypeError):
        result.allocations["A"] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.scale = 0.0  # type: ignore[misc]


def test_credit_models_are_frozen_and_validate_finite_values() -> None:
    balance = CreditBalance(balance=2.5, pool=-0.5)
    assert balance.balance == 2.5
    assert balance.pool == -0.5
    assert balance.available == 2.5
    with pytest.raises(FrozenInstanceError):
        balance.balance = 3.0  # type: ignore[misc]

    with pytest.raises(ValueError, match="有限"):
        CreditBalance(balance=math.inf)

    entry = CreditLedgerEntry(
        ledger_id="led_1",
        task_id="task_1",
        round_id="round_1",
        entry_kind="input_reservation",
        amount=-2.5,
        metadata_json='{"price_source":"host_config"}',
    )
    with pytest.raises(FrozenInstanceError):
        entry.amount = 0.0  # type: ignore[misc]


def test_reserve_input_emits_usage_and_research_ledger_commands() -> None:
    reservation = reserve_input(
        estimated_charge=2.5,
        task_id="task_1",
        round_id="round_1",
        branch_id="branch_1",
        call_id="call_1",
        usage_id="usage_1",
        ledger_id="ledger_1",
        role="agent",
        selector="task:research",
        estimated_model_name="model-a",
        price_source="host_config",
        price_fingerprint="fingerprint",
        prompt_tokens=100,
        completion_tokens=0,
        cache_hit_tokens=0,
        cache_miss_tokens=100,
    )

    assert [command.kind for command in reservation.commands] == [
        "insert_llm_usage",
        "insert_credit_ledger",
    ]
    usage_values = reservation.commands[0].values
    assert usage_values["reconciliation_status"] == "reserved"
    assert usage_values["estimated_charge"] == 2.5
    ledger_values = reservation.commands[1].values
    assert ledger_values["entry_kind"] == "input_reservation"
    assert ledger_values["amount"] == -2.5


def test_reconcile_usage_writes_actual_charge_and_signed_adjustment() -> None:
    catalog = PriceCatalog.from_sources(
        {},
        {"model-a": PriceProfile(price_in=10.0, price_out=20.0)},
        {"research": ["model-a"]},
    )
    reservation = reserve_input(
        estimated_charge=3.0,
        task_id="task_1",
        round_id="round_1",
        branch_id="branch_1",
        call_id="call_1",
        usage_id="usage_1",
        ledger_id="ledger_1",
        role="agent",
        selector="task:research",
        estimated_model_name="model-a",
        price_source="host_config",
        price_fingerprint=catalog.fingerprint,
        prompt_tokens=100,
        cache_miss_tokens=100,
    )

    result = reconcile_usage(
        reservation,
        catalog=catalog,
        actual_model_name="model-a",
        usage=TokenUsage(10_000, 0, 0, 10_000, source="actual"),
    )

    assert result.status == "actual"
    assert result.actual_charge == pytest.approx(10.0)
    assert result.adjustment == pytest.approx(-7.0)
    assert result.usage_command.values["actual_charge"] == pytest.approx(10.0)
    assert result.usage_command.values["adjustment"] == pytest.approx(-7.0)
    assert result.ledger_command is not None
    assert result.ledger_command.values["entry_kind"] == "input_reconciliation"
    assert result.ledger_command.values["amount"] == pytest.approx(-7.0)


def test_summarizer_reconciliation_has_usage_telemetry_without_research_ledger() -> None:
    catalog = PriceCatalog.from_sources({}, {"model-a": PriceProfile(price_in=10.0)}, {})
    reservation = reserve_input(
        estimated_charge=1.0,
        task_id="task_1",
        round_id="round_1",
        call_id="call_1",
        role="summarizer",
        selector="model:model-a",
        estimated_model_name="model-a",
        price_source="host_config",
        price_fingerprint=catalog.fingerprint,
        prompt_tokens=100,
    )
    assert len(reservation.commands) == 1
    result = reconcile_usage(
        reservation,
        catalog=catalog,
        actual_model_name="model-a",
        usage=TokenUsage(100_000, 0, 0, 100_000, source="actual"),
    )
    assert result.ledger_command is None
    assert result.commands == (result.usage_command,)
