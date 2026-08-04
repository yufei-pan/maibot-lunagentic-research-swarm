from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from lunagentic_research_swarm.runtime.credits import (
    allocate_children,
    assert_task_credit_equation,
)


@given(
    balance=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    requests=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        max_size=8,
    ),
)
def test_allocation_conserves_parent_balance(balance: float, requests: list[float]) -> None:
    result = allocate_children(balance, [(str(index), value) for index, value in enumerate(requests)])
    assert math.fsum(result.allocations.values()) + result.returned_to_pool == pytest.approx(balance)
    assert all(value >= 0 for value in result.allocations.values())


@given(
    initial=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    adjustments=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        max_size=20,
    ),
    charges=st.lists(
        st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_infinity=False),
        max_size=20,
    ),
)
def test_task_credit_equation(initial: float, adjustments: list[float], charges: list[float]) -> None:
    expected = initial + math.fsum(adjustments) - math.fsum(charges)
    active = {"A": expected / 3.0, "B": expected / 3.0}
    pool = expected - math.fsum(active.values())
    assert_task_credit_equation(
        initial=initial,
        signed_adjustments=adjustments,
        charged_agent_costs=charges,
        active_balances=active,
        dormant_pool=pool,
    )
