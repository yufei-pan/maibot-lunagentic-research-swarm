from __future__ import annotations

import math

import pytest

from lunagentic_research_swarm.extensions.contracts import ProcedureResult
from lunagentic_research_swarm.procedures.billing import extract_research_credits_charged


def test_extract_research_credits_charged_from_model() -> None:
    result = ProcedureResult(
        success=True,
        data={},
        error=None,
        metadata={},
        research_credits_charged=3.5,
    )
    assert extract_research_credits_charged(result) == pytest.approx(3.5)


def test_extract_research_credits_charged_from_mapping_default() -> None:
    assert extract_research_credits_charged({"success": True}) == 0.0


def test_extract_research_credits_charged_rejects_negative_and_nonfinite() -> None:
    assert extract_research_credits_charged({"research_credits_charged": -1.0}) == 0.0
    assert extract_research_credits_charged({"research_credits_charged": math.inf}) == 0.0
    assert extract_research_credits_charged({"research_credits_charged": "x"}) == 0.0
