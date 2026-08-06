"""Procedure 研究额度扣费辅助。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def extract_research_credits_charged(result: Any) -> float:
    """从 ProcedureResult 或 mapping 读取非负有限的 research_credits_charged。"""

    raw = getattr(result, "research_credits_charged", None)
    if raw is None and isinstance(result, Mapping):
        raw = result.get("research_credits_charged", 0.0)
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value
