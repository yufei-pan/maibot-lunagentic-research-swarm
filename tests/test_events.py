from __future__ import annotations

import json

import pytest

from lunagentic_research_swarm.runtime.events import AgentCallCompleted, event_from_json, event_to_json


def test_event_round_trip_keeps_generation_and_usage() -> None:
    """若事件编码遗漏代数或完成结果标识，本测试应失败。"""

    event = AgentCallCompleted(
        event_id="evt_1",
        task_id="lrs_1",
        round_id="rnd_1",
        generation=3,
        branch_id="br_1",
        call_id="call_1",
        result_id="result_1",
    )

    assert event_from_json(event_to_json(event)) == event


def test_event_from_json_rejects_unregistered_event_type() -> None:
    """若未知事件被静默降级或接纳，本测试应失败。"""

    payload = json.dumps(
        {
            "event_type": "UnknownEvent",
            "event_id": "evt_1",
            "task_id": "lrs_1",
            "round_id": "rnd_1",
            "generation": 0,
            "occurred_at": "2026-08-03T00:00:00+00:00",
        }
    )

    with pytest.raises(ValueError, match="未知事件类型"):
        event_from_json(payload)


def test_each_event_receives_its_own_occurrence_time() -> None:
    """若模块加载时冻结 occurred_at，事件排序会把新输入误判为同一时刻。"""

    first = AgentCallCompleted("evt_1", "lrs_1", "rnd_1", 0)
    second = AgentCallCompleted("evt_2", "lrs_1", "rnd_1", 0)

    assert first.occurred_at != second.occurred_at
