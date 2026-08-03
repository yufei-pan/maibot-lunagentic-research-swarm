from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from lunagentic_research_swarm.runtime.events import AgentCallCompleted, Event, event_from_json, event_to_json


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
        usage={"input_tokens": 123, "cache": {"hit_tokens": 45}, "models": ["model.a"]},
    )

    encoded = event_to_json(event)
    decoded = event_from_json(encoded)

    assert json.loads(encoded)["event_type"] == "AgentCallCompleted"
    assert decoded == event
    assert decoded.usage == {"input_tokens": 123, "cache": {"hit_tokens": 45}, "models": ("model.a",)}
    with pytest.raises(TypeError):
        assert decoded.usage is not None
        decoded.usage["input_tokens"] = 0
    with pytest.raises(TypeError):
        assert decoded.usage is not None
        decoded.usage["cache"]["hit_tokens"] = 0


def test_event_to_json_rejects_unregistered_event_class() -> None:
    """若未注册输入可持久化，重启后事件流会无法解码。"""

    @dataclass(frozen=True, slots=True)
    class UnregisteredEvent(Event):
        payload: str = ""

    event = UnregisteredEvent("evt_1", "lrs_1", "rnd_1", 0, payload="不可持久化")

    with pytest.raises(ValueError, match="未注册事件类型"):
        event_to_json(event)


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
