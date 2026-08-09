"""effect_runner 对 reducer 已构建的 correction 不得再 prepare。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.events import AgentCallCompleted
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall


class _RecordingTurnWorker:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def perform_agent_call(self, effect: PerformAgentCall) -> AgentCallCompleted:
        self.calls.append(effect)
        return AgentCallCompleted(
            "evt-done",
            effect.task_id,
            str(effect.round_id or ""),
            effect.generation,
            branch_id=str(effect.payload.get("branch_id", "")),
            call_id=str(effect.payload.get("call_id", "")),
            protocol=str(effect.payload.get("protocol", "json_envelope")),
            protocol_result={"report": "ok", "procedures": [], "delegations": []},
            correction_count=int(effect.payload.get("correction_count", 0)),
            messages=tuple(effect.payload.get("messages", ())),
            estimated_charge=float(effect.payload.get("estimated_charge", 0.0)),
            balance_before_reconciliation=float(effect.payload.get("credits_after_reservation", 0.0)),
            actual_model_name="physical-v1",
            pinning_supported=True,
        )


class _RecordingManager:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.handled: list[Any] = []

    async def prepare_agent_effect(self, effect: PerformAgentCall) -> PerformAgentCall:
        self.prepare_calls += 1
        raise AssertionError("correction 路径不应再调用 prepare_agent_effect")

    async def handle_runtime_event(self, event: Any) -> None:
        self.handled.append(event)


@pytest.mark.asyncio
async def test_effect_runner_skips_prepare_for_correction_turns() -> None:
    worker = _RecordingTurnWorker()
    runner = RuntimeEffectRunner(worker)
    manager = _RecordingManager()
    runner.bind_manager(manager)

    messages = (
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "task"},
        {
            "role": "user",
            "content": '你上一次返回的 swarm turn 协议无效。最小正确格式：{"report":"","procedures":[],"delegations":[]}',
        },
    )
    effect = PerformAgentCall(
        "task-1",
        "round-1",
        0,
        event_id="evt-correction",
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1:correction",
            "selector": "model:physical-v1",
            "protocol": "json_envelope",
            "messages": messages,
            "estimated_charge": 0.1,
            "credits_after_reservation": 4.9,
            "correction_count": 1,
            "max_correction_turns": 1,
            "pinning_supported": True,
        },
    )

    completed = await runner.run(effect)
    assert manager.prepare_calls == 0
    assert len(worker.calls) == 1
    assert worker.calls[0].payload["selector"] == "model:physical-v1"
    assert worker.calls[0].payload["messages"][-1]["content"].startswith("你上一次返回")
    assert isinstance(completed, AgentCallCompleted)
    assert manager.handled and manager.handled[0] is completed


@pytest.mark.asyncio
async def test_effect_runner_still_prepares_ordinary_agent_calls() -> None:
    worker = _RecordingTurnWorker()
    runner = RuntimeEffectRunner(worker)
    prepared_flag = {"done": False}

    class _Manager:
        async def prepare_agent_effect(self, effect: PerformAgentCall) -> PerformAgentCall:
            prepared_flag["done"] = True
            payload = dict(effect.payload)
            payload["messages"] = ({"role": "user", "content": "prepared"},)
            payload["selector"] = "task:mid_memory"
            return PerformAgentCall(
                effect.task_id,
                effect.round_id,
                effect.generation,
                event_id=effect.event_id,
                payload=payload,
            )

        async def handle_runtime_event(self, event: Any) -> None:
            return None

    runner.bind_manager(_Manager())
    effect = PerformAgentCall(
        "task-1",
        "round-1",
        0,
        payload={"branch_id": "branch-1", "call_id": "call-1"},
    )
    await runner.run(effect)
    assert prepared_flag["done"] is True
    assert worker.calls[0].payload["selector"] == "task:mid_memory"
