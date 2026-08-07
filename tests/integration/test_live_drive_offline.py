# tests/integration/test_live_drive_offline.py
from __future__ import annotations

import pytest
from fakes import FakeLLMResponse
from live_harness import drive_until_terminal

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID


@pytest.mark.asyncio
async def test_drive_until_terminal_with_scripted_terminate(runtime_harness) -> None:
    harness = runtime_harness
    await harness.start("离线驱动自测", credits=40.0, time_budget=60)
    await harness.formalize(
        "正式任务：用一次 turn 调用 core.terminate 并给出简短 report。"
    )
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "离线完成",
                "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}, "credits": 0}],
                "delegations": [],
            }
        )
    )
    status = await drive_until_terminal(harness, timeout_seconds=30)
    raw = status.get("status")
    value = raw.value if hasattr(raw, "value") else str(raw)
    assert value in {TaskStatus.COMPLETED.value, TaskStatus.COMPLETED_WITH_ERRORS.value}
