"""True offline E2E: plugin toolcall → real manager/effects → report.

No ``root_delegates`` synthetic graph, no ``launch_delegation`` stubs, no injected
grace/deadline events. Scripted FakeLLM + FakeScheduler drain through
``RuntimeEffectRunner`` (same effect path as production FairScheduler workers).
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from fakes import FakeLLMGateway, FakeLLMResponse, RuntimeHarness
from live_harness import drive_until_terminal

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID

_BRANCH_RE = re.compile(r"branch=([^;]+);")


def _branch_id_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, (list, tuple)):
        return None
    for item in reversed(list(messages)):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        if "[LRS runtime]" not in content:
            continue
        matched = _BRANCH_RE.search(content)
        if matched:
            return matched.group(1).strip()
    return None


class BranchScriptedLLM(FakeLLMGateway):
    """FIFO per branch-id (from ``[LRS runtime]``), so sibling turns stay deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.by_branch: dict[str, list[FakeLLMResponse | Exception]] = {}

    def enqueue_for(self, branch_key: str, *responses: FakeLLMResponse | Exception) -> None:
        self.by_branch.setdefault(str(branch_key), []).extend(responses)

    async def generate(self, request=None, *, selector=None, messages=None, **kwargs):
        msgs = request.messages if request is not None else messages
        branch_id = _branch_id_from_messages(msgs)
        # Root's first call has an empty history before runtime append; fall back to
        # a dedicated "root" script or the legacy FIFO queue.
        queue = None
        if branch_id and self.by_branch.get(branch_id):
            queue = self.by_branch[branch_id]
        elif branch_id is None and self.by_branch.get("__root__"):
            queue = self.by_branch["__root__"]
        if queue is not None and queue:
            # Temporarily park the shared FIFO so super().generate pops from our queue.
            previous = self.responses
            self.responses = queue
            try:
                return await super().generate(request, selector=selector, messages=messages, **kwargs)
            finally:
                self.responses = previous
        return await super().generate(request, selector=selector, messages=messages, **kwargs)


def _terminate(report: str) -> dict[str, Any]:
    return {
        "report": report,
        "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}, "credits": 0}],
        "delegations": [],
    }


@pytest.mark.asyncio
async def test_e2e_plugin_tool_retire_parent_in_frontier_still_reports(runtime_harness, plugin_module) -> None:
    """Reproduce the production wedge path through real tool → drain → report.

    1. ``start_deep_research`` tool (real manager)
    2. Root delegates two children (real materialize / retire_parent on root)
    3. First child terminates → early open_epoch freezes the sibling
    4. Sibling delegates a grandchild (retire_parent while in frontier)
    5. Grandchild terminates
    6. Must reach COMPLETED with at least one persisted report — not REPORTING forever
    """

    harness: RuntimeHarness = runtime_harness
    scripted = BranchScriptedLLM()
    harness.llm = scripted

    await harness.open()
    # Root + self-delegation targets share the integration catalog root agent id.
    plugin = plugin_module.create_plugin()
    plugin._manager = harness.manager

    gate = harness.summarizer.formalization_gate
    gate.clear()
    started = await plugin.start_deep_research(
        objective="E2E：frontier 内 retire_parent 后仍须产出报告",
        time_budget_seconds=30,
        effort_level=1.0,
        stream_id=harness.stream_id,
    )
    assert started.get("success") is True, started
    harness.task_id = str(started["task_id"])
    harness.round_id = str((await harness.manager.status(harness.task_id))["round_id"])
    harness.coordinator = harness.manager.report_coordinators.get(harness.task_id)

    await harness.formalize("正式任务：两条分支取证后汇总；验证退休父节点不卡住 REPORTING。")
    root_id = next(iter(harness.coordinator.branches))

    # Root: spawn two siblings.
    scripted.enqueue_for(
        root_id,
        FakeLLMResponse(
            payload={
                "report": "拆成 A/B 两条线",
                "procedures": [],
                "delegations": [
                    {
                        "agent_id": "builtin.quick_thinker",
                        "task": "轨道 A：收集证据并终结",
                        "credits": 40,
                    },
                    {
                        "agent_id": "builtin.quick_thinker",
                        "task": "轨道 B：先继续再交孙子分支",
                        "credits": 40,
                    },
                ],
            }
        ),
    )
    # Child scripts use branch id suffixes after materialize (:1 / :2).
    # We do not know the root branch id prefix until formalize; register by pattern
    # after first drain wave by using wildcard queues filled once children exist.
    # Instead: respond based on assignment text markers in runtime block.
    scripted.enqueue_for(
        "__pending_a__",
        FakeLLMResponse(payload=_terminate("A 证据已齐，终结本叶")),
    )
    scripted.enqueue_for(
        "__pending_b__",
        FakeLLMResponse(
            payload={
                "report": "B 交棒给下一跳",
                "procedures": [],
                "delegations": [
                    {
                        "agent_id": "builtin.quick_thinker",
                        "task": "轨道 B 子分支：收尾并终结",
                        "credits": 20,
                    },
                ],
            }
        ),
    )
    scripted.enqueue_for(
        "__pending_b_child__",
        FakeLLMResponse(payload=_terminate("B 子分支收尾完成")),
    )

    # Remap pending queues onto real branch ids as soon as children materialize.
    original_generate = scripted.generate

    async def _generate(request=None, *, selector=None, messages=None, **kwargs):
        msgs = request.messages if request is not None else messages
        text = ""
        if isinstance(msgs, (list, tuple)):
            for item in msgs:
                if isinstance(item, dict):
                    text += str(item.get("content") or "")
        branch_id = _branch_id_from_messages(msgs) or ""
        if branch_id and branch_id not in scripted.by_branch:
            if "轨道 A" in text and scripted.by_branch.get("__pending_a__"):
                scripted.by_branch[branch_id] = scripted.by_branch.pop("__pending_a__")
            elif "轨道 B 子分支" in text and scripted.by_branch.get("__pending_b_child__"):
                scripted.by_branch[branch_id] = scripted.by_branch.pop("__pending_b_child__")
            elif "轨道 B" in text and scripted.by_branch.get("__pending_b__"):
                scripted.by_branch[branch_id] = scripted.by_branch.pop("__pending_b__")
        return await original_generate(request, selector=selector, messages=messages, **kwargs)

    scripted.generate = _generate  # type: ignore[method-assign]

    status = await drive_until_terminal(harness, timeout_seconds=60, auto_advance_clock=True)
    raw = status.get("status")
    value = raw.value if hasattr(raw, "value") else str(raw)
    assert value in {TaskStatus.COMPLETED.value, TaskStatus.COMPLETED_WITH_ERRORS.value}, status

    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None
    assert layer.reports, "E2E must persist at least one report (not wedge in REPORTING)"
    assert layer.summaries, "expected branch summaries"
    kinds = {str(item.get("kind")) for item in layer.summaries}
    assert "BRANCH_FINAL" in kinds or "CHECKPOINT" in kinds
