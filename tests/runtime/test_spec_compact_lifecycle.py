"""Design §12.2 / §23.2 — oversize / auto-compact failure durable lifecycle.

Deepens ``test_spec_compact_and_context`` beyond bare ``_maybe_auto_compact`` raises:
real ``prepare_agent_effect`` + ``RuntimeEffectRunner`` must finalize the branch,
never hand the turn worker a prepared PerformAgentCall, and leave formalized User1
byte-identical.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from lunagentic_research_swarm.llm.gateway import GenerationError
from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.llm.tokens import estimate_prompt_tokens
from lunagentic_research_swarm.models import BranchLifecycle, FormalizedTask
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.reducer import PerformAgentCall, PerformBranchSummary

from .test_controller_start import harness  # noqa: F401


FORMALIZED_BASE = "正式任务：compact lifecycle α"
HUGE_FORMALIZED = "不可变正式任务-" + ("超长前缀" * 4000)


class _CaptureTurnWorker:
    """Records any agent call that would have reached the LLM boundary."""

    def __init__(self) -> None:
        self.agent_calls: list[Any] = []

    async def perform_agent_call(self, effect: Any) -> Any:
        self.agent_calls.append(effect)
        raise AssertionError("oversized/failed compact must not reach turn worker")

    async def perform_procedure_batch(self, effect: Any) -> Any:
        raise AssertionError(f"unexpected procedure batch: {effect!r}")


class _OkCompactSummarizer:
    async def compact_branch(self, _request: Any) -> SummaryResult:
        return SummaryResult(True, "短摘要", "model:fake", None, None)


class _FailCompactSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def compact_branch(self, _request: Any) -> SummaryResult:
        self.calls += 1
        return SummaryResult(
            False,
            "",
            "model:fake",
            None,
            GenerationError(code="provider_error", message="compact unavailable"),
        )


def _branch_messages(*, formalized: str, mutable: str = "可变证据历史") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "swarm catalog prefix"},
        {"role": "user", "content": formalized},
        {"role": "assistant", "content": mutable},
        {"role": "assistant", "content": "更多可变历史 " + ("x" * 200)},
    ]


async def _start_root_agent_effect(manager: Any, scheduler: Any) -> tuple[str, PerformAgentCall, str]:
    result = await manager.start(objective="调查 compact lifecycle", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    agent_effect = next(
        effect for effect in reversed(scheduler.enqueued) if isinstance(effect, PerformAgentCall)
    )
    branch_id = str(agent_effect.payload["branch_id"])
    return task_id, agent_effect, branch_id


async def _drive_fail_and_finalize(
    manager: Any,
    scheduler: Any,
    agent_effect: PerformAgentCall,
    *,
    branch_id: str,
) -> _CaptureTurnWorker:
    worker = _CaptureTurnWorker()
    runner = RuntimeEffectRunner(worker)
    runner.bind_manager(manager)
    before = len(scheduler.enqueued)
    outcome = await runner.run(agent_effect)
    assert outcome is None
    assert worker.agent_calls == []

    summaries = [
        effect
        for effect in scheduler.enqueued[before:]
        if isinstance(effect, PerformBranchSummary) and effect.payload.get("branch_id") == branch_id
    ]
    assert summaries, "compact prepare failure must durable-finalize via PerformBranchSummary"
    assert summaries[0].payload["reason"] == "agent_effect_failed"
    further_agents = [
        effect
        for effect in scheduler.enqueued[before:]
        if isinstance(effect, PerformAgentCall) and effect.payload.get("branch_id") == branch_id
    ]
    assert further_agents == [], "no further PerformAgentCall may be enqueued for the failed branch"

    await manager.handle_branch_summary_effect(summaries[0])
    coordinator = manager.report_coordinators[agent_effect.task_id]
    assert coordinator.branches[branch_id].lifecycle is BranchLifecycle.FINALIZED
    assert branch_id not in manager._controllers[agent_effect.task_id].state.active_leaves
    return worker


@pytest.mark.asyncio
async def test_spec_12_2_oversize_prefix_prepare_terminates_branch_without_agent_call(
    harness,  # noqa: F811
) -> None:
    """§12.2 — immutable prefix > usable window: prepare fails → branch FINALIZED, no LLM call."""

    manager, _store, summarizer, scheduler, *_ = harness
    summarizer.compact_branch = _OkCompactSummarizer().compact_branch  # type: ignore[method-assign]

    task_id, agent_effect, branch_id = await _start_root_agent_effect(manager, scheduler)
    formalized = FormalizedTask.create(HUGE_FORMALIZED)
    controller = manager._controllers[task_id]
    controller.state = replace(controller.state, formalized_task=formalized)

    messages = _branch_messages(formalized=formalized.text)
    prefix_tokens = estimate_prompt_tokens(
        (
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": formalized.text},
        )
    ).prompt_tokens
    model_limit = max(32, prefix_tokens // 2)
    manager._runtime_limits.update(
        {
            "auto_compact_tokens": 10**9,
            "model_context_window": model_limit,
            "reserved_output_tokens": 0,
            "safety_margin_tokens": 0,
        }
    )
    manager._branches[task_id][branch_id]["messages"] = [dict(item) for item in messages]
    coordinator = manager.report_coordinators[task_id]
    if branch_id in coordinator.branches:
        coordinator.branches[branch_id].messages = [dict(item) for item in messages]

    with pytest.raises(RuntimeError, match="仍超过安全窗口|终止该分支"):
        await manager.prepare_agent_effect(agent_effect)

    # Formalized User1 stays byte-identical even if branch history was rewritten pre-raise.
    assert controller.state.formalized_task is not None
    assert controller.state.formalized_task.text.encode("utf-8") == formalized.text.encode("utf-8")
    user1 = next(
        item["content"]
        for item in manager._branches[task_id][branch_id]["messages"]
        if item.get("role") == "user"
    )
    assert user1.encode("utf-8") == formalized.text.encode("utf-8")

    # Re-seed pre-compact history so the durable runner path exercises prepare→fail independently.
    manager._branches[task_id][branch_id]["messages"] = [dict(item) for item in messages]
    if branch_id in coordinator.branches:
        coordinator.branches[branch_id].messages = [dict(item) for item in messages]
    assert branch_id in controller.state.active_leaves
    await _drive_fail_and_finalize(manager, scheduler, agent_effect, branch_id=branch_id)
    assert controller.state.formalized_task.text.encode("utf-8") == formalized.text.encode("utf-8")


@pytest.mark.asyncio
async def test_spec_23_2_auto_compact_failure_leaves_history_and_skips_agent_call(
    harness,  # noqa: F811
) -> None:
    """§23.2 — auto-compact failure: history unchanged; turn worker never sees a rewrite."""

    manager, _store, summarizer, scheduler, *_ = harness
    fail_compact = _FailCompactSummarizer()
    summarizer.compact_branch = fail_compact.compact_branch  # type: ignore[method-assign]

    task_id, agent_effect, branch_id = await _start_root_agent_effect(manager, scheduler)
    formalized = FormalizedTask.create(FORMALIZED_BASE)
    controller = manager._controllers[task_id]
    controller.state = replace(controller.state, formalized_task=formalized)

    messages = _branch_messages(formalized=formalized.text)
    before = [dict(item) for item in messages]
    manager._runtime_limits.update(
        {
            "auto_compact_tokens": 100,
            "reserved_output_tokens": 0,
            "safety_margin_tokens": 0,
        }
    )
    manager._branches[task_id][branch_id]["messages"] = [dict(item) for item in messages]
    coordinator = manager.report_coordinators[task_id]
    if branch_id in coordinator.branches:
        coordinator.branches[branch_id].messages = [dict(item) for item in messages]

    with pytest.raises(RuntimeError, match="自动 compact 失败"):
        await manager.prepare_agent_effect(agent_effect)

    assert fail_compact.calls >= 1
    assert manager._branches[task_id][branch_id]["messages"] == before
    assert all("分支压缩摘要" not in str(item.get("content", "")) for item in before)
    assert controller.state.formalized_task.text.encode("utf-8") == formalized.text.encode("utf-8")

    worker = await _drive_fail_and_finalize(manager, scheduler, agent_effect, branch_id=branch_id)
    assert worker.agent_calls == []
    # Durable terminate: no subsequent prepare can resurrect an oversized rewrite feed.
    assert branch_id not in controller.state.active_leaves
    with pytest.raises(LookupError):
        await manager.prepare_agent_effect(agent_effect)
