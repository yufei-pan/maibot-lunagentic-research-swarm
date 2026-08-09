"""针对本轮 review 修复的回归测试。

每个用例对应一处“规格要求存在、实现却缺失”的行为，防止再次退化。
"""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CoreProcedureDecision
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.reporting import render_report
from lunagentic_research_swarm.models import ReportKind, SummaryKind
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.events import AgentCallCompleted, ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    PerformAgentCall,
    PerformBranchSummary,
    RuntimeState,
    reduce_event,
)

from .test_controller_start import harness  # noqa: F401


# --- spec §8.2/§9.1：agent 的 report 必须进入可变 history ---------------------


@pytest.mark.asyncio
async def test_agent_report_enters_branch_history_before_procedure_results() -> None:
    executor = ProcedureExecutor(catalog=None, api=None)
    effect = type(
        "E",
        (),
        {
            "task_id": "t",
            "round_id": "r",
            "generation": 0,
            "event_id": "e",
            "payload": {
                "branch_id": "br",
                "call_id": "c",
                "messages": ({"role": "user", "content": "任务"},),
                "report": "我核对了三个来源，结论是 X。",
                "requests": (),
            },
        },
    )()

    completed = await executor.invoke_many(effect)

    assert completed.parent_messages[-1] == {
        "role": "assistant",
        "content": "我核对了三个来源，结论是 X。",
    }


# --- spec §17.1：add_research_context 广播到每个活动分支的下一次调用 ---------


@pytest.mark.asyncio
async def test_supplied_context_is_injected_into_the_next_agent_call(harness) -> None:  # noqa: F811
    manager, _store, _summarizer, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    await manager.add_context(task_id, "新事实：供应商已更换")

    agent_effect = next(
        effect for effect in reversed(scheduler.enqueued) if isinstance(effect, PerformAgentCall)
    )
    prepared = await manager.prepare_agent_effect(agent_effect)

    contents = [str(message.get("content", "")) for message in prepared.payload["messages"]]
    assert any("新事实：供应商已更换" in item for item in contents)
    # 只注入一次。
    branch_id = prepared.payload["branch_id"]
    assert manager._branches[task_id][branch_id]["pending_context"] == []


# --- spec §8.2：每次调用追加一个新的 runtime 块，历史只追加不改写 -------------


@pytest.mark.asyncio
async def test_runtime_block_is_appended_per_call_and_history_stays_append_only(harness) -> None:  # noqa: F811
    """每次调用在末尾追加一个新块；已发送的消息一个都不动。

    上一次请求必须是这一次请求的逐字节前缀，provider 的前缀缓存才能覆盖整条链；
    块内自带「只有最后一个有效」的说明，历史里的旧块因此不会误导模型。
    """

    manager, _store, _summarizer, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    agent_effect = next(
        effect for effect in reversed(scheduler.enqueued) if isinstance(effect, PerformAgentCall)
    )
    branch_id = str(agent_effect.payload.get("branch_id") or "")

    first = await manager.prepare_agent_effect(agent_effect)
    # 让上一 turn 的输出落进分支历史，模拟真实的 turn 结算。
    manager._branches[task_id][branch_id]["messages"] = [
        *first.payload["messages"],
        {"role": "assistant", "content": "上一 turn 的 report"},
    ]
    second = await manager.prepare_agent_effect(agent_effect)

    def blocks(payload):
        return [
            str(item["content"])
            for item in payload["messages"]
            if str(item.get("content", "")).startswith("[LRS runtime]")
        ]

    assert len(blocks(first.payload)) == 1
    assert "turn=1" in blocks(first.payload)[0]
    # 第二次调用保留了第一次的块，并在末尾追加了新的一个。
    assert len(blocks(second.payload)) == 2
    assert "turn=2" in blocks(second.payload)[-1]
    assert str(second.payload["messages"][-1]["content"]).startswith("[LRS runtime]")
    assert "只有最后一个" in blocks(second.payload)[-1]
    previous = [dict(item) for item in first.payload["messages"]]
    current = [dict(item) for item in second.payload["messages"]]
    assert current[: len(previous)] == previous, "上一次请求必须是这一次请求的逐字节前缀"


# --- spec §14.3 + §11.7：全部委派被拒时不悬空、不销毁 credits ----------------


def _rejected_event(*, delegations, live_agent_ids, parent_depth=0, max_depth=32):
    return ProcedureBatchCompleted(
        "evt",
        "task",
        "round",
        0,
        branch_id="parent",
        call_id="call",
        result_id="result",
        results=(),
        controls=CoreProcedureDecision(),
        delegations=delegations,
        credits_after=10.0,
        parent_messages=({"role": "user", "content": "任务"},),
        parent_depth=parent_depth,
        live_agent_ids=live_agent_ids,
        max_branch_depth=max_depth,
        agent_calls_started=0,
    )


def test_depth_limit_rejecting_every_edge_finalizes_the_parent_instead_of_stranding_it() -> None:
    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 10.0}),
        _rejected_event(
            delegations=({"agent_id": "a", "task": "t", "credits": 4.0},),
            live_agent_ids=("a",),
            parent_depth=32,
            max_depth=32,
        ),
    )

    summaries = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    assert len(summaries) == 1
    assert summaries[0].payload["branch_id"] == "parent"
    assert summaries[0].payload["reason"] == "branch_depth_exceeded"
    # 深度上限是确定性的，不重试。
    assert not [effect for effect in transition.effects if isinstance(effect, PerformAgentCall)]


def test_partially_rejected_edges_do_not_take_credits_from_live_siblings() -> None:
    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 10.0}),
        _rejected_event(
            delegations=(
                {"agent_id": "gone", "task": "t1", "credits": 5.0},
                {"agent_id": "live", "task": "t2", "credits": 5.0},
            ),
            live_agent_ids=("live",),
        ),
    )

    edges = [
        effect
        for effect in transition.effects
        if isinstance(effect, PerformBranchSummary) and effect.payload.get("edge_finalization")
    ]
    assert len(edges) == 1 and edges[0].payload["credits"] == 0.0
    children = [
        effect
        for effect in transition.effects
        if effect.payload.get("action") == "materialize_child"
    ]
    # 存活兄弟拿到自己请求的 5.0，未被死边稀释；剩余 5.0 随父节点退休回 pool。
    assert len(children) == 1
    assert children[0].payload["credits"] == pytest.approx(5.0)
    assert children[0].payload["pool_return"] == pytest.approx(5.0)


# --- spec §23：worker 崩溃必须终结分支，而不是让它永远悬空 -------------------


@pytest.mark.asyncio
async def test_crashed_agent_effect_finalizes_the_branch(harness) -> None:  # noqa: F811
    manager, _store, _summarizer, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    agent_effect = next(
        effect for effect in reversed(scheduler.enqueued) if isinstance(effect, PerformAgentCall)
    )
    branch_id = agent_effect.payload["branch_id"]

    class _Boom:
        async def perform_agent_call(self, effect):
            raise RuntimeError("provider 崩溃")

    runner = RuntimeEffectRunner(_Boom())
    runner.bind_manager(manager)
    before = len(scheduler.enqueued)
    await runner.run(agent_effect)

    # 崩溃不再被静默吞掉：分支被显式终结，而不是永远留在 active_leaves。
    summaries = [
        effect
        for effect in scheduler.enqueued[before:]
        if isinstance(effect, PerformBranchSummary) and effect.payload.get("branch_id") == branch_id
    ]
    assert summaries and summaries[0].payload["reason"] == "agent_effect_failed"


# --- spec §7.5：省略 time_budget_seconds 时复用 Task 保存的间隔 --------------


@pytest.mark.asyncio
async def test_continue_reuses_the_stored_time_budget(harness) -> None:  # noqa: F811
    manager, _store, _summarizer, _scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=600)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)

    assert await manager._stored_time_budget(task_id) == 600


# --- spec §13.3：中间报告也要带 token/cache/credits/失败统计 -----------------


def test_intermediate_report_header_carries_statistics() -> None:
    text = render_report(
        kind=ReportKind.INTERMEDIATE,
        body="进展",
        task_id="lrs_1",
        round_id="rnd_1",
        epoch=2,
        running_branch_count=1,
        queued_branch_count=0,
        unavailable_count=0,
        elapsed_seconds=12.0,
        next_interval_seconds=120,
        credit_balance=5.0,
        credit_pool=1.0,
        stats={"prompt_tokens": 100, "cache_hit_tokens": 40, "error_count": 1},
    )

    assert "中间报告" in text
    assert "prompt_tokens=100" in text
    assert "cache_hit_tokens=40" in text
    assert "error_count=1" in text


# --- spec §13.5：广播给“其他”活动分支，不回灌自己 --------------------------


def test_branch_does_not_receive_its_own_summary_broadcast() -> None:
    from lunagentic_research_swarm.reporting import CoverageSummary

    class _Coordinator:
        pending_summary_messages = None

    from lunagentic_research_swarm.runtime.epochs import ReportCoordinator

    coordinator = object.__new__(ReportCoordinator)
    coordinator._seen = {}
    coordinator._summaries = [
        CoverageSummary("s1", "br_self", SummaryKind.CHECKPOINT, 1, "自己的摘要", 1.0),
        CoverageSummary("s2", "br_other", SummaryKind.CHECKPOINT, 1, "兄弟的摘要", 2.0),
    ]

    messages = ReportCoordinator.pending_summary_messages(coordinator, "br_self")

    contents = [item["content"] for item in messages]
    assert len(contents) == 1
    assert "兄弟的摘要" in contents[0]
    assert "自己的摘要" not in contents[0]
    # 广播必须自带出处，否则接收方会把兄弟分支的结论当成自己做过的工作。
    assert "br_other" in contents[0]
    assert "其它分支" in contents[0]
    # OpenAI 兼容服务端对 `name` 有 ^[a-zA-Z0-9_-]+$ 限制；带冒号的名字会被 400 拒绝。
    assert all("name" not in item for item in messages)


# --- spec §9.2：纠正 turn 使用同一协议 --------------------------------------


# --- 暂存 checkpoint 的释放必须与普通委派走同一套上限判定 -------------------


def _held_release_harness(harness, *, limits: dict[str, int]):
    manager, _store, _summarizer, scheduler, *_ = harness
    manager._runtime_limits.update(limits)
    return manager, scheduler


@pytest.mark.asyncio
async def test_held_release_applies_branch_depth_limit(harness) -> None:  # noqa: F811
    manager, scheduler = _held_release_harness(harness, limits={"max_branch_depth": 1})
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    parent_id = (await manager.status(task_id))["active_leaves"][0]["branch_id"]
    manager._branches[task_id][parent_id]["depth"] = 1

    scheduler.enqueued.clear()
    await manager._release_held_delegations(
        task_id,
        parent_id,
        ({"agent_id": "child", "task": "太深了", "credits": 10.0},),
    )

    # 深度上限对暂存路径同样生效：不再无条件物化子分支。
    assert not [
        item for item in scheduler.enqueued if item.payload.get("action") == "materialize_child"
    ]
    edges = [
        item
        for item in scheduler.enqueued
        if isinstance(item, PerformBranchSummary) and item.payload.get("edge_finalization")
    ]
    assert edges and edges[0].payload["reason"] == "branch_depth_exceeded"
    parent_final = [
        item
        for item in scheduler.enqueued
        if isinstance(item, PerformBranchSummary) and item.payload.get("branch_id") == parent_id
    ]
    assert parent_final and parent_final[0].payload["reason"] == "branch_depth_exceeded"


@pytest.mark.asyncio
async def test_held_release_rejects_unknown_agent_and_retries_parent(harness) -> None:  # noqa: F811
    manager, scheduler = _held_release_harness(harness, limits={})
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    parent_id = (await manager.status(task_id))["active_leaves"][0]["branch_id"]

    scheduler.enqueued.clear()
    await manager._release_held_delegations(
        task_id,
        parent_id,
        ({"agent_id": "provider.removed", "task": "已下线", "credits": 10.0},),
    )

    assert not [
        item for item in scheduler.enqueued if item.payload.get("action") == "materialize_child"
    ]
    retries = [
        item
        for item in scheduler.enqueued
        if isinstance(item, PerformAgentCall) and item.payload.get("branch_id") == parent_id
    ]
    assert len(retries) == 1
    assert "agent_unavailable" in retries[0].payload["appended_messages"][0]["content"]


@pytest.mark.asyncio
async def test_held_release_consumes_the_task_call_budget(harness) -> None:  # noqa: F811
    from lunagentic_research_swarm.runtime.reducer import NotifyToolWaiter

    manager, scheduler = _held_release_harness(harness, limits={})
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await manager.wait_idle(task_id)
    parent_id = (await manager.status(task_id))["active_leaves"][0]["branch_id"]
    before = manager._controllers[task_id].state.agent_calls_started

    scheduler.enqueued.clear()
    await manager._release_held_delegations(
        task_id,
        parent_id,
        ({"agent_id": "child", "task": "a", "credits": 10.0},),
    )
    for item in [
        entry
        for entry in scheduler.enqueued
        if isinstance(entry, NotifyToolWaiter) and entry.payload.get("action") == "materialize_child"
    ]:
        await manager.materialize_child_effect(item)

    # 暂存路径过去完全绕过 max_agent_calls_per_task；现在与普通委派一样计入。
    assert manager._controllers[task_id].state.agent_calls_started == before + 1


def test_correction_turn_keeps_the_original_protocol() -> None:
    event = AgentCallCompleted(
        "evt",
        "task",
        "round",
        0,
        branch_id="br",
        call_id="call",
        actual_model_name="physical-1",
        actual_charge=1.0,
        estimated_charge=1.0,
        balance_before_reconciliation=10.0,
        usage={"prompt_tokens": 10, "completion_tokens": 1, "cache_hit_tokens": 0, "cache_miss_tokens": 10},
        protocol="native_tools",
        protocol_error={"message": "bad", "errors": ()},
        max_correction_turns=1,
    )

    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"br": 10.0}),
        event,
    )

    calls = [effect for effect in transition.effects if isinstance(effect, PerformAgentCall)]
    assert len(calls) == 1
    assert calls[0].payload["protocol"] == "native_tools"
    # 纠正 turn 计入调用次数。
    assert transition.next_state.agent_calls_started == 1


def test_checkpoint_without_delegations_ends_the_branch_instead_of_holding_it() -> None:
    """checkpoint 不产生待执行工作，因此不能把分支挂到一个无事可放的 epoch 边界。"""

    event = ProcedureBatchCompleted(
        "evt",
        "task",
        "round",
        0,
        branch_id="br",
        call_id="call",
        result_id="result",
        results=(),
        controls=CoreProcedureDecision(checkpoint=True),
        delegations=(),
        credits_after=5.0,
        parent_messages=({"role": "user", "content": "任务"},),
    )

    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"br": 5.0}),
        event,
    )

    summaries = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    assert len(summaries) == 1
    assert summaries[0].payload["reason"] == "no_further_work"
    assert summaries[0].payload["held_delegations"] == ()


def test_checkpoint_with_delegations_still_holds_them_for_the_epoch_boundary() -> None:
    event = ProcedureBatchCompleted(
        "evt",
        "task",
        "round",
        0,
        branch_id="br",
        call_id="call",
        result_id="result",
        results=(),
        controls=CoreProcedureDecision(checkpoint=True),
        delegations=({"agent_id": "a", "task": "t", "credits": 1.0},),
        credits_after=5.0,
        parent_messages=({"role": "user", "content": "任务"},),
    )

    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"br": 5.0}),
        event,
    )

    summaries = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    assert len(summaries) == 1
    assert summaries[0].payload["reason"] == "checkpoint"
    assert len(summaries[0].payload["held_delegations"]) == 1


def test_correction_turn_is_disabled_when_max_correction_turns_is_zero() -> None:
    event = AgentCallCompleted(
        "evt",
        "task",
        "round",
        0,
        branch_id="br",
        call_id="call",
        actual_model_name="physical-1",
        actual_charge=1.0,
        estimated_charge=1.0,
        balance_before_reconciliation=10.0,
        usage={"prompt_tokens": 10, "completion_tokens": 1, "cache_hit_tokens": 0, "cache_miss_tokens": 10},
        protocol_error={"message": "bad", "errors": ()},
        max_correction_turns=0,
    )

    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"br": 10.0}),
        event,
    )

    summaries = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    assert summaries and summaries[0].payload["reason"] == "protocol_invalid"
