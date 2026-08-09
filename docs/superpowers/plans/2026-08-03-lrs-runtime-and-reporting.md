# 麦麦深度调查组：运行时与报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已冻结的基础契约上实现可运行的异步调查图：credits 账本、单一 reducer、公平有界调度、turn/Procedure 执行、暂停/继续/停止、checkpoint/grace 报告与 Maisaka 交付。

**Architecture:** 每个 Task 有一个 `TaskController` 和串行事件 inbox，worker 只返回 event；reducer 先生成 SQLite commands 和后续 effects，controller 先提交事务再发布新内存状态与调度 effect。普通 LLM、总结器和 Procedure 由 task-aware scheduler 限流；报告由显式 epoch/frontier 状态机生成，不通过暂停全图实现。

**Tech Stack:** 第一阶段全部依赖、Python `asyncio`、stdlib `heapq`/`deque`、MaiBot `@Tool`/Maisaka capability、pytest-asyncio、Hypothesis。

## Global Constraints

- 必须先完成 `docs/superpowers/plans/2026-08-03-lrs-foundation-and-contracts.md`，并复用其中已经冻结的类型和接口名称。
- 不修改 Host/SDK；首发 stop 只取消本地等待、禁止新调度并丢弃 generation 不匹配的迟到结果，不宣称取消了上游调用。
- 所有权威状态只由 reducer 产生，worker、scheduler、Tool handler 和 outbox worker 不得直接修改 `TaskSnapshot`/`RoundSnapshot`。
- SQLite 关键事务必须成功后才能启动 effect；事务失败时 Task 显式 `FAILED`，不保留“内存成功”。
- 默认时间预算 120 秒、grace 60 秒、暂停过期 1200 秒；时间预算只控制报告节奏，不终止研究。
- credits 单位和算法必须逐条遵守批准规格：零余额可委派、负余额才终止、pool 只由 continue 触发再分配、Procedure/总结器不扣研究 credits。
- Turn 顺序固定为：拒绝迟到 generation → 核销 LLM → 协议校验/最多一次纠正 → 普通 Procedure → continue redistribution barrier → control Procedure → terminate/negative → checkpoint hold → children。
- 每次报告合成开始时冻结 report kind；使用任何 checkpoint 输入一定是 intermediate，不能在生成途中“升级”为 final。
- 正式任务描述通过 delegation、compact、checkpoint、report、continue/new round 后必须逐字节相同。
- 不保存/传递 reasoning；协议纠正以实际物理模型名固定到同一模型，不可固定时明确终结，不退回 task alias。
- 本阶段集成测试使用 deterministic fake LLM/Procedure/clock；不依赖网络或真实 Host。

---

### Task 1: 实现 credits 数学与不可变账本条目

**Files:**
- Create: `lunagentic_research_swarm/runtime/credits.py`
- Modify: `lunagentic_research_swarm/models.py`
- Test: `tests/runtime/test_credits.py`
- Test: `tests/runtime/test_credits_properties.py`

**Interfaces:**
- Consumes: 第一阶段 `PriceCatalog.charge_*()` 与 `TokenUsage`。
- Produces: `CreditBalance`、`CreditLedgerEntry`、`reserve_input()`、`reconcile_usage()`、`allocate_children()`、`settle_branch()`、`redistribute_pool()`、`assert_task_credit_equation()`。

- [ ] **Step 1: 写八个批准场景的失败测试**

```python
from lunagentic_research_swarm.runtime.credits import (
    allocate_children,
    assert_task_credit_equation,
    redistribute_pool,
)


def test_root_exact_allocation() -> None:
    result = allocate_children(100.0, [("A", 50.0), ("B", 25.0), ("C", 25.0)])
    assert result.allocations == {"A": 50.0, "B": 25.0, "C": 25.0}
    assert result.returned_to_pool == 0.0


def test_oversubscription_scales_proportionally() -> None:
    result = allocate_children(2.0, [("A", 2.0), ("B", 1.0), ("C", 1.0)])
    assert result.allocations == {"A": 1.0, "B": 0.5, "C": 0.5}


def test_zero_balance_still_launches_zero_credit_children() -> None:
    result = allocate_children(0.0, [("A", 9.0), ("B", 1.0)])
    assert result.allocations == {"A": 0.0, "B": 0.0}
    assert result.launch_allowed


def test_negative_balance_cannot_launch_children() -> None:
    result = allocate_children(-0.01, [("A", 0.0)])
    assert not result.launch_allowed
    assert result.allocations == {}


def test_negative_pool_is_dormant_until_continue() -> None:
    # 分支 -40 结算到既有 0.2 后只是 pool=-39.8，不影响别的叶子。
    assert 0.2 + (-40.0) == pytest.approx(-39.8)


def test_continue_distributes_signed_pool_proportionally() -> None:
    result = redistribute_pool(pool=-12.0, adjustment=0.0, leaves={"A": 9.0, "B": 3.0})
    assert result.balances == {"A": 0.0, "B": 0.0}
    assert result.pool_after == 0.0


def test_continue_distributes_evenly_when_all_active_leaves_are_zero() -> None:
    assert redistribute_pool(6.0, -3.0, {"A": 0.0, "B": 0.0}).balances == {"A": 1.5, "B": 1.5}
    assert redistribute_pool(-6.0, 0.0, {"A": 0.0, "B": 0.0}).balances == {"A": -3.0, "B": -3.0}


def test_no_active_leaf_restart_funds_can_be_negative() -> None:
    result = redistribute_pool(-5.0, 2.0, {})
    assert result.restart_balance == -3.0
    assert not result.can_start_root
```

- [ ] **Step 2: 运行 credits 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_credits.py -v`

Expected: FAIL because `runtime.credits` does not exist。

- [ ] **Step 3: 实现分配、结算和有符号再分配**

使用 `math.fsum` 计算总和并在最终一项吸收浮点余差；所有请求 credits 必须有限且 `>=0`。算法固定如下：

```python
def allocate_children(balance: float, requests: Sequence[tuple[str, float]]) -> AllocationResult:
    if balance < 0:
        return AllocationResult({}, 0.0, launch_allowed=False, scale=0.0)
    requested_total = math.fsum(value for _, value in requests)
    if requested_total == 0:
        return AllocationResult({key: 0.0 for key, _ in requests}, balance, True, 1.0)
    scale = min(1.0, balance / requested_total)
    allocations = _scaled_with_last_item_remainder(requests, scale, min(balance, requested_total))
    returned = balance - math.fsum(allocations.values())
    return AllocationResult(allocations, returned, True, scale)
```

`settle_branch(balance, pool)` 无论 balance 正/负都返回 `pool + balance`。`redistribute_pool(pool, adjustment, leaves)` 先 `available=pool+adjustment`；有 active leaves 时，正余额之和大于 0 就按正余额比例把 signed available 加到所有正余额叶子，零/负叶子权重 0；所有叶子都没有正余额时平均加到所有 active leaves。完成后 pool=0。无 active leaves 时把 available 保存为新 pool；`>=0` 可启动新 root，`<0` 返回 `task_finished_insufficient_funds`，但负 pool 仍持久化，后续正 adjustment 可以偿还。

任何 branch 以 terminate、natural no-further-work、negative credit、missing edge 或结构上限结束时，都必须在 terminal summary 或 `summary_unavailable` 终结记录成功提交的同一 transaction 把最终 balance（正、零或负）结算进 dormant pool 并将 branch 标 FINALIZED；pool 变化本身绝不触发 redistribution。

- [ ] **Step 4: 实现 reserve/reconcile ledger**

`reserve_input()` 生成两个同 transaction commands：`llm_usage(reconciliation_status="reserved")` 与 `credit_ledger(entry_kind="input_reservation", amount=-estimated_charge)`。调用完成后 `reconcile_usage()` 以实际 `model_name`/token 计算 whole-call actual charge，写 `actual_charge` 与 `adjustment=estimated_charge-actual_charge`，并写 ledger adjustment。失败且 Host 没 usage 时不释放 reservation，标 `estimated_unreconciled`；总结器 role 写 usage telemetry 但不写 research credit ledger。

- [ ] **Step 5: 写 Hypothesis 守恒性质并运行**

```python
@given(
    balance=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    requests=st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False), max_size=8),
)
def test_allocation_conserves_parent_balance(balance, requests) -> None:
    result = allocate_children(balance, [(str(i), value) for i, value in enumerate(requests)])
    assert math.fsum(result.allocations.values()) + result.returned_to_pool == pytest.approx(balance)
    assert all(value >= 0 for value in result.allocations.values())


@given(
    initial=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    adjustments=st.lists(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False), max_size=20),
    charges=st.lists(st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_infinity=False), max_size=20),
)
def test_task_credit_equation(initial, adjustments, charges) -> None:
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
```

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_credits.py tests/runtime/test_credits_properties.py -v`

Expected: 全部 PASS。

- [ ] **Step 6: 提交 credits 引擎**

```bash
git add lunagentic_research_swarm/runtime/credits.py lunagentic_research_swarm/models.py tests/runtime/test_credits.py tests/runtime/test_credits_properties.py
git commit -m "feat: implement LRS credit accounting"
```

### Task 2: 实现纯 reducer 与 transaction-before-effect 驱动

**Files:**
- Create: `lunagentic_research_swarm/runtime/reducer.py`
- Modify: `lunagentic_research_swarm/runtime/events.py`
- Test: `tests/runtime/test_reducer.py`
- Test: `tests/runtime/test_reducer_persistence.py`

**Interfaces:**
- Consumes: event union、`StoreCommand`、credits helpers。
- Produces: `reduce_event(state, event) -> Transition`、`Transition(next_state, commands, effects)`、显式 `Effect` union。

- [ ] **Step 1: 写 lifecycle matrix 与 persistence ordering 失败测试**

```python
@pytest.mark.parametrize(
    ("status", "event_type", "expected"),
    [
        ("FORMALIZING", "FormalizationSucceeded", "RUNNING"),
        ("FORMALIZING", "FormalizationFailed", "FAILED"),
        ("RUNNING", "PauseRequested", "PAUSING"),
        ("PAUSING", "AllInflightSettled", "PAUSED"),
        ("RUNNING", "StopRequested", "STOPPED"),
        ("REPORTING", "ReportCompleted", "RUNNING"),
        ("FINALIZING", "FinalReportCompleted", "COMPLETED"),
        ("FINALIZING", "FinalReportFailed", "COMPLETED_WITH_ERRORS"),
    ],
)
def test_lifecycle_transitions(status, event_type, expected, state_factory, event_factory) -> None:
    transition = reduce_event(state_factory(status), event_factory(event_type))
    assert transition.next_state.status.value == expected
```

```python
@pytest.mark.asyncio
async def test_effect_is_not_launched_when_transaction_fails(controller_factory) -> None:
    controller, store, executor = controller_factory(store_fails=True)
    await controller.submit(event_factory("FormalizationSucceeded"))
    await controller.drain_once()
    assert executor.launched == []
    assert controller.state.status.value == "FAILED"
```

- [ ] **Step 2: 运行 reducer 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py -v`

Expected: FAIL because reducer is missing。

- [ ] **Step 3: 实现纯 transition 与状态白名单**

`reduce_event()` 不 await、不访问 ctx/store/time/random；ID/time 都由 event 携带。它先验证 `task_id/round_id/generation`，旧 generation 返回 `Transition.ignored(reason="late_generation")`。每个 status 只接受明确事件；非法组合返回 `LRSError("invalid_state")` effect 供 Tool caller，不能猜测转移。

`Effect` 至少包括：`PerformFormalization`、`PerformAgentCall`、`PerformProcedureBatch`、`PerformBranchSummary`、`PerformTaskSummary`、`OpenReportEpoch`、`DeliverOutbox`、`ArmDeadline`、`ArmPauseExpiry`、`ReleaseRawContext`、`NotifyToolWaiter`。Reducer 只描述 effect payload，不创建 asyncio task。

`TaskController._apply(event)` 固定顺序：

```python
transition = reduce_event(self.state, event)
try:
    await self.store.transact(transition.commands)
except Exception as exc:
    await self._fail_after_storage_error(event, exc)
    return
self.state = transition.next_state
for effect in transition.effects:
    await self.scheduler.enqueue(effect)
```

若关键 transaction 失败，`_fail_after_storage_error` 尝试单条 best-effort FAILED update；第二次也失败则停止 controller 并向 plugin health 登记，不能继续启动任何 work。

- [ ] **Step 4: 补全 terminal/new-round 规则**

六个 round 终态都允许 ContinueRequested；active leaves 走 redistribution barrier，no leaves 走 new round 或 insufficient。STOPPED generation 立即递增并释放 raw context；任何旧 generation Agent/Summary completion 被忽略且不核销到新 round。PAUSED 超时只转 EXPIRED，不总结、不反馈提醒（反馈逻辑第三阶段）。

- [ ] **Step 5: 运行 reducer 测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/runtime/reducer.py lunagentic_research_swarm/runtime/events.py tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py
git commit -m "feat: add transactional LRS event reducer"
```

### Task 3: 实现 task-aware 有界公平调度器

**Files:**
- Create: `lunagentic_research_swarm/runtime/scheduler.py`
- Test: `tests/runtime/test_scheduler.py`

**Interfaces:**
- Consumes: reducer `Effect`；每个 effect 标注 `task_id/kind/priority/generation`。
- Produces: `FairScheduler.start/close/enqueue/cancel_generation/stats`，按 task round-robin 启动 worker。

- [ ] **Step 1: 写并发、公平、优先级与暂停失败测试**

```python
@pytest.mark.asyncio
async def test_wide_task_cannot_starve_other_task(fake_worker) -> None:
    scheduler = FairScheduler(global_llm=2, per_task_llm=2, per_task_procedure=2, worker=fake_worker)
    await scheduler.start()
    for index in range(20):
        await scheduler.enqueue(effect("A", f"a{index}", kind="agent"))
    await scheduler.enqueue(effect("B", "b0", kind="agent"))
    await fake_worker.wait_started(3)
    assert "b0" in fake_worker.started[:3]


@pytest.mark.asyncio
async def test_control_barrier_precedes_child_launch(fake_worker) -> None:
    await scheduler.enqueue(effect("A", "child", kind="agent", priority="normal"))
    await scheduler.enqueue(effect("A", "continue", kind="control", priority="barrier"))
    await fake_worker.wait_started(1)
    assert fake_worker.started[0] == "continue"
```

- [ ] **Step 2: 运行 scheduler 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_scheduler.py -v`

Expected: FAIL because scheduler is missing。

- [ ] **Step 3: 实现三类容量与 round-robin 队列**

维护 `priority -> OrderedDict[task_id, deque[Effect]]`，每次从最高非空 priority 轮转一个 task，再从该 task 取一个 effect。容量规则：普通 agent 同时占 global LLM + per-task LLM；summarizer 只占 global LLM；Procedure 只占 per-task Procedure。默认分别 16/8/16，来自 live config；收紧值只阻止新启动，不取消在途。

`pause_task(task_id)` 阻止新的 agent/summarizer effect，但允许已经开始的普通 LLM完成和该 turn 的 Procedure effect；`stop_generation()` 移除未启动 effect并给 worker generation token，无法取消 Host call 时结果仍由 reducer 丢弃。report/stop/continue barrier priority 高于 ordinary child；fairness 仍在同 priority 内生效。

- [ ] **Step 4: 实现可观测统计与正常关闭**

`stats()` 返回全局/每 task active、queued、kind、wait latency，不包含 prompt。`close()` 停止取新 work、取消本地 queued futures、等待已启动 wrapper；wrapper 对 Host call 不做 HTTP retry。

- [ ] **Step 5: 运行 scheduler 测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_scheduler.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/runtime/scheduler.py tests/runtime/test_scheduler.py
git commit -m "feat: add fair bounded swarm scheduler"
```

### Task 4: 实现 Procedure 执行器与三项核心控制 Procedure

**Files:**
- Create: `lunagentic_research_swarm/procedures/executor.py`
- Create: `lunagentic_research_swarm/procedures/core.py`
- Test: `tests/procedures/test_executor.py`
- Test: `tests/procedures/test_core.py`

**Interfaces:**
- Consumes: frozen Procedure catalog、`ctx.api.call`、`SummarizerService`。
- Produces: `ProcedureExecutor.invoke_many()`、纯函数 `split_procedure_requests()`、`CoreProcedureDecision(compact, checkpoint, terminate)`、`ProcedureBatchCompleted` event；余额/control precedence仍由 reducer决定。

- [ ] **Step 1: 写调用顺序、timeout 与 control precedence 失败测试**

```python
@pytest.mark.asyncio
async def test_executor_runs_only_ordinary_batch_and_returns_event(executor) -> None:
    ordinary, controls = split_procedure_requests(
        [request("builtin.search", {"q": "x"}), request("core.checkpoint", {})]
    )
    event = await executor.invoke_many(effect_with(ordinary))
    assert [item.procedure_id for item in event.results] == ["builtin.search"]
    assert controls.checkpoint


@pytest.mark.asyncio
async def test_terminate_dominates_other_control_procedures() -> None:
    ordinary, controls = split_procedure_requests(
        [request("core.compact", {}), request("core.terminate", {"reason": "done"})]
    )
    assert ordinary == []
    assert controls.terminate
    assert not controls.compact
    assert controls.ignored_controls == ["core.compact"]
```

- [ ] **Step 2: 运行 Procedure 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_executor.py tests/procedures/test_core.py -v`

Expected: FAIL because executor/core modules are missing。

- [ ] **Step 3: 实现外部/内置统一调用与 retry 规则**

每次调用创建稳定 `request_id`，按 provider snapshot 的完整 `plugin_id.invoke_procedure` + version 1 调用。scoped metadata 只含 task/round/branch/turn/agent，不含其他 branch context。结果用 `ProcedureResult.model_validate()`；invalid result 转结构化 `provider_contract_invalid`。

非幂等 Procedure timeout/不确定失败绝不重试。幂等 Procedure 只对明确 `retryable=true` 且尚未返回业务结果的错误重试一次，复用同一 request_id；任何 retry 都写 attempt metadata。所有 Procedure 不改变 research credits，external cost 只进 telemetry。

- [ ] **Step 4: 实现 core compact/checkpoint/terminate**

核心 ID 固定 `core.compact/core.checkpoint/core.terminate`，不进入第三方 API。解析时将 ordinary 与 control 分开，ordinary 并发执行但结果按原请求顺序写入 `ProcedureBatchCompleted` event。控制规则由 reducer在该 event持久化后执行：terminate 优先，忽略 compact/checkpoint和全部 delegations；否则 compact可与 ordinary/children共存，成功 summary event后替换可变 history，失败把 error append且若无 context空间则 finalize；negative credit在 compact之后检查并 finalize；checkpoint仅在非负时运行并 hold children。自动 compact由 threshold触发，与显式 compact共用同一 summary effect且一次 turn最多调用一次。

- [ ] **Step 5: 运行 Procedure 测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_executor.py tests/procedures/test_core.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/procedures/executor.py lunagentic_research_swarm/procedures/core.py tests/procedures
git commit -m "feat: add procedure execution and core controls"
```

### Task 5: 实现普通智能体 event-driven turn pipeline

**Files:**
- Create: `lunagentic_research_swarm/runtime/turns.py`
- Create: `lunagentic_research_swarm/runtime/context.py`
- Modify: `lunagentic_research_swarm/runtime/events.py`
- Modify: `lunagentic_research_swarm/runtime/reducer.py`
- Test: `tests/runtime/test_turns.py`
- Test: `tests/runtime/test_context_invariance.py`

**Interfaces:**
- Consumes: LLM gateway/protocol/credits/Procedure executor/catalog snapshots。
- Produces: `TurnWorker.perform_agent_call(PerformAgentCall) -> AgentCallCompleted`、`perform_procedure_batch(PerformProcedureBatch) -> ProcedureBatchCompleted`、stable prompt builder；reducer串联 reservation/call/correction/procedure/control/children。

- [ ] **Step 1: 写完整顺序与 missing-agent 失败测试**

```python
@pytest.mark.asyncio
async def test_zero_credit_agent_can_delegate_but_paid_children_finish_negative(turn_harness) -> None:
    turn_harness.branch.credits = 0.0
    turn_harness.llm.respond(envelope(delegations=[delegation("agent.a", 10), delegation("agent.b", 5)]), cost=0.0)
    result = await turn_harness.run()
    assert [child.credits for child in result.children] == [0.0, 0.0]


@pytest.mark.asyncio
async def test_procedures_finish_before_negative_credit_blocks_children(turn_harness) -> None:
    turn_harness.llm.respond(
        envelope(procedures=[procedure("builtin.echo")], delegations=[delegation("agent.a", 1)]),
        actual_charge=5.0,
    )
    turn_harness.branch.credits = 1.0
    result = await turn_harness.run()
    assert turn_harness.procedures.calls == ["builtin.echo"]
    assert result.children == []
    assert result.finalize_reason == "negative_credit"


@pytest.mark.asyncio
async def test_removed_agent_edge_is_summarized_without_blocking_sibling(turn_harness) -> None:
    turn_harness.live_agents.remove("agent.removed")
    result = await turn_harness.run_with_delegations([("agent.removed", 1), ("agent.valid", 1)])
    assert [child.agent_id for child in result.children] == ["agent.valid"]
    assert result.edge_finalizations[0].reason == "agent_unavailable"
```

- [ ] **Step 2: 运行 turn 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_turns.py tests/runtime/test_context_invariance.py -v`

Expected: FAIL because turn/context modules are missing。

- [ ] **Step 3: 实现稳定 prefix 与 runtime header**

`context.py` 系统消息只由 swarm identity + bot nickname/personality/behavior_style/reply_style + architecture/credits rules + frozen agent/procedure catalog canonical JSON 构成。frozen catalog为每个 agent附带 selector、protocol、角色、allowed procedures，以及基于 task首模型/物理模型解析出的无 secret price profile/fingerprint与 cache/miss估算说明，使 agent能据预算路由。system中不称 quick thinker为“编排器”。User 1 永远是 immutable formalized task；root 才有 User 2 “本轮起始协调者”；child 继承 parent messages 并 append assignment。每次 call 最后 append runtime header，含 branch/turn、agent capability、剩余报告秒数、input reservation 后 credits、active/queued counts。余额负写“本 turn 后不能启动后代”；零写“仍可零 credits 委派”。

自动 compact threshold 取 agent override > definition > global 258000，并同时检查已知 model context limit（snapshot 若有）减 reserved output 8192 与 safety 8192；任一触发就 compact，formalized task 单独重插。

- [ ] **Step 4: 实现严格 turn 顺序和同模型 correction**

Reducer收到 `AgentCallRequested` 后先生成 input estimate，并在同一 transition写 reservation ledger/usage + `AgentCallReserved`，事务成功才 enqueue `PerformAgentCall`。`TurnWorker.perform_agent_call()` 只调用 LLM/解析 envelope并返回 `AgentCallCompleted`，不改 branch、不写 store；event含 normalized usage、actual model、protocol result/error与 correction_count。

Reducer收到 completion先整体核销 actual usage；若 protocol invalid且核销后余额 `>=0`，追加 schema error user message、为 `model:<first_result.model_name>` 创建唯一 correction reservation/effect；pinning不支持或第二次非法就 effect finalize `protocol_invalid`。余额 `<0` 时不 correction。这样 input reservation在真实调用前已 durable，所有核销/控制仍由单一 reducer完成。

协议成功的 completion使 reducer enqueue ordinary Procedure batch；worker返回 `ProcedureBatchCompleted`。Reducer提交procedure metadata/result后，若controller已挂continue barrier就在此安全点把 pool/adjustment分配到叶子；再触发 control summary effects；terminate或negative直接branch finalizer。checkpoint summary event成功后state `WAITING_REPORT_WITH_CHECKPOINT`且持有delegations；no delegations且本 turn 已执行非控制 Procedure 时自动自委派（同 agent / 剩余 credits）；否则无 checkpoint/terminate 时自然finalize `no_further_work`。有delegations则用 Task 1分配，父余额/债务结算到pool，分别创建children；missing agent edge把原assignment与`agent_unavailable`原因作为最后一条消息追加到该edge克隆上下文，再独立调用branch finalizer，不启动普通agent，也不影响valid sibling。

在创建边前强制三个结构上限：每 turn最多8个 delegation、branch depth最多32、每 Task普通 agent calls最多256。超出上限的每一条边都生成带 `delegation_limit_exceeded` / `branch_depth_exceeded` / `agent_call_limit_exceeded` 原因的 edge finalization summary，不静默裁剪；同 envelope仍在限额内的边继续。这样零成本/零 credits agent也不能无限递归。

- [ ] **Step 5: 验证正式任务 byte invariance**

测试 root → delegation → compact → checkpoint → restart prompt，每一步收集所有 User 1 content 的 UTF-8 bytes，断言唯一集合等于 `{formalized.text.encode("utf-8")}`。branch finalized 后 raw messages 从 runtime graph 删除，测试用 weakref 或显式 `branch.messages == []` 验证。

- [ ] **Step 6: 运行 turn 测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_turns.py tests/runtime/test_context_invariance.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/runtime/turns.py lunagentic_research_swarm/runtime/context.py lunagentic_research_swarm/runtime/events.py lunagentic_research_swarm/runtime/reducer.py tests/runtime/test_turns.py tests/runtime/test_context_invariance.py
git commit -m "feat: implement agent turn pipeline"
```

### Task 6: 实现 TaskController 启动、形式化与生命周期控制

**Files:**
- Create: `lunagentic_research_swarm/runtime/controller.py`
- Create: `lunagentic_research_swarm/runtime/manager.py`
- Modify: `lunagentic_research_swarm/services.py`
- Test: `tests/runtime/test_controller_start.py`
- Test: `tests/runtime/test_controller_controls.py`

**Interfaces:**
- Consumes: reducer/scheduler/turn/summarizer/store、SDK message/config capabilities。
- Produces: `ResearchManager.start/pause/continue/stop/add_context/status/list_tasks` 与每 Task `TaskController`。

- [ ] **Step 1: 写 start 立即返回与 formalization failure 测试**

```python
@pytest.mark.asyncio
async def test_start_returns_after_durable_create_without_waiting_for_formalizer(manager, fake_summarizer) -> None:
    fake_summarizer.block()
    result = await manager.start(
        objective="调查问题", stream_id="stream-1", time_budget_seconds=90, effort_level=1.5,
    )
    assert result["task_id"].startswith("lrs_")
    assert result["status"] == "FORMALIZING"
    assert result["initial_credits"] == 150.0
    assert await manager.store.load_task(result["task_id"])


@pytest.mark.asyncio
async def test_formalizer_failure_marks_task_failed_without_using_raw_objective(manager, fake_summarizer) -> None:
    fake_summarizer.fail("provider error")
    task = await manager.start(objective="raw", stream_id="s", time_budget_seconds=120, effort_level=1.0)
    await manager.wait_idle(task["task_id"])
    stored = await manager.store.load_task(task["task_id"])
    assert stored.current_round.status.value == "FAILED"
    assert stored.formalized_task is None
```

- [ ] **Step 2: 运行 controller 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_controller_start.py tests/runtime/test_controller_controls.py -v`

Expected: FAIL because controller/manager are missing。

- [ ] **Step 3: 实现 context collector 与异步 start**

`ResearchManager.start()` 同步校验 nonblank objective、time >0、effort_level >=0、root/summarizer selector 和 live root；创建 `task_id/round_id`，初始 credits=`default_effort_credits*effort_level` 100% 放 root reservation 之前的 branch，transactionally 插 tasks/round，enqueue `PerformFormalization` 后立即返回。每次 start 都用 root task首模型的500k cache-miss input + 50k output重新计算低预算阈值；有效初始 credits更低时打印一次该 Task warning但仍启动。

formalizer context 使用 `ctx.message.get_recent(stream_id, configured_limit)` + `ctx.message.build_readable()`；分别读取 `ctx.config.get("bot.nickname")`、`personality.personality`、`personality.behavior_style`、`personality.reply_style`。没有公开 Maisaka read API，因此只使用 Tool invocation 显式传入的 `planner_context`（若 Host kwargs 提供）并在 health/文档说明；不读取私有 Maisaka DB。原始 objective/recent messages/personality 只驻留到 formalization 完成，不写 SQLite。

成功后 transactionally 保存 FormalizedTask、vector job、frozen catalog fingerprint、root branch，再启动 root。形式化失败标 FAILED，不用 raw objective 冒充。

- [ ] **Step 4: 实现 pause/stop/add_context**

pause：RUNNING/REPORTING → PAUSING，允许在途 LLM完成及其 ordinary Procedure，禁止新 LLM/summary/children；全部在途安全结算后 PAUSED，保留内存 branch；arm 1200s expiry，超时 EXPIRED/释放 raw。stop：generation++、STOPPED、取消 queued/local waits、丢弃迟到结果、不总结、立即释放 raw。add_context：已存在 Task 才接受，保存为 summary-layer `supplied_context` 记录并广播到所有 active branch 的下一 call；terminal task 保存供下一 round。

- [ ] **Step 5: 实现 continue barrier 与 new round**

active leaves：设置 continue barrier，重置 report timer（显式新 time 或原 time），等待每个在途 turn 完成 cost + ordinary procedures，然后一次 transaction 用 pool+signed adjustment 分配；余额因此变负的 branch finalizes，nonnegative branch 继续处理 held children。pool 不在其他任何事件自动分配。

no active leaves：把 pool+adjustment 持久化；负数返回 `task_finished_insufficient_funds`，不建 root；非负创建 round_number+1/generation+1，输入只含 immutable formalized task、所有 summaries/reports/feedback/supplied context，重新冻结 catalog，创建新 root。不得读取 debug transcript/payload。

- [ ] **Step 6: 运行 controller 测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_controller_start.py tests/runtime/test_controller_controls.py -v`

Expected: 全部 PASS，包括 pause 不总结、stop late result、zero leaves positive/negative restart、all-zero active redistribution。

```bash
git add lunagentic_research_swarm/runtime/controller.py lunagentic_research_swarm/runtime/manager.py lunagentic_research_swarm/services.py tests/runtime/test_controller_start.py tests/runtime/test_controller_controls.py
git commit -m "feat: add research task lifecycle controller"
```

### Task 7: 实现 checkpoint、report epoch、grace 与 coverage set

**Files:**
- Create: `lunagentic_research_swarm/runtime/epochs.py`
- Create: `lunagentic_research_swarm/reporting.py`
- Modify: `lunagentic_research_swarm/runtime/controller.py`
- Test: `tests/runtime/test_report_epochs.py`
- Test: `tests/runtime/test_grace_period.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Consumes: clock、branch finalizer/task finalizer、store summary layer。
- Produces: `ReportEpoch`、`FrontierEntry`、`ReportCoverage`、`ReportCoordinator.open_epoch/on_branch_safe_point/on_grace_expired/synthesize`。

- [ ] **Step 1: 写主动 checkpoint 与提前报告失败测试**

```python
@pytest.mark.asyncio
async def test_manual_checkpoint_holds_children_until_report(report_harness) -> None:
    branch = report_harness.branch("A")
    await report_harness.agent_returns_checkpoint(branch, children=["B"])
    assert branch.lifecycle.value == "WAITING_REPORT_WITH_CHECKPOINT"
    assert not report_harness.was_launched("B")
    report_harness.task_finalizer.block()
    await report_harness.open_due_epoch()
    assert report_harness.was_launched("B")
    report_harness.task_finalizer.release()
    await report_harness.wait_report()


@pytest.mark.asyncio
async def test_all_active_branches_checkpointed_reports_early(report_harness) -> None:
    report_harness.clock.set(10)
    await report_harness.checkpoint_all()
    assert report_harness.reports[-1].kind.value == "INTERMEDIATE"
    assert report_harness.reports[-1].created_at < report_harness.deadline_at
```

- [ ] **Step 2: 写 grace clone 与类型冻结失败测试**

```python
@pytest.mark.asyncio
async def test_agent_return_during_grace_is_checkpointed_but_original_continues(report_harness) -> None:
    epoch = await report_harness.deadline_with_inflight("A")
    await report_harness.return_agent("A", delegations=["B"])
    assert epoch.frontier["A"].checkpoint_requested
    assert report_harness.was_launched_for_next_epoch("B")


@pytest.mark.asyncio
async def test_intermediate_does_not_upgrade_if_branches_finish_during_synthesis(report_harness) -> None:
    await report_harness.begin_intermediate_with_checkpoint()
    report_harness.task_finalizer.block()
    await report_harness.finalize_all_branches()
    report_harness.task_finalizer.release()
    assert report_harness.reports[-2].kind.value == "INTERMEDIATE"
    assert report_harness.reports[-1].kind.value == "FINAL"
```

- [ ] **Step 3: 运行 report tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_report_epochs.py tests/runtime/test_grace_period.py tests/test_reporting.py -v`

Expected: FAIL because epoch/reporting modules are missing。

- [ ] **Step 4: 实现 manual checkpoint hold 状态机**

普通 Procedures结束后，agent 主动 checkpoint 立即调用 branch finalizer 角色生成 `CHECKPOINT` summary；成功/失败都持久化并广播。成功时 branch 进入 `WAITING_REPORT_WITH_CHECKPOINT`，held delegations 不启动；它等待“下一个 report/grace 开始”。如果所有 active branches 都已有当前 checkpoint 或 terminal summary，立即提前打开 epoch，而不是等 deadline；epoch 一打开就 release held work，original branch可与 task-level synthesis并行。checkpoint 在 grace 内发生时，它属于当前 epoch，summary 完成后 original branch进入下一 epoch运行。

- [ ] **Step 5: 实现 deadline/frontier/grace clone**

deadline 到达时创建 `ReportEpoch(epoch=N, frozen_at, frontier={active leaf ids})` 并把 round 标 REPORTING，但不全局暂停。预先 checkpoint 的 frontier entry 标 ready 并恢复 original。READY branch 立即 clone stable context总结，original 继续；IN_FLIGHT branch等待安全点。

任何 frontier agent 在 60s grace 内返回：先按正常 turn做核销/协议/Procedures，再 clone 此时 stable context触发 checkpoint；original继续 compact/credits/children，children归下一 epoch。若 frontier branch finalizes，entry 使用 terminal summary，不再需要 checkpoint。所有 entry ready/failed/terminal 就提前结束 grace。

60s 到点仍 in-flight：clone 该 branch 的 last stable pre-call context；late call/result归下一 epoch。grace 结束后仍等待这些 checkpoint summarizer success/failure，再 synthesize；总结器占 global LLM capacity但不扣 credits。

- [ ] **Step 6: 实现 coverage set 与不可升级 kind**

`build_coverage()` 固定包含：全 Task/round 已提交 terminal summaries（含早期 epoch）、当前每个 active frontier branch 最新 checkpoint；更老且已被 supersede 的 checkpoint保留 DB但不传 task finalizer。若 synthesis 开始时 active branches>0 或任一输入为 checkpoint，kind=INTERMEDIATE；只有 active=0 且所有 input 都是 terminal 才 FINAL。FINAL 只传 terminal summaries，不传历史 checkpoint。kind 写入 report row 后才调用 finalizer，生成中绝不改。

coverage 按 `created_at, summary_id` 稳定排序，每项携带 branch ID/type/epoch/terminal 标记并作为独立 assistant message 传入。coverage 只有一个可用 summary 时直接使用该正文并由插件加 report header/stats，不额外调用 `finalize_task`；多个可用 summaries 才调用 task finalizer。Intermediate 没有任何可用 summary 时只发送插件确定性生成的进度/错误状态，不能让 LLM 凭空补结论。所有 branch terminal/checkpoint summary 在 SQLite 提交后按 summary ID 广播给其他 active branches，只在它们下一次 LLM call append，不打断在途；每个 branch 记录 seen summary IDs 避免重复。

Intermediate 正文必须标“中间报告”、仍运行/排队 branch 数、coverage unavailable 数、task/round/epoch、已用时间、下一报告间隔、当前余额/pool与主要未决工作；Final 标“最终结论”，包含 calls、branches、最大深度、compact/checkpoint/protocol correction/continue次数、token/cache、credits consumed/pool/debt、总结器 cost-equivalent、Procedure现实费用、duration、errors 等确定性 stats。报告 synthesis 失败时仍保存 coverage summaries 与 failure stats；final failure转 COMPLETED_WITH_ERRORS。

Intermediate report持久化并写入outbox后，把下一 deadline设为 `report_created_at + effective_time_budget_seconds`，round回 RUNNING；`continue_deep_research` 同样从调用成功时重置 deadline。全部 branches finalized时不等 deadline，立即 finalizing。

- [ ] **Step 7: 运行所有报告测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_report_epochs.py tests/runtime/test_grace_period.py tests/test_reporting.py -v`

Expected: 全部 PASS，包括 60s timeout clone、all-ready early end、旧 terminal coverage、checkpoint never final、finalizer in-flight race。

```bash
git add lunagentic_research_swarm/runtime/epochs.py lunagentic_research_swarm/reporting.py lunagentic_research_swarm/runtime/controller.py tests/runtime/test_report_epochs.py tests/runtime/test_grace_period.py tests/test_reporting.py
git commit -m "feat: add branch-aware report epochs"
```

### Task 8: 实现 Maisaka durable outbox

**Files:**
- Create: `lunagentic_research_swarm/storage/outbox.py`
- Modify: `lunagentic_research_swarm/services.py`
- Test: `tests/storage/test_outbox.py`

**Interfaces:**
- Consumes: `ctx.maisaka.context.append`、`ctx.maisaka.proactive.trigger`、SQLite outbox rows。
- Produces: `MaisakaOutbox.start/close/wake/deliver_once`，报告先 append 后 trigger 的 durable workflow。

- [ ] **Step 1: 写 append/trigger 部分失败测试**

```python
@pytest.mark.asyncio
async def test_trigger_failure_does_not_repeat_completed_append(outbox_harness) -> None:
    outbox_harness.maisaka.append_succeeds()
    outbox_harness.maisaka.trigger_fails_once()
    await outbox_harness.deliver_until_idle()
    assert outbox_harness.maisaka.append_calls == 1
    assert outbox_harness.maisaka.trigger_calls == 2
    assert outbox_harness.row.status == "delivered"
```

- [ ] **Step 2: 运行 outbox 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_outbox.py -v`

Expected: FAIL because outbox is missing。

- [ ] **Step 3: 实现两阶段 outbox 与稳定幂等键**

报告 transaction 同时写 report + `append_report` outbox，idempotency key=`lrs:{task}:{round}:{report}:append`。append 成功 transaction标 delivered并插 `trigger_report_review` row，key 尾缀 trigger；trigger intent 明确说明 intermediate/final、running count、task ID，metadata 含 report ID，reason不含 raw transcript。

worker 只 retry capability delivery，不重复 LLM；失败后 attempt+1，退避 `min(300, 2 ** min(attempt, 8))` 秒。Host 没有跨 crash exactly-once receipt，README/health明确“稳定 idempotency key + at-least-once”；append成功状态已提交后不会因 trigger失败重做 append。

- [ ] **Step 4: 运行 outbox 测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_outbox.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/storage/outbox.py lunagentic_research_swarm/services.py tests/storage/test_outbox.py
git commit -m "feat: add durable Maisaka report delivery"
```

### Task 9: 暴露 Planner-facing Tools

**Files:**
- Create: `lunagentic_research_swarm/tools.py`
- Modify: `plugin.py`
- Test: `tests/test_planner_tools.py`

**Interfaces:**
- Consumes: `ResearchManager` public methods。
- Produces: 七个运行时 tool 名称和稳定 JSON result shapes；`submit_research_feedback` 与它的正式 schema/service 在第三阶段作为完整单元加入，不在本阶段注册半成品组件。

- [ ] **Step 1: 写工具名、schema 与 non-blocking start 失败测试**

```python
def test_runtime_planner_tool_names_are_plain(plugin_module) -> None:
    names = {
        item["name"] for item in plugin_module.create_plugin().get_components()
        if item["type"] == "TOOL"
    }
    assert names == {
        "start_deep_research", "pause_deep_research", "continue_deep_research",
        "stop_deep_research", "add_research_context", "get_research_status",
        "list_research_tasks",
    }


@pytest.mark.asyncio
async def test_start_tool_forwards_stream_and_returns_immediately(fake_plugin) -> None:
    result = await fake_plugin.start_deep_research(
        objective="调查", time_budget_seconds=60, effort_level=1.0, stream_id="s1"
    )
    assert result["success"]
    assert result["task_id"].startswith("lrs_")
    assert result["status"] == "FORMALIZING"
```

- [ ] **Step 2: 运行 tools 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_planner_tools.py -v`

Expected: FAIL because tools are not registered。

- [ ] **Step 3: 写参数 schemas 与 handler forwarding**

`tools.py` 导出七个 JSON schema constants。关键参数：

- `start_deep_research(objective: string required, time_budget_seconds: integer optional, effort_level: number default 1.0)`；stream_id 从 Host kwargs 取，空则明确 `stream_id_required`。
- pause/status：`task_id required`；stop 另接受 optional `reason` 并写入结构化 lifecycle event。
- continue：`task_id required, time_budget_seconds optional, credit_adjustment number default 0`，adjustment 可正可负。
- add context：`task_id/information required`。
- list：`status optional, created_after/created_before optional ISO timestamp, limit integer 1..100 default 20`。
在 `plugin.py` 分别用 `@Tool("start_deep_research", parameters=START_SCHEMA, core_tool=True, visibility="visible")` 等七个明确 decorator 声明；每个 handler 只校验/forward，不直接改状态。Tool description 告诉 Planner start 是 async，需用 status/stop/pause/continue/add 交互。所有 mutating result固定返回 `task_id/round/status/effective_time_budget_seconds/effective_credits_or_adjustment`；错误使用基础计划的结构化 code。

- [ ] **Step 4: 运行 tool tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_planner_tools.py tests/test_plugin_contract.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/tools.py plugin.py tests/test_planner_tools.py
git commit -m "feat: expose deep research planner tools"
```

### Task 10: 完成本阶段确定性端到端测试

**Files:**
- Create: `tests/fakes.py`
- Create: `tests/integration/test_research_flow.py`
- Create: `tests/integration/test_control_races.py`
- Create: `tests/integration/test_extension_removal.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: 本阶段全部 runtime/services/plugin interfaces。
- Produces: 无 Host/网络的 deterministic test harness 与核心调查 flow regression suite。

- [ ] **Step 1: 实现 deterministic fake clock/providers**

`FakeClock` 由测试手动 `advance(seconds)`；`FakeLLMGateway` 按 call queue 返回模型/usage/envelope并可 block；`FakeProcedureProvider` 记录 request_id/scoped metadata；`FakeMaisaka` 分开 append/trigger failure；`RuntimeHarness` 用 temp SQLite，关闭 periodic real timers。

- [ ] **Step 2: 写 root→三分支→checkpoint→intermediate→final 测试**

```python
@pytest.mark.asyncio
async def test_complete_branching_research_flow(harness) -> None:
    task = await harness.start("比较两个方案", credits=100, time_budget=120)
    await harness.formalize("正式任务")
    await harness.root_delegates({"A": 50, "B": 25, "C": 25})
    await harness.branch_checkpoint("A")
    harness.clock.advance(120)
    await harness.run_until_idle()
    assert harness.reports[0].kind.value == "INTERMEDIATE"
    assert harness.reports[0].running_branch_count > 0
    await harness.finalize_all()
    assert harness.reports[-1].kind.value == "FINAL"
    assert harness.task_status.value == "COMPLETED"
    assert harness.raw_context_count == 0
```

- [ ] **Step 3: 写 control/extension race 测试**

覆盖：pause 时在途返回但 children 不启动；continue negative adjustment barrier 后 branch finalizes；stop 后迟到 LLM/summary丢弃；extension agent 在途移除后有效 sibling继续、self/missing边总结；宽 fan-out 不饿死另一 Task；report synthesis中全部结束仍先 intermediate再 final。

- [ ] **Step 4: 运行 runtime 全套与窄 lint**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime tests/procedures tests/storage/test_outbox.py tests/integration tests/test_planner_tools.py -v`

Expected: 全部 PASS。

Run: `python -m ruff check plugin.py lunagentic_research_swarm tests`

Expected: exit 0。

- [ ] **Step 5: 提交运行时集成验收**

```bash
git add tests/fakes.py tests/conftest.py tests/integration
git commit -m "test: cover end-to-end swarm runtime"
```

## 本计划完成门槛

- credits 八个指定场景与 property 守恒测试通过。
- reducer transaction-before-effect、generation late-result rejection 与全生命周期 matrix 通过。
- 并发限额、公平性、barrier priority 与 shutdown 测试通过。
- Turn 严格顺序、单次同物理模型纠正、zero/negative、missing agent edge 与 compact invariance 通过。
- start durable/non-blocking，pause/continue/stop/add/status/list tools 可用。
- checkpoint/grace/coverage/type-freeze 语义逐项通过，报告包含 running count 与 stats。
- 报告 durable 后再 append/trigger，部分失败不重复已完成阶段。
- 完整 feedback tool、默认扩展、向量索引与用户命令仍由第三阶段交付；在完成第三阶段前不得发布 0.1.0。
