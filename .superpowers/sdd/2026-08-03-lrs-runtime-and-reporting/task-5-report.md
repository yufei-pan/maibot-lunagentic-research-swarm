# Task 5 报告：普通智能体 event-driven turn pipeline

## Status

完成。新增稳定 prompt/context builder 与普通智能体 `TurnWorker`，正式任务始终作为独立
User 1 重插；system prefix 使用 canonical JSON，runtime header 只追加在每次调用末尾。
turn resolver 固定执行 LLM 核销后的 ordinary Procedure、control、负余额检查、checkpoint
hold 与 children，并对 missing agent 和三个结构上限逐 edge 生成明确终结结果。

Reducer 现在在发布 `PerformAgentCall` 前同一 transition 写入 input reservation usage、credits
ledger 与 reservation lifecycle；completion 整体核销实际 usage，协议失败仅允许一次
`model:<actual physical model>` correction。Procedure completion 持久化可审计 metadata，并按
continue barrier、control、negative credit、checkpoint/no-work/children 的安全点继续。

## TDD 证据

- RED：首次 focused run 在 collection 阶段以两个 `ModuleNotFoundError` 失败，分别证明
  `runtime.turns` 与 `runtime.context` 尚不存在。
- RED/GREEN 增量：worker contract 与 Procedure completion precedence 均先单独观察到失败，
  再加入最小实现并通过。
- GREEN 覆盖零 credits 委派、Procedure-before-negative、missing-agent sibling 隔离、每 turn
  delegation/depth/task-call 三项上限、durable reservation、同物理模型 correction、稳定 prefix /
  runtime suffix、自动 compact 双阈值、正式任务 byte invariance 与 raw context release。

## 验证

```text
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_turns.py tests/runtime/test_context_invariance.py -v
→ 14 passed in 0.15s

PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py \
  tests/procedures/test_executor.py tests/procedures/test_core.py -q
→ 36 passed in 0.21s
```

另外通过 `.venv/bin/python -m compileall -q lunagentic_research_swarm/runtime` 与
`git diff --check`。按 Task 5 指示未运行完整 sandbox suite。

## Scope

只修改 Task 5 指定的 `runtime/events.py`、`runtime/reducer.py`，新增
`runtime/turns.py`、`runtime/context.py` 和两项 focused test；未修改 controller、reporting
或后续任务文件。

## Commit

本报告随 `feat: implement agent turn pipeline` 提交。

## Review fix round 1

复审发现的四项阻塞已在 Task 5 范围内修复：

- `AgentCallRequested` 的 reservation lifecycle metadata 现在剥离完整 `messages`，不再持久化
  raw prompt/transcript；selector、模型、token、价格、余额与结构限制等 routing/accounting 字段保留。
- `TurnWorker` 把原始 request messages 写入 `AgentCallCompleted`；同物理模型 correction 因而在原上下文
  末尾追加 schema error，而不是从空消息开始。
- `AgentCallFailed` 携带 normalized usage、实际模型、实际/预估费用与预留后余额。失败有 usage 时 reducer
  写 whole-call reconciliation 与 signed ledger adjustment；无 usage 时写
  `estimated_unreconciled` telemetry，保留 reservation 且不产生补偿 ledger。
- delegation/depth/task-call 上限与 missing-agent 判断已经接入 `ProcedureBatchCompleted` durable
  materialization boundary。合法 sibling 逐个发出 immutable `materialize_child` effect；被拒绝的 edge
  逐个发出带原 assignment、credits、clone messages 与明确原因的 immutable branch-finalization effect。

### Fix-round TDD 与验证

```text
RED: PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest tests/runtime/test_turns.py -q
→ 8 failed, 10 passed

GREEN focused: PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_turns.py tests/runtime/test_context_invariance.py -q
→ 22 passed

Impacted: PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime tests/procedures tests/test_events.py -q
→ 96 passed
```

本轮仍未修改 controller、reporting 或 Task 6+ 文件。

## Review fix round 2

第二轮复审指出 task-call 配额仍有两个相互关联的竞争边界，本轮在 Task 5 runtime
materialization boundary 内完成修复：

- delegation 先按 per-turn/depth 与 live-agent availability 分类，missing agent 仍逐 edge
  终结并保留原 credits/messages，但不再消耗 task-call slot；因此仅剩一个 slot 时，位于
  missing edge 后的合法 sibling 仍能物化。
- `RuntimeState.agent_calls_started` 成为 reducer 的任务级单调权威计数。每个
  `ProcedureBatchCompleted` 先取 state/event 快照最大值，只为实际发出的 live
  `materialize_child` edge 原子预留，并把批次预留后的权威计数及三项结构限制传播到每个
  child effect。两个携带相同旧快照的 completion 顺序归约时，第二个不会超过任务上限。
- continue-barrier redistribution 从已同步的 `next_state` 派生，避免覆盖单调计数。
- `resolve_completed_turn` 的前置纯解析路径采用同一排序，并保持 edge finalization 的原输入顺序。

### Fix-round 2 TDD 与验证

```text
RED missing-before-valid:
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_turns.py::test_missing_agent_does_not_consume_last_task_call_slot_before_valid_sibling -q
→ 1 failed（children == []，预期 ["agent.valid"]）

RED atomic state:
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_turns.py::test_reducer_atomically_reserves_task_call_slot_across_stale_procedure_events -q
→ 1 failed（RuntimeState 尚无 agent_calls_started）

RED continue barrier:
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_turns.py::test_procedure_completion_preserves_monotonic_call_counter_through_continue_barrier -q
→ 1 failed（得到 254，预期 255）

GREEN focused:
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime/test_turns.py tests/runtime/test_context_invariance.py -q
→ 25 passed

Impacted:
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/python -m pytest \
  tests/runtime tests/procedures tests/test_events.py -q
→ 99 passed
```

另外通过 `.venv/bin/python -m compileall -q lunagentic_research_swarm/runtime` 与
`git diff --check`。本轮仍未修改 controller、reporting 或 Task 6+ 文件。
