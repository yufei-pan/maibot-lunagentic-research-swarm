# Task 6 报告：TaskController 启动与生命周期控制

## Status

完成。新增独立的 `runtime/controller.py`，保留 reducer 的兼容导入路径；新增
`ResearchManager` 提供 durable start、异步形式化、pause/continue/stop、context 与
status/list API。开始任务只持久化 ID、round、credits 与目录指纹；objective、最近消息、
人格及显式 planner context 仅在形式化协程中存在。成功时同一 transaction 保存正式任务、
vector job、root branch 和 RUNNING；失败则标记 FAILED，不以原始目标替代。

暂停禁止 scheduler 启动新的 LLM/summary 并等待 in-flight work 结算，停止递增并持久化
generation，迟到结果被忽略。continue 在 pause barrier 重新分配 pool/adjustment；没有叶子
时仅以 summary layer 创建新的 round/root，负余额返回明确错误。暂停超时释放 raw context。

## TDD 证据

- RED：首次运行 controller 测试在 collection 阶段失败：`ModuleNotFoundError:
  lunagentic_research_swarm.runtime.manager`；并发现 runtime tests 的相对 fake import
  需要 package marker。
- GREEN：实现最小 controller/manager 并添加 `tests/runtime/__init__.py` 后，controller
  focused suite 通过。

## Review fix round 1

- RED：新增当前 generation 的 `AgentCallCompleted` 回归测试，确认此前 manager 会直接
  丢弃它，既不写 reconciliation command，也不调度 `PerformProcedureBatch`；同时改为从
  `runtime.controller` 导入 controller failure 测试，确认较弱的重复实现不会在 fallback
  persistence 同样失败后停止接收事件。
- GREEN：`ResearchManager.handle_runtime_event()` 现在把所有当前 round/generation 的 worker
  event 提交给唯一 `TaskController` 并 drain，因此 completion 会先持久化核销、后调度
  procedure/materialization effect；late generation 和 terminal task guard 保持不变。
  pause、settle、expiry、stop 及 paused-leaf continue 也改走 reducer/controller。
- `runtime/controller.py` 现在承载完整的 transaction-before-effect、fallback FAILED write、
  health 和 stopped safeguards；`runtime.reducer.TaskController` 继续经兼容 re-export 指向
  同一个类，避免两份实现漂移。

## Review fix round 2

- RED：stop transaction 被故意阻塞时并发提交当前 generation 的 completion；旧实现会在
  STOPPED 写入期间按旧 generation 规约并排入 procedure effect。controller 现以单一
  asyncio lock 串行 inbox、reducer、durable transaction 与 effect 发布，completion 在
  stop 后只会作为 late generation 被忽略。
- formalization 成功/失败和 terminal round restart 都通过 controller 的原子 event path
  提交；manager 不再绕过 controller 直接写权威状态或调度 effect。失败 restart 不会插入
  负余额 root branch。
- pause 兼容 FairScheduler 的公开 `stats().tasks[task_id].active`，可选使用专用
  `wait_task_idle`，不再假设 scheduler 有未声明的 task_inflight_count API。
- continue 的 active leaf balances、credit pool、time budget/deadline reset 以及无 leaf
  的负 pool+adjustment 均由 reducer 和同一 transaction 持久化。

## Review fix round 3

- pause barrier 改为从 FairScheduler 的公开 task telemetry 同时检查 `active` 与
  `queued`，覆盖 AgentCallCompleted 持久化后、ProcedureBatch 尚在队列中的交接窗口；
  真实 FairScheduler 回归测试确认 procedure 开始且未结束时状态仍为 `PAUSING`。
- reducer 将 agent 核销、失败核销和 procedure 完成后的 branch balance 写回
  `RuntimeState.active_leaves`；manager 在每个当前 runtime event 后以该权威状态刷新
  仅用于 status/pending-context 的 `_branches` cache。continue 因此持久化核销后的
  80 credits，而不会使用过期的 100。
- controller 遇到 `transition.error` 时只提交 reducer 自带的补偿 commands/effects，
  保持 reducer 的 next state 并返回 `False`；不再合并 manager 的 state_changes、extra
  commands 或 effect override。负余额 restart 不会产生 phantom leaf 或重置
  `raw_context_released`。

## 验证

- `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/runtime/test_controller_start.py tests/runtime/test_controller_controls.py -v` → `14 passed`。
- `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/runtime/test_reducer_persistence.py tests/runtime/test_reducer.py tests/runtime/test_scheduler.py -v` → `35 passed`。
- `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -v` → `349 passed`。
- `.venv/bin/python -m compileall -q lunagentic_research_swarm tests/runtime` 与 `git diff --check` 通过。
- Fix round verification: `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest
  tests/runtime/test_controller_start.py tests/runtime/test_controller_controls.py
  tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py
  tests/runtime/test_scheduler.py tests/runtime/test_turns.py
  tests/runtime/test_context_invariance.py -v` → `75 passed`。
