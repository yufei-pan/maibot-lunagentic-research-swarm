# LRS Production Runtime Composition Design

## Goal

让 `LRSServiceContainer` 在真实插件生命周期中创建并公开可用的 `ResearchManager`，并用一个最小、
可测试的 effect runner 把 scheduler effect 回投为 runtime event，使 Planner Tool 启动的调查能够从
root agent 推进到 Procedure、child branch 与 branch summary。

## Architecture

新增 `runtime/effect_runner.py`。`RuntimeEffectRunner` 是 `FairScheduler` 的 worker，只负责执行已经
durable commit 后发布的 effect，并把结果交还 `ResearchManager`；它不直接拥有 task 状态机。
runner 创建时可先不绑定 manager，以解除 `manager → scheduler → runner → manager` 的构造环，service
完成对象组合后调用 `bind_manager()` 一次。

`LRSServiceContainer` 在 extension catalog 与价格快照初始化后创建 `LLMGateway`、
`SummarizerService`、`ProcedureExecutor`、`TurnWorker`、`RuntimeEffectRunner`、`FairScheduler` 和
`ResearchManager`，并公开只读 `manager` 属性。关闭时先停止 manager/scheduler 的新工作，再关闭
outbox/discovery/store。由于 manager 在 `services.start()` 内完成构造，plugin 必须在 `await start()`
成功后再缓存 `services.manager`。

`ResearchManager` 为每个 active task 保存启动时已经冻结的 `NextRoundSnapshot`，并向 runner 提供小型
runtime adapter 方法；runner 不读取 live catalog 来替换 round snapshot。child materialization 的 durable
branch 写入和 authoritative active-leaf 更新由 manager 统一完成，避免 runner 直接篡改 controller。

## Effect behavior

- `PerformFormalization`：no-op。原始 objective/context 只由 `ResearchManager._formalize()` 的短生命周期
  coroutine 持有，runner 不复制或持久化它。
- `PerformAgentCall`：runner 为 root/child 补齐冻结 round 的 agent definition、selector、protocol、
  prompt、live agent IDs 与 safety limits，然后调用 `TurnWorker.perform_agent_call()`；完成 event 回投
  `ResearchManager.handle_runtime_event()`。
- `PerformProcedureBatch`：调用真实 `ProcedureExecutor`/`TurnWorker.perform_procedure_batch()`，完成 event
  回投 manager。
- `NotifyToolWaiter(action="materialize_child")`：验证目标 agent 仍 live；原子持久化 child branch，更新
  controller active leaves、manager branch cache 和 report coordinator branch graph，再 enqueue child
  `PerformAgentCall`。目标已移除时转成 `PerformBranchSummary(reason="agent_unavailable")`。
- `PerformBranchSummary`：把 effect payload 中的 messages/reason/held delegations 映射到 coordinator 的
  checkpoint 或 terminal safe point；总结和报告继续由 `ReportCoordinator` 持久化与广播。
- `OpenReportEpoch` 继续交给 manager 已有 bridge；`PerformTaskSummary` 由 coordinator synthesis 管理；
  `DeliverOutbox` 继续由 `MaisakaOutbox` 管理。本轮不新增 deadline/feedback timer。

## Error and lifecycle rules

worker exception 由 `FairScheduler` 的既有错误隔离捕获，不破坏 dispatcher。generation/round 晚到结果仍
由 manager/controller 拒绝。child materialization 必须先 SQLite commit，再 enqueue agent；commit 失败
不得产生 phantom child。关闭顺序必须避免 scheduler worker 在 store 关闭后继续提交事件。

## Tests

测试先证明：service load 后 `manager` 非空且 Planner start 不再返回 `manager_unavailable`；真实 runner
能将 Agent completion 回投并启动 Procedure；materialize child 先持久化再启动；missing agent 转 summary；
branch summary 走 coordinator；service close 不遗留 scheduler/store resource。最后运行 runtime、integration、
lifecycle/plugin tool 与完整 suite。
