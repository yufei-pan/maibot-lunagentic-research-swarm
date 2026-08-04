# Task 2 报告：纯 reducer 与 transaction-before-effect 驱动

## Status

完成。新增不可变 `RuntimeState`、`Transition` 与显式 `Effect` union，`reduce_event()`
只依赖事件携带的 task/round/generation/time，不访问 store、ctx、clock、random，也不创建
asyncio task。生命周期白名单覆盖形式化、运行、暂停/暂停超时、报告、停止与 final report；
迟到或跨 round/generation 结果返回 ignored transition，非法状态返回结构化
`LRSError(code="invalid_state")` 和 `NotifyToolWaiter` effect。

`TaskController` 的最小驱动固定为 `transact(commands)` → 发布 `next_state` → enqueue
effects。事务失败时先把内存快照置为 `FAILED`，再做单条 best-effort FAILED update；该 update
也失败时停止 controller 并写入 health，不会启动任何 effect。STOPPED 的 generation 通过
新增 SQLite `update_round_generation` command 持久化。

## TDD 证据

- RED：新增 reducer/persistence 测试后首次运行在 collection 阶段失败，报
  `ModuleNotFoundError: lunagentic_research_swarm.runtime.reducer`。
- GREEN：实现最小 reducer、effect 值对象和事务驱动后 focused suite 为 `15 passed`。

## 验证

- `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py -q`
  → `15 passed`。
- `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/test_events.py tests/test_models.py tests/runtime -q`
  → `40 passed`。
- `git diff --check` 与 `python -m compileall -q lunagentic_research_swarm` 通过。
- 存储/完整回归在当前沙箱的既有异步 SQLite/extension persistence 测试处无法稳定结束，
  已停止等待；由控制器在沙箱外复跑完整 suite。

## Concerns

- `TaskController` 这里只提供 Task 2 所需的串行 inbox/提交顺序；scheduler、turn worker、
  manager 和完整报告协调仍留给后续任务。
- `ContinueRequested` 的新 round ID、generation、时间预算等均需由调用方写入事件；reducer
  不生成 ID，也不会猜测缺失的 round identity。
- `update_round_generation` 是 foundation SQLite schema 的兼容 command handler，无迁移版本
  变化；后续 controller 应继续把它与 status/lifecycle 写入同一 transaction。
