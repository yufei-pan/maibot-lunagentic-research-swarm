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
