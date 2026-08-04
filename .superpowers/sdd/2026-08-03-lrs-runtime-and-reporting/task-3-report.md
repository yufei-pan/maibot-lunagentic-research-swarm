# Task 3 报告：task-aware 有界公平调度器

## Status

完成。新增 `FairScheduler`，按 priority 分组并在同优先级内按 task round-robin
取 work；普通 agent 同时受 global/per-task LLM 容量约束，summarizer 只占 global
LLM，Procedure 只占 per-task Procedure。调度器支持 live limit 收紧、task
pause/resume、generation stop token、无 prompt 的统计与等待在途 wrapper 的正常关闭；
worker wrapper 不实现 Host HTTP retry。

## TDD 证据

- 接管说明和现有测试表明 RED 已由前代理在 `scheduler` 模块缺失时建立；接管时共享
  worktree 已同时存在未提交的 `scheduler.py`，因此未伪造或重做历史 RED。
- 首次使用系统 pytest 的运行因缺少 `pytest-asyncio` 将 8 项全部 SKIP，不作为验证结果。
- 使用项目 `.venv` 的有效 GREEN：focused scheduler suite 为 `8 passed`。

## 验证

- `timeout 30s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/runtime/test_scheduler.py -q`
  → `8 passed in 0.09s`。
- `timeout 30s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/runtime/test_credits.py tests/runtime/test_credits_properties.py tests/runtime/test_reducer.py tests/runtime/test_reducer_persistence.py -q`
  → `39 passed in 0.21s`。
- `python -m compileall -q lunagentic_research_swarm` 与 `git diff --check` 通过。
- 按控制器要求未运行完整 suite；完整 suite 由控制器在沙箱外复核。

## Concerns

- 当前 worktree 的 `.venv` 没有 ruff 可执行文件，因此本任务没有独立 ruff 结果；
  Python 编译和 diff whitespace 检查已通过。
- generation token 是本地 advisory stop：它标记在途调用已取消并清理未启动 effect，
  不会中断已经进入 Host 的请求；迟到结果仍必须由 reducer 的 generation 检查丢弃。
- scheduler 只实现本任务的队列与 wrapper；Procedure/turn/controller/reporting 接线留给
  后续计划任务。
