# Task 4 报告：Procedure 执行器与核心控制

## Status

完成。新增 `ProcedureExecutor.invoke_many()`、稳定 request ID 与 provider contract 校验，
普通 Procedure 并发执行后按原请求顺序生成 `ProcedureBatchCompleted`；新增
`split_procedure_requests()` 与 `CoreProcedureDecision`，固定识别
`core.compact/core.checkpoint/core.terminate`，terminate 优先且 core 不走第三方 API。
compact/checkpoint 通过 `SummarizerService` 本地调用，Procedure/core 结果不进入 research
credits ledger；provider 错误、超时和 contract invalid 均转换为结构化结果，不保留原始异常
或 payload。

## TDD 证据

- RED：首次运行 focused tests 在 collection 阶段按预期失败，报
  `ModuleNotFoundError`（`procedures.executor`、`procedures.core` 缺失）。
- GREEN：最小实现后 focused suite 覆盖调用顺序、控制 precedence、并发结果排序、稳定
  request ID、明确 retryable 幂等重试、timeout 不重试、结果 contract invalid、事件 round-trip
  与 core summarizer 调用。

## 验证

```text
timeout 30s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/procedures/test_executor.py tests/procedures/test_core.py \
  tests/runtime/test_credits.py tests/runtime/test_reducer.py tests/runtime/test_scheduler.py -q
→ 59 passed in 0.22s
```

另外通过 `git diff --check` 与 `python -m compileall -q lunagentic_research_swarm tests/procedures`。
未运行完整 suite（按 Task 4 指示由控制器在外部环境复核）。

## Concerns

- `ProcedureBatchCompleted` 的 `results`/`controls` 字段已加入 runtime event union，后续
  Task 5 turn reducer 应消费控制决定并执行余额、checkpoint hold、delegation precedence；
  Task 4 不越界修改 reducer 的控制策略。
- `ProcedureExecutor` 只把 task/round/branch/turn/agent 放入 scoped metadata；retry attempt
  写入结果 telemetry metadata，request ID 在 retry 中复用。上游 API 若要求其它 envelope
  字段，应保持 `invoke_procedure@1` 契约而不要把 branch raw context 加入 metadata。

## Commit

`46f3dd1 feat: add procedure execution and core controls`

## 审查修复 Round 1

### RED

新增四项 Important 回归后，修复前 focused suite 为 `10 passed, 4 failed`：

- 合法 provider `data/error` 的 nested `reasoning/raw_payload/provenance/payload` 仍进入 event JSON；
- provider 返回的伪造 provenance metadata 被 `setdefault` 接受；
- `retryable="true"` 被 truthiness 误判为可重试；
- core `control_requests` 可变且 `as_dict`/event round-trip 丢失控制请求。

### GREEN

- 新增递归 JSON sanitizer，作用于 ProcedureResult 的 data/error/metadata；只移除明确
  sensitive key，保留普通业务字段，并在 event 边界再次验证冻结结果；
- provider plugin/API/version/request_id/procedure_id 由 frozen catalog 和稳定 request ID 强制覆盖；
- exception retry 只接受 `retryable is True`；
- 新增 `FrozenControlRequest`，递归冻结 arguments；CoreProcedureDecision 序列化并恢复
  control requests，控制语义在 round-trip 后保持。

修复后 focused + Task1–3 回归：`68 passed in 0.35s`。

修复提交：本报告随 `fix: harden Procedure privacy, provenance, retry and core event immutability` 提交。

## 审查修复 Round 2

### RED/GREEN

新增回归先证明 `raw_result`、`raw_arguments`、`transcript`、`messages` 会从
`ProcedureResult.data` 进入 event JSON；另证明 core control arguments 中同类字段以及
`reasoning` 会泄露。递归 sanitizer 与 frozen control request 过滤后，普通业务字段和必要的
`reason` 保留，focused + Task1–3 回归为 `70 passed in 0.33s`。

修复提交：本报告随 `fix: remove raw procedure transcripts from events` 提交。
