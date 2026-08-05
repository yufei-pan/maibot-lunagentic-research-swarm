# Task 7 报告：checkpoint、report epoch、grace 与 coverage

## Status

完成。新增 `runtime/epochs.py` 与 `reporting.py`，提供 `ReportEpoch`、frontier
快照、checkpoint hold/release、60 秒 grace clone、coverage 选择、稳定的
intermediate/final kind 冻结、报告 header/stats、SQLite report/outbox 持久化和
summary broadcast。`ResearchManager` 在形式化成功后默认注册 coordinator；deadline、grace
和安全点通过显式 bridge 进入 coordinator，报告事务提交后 callback 回到
`TaskController`。

## Review fix rounds

- Round 1（`a21ad91..e0d3acb`）：修复无既有 epoch 的最后 terminal branch 未生成 final、
  synthesis coverage race、grace clone 在 child launch 后才取 history，以及 manager
  coordinator/effect bridge。
- Round 2（`e0d3acb..458aa95`）：intermediate synthesis 完成后若所有分支已终结，自动
  开启独立 FINAL epoch；默认构造/注册 coordinator 并将报告完成事件提交回 controller；
  ReportDeadlineReached/GraceExpired 增加单调 epoch 校验，旧/跳跃事件被忽略。
- Round 3（`458aa95..b045676`）：最终报告 FAILED/DEGRADED 时发 `FinalReportFailed`，
  任务进入 `COMPLETED_WITH_ERRORS`；中间报告失败仍恢复 RUNNING。错误码限制为安全字符，
  错误消息限制长度并过滤正式任务/coverage 文本，避免 prompt 泄露。

## Verification

- Report/epoch/reducer/bridge focused tests：`31 passed`；最终失败路径 focused：`11 passed`。
- 完整 `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q`：`378 passed`。
- `git diff --check` 与 `compileall` 通过；正式 scoped review round 3：`PASS`。
