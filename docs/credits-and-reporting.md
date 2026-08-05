# Credits、时间报告与 Feedback

## Credits 数学

- 模型配置价格数值 **`1.0` = 100 credits**。忽略配置界面的货币标签。
- 默认 `budget.default_effort_credits = 100`。
- 启动时：`initial_credits = default_effort_credits × effort_level`。
- **Host 价格优先**。仅当插件 `[pricing.models."<ModelInfo.name>"]` 存在该模型条目时，才完整覆盖 Host 同模型全部价格字段；条目内未写字段按 **0（免费）**。
- Host 与插件均无价格 → 按免费；低预算警告阈值约 **500k cache-miss 输入 token / 50k 输出 token**（可配置），只警告不拒绝。
- 根智能体获得 100% 初始 credits。普通智能体先付自身输入/输出，再从剩余 `R` 按比例分配子请求；`R < 0` 时 Procedure 仍完成，但不启动新委派。
- 零余额仍可零 credits 委派；负余额触发 credits 终止。
- **总结器与普通 Procedure 不扣研究 credits**；其 token / cost-equivalent / 外部费用另计。

分配是转移，不能制造 credits。账本与任务状态在同一 SQLite 事务中提交。

## 时间与报告

| 项 | 默认 | 语义 |
|---|---:|---|
| `default_time_budget_seconds` | 120 | 报告间隔 / 时间提示 |
| `grace_period_seconds` | 60 | 超时后收尾；frontier 齐备可提前结束 |
| `pause_timeout_seconds` | 1200 | 暂停过期 → `EXPIRED` |
| `feedback_wait_seconds` | 600 | 终态后等待反馈 |

控制行为：

- **pause**：在途调用到安全边界后暂停新 LLM 调度。
- **continue**：重置计时；有叶子则重分配 pool；无叶子则只从 summary layer 开新 round。
- **stop**：停止调度、丢弃迟到结果、不总结；可触发反馈提醒。
- **checkpoint / compact / terminate**：core Procedure，由总结器执行；不扣研究 credits。

中间报告先写入 SQLite + outbox，再 Maisaka `context.append` 与 `proactive.trigger`。最终报告正文含插件生成的**确定性统计**（不得由 LLM 臆造）。

## Feedback 与提醒

- 提交：`accepted | mixed | rejected | superseded`，可选评分、纠错、outcome、notes。
- 事件不可变；新反馈可 supersede，不删历史。
- **学习边界：** 只用于透明检索、排序、统计与可见 lesson；**绝不**自动改 prompt、selector 或路由。
- 提醒：`COMPLETED` / `COMPLETED_WITH_ERRORS` / 手动 `STOPPED` 后启动；到期一次 Maisaka 触发；`continue` 或提交 feedback 取消；`EXPIRED` / `INTERRUPTED` 不提醒。
