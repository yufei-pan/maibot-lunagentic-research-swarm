# 隐私、存储与崩溃恢复

## 存储布局

全部持久化位于 `ctx.paths.data_dir`：

```text
data_dir/
├── lrs-state.sqlite3      # 权威状态
├── vectors/lancedb/       # 可删除重建的派生索引
└── debug/                 # 仅 raw 调试开关开启时
```

SQLite 是权威源；LanceDB 失败不影响任务结论，只影响 `past_cases` 检索。

## 默认隐私

始终保存：任务 / round / 正式任务描述、branch 与 checkpoint 总结、中间/最终报告、credits 与 usage 账本、Procedure 最小元数据、extension fingerprint、feedback / reminder / outbox / vector job。

默认关闭：

```toml
[storage]
store_agent_transcripts = false
store_raw_procedure_payloads = false
```

两个开关独立。Raw chat、普通智能体转录、Procedure 原始载荷、compact 中间文本、provider reasoning **默认不进** SQLite 调试表，也**永不**进入 LanceDB。

默认只索引：正式任务、branch/checkpoint terminal summary、中间/最终报告、feedback lesson。

## Embedding 与 rebuild

每个 generation 记录 selector、实际模型名（若有）、fingerprint、维度、schema。任一变化 → `embedding_generation_mismatch`，自动（`embedding.auto_rebuild`）或 `/swarm vectors rebuild [--force]` 手动重建：

1. 新建 generation / table  
2. 从 SQLite 总结层重嵌  
3. 逐批校验维度；不一致立即失败该 generation  
4. 验证后原子切换 active  
5. 旧 generation 延迟清理  

禁止截断 / padding 强行适配。重建期间 `past_cases` 返回明确 `vector_index_rebuilding`。

## 崩溃恢复

- 启动时将活动 round 标为 **`INTERRUPTED`**。
- 在途 input reservation 保留为 `estimated_unreconciled`。
- 默认**不**为 `INTERRUPTED` 调度反馈提醒。
- Maisaka 交付用 outbox 分阶段重试；报告含稳定 report ID 供消费者去重。
- 取舍说明：相对「崩溃极少」假设，优先默认隐私与较低持久化成本；需要排障时再打开 raw 开关。
