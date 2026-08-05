# Changelog

## 0.1.0 — 2026-08-05

首发扩展与发布候选。

### 新增

- 九个内置研究智能体，经同一 extension validator/registry 装载（默认 root：`builtin.quick_thinker`）
- 内置 memory / web 搜索 / past-cases / 分析 / provenance Procedures，以及 core `compact` / `checkpoint` / `terminate`
- 八个 Planner tools 与九个 `/swarm` 用户命令
- SQLite 权威状态 + LanceDB 可重建向量索引；embedding mismatch 显式 rebuild
- credits 账本、时间预算 / 60s grace、中间与最终报告（含确定性统计）
- feedback 事件、lesson 索引与 600 秒提醒（不自动改 prompt/路由）
- 推荐集成 `maibot-fetch-url-plugin`（非硬依赖）；缺失时 health 显示 `recommended_missing`

### 已知问题

- LanceDB 原生层在极端并发下仍可能不稳定；0.1.0 已将插件内 Lance 调用串行化到专用单线程池以降低风险。若仍遇崩溃，请保留 SQLite 权威数据并用 `/swarm vectors rebuild` 重建索引。
