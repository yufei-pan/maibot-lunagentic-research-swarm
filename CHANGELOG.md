# Changelog

## 0.2.0 — 2026-08-09

运行时硬化、网页搜索可用性与文档更新。

### 修复

- 报告协调：`retire_parent` 不再卡住已打开的报告 frontier；deadline 对已打开 epoch 也会武装 grace；合成崩溃可有限次重试，避免永久卡在 `REPORTING`
- Procedure 调用：把 catalog `timeout_seconds` 透传为 Host `api.call` 的 `timeout_ms`，避免慢 fetch/summarize 被默认 30s RPC 提前杀掉并记成 `provider_call_failed`
- 网页搜索：引擎 id `duckduckgo` → `ddgs`，改用 ddgs 自动 backends（旧 HTML backend 易被拦）；透传 timeout / region / safesearch / backend，并把 procedure `recency` 映射为 `timelimit`；配置版本 **1.4.0** 迁移旧引擎名

### 变更

- 内置角色默认模型选择：工具向角色用 `task:utils`，规划/核验向角色用 `task:planner`（摘要仍为 `task:mid_memory`），便于缓存友好的快慢模型混用
- README / marketplace 描述：说明 Lunagentic 命名与 swarm 架构，并补充记忆、embedding 历史经验、多引擎搜索、credits 价格警告等

### 测试

- 增加 retire_parent frontier、procedure RPC timeout、web_search / ddgs 与相关集成用例

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
