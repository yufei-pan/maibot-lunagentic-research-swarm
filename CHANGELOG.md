# Changelog

## 0.2.2 — 2026-10-14

### 修复

- `model:` selector（物理模型 pinning）在 Host 1.2.0+ 上全部失败，任务在形式化阶段即报「当前 Host 的物理模型固定接口签名不兼容」：Host 1.2.0 把内部 `generate_response_with_message_async` / `_build_prompt_message_factory` 改名为 `…_with_context_async` / `_build_prompt_context_factory`，1.3.0 又删除了 `TaskConfig.slow_threshold`（`extra="forbid"` 直接拒绝）。pinning 改走公开 `ctx.llm.generate(_with_tools)` 的 `task_name="utils"` + `model_name=<物理名>`，不再 import Host 内部 LLM 实现；Host 对显式 `model_name` 失败直接报错，不会换用其他模型
- 加载期 pinning 检查失败时写 warning 日志并列出受影响的 `model:` selector，不再只记在 `/swarm health` 里等任务失败才发现
- 形式化时 Host `message.build_readable` 失败（返回 `{success: False}`，例如 WebUI 本地聊天的消息缺 `user_nickname`）不再被当成聊天消息送进 LLM 导致「消息缺少字符串类型的 `role` 字段」整任务失败；改为写 warning 并不带聊天上下文继续

### 变更

- Host 最低版本升至 **1.2.5**（公开 LLM 能力区分 `task_name` 与 `model_name` 的首个版本）；SDK 仍为 **≥2.8.1**
- `model:` selector 的正数 `max_tokens` 现在与 `task:` 一致地显式生效；为 0/留空时仍依次继承 Host 模型、`extra_params` 与 `utils` 任务的 `max_tokens`。请求超时（`hard_timeout`）取 Host `utils` 任务配置

## 0.2.1 — 2026-09-20

### 修复

- `task:` selector 的公开 LLM 调用改为传 `task_name` 而非 `model`。SDK 2.8.1 会默认附带 `task_name="utils"`，若再把 `utils`/`planner` 放进 `model`，Host 会当成具体模型名并报「未找到名为 … 的模型」。`model:` selector 仍走物理 pinning，不受影响

## 0.2.0 — 2026-08-09

运行时硬化、网页搜索可用性与文档更新。

### 修复

- 报告协调：`retire_parent` 不再卡住已打开的报告 frontier；deadline 对已打开 epoch 也会武装 grace；合成崩溃可有限次重试，避免永久卡在 `REPORTING`
- Procedure 调用：把 catalog `timeout_seconds` 透传为 Host `api.call` 的 `timeout_ms`，避免慢 fetch/summarize 被默认 30s RPC 提前杀掉并记成 `provider_call_failed`
- 网页搜索：引擎 id `duckduckgo` → `ddgs`，改用 ddgs 自动 backends（旧 HTML backend 易被拦）；透传 timeout / region / safesearch / backend，并把 procedure `recency` 映射为 `timelimit`；配置版本 **1.4.0** 迁移旧引擎名

### 变更

- Host 最低版本升至 **1.1.0**（需要 `ctx.paths.data_dir` 统一持久化，以及 Docker `MaiMBot-plugin-data` 数据目录布局）；SDK 仍为 **≥2.7.1**
- Prompt 资源迁至仓库根目录 `i18n/zh-CN/`，manifest 声明 `locales_path = "i18n"`
- 内置角色默认模型选择：工具向角色用 `task:utils`，规划/核验向角色用 `task:planner`（摘要仍为 `task:mid_memory`），便于缓存友好的快慢模型混用
- README / marketplace 描述：说明 Lunagentic 命名与 swarm 架构，并补充记忆、embedding 历史经验、多引擎搜索、credits 价格警告等；注明记忆研究员可读当前聊天流历史消息

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
