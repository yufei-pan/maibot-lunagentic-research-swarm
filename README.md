# 麦麦深度调查组（Lunagentic Research Swarm）

面向 MaiBot 的深度研究智能体蜂群插件。

**Lunagentic** 是 **Luna** 与 **agentic** 的组合词。本插件以 *agentic research swarm* 架构协调多代专职智能体：拆分问题、检索证据、互相质疑、压缩上下文，并在时间提示与 credits 预算下持续产出带统计信息的中间报告与最终结论。

部署时可把价格有竞争力的快速模型（例如 GPT-5.6 Luna、DeepSeek-V4-Flash-0731）与大型知识模型（例如 GPT-5.6 Sol、Claude Opus 5，可降低 reasoning 档位）组合使用。**这些名称只是部署示例，不构成依赖或默认固定路由。** 插件兼容任何满足 MaiBot LLM 调用契约的模型。

## 安装

1. 将本仓库放入 Host 的 `plugins/`（或符号链接）。
2. 复制 `config.default.toml` 为插件 live `config.toml`，按需填写搜索引擎密钥等；**不要提交含密钥的 live config**。
3. 在 Host 中配置 `task:` / `model:` selector（见下）。依赖见 `_manifest.json` / `pyproject.toml` / `requirements.txt`（三者已同步）。
4. **推荐（非必需）** 同时安装 [`maibot-fetch-url-plugin`](https://github.com/yufei-pan/maibot-fetch-url-plugin)，以提供网页全文抓取 Procedure。缺失时插件仍可正常加载，`/swarm health` 会显示 `recommended_fetch: recommended_missing`。

运行期依赖：`pydantic`、`httpx`、`ddgs`、`lancedb`（见 manifest）。**不**把 fetch-url 列为硬依赖。

## 模型选择与物理 pinning

- Selector 必须写成 `task:…` 或 `model:…`，禁止裸字符串。
- 全局 `llm.default_selector`、逐 agent override、总结器 selector 可分别配置；优先级为 agent/摘要器显式配置 > 非空 default_selector > 内置默认；embedding selector 独立。
- `model:` 物理 pinning 走 Host 内部路径，**脆弱且非正式 SDK 合约**。加载时做兼容检查，健康状态见 `/swarm health` 的 `physical_pinning`。不可用时该 selector 被显式拒绝，不会静默回退。
- **`@LLMProvider` 不用于 pinning**；它只注册新的 backend `client_type`。

## Credits 与价格

- 模型配置价格数值 `1.0` = **100 credits**；界面货币标签（¥ / $ / € 等）一律忽略。
- 默认基础预算 **100 credits**；`initial = default_effort_credits × effort_level`。
- **Host 价格优先**；仅当插件 `[pricing.models."<name>"]` 写了该模型条目时，才**完整覆盖** Host 同模型全部价格字段（未写字段按 0 / 免费）。详见 `config.default.toml` 中醒目 Note，以及 [docs/credits-and-reporting.md](docs/credits-and-reporting.md)。
- Host 与插件均无价格时按免费（0）计；低于约 500k cache-miss 输入 / 50k 输出 token 估算阈值时发出低预算警告，但不拒绝启动。

## 时间、控制与报告

- 默认时间预算 120s，超时后 **60s grace**；frontier 齐备可提前结束 grace。
- 用户命令与 Planner 工具见下方「用户命令」；`start_deep_research` **立即返回** `task_id`。
- `pause_deep_research`：等待在途调用结算后进入 `PAUSED`；超时未 `continue_deep_research` → `EXPIRED` 并释放 raw 上下文。
- `stop_deep_research`：取消当前 generation，释放 raw 上下文，进入 `STOPPED`。
- 中间报告：先持久化到 SQLite / outbox，再 Maisaka append + trigger；最终报告含确定性统计区块。
- `continue_deep_research` 在无活动叶子时只从 **summary layer** 开新 round，不回放 raw。
- Core procedures：`core.compact` / `core.checkpoint` / `core.terminate`。
- Procedure 请求可带外层 `credits`（预算提示，不预扣）；handler 经 `research_credits_charged` 事后扣研究余额。自动 compact 不扣；智能体请求的 `core.compact` 会扣。
- **Turn 语义：** Procedure 结果进入分支上下文供子委派继承，不会像聊天 tool-call 自动回传同一气泡。若 agent 要自己阅读结果，应**显式自委派**。若本 turn 已执行非控制 Procedure 且 `delegations` 为空，runtime 会静默注入一次自委派作为兜底（余额规则与显式委派相同；不在 agent prompt 中宣传）。鼓励同一 envelope 内并行多条 procedure / 委派。
- **System 契约：** 冻结目录以可读 Markdown 呈现（可用 Procedure：控制 + 研究；可委派智能体身份卡；有用时附模型费用参考，按「每 100 万 token 多少 credits」渲染）。共同身份段来自 `prompts/zh-CN/swarm_system.txt`。`character_prompt` 只出现在子任务 assignment。工作方式与输出协议在 system 末尾单独说明；不向 agent 暴露实现细节（如 external_cost、目录指纹、第三方 provider 边界）。
- **`[LRS runtime]` 块：** system 前缀对全轮所有智能体逐字节相同（cache 要求），因此逐 agent 的信息全部集中在每次调用末尾追加的这一个块里：身份（`agent_id` / 职责 / 自委派该写什么）、任务分配（`character_prompt` + 子任务；根调用为协调者说明）、以及运行时状态（可调用 Procedure 子集、剩余时间、credits、上一 turn 实际扣费、活动/排队数）。
- **历史只追加：** 旧的 `[LRS runtime]` 块保留原位不删改，上一次请求因此始终是下一次请求的逐字节前缀，最大化 provider 前缀 cache；块内与 system 都声明「只有最后一个有效」。任务分配每 turn 重新渲染，故 compact 后分支仍知道自己的角色与子任务。

## 九个默认智能体与内置 Procedures

| 智能体 | 默认 selector | 角色摘要 |
|---|---|---|
| `builtin.quick_thinker` | `task:utils` | 默认 root：快速建图与委派 |
| `builtin.deep_thinker` | `task:planner` | 复杂推理与长期约束 |
| `builtin.debater` | `task:replyer` | 第二意见与反例 |
| `builtin.researcher` | `task:utils` | 外部搜索与证据 |
| `builtin.memory_researcher` | `task:mid_memory` | 专职记忆族查询 |
| `builtin.knowledge_reporter` | `task:replyer` | 报告模型已知知识 |
| `builtin.past_case_researcher` | `task:utils` | 专职相似历史案例 |
| `builtin.evidence_verifier` | `task:planner` | 证据核验 |
| `builtin.quantitative_analyst` | `task:planner` | 数值与量级 |

内置 Procedures（节选）：`builtin.web_search`、memory 六件套、`builtin.past_cases`、`builtin.calculate` / `statistics` / `convert_units`、`builtin.normalize_urls` / `organize_provenance`、`builtin.contractor`（旁路承包商：目录智能体作 outsider 工具，新鲜上下文、无子委派）。四搜索引擎（DuckDuckGo / SearXNG / Tavily / You）仅在配置有效时进入目录。计费语义见 [docs/credits-and-reporting.md](docs/credits-and-reporting.md)。

## 用户命令

聊天中通过 `/swarm …` 调用（可由 `[commands].enabled` 关闭）。复杂任务控制与完整反馈请用下方 Planner 工具。

| 命令 | 功能 |
|---|---|
| `/swarm status` | 插件运行概要：当前忙任务、根智能体与健康摘要 |
| `/swarm status <task_id>` | 指定任务的状态、round、分支、队列与报告概况 |
| `/swarm tasks` | 列出本会话最近深度调查任务 |
| `/swarm tasks <status>` | 按状态过滤任务列表（如 `RUNNING`、`PAUSED`） |
| `/swarm stats` | 插件聚合统计 |
| `/swarm stats <task_id>` | 指定任务的统计 |
| `/swarm agents` | 列出 live 智能体目录（不含 prompt / 密钥） |
| `/swarm procedures` | 列出 live Procedure 目录 |
| `/swarm health` | SQLite / 向量 / pinning / 扩展 / 推荐 fetch / 队列 / outbox / 提醒等健康度 |
| `/swarm vectors status` | 向量索引 generation、selector 与重建 job 状态 |
| `/swarm vectors rebuild` | 手动重建向量索引（维护命令，见下） |
| `/swarm vectors rebuild --force` | 即使 fingerprint 未变也强制创建新 generation |
| `/swarm feedback <task_id> accepted\|mixed\|rejected [notes…]` | 提交简化反馈；复杂反馈请用 Planner 工具 |

维护命令（`/swarm vectors rebuild`）受 `[commands]` 约束：

- `allow_vector_rebuild` 默认 **false**；设为 true 才允许手动重建。
- `maintenance_allowed_user_ids` 每项可填 Host 命令 RPC 的 **`user_id`**（平台用户 ID），也可填 MaiBot **`person_id`**（`md5(f"{platform}_{user_id}")` 的 32 位 hex，跨适配器唯一）。两种格式不会碰撞，可混填；person_id 由插件用 `ctx.person.get_id` 现算比对。
- **空列表 = 不限制**（任何聊天成员都可通过白名单检查）。生产环境请填入维护者身份，或保持 `allow_vector_rebuild = false`。

### Planner 工具（Maisaka / LLM）

| 工具 | 功能 |
|---|---|
| `start_deep_research` | 异步启动一次深度调查；立即返回 `task_id`，不等 formalize / 总结器 |
| `pause_deep_research` | 暂停：等待在途工作结算，不启动新的研究调用 |
| `continue_deep_research` | 继续或重启；可增减 signed credits、重置报告时间预算；无活动叶子时只从 summary layer 开新 round |
| `stop_deep_research` | 停止并丢弃迟到结果；不调用总结器；释放 raw 上下文 |
| `add_research_context` | 向仍在运行或可继续的任务广播补充信息 |
| `get_research_status` | 查询指定任务的运行状态、round、分支与上下文释放状态 |
| `list_research_tasks` | 列出任务；可按状态与 ISO 时间范围过滤（最多 100 条） |
| `submit_research_feedback` | 提交完整质量反馈、纠正与 outcome；生成可检索 lesson |

公开 API：`refresh_extensions@1` — 请求重新扫描智能体与 Procedure provider。

## 协议

- 默认 **JSON envelope**；可按 agent 覆写为 **native tools**。
- Native 允许 toolcall 无正文。
- 格式错误：有限本地修复 + **同模型一次** correction turn；之后仍非法则终结分支。

## 存储、隐私与恢复

- SQLite 权威；LanceDB 仅存可重建向量。默认不落盘 agent transcript / raw procedure payload。
- Embedding selector / 模型 fingerprint / 维度 / schema 任一变化 → 明确 mismatch，自动或 `/swarm vectors rebuild` 手动重建，原子切换 generation。
- 进程崩溃后活动 round → `INTERRUPTED`（默认不发 feedback 提醒）。
- 详见 [docs/privacy-and-recovery.md](docs/privacy-and-recovery.md)。

## Feedback 与学习边界

- Feedback 为不可变事件；可用于检索排序、统计与可见 lesson。
- **不**自动改写内置 prompt、agent selector 或路由。
- `COMPLETED` / `COMPLETED_WITH_ERRORS` / 手动 `STOPPED` 后默认 600s 无反馈提醒一次；`continue` 或提交 feedback 取消；`EXPIRED` / `INTERRUPTED` 不提醒。
- 详见 [docs/credits-and-reporting.md](docs/credits-and-reporting.md)。

## 可选集成

| 集成 | 状态 |
|---|---|
| `maibot-fetch-url-plugin` | **推荐**，非硬依赖 |
| 文件仓库（file depot） | **独立未来 provider**；LRS 不提供 shell / 任意路径访问 |
| `@LLMProvider` | 不用于物理 pinning |

扩展作者契约见 [docs/extension-authoring.md](docs/extension-authoring.md)。

## 测试

```bash
# 离线 smoke（最后一行应为 ok: …）
PYTHONPATH=../maibot-plugin-sdk python tests/smoke_test.py

# 完整 pytest
PYTHONPATH=.:../maibot-plugin-sdk pytest -q

# 窄范围 ruff
python -m ruff check plugin.py lunagentic_research_swarm tests
```

仅在明确平台条件（例如缺少 Lance wheel）时允许 pytest marker skip，并应在此 README 说明。默认离线套件（不含 live markers）在 Linux x86_64 / aarch64 常规环境期望 **0 skipped**。

### Live LLM tests

可选真实 LLM / 工具层，按 marker 分档。默认 `pytest` 通过 `addopts` **排除**全部 `live_llm*` markers（即使存在可用凭证也不会跑慢速 live 测试）。显式 `-m live_llm…` 覆盖默认过滤以 opt-in；无可用凭证时 **skip**（不算失败）。模板见仓库根 `.debug_api_call_credentials.example`（复制为 gitignored 的 `.debug_api_call_credentials`）。**不要**提交真实 endpoint / API key。

| Marker | 内容 |
|---|---|
| `live_llm` | 协议 smoke（JSON envelope） |
| `live_llm_e2e` | 薄垂直切片 A/B + light judge |
| `live_llm_thorough` | 真实九智能体目录 + 完整内置 Procedure 目录；`web_search` 用 stub fixture；真实 `prompts/zh-CN` 总结器 + deep judge |
| `live_llm_live_tools` | 同上，但真实 `web_search`（需 `web_search_enabled = true`）+ deep judge；含 `test_live_agent_identity`（身份 / 可调用 ID / 委派角色卡） |

```bash
# 默认：仅离线（live markers 被 deselect）
pytest -q

# opt-in 各档（-m 覆盖 pyproject addopts）
pytest -m live_llm -v
pytest -m live_llm_e2e -v
pytest -m live_llm_thorough -v
pytest -m live_llm_live_tools -v
```

## 许可

MIT — 见 [LICENSE](LICENSE)。
