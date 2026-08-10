# 麦麦深度调查组（Lunagentic Research Swarm）

面向 MaiBot 的深度研究智能体蜂群插件。

**Lunagentic** 是 **Luna** 与 **agentic** 的组合词。本插件以 *agentic research swarm* 架构协调多代专职智能体：拆分问题、检索证据、互相质疑、压缩上下文，并在时间提示与 credits 预算下持续产出带统计信息的中间报告与最终结论——把「帮我认真查清楚」变成可启动、可暂停、可追踪的深度调查。

你用自然语言提出目标即可。后台角色会**上网检索**、查麦麦记忆与知识库、交叉核验、算数量级、整理来源；中间进展与最终结论推送到对话，也可用 `/swarm` 看状态、交反馈。搜索引擎可配 **ddgs**、自建 **SearXNG**、**Tavily**、**You** 等，并在装了推荐抓取插件时打开原文核对——不是只靠模型「凭印象回答」。插件还带有 **embedding 向量索引**，能检索相似的历史调查与你留下的反馈（含踩过的坑），在后续任务里主动参考，越用越会避开重复失误。

部署时可把价格有竞争力的快速模型（例如 GPT-5.6 Luna、DeepSeek-V4-Flash）与大型知识模型（例如 GPT-5.6 Sol、Claude Opus 5，可酌情降低 reasoning 档位）组合使用：轻量角色走快模型，重推理与核验走大模型。

**开箱即用（batteries included）**

- **九个内置角色：** 快速/深度思考、辩手、外部与记忆研究员、知识报告、历史案例、证据核验、定量分析  
- **网页搜索与取证：** 多引擎搜索开箱可用；可选 SearXNG / Tavily / You；配合推荐的 [fetch-url](https://github.com/yufei-pan/maibot-fetch-url-plugin) 抓取全文  
- **记忆与知识库：** 可查询聊天、消息、人物与 Host 知识库（记忆研究员）  
- **历史经验与 embedding：** 索引过往调查与质量反馈；新任务可检索相似案例与 lesson，从成功与失误中改进  
- **其它研究工具：** 计算/统计/换算、来源整理  
- **任务全周期：** 启动后立刻有 `task_id`；支持暂停/继续/停止、补充背景、中间报告与最终报告、结束后的质量反馈  
- **可预期的花费与节奏：** 按 Host 模型价格计 credits，时间预算与宽限可配；快慢模型可混用控成本  

**可扩展：** 其它 MaiBot 插件可通过公开 API 注册自定义角色与工具，装上即出现在调查组的可用目录里（见文末「进阶」）。你只安装启用，不必改本插件代码。

- **插件 ID：** `com.0-hz.lunagentic-research-swarm`
- **许可：** MIT（见 [LICENSE](LICENSE)）
- **仓库：** <https://github.com/yufei-pan/maibot-lunagentic-research-swarm>

---

## 1. 快速开始

1. 在 MaiBot **插件市场**安装并启用本插件。  
2. 打开 Host **WebUI → 本插件配置**，确认模型与（可选）网页搜索等设置合理（见 §4）。  
3. （推荐）同样安装 [网页抓取插件（fetch-url）](https://github.com/yufei-pan/maibot-fetch-url-plugin)，调查时才能稳定取网页全文。  
4. 在聊天里用自然语言提出调查需求，例如：「帮我深度调研……」。  
5. 等待对话中的中间报告 / 最终报告；需要时用下面的 `/swarm` 命令查看状态或提交反馈。

首次使用建议先发一条 `/swarm health`，确认没有明显错误。

> **和麦麦怎么配合：** 麦麦能听懂这类请求并替你启动、查询、暂停/继续/停止调查，或把你补充的背景写进进行中的任务。你也可以不经过它、自己发 `/swarm …` 命令。两种方式可以混用。

---

## 2. 日常怎么用

### 2.1 发起与跟踪

- **发起：** 直接在对话里说明你要查什么、有哪些约束或必须覆盖的点（麦麦会据此开一次调查）。  
- **跟踪：**  
  - 看插件推送到对话里的报告；或  
  - 自己发送 `/swarm status`、`/swarm status <task_id>`、`/swarm tasks`；或  
  - 问麦麦：「那个调研现在怎么样了？」  
- **补充材料：** 继续发背景、链接或纠正；需要写进任务时，可以说「把这些补进正在做的调查」。  
- **暂停 / 继续 / 停止：** 直接说「先暂停 / 继续做 / 停掉这个调研」即可；也可用 `/swarm status` 确认是否已变成 `PAUSED`、`STOPPED` 等。

启动后会有一个 `task_id`（出现在状态输出或麦麦的回复里），之后查进度、反馈都用得到它。

### 2.2 常见任务状态

| 状态 | 含义 |
|---|---|
| 进行中（如 `RUNNING`） | 正在调查 |
| `PAUSED` | 已暂停；在时限内需要继续，否则可能变为 `EXPIRED` |
| `COMPLETED` / `COMPLETED_WITH_ERRORS` | 已结束（后者表示过程中有错误但仍产出了结论） |
| `STOPPED` | 已手动停止 |
| `EXPIRED` | 暂停过久未继续 |
| `INTERRUPTED` | Host 重启等导致中断 |

### 2.3 反馈

调查结束后，可用：

```text
/swarm feedback <task_id> accepted|mixed|rejected [备注…]
```

也可以在对话里更详细地跟麦麦说哪些有用、哪些错了、缺了什么，让它帮你记反馈。反馈会留下可检索的经验记录，**不会**自动改插件的默认提示词或模型路由。

---

## 3. 聊天命令一览

在聊天框**由你直接输入**（可在配置里关闭整组命令）。下面这些不依赖麦麦是否主动调用工具：

| 命令 | 作用 |
|---|---|
| `/swarm status` | 插件是否在忙、根智能体、健康摘要 |
| `/swarm status <task_id>` | 某个任务的详细状态 |
| `/swarm tasks` | 最近任务列表 |
| `/swarm tasks <status>` | 按状态筛选（如 `RUNNING`） |
| `/swarm stats` / `/swarm stats <task_id>` | 统计 |
| `/swarm agents` | 当前可用的研究角色列表 |
| `/swarm procedures` | 当前可用的研究工具列表 |
| `/swarm health` | 健康检查（存储、向量、扩展、推荐抓取插件等） |
| `/swarm vectors status` | 向量索引状态 |
| `/swarm vectors rebuild` | 重建向量索引（默认关闭，见下） |
| `/swarm vectors rebuild --force` | 强制重建 |
| `/swarm feedback <task_id> …` | 提交简化反馈 |

**维护命令：** `/swarm vectors rebuild` 默认不允许。若要在 WebUI 打开：

- 将 `allow_vector_rebuild` 设为 `true`；  
- 建议同时填写 `maintenance_allowed_user_ids`（平台用户 ID 或 MaiBot `person_id`）。列表为空表示不限制调用者——生产环境请谨慎。

> **提示：** 列任务、看健康、交简单反馈等，用命令往往最快；需要麦麦结合上下文理解「刚才那个」并操作时，用自然语言即可。

---

## 4. 配置（WebUI）

在 Host WebUI 编辑本插件配置即可。完整字段说明与默认值见插件自带的 `config.default.toml`（带注释）。下面是最常改的几块。

### 4.1 模型

本插件通过 **selector** 选用 Host 里已配置的模型，格式为 `task:名称` 或 `model:物理名`。

请先保证 Host 模型配置里存在对应任务/模型，再改本插件：

| 配置 | 作用 |
|---|---|
| `llm.default_selector` | 非空时，统一覆盖各研究角色与摘要器的默认模型（不含向量化） |
| 各 `[agents."…"].selector` | 只改某一个角色 |
| `summarizer.selector` | 摘要用模型；留空则跟随上面的默认，再否则为 `task:mid_memory` |
| `embedding.selector` | 向量化，默认 `task:embedding` |
| `plugin.root_agent` | 默认从哪个角色开局，一般保持 `builtin.quick_thinker` |

出厂时研究角色大致分成两档（可在 WebUI 改）：

- **`task:utils`：** 快速思考者、外部研究员、记忆研究员、历史案例研究员  
- **`task:planner`：** 深度思考者、辩手、知识报告员、证据核验员、定量分析员  

摘要默认 `task:mid_memory`。若 `/swarm health` 里 `physical_pinning` 报错，说明某个 `model:` 在当前 Host 不可用，需要改回 `task:` 或修好 Host 侧配置。

### 4.2 时间与花费

| 配置 | 默认 | 含义 |
|---|---:|---|
| `timing.default_time_budget_seconds` | 600 | 单次调查默认时间预算（秒） |
| `timing.grace_period_seconds` | 60 | 超时后的收尾宽限 |
| `timing.pause_timeout_seconds` | 1200 | 暂停过久未继续则过期 |
| `timing.feedback_wait_seconds` | 600 | 结束后多久提醒你反馈 |
| `budget.default_effort_credits` | 100 | 基础 credits；实际初始预算还会乘上启动时的努力程度 |

Credits 与价格细节见 [docs/credits-and-reporting.md](docs/credits-and-reporting.md)。简要规则：Host 模型价格优先；只有你在本插件里为某模型单独写了价格覆盖时才改用插件价格；`1.0` 价格单位 = **100 credits**。

> **警告：模型价格配错时，credits 预算可能形同虚设。**  
> 调查组按「模型单价 × 用量」扣 credits 来约束深度与轮次。若 Host（或本插件的价格覆盖）把**付费模型**的 `price_in` / `price_out` 写成 `0`，或根本没填价格，系统会按**免费**计量——任务仍会按预算继续跑，但真实 API 账单可能迅速膨胀。请务必在 Host 里为实际在用的模型填好单价；只有真正免费的模型才应保持为 `0`。时间预算仍会限制墙钟时间，但挡不住「单价被当成 0」时的费用失控。

### 4.3 网页搜索

调查过程中由研究角色自动搜索，无需你在聊天里单独点「搜索」。

| 配置 | 说明 |
|---|---|
| `enabled_engines` | 启用 `ddgs` / `searxng` / `tavily` / `you` 中的哪些 |
| `timeout_seconds` / `max_results` | 超时与条数 |
| `ddgs_region` / `ddgs_safesearch` / `ddgs_backend` | ddgs 选项 |
| `searxng_url` | 自建 SearXNG 地址 |
| `tavily_api_key`、`you_*` | 对应服务的密钥与地址 |

密钥只保存在你的 Host 上，不要发到公开场合。

### 4.4 报告、隐私与其它

- **报告：** `[reporting]` 控制是否投递中间/最终报告及长度上限。  
- **反馈提醒：** `[feedback]`。  
- **隐私：** 默认不把智能体全文对话和工具原始载荷写入调试存储；需要排障再开 `[storage]` 相关开关。说明见 [docs/privacy-and-recovery.md](docs/privacy-and-recovery.md)。  
- **扩展刷新间隔：** `[extensions].refresh_interval_seconds`（一般保持默认即可）。

---

## 5. 内置研究角色与能力

用 `/swarm agents`、`/swarm procedures` 可看当前实例里**实际启用**的列表（含你另外装的扩展插件）。

### 5.1 角色（默认九个）

| 角色 | 做什么 |
|---|---|
| 快速思考者 | 默认开局：拆问题、分工、收敛 |
| 深度思考者 | 复杂推理与长期约束 |
| 辩手 | 找反例、风险与替代解释 |
| 外部研究员 | 外网检索与证据收集 |
| 记忆研究员 | 查聊天、消息、人物、知识库 |
| 知识报告员 | 整理模型已知知识与不确定点 |
| 历史案例研究员 | 查以往类似调查与反馈 |
| 证据核验员 | 核对来源是否支撑结论 |
| 定量分析员 | 数量级、单位与计算核对 |

### 5.2 常用能力

| 能力 | 说明 |
|---|---|
| 网页搜索 | ddgs / SearXNG / Tavily / You（取决于你的配置） |
| 网页全文 | 需安装 fetch-url 插件 |
| 记忆与知识库查询 | 仅记忆研究员使用 |
| 历史案例 | 仅历史案例研究员使用 |
| 计算 / 统计 / 单位换算 | 定量分析等场景 |
| 来源整理 | URL 规范化与 provenance 整理 |

---

## 6. 可选集成

| 集成 | 建议 |
|---|---|
| fetch-url 插件 | 强烈推荐 |
| 自建 SearXNG | 适合不想用公网聚合搜索、或要内网源的情况 |
| Tavily / You | 在配置里填密钥即可 |
| 其它扩展插件 | 安装后若作者已对接本插件，角色/工具会出现在 `/swarm agents`、`/swarm procedures` |

---

## 7. 进阶：为调查组增加自定义角色与工具（插件开发者）

如果你在写**另一个** MaiBot 插件，希望把自定义智能体或工具提供给深度调查组调用，按下列公开 API 对接。普通使用者只需安装你的插件，不必阅读本节。

完整备忘：[docs/extension-authoring.md](docs/extension-authoring.md)。

### 7.1 发现

本插件会扫描带元数据的公开 API。你的插件可在加载后请求刷新：

```python
await self.ctx.api.call(
    "com.0-hz.lunagentic-research-swarm.refresh_extensions",
    version="1",
)
```

也可等待配置的周期刷新。密钥留在你自己的插件配置中。

### 7.2 提供智能体：`describe_agents@1`

```python
from maibot_sdk import API

@API(
    "describe_agents",
    description="向深度调查组描述本插件提供的智能体",
    version="1",
    public=True,
    metadata={
        "lunagentic_extension": "agents",
        "lunagentic_contract": "1",
    },
)
async def describe_agents(self) -> list[dict]:
    return [
        {
            "agent_id": "myplugin.researcher",
            "version": "1",
            "display_name": "自定义研究员",
            "description": "能力与边界",
            "character_prompt": "角色说明……",
            "model_selector": "task:utils",
            "protocol": "json_envelope",
            "allowed_procedures": ["*"],
            "can_be_root": False,
            "auto_compact_tokens": None,
            "enabled": True,
        }
    ]
```

`agent_id` 命名空间须与你的插件一致。用户可在本插件配置里覆盖 selector / 启用状态等。

### 7.3 提供工具：`describe_procedures@1` + `invoke_procedure@1`

```python
@API(
    "describe_procedures",
    description="向深度调查组描述 Procedures",
    version="1",
    public=True,
    metadata={
        "lunagentic_extension": "procedures",
        "lunagentic_contract": "1",
    },
)
async def describe_procedures(self) -> list[dict]:
    return [
        {
            "procedure_id": "myplugin.lookup",
            "version": "1",
            "display_name": "查找",
            "description": "……",
            "arguments_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            "result_schema": {"type": "object"},
            "idempotent": True,
            "timeout_seconds": 30,
            "external_cost_kind": "none",
            "allowed_agents": ["*"],
            "enabled": True,
        }
    ]


@API(
    "invoke_procedure",
    description="执行深度调查组请求的 Procedure",
    version="1",
    public=True,
)
async def invoke_procedure(
    self,
    *,
    procedure_id: str,
    request_id: str,
    arguments: dict,
    scoped_metadata: dict,
) -> dict:
    return {
        "success": True,
        "data": {"ok": True},
        "error": None,
        "research_credits_charged": 0.0,
        "metadata": {
            "provider_plugin_id": self.plugin_id,
            "duration_ms": 0,
            "provenance": [],
            "external_cost": None,
        },
    }
```

`allowed_agents` 限制哪些研究角色能调用该工具。对接完成后，用一次真实调查验证即可。

---

## 8. 本仓库开发（维护者）

仅用于开发本插件源码，与插件市场使用无关。

```bash
PYTHONPATH=../maibot-plugin-sdk python tests/smoke_test.py
PYTHONPATH=.:../maibot-plugin-sdk pytest -q
```

Live 测试见 `.debug_api_call_credentials.example`，勿提交真实密钥。

---

## 许可

MIT — 见 [LICENSE](LICENSE)。
