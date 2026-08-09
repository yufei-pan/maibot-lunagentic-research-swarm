# Procedure 访问权与 Prompt 表面

日期：2026-08-07  
状态：已定稿（实现见同日变更）

## 问题

原先由智能体 `allowed_procedures` 声明「我能用哪些工具」。专职智能体只拿到窄列表，而一般智能体拿 `*`。结果是：

- 网页搜索 / 抓取等「通用智力」工具被绑在研究员身上；
- 记忆与历史案例工具的存在感主要来自 agent 卡上的窄列表，对协调者不直观；
- 冻结 system 对全员相同，逐 agent 可调用集只能写在 `[LRS runtime]`，但叙事仍像「agent 拥有 allowlist」。

## 决策

1. **访问权归 Procedure**：默认 `allowed_agents: ["*"]`（全体可用）；少数工具显式列出可调用的 `agent_id`。
2. **首发限制**：记忆族六项仅 `builtin.memory_researcher`；`builtin.past_cases` 仅 `builtin.past_case_researcher`。其余内置研究工具与 add-on（含 `fetch_url.fetch`）默认开放。`core.*` 仍对全体可用（不进 registry）。
3. **专职也拿到全部一般工具**；角色差异靠 `character_prompt` / 任务分配，而不是剥夺通用 Procedure。
4. **Prompt**：冻结 system 列出全部启用 Procedure 定义与 schema；受限项注明「仅某 agent 可调用，请委派」。`[LRS runtime]` 只列本智能体本 turn 可调用的 **ID**（含 core），供注意力机制对齐 system 里的定义。
5. Agent 侧 `allowed_procedures` 保留字段兼容，**不再用于执行与目录卡片**；内置定义统一为 `["*"]`。
6. 其它清理：`deep_thinker.description` 去掉运维向「建议大模型」措辞；`memory_researcher` 默认 selector 改为 `task:mid_memory`；`past_case_researcher` 保持 `task:utils`；runtime「只有最后一个有效」去掉「也就是本块」类赘语。

## 解析

```
callable(agent_id) =
  { p in frozen_procedure_catalog
    | p.allowed_agents == ["*"] or agent_id in p.allowed_agents }
```

执行器仍对普通 Procedure 做 `procedure_not_allowed`；core 仍走 `split_procedure_requests` 控制路径。

## Prompt 不变量

- 同一 round 内冻结 system 对所有智能体逐字节相同（含受限 Procedure 的完整定义与限制说明）。
- 身份与可调用 ID 子集只出现在每 turn 末尾的 `[LRS runtime]`。
- 对话历史中可能有多个 runtime 块；**只有最后一个有效**。

## 非目标

- 不为智能体引入 LRS 侧 temperature 覆盖（仍跟 Host task / 模型配置）。
- 不改变 summarizer 默认 `task:mid_memory`。
- 不要求 fetch-url 插件改 descriptor（缺省 `allowed_agents=["*"]` 即可）。
