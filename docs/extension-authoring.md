# 扩展作者指南

第三方插件可通过公开 API 向麦麦深度调查组（LRS）提供智能体或 Procedure。配置与密钥留在**提供方自己的** `config.toml`；LRS 只发现定义并调用，不托管对方私有配置。

## 发现与刷新

LRS 通过 `ctx.api.list()` 扫描带元数据的公开 API，再以完整名 `plugin_id.<api>@1` 拉取定义。触发时机：

- LRS 加载
- 新 Task / new round
- 提供方调用 LRS 的 `refresh_extensions@1`
- 可配置周期刷新（`extensions.refresh_interval_seconds`）

```python
# 在你的插件中（加载完成后）请求 LRS 重扫
await self.ctx.api.call(
    "com.0-hz.lunagentic-research-swarm.refresh_extensions",
    version="1",
)
```

卸载或禁用提供方后，定义在下次扫描时消失；**不要**向 LRS 推入常驻 Python 对象。

## `describe_agents@1`

```python
from maibot_sdk import API

@API(
    "describe_agents",
    description="向 LRS 描述本插件提供的智能体",
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
            "agent_id": "myplugin.researcher",  # 命名空间 = 授权前缀
            "version": "1",
            "display_name": "自定义研究员",
            "description": "能力与边界说明",
            "character_prompt": "角色偏好……",
            "model_selector": "task:utils",  # 必须 task: 或 model:
            "protocol": "json_envelope",     # 或 native_tools
            "allowed_procedures": ["*"],
            "can_be_root": False,
            "auto_compact_tokens": None,
            "enabled": True,
        }
    ]
```

**校验要点：** ID 命名空间、selector 格式、protocol 枚举、禁止冒充 `core` / summarizer；整批失败时 provider 在 health 中为 `invalid`，不会静默兜底。

## `describe_procedures@1` / `invoke_procedure@1`

```python
@API(
    "describe_procedures",
    description="向 LRS 描述本插件提供的 Procedures",
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
            "arguments_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            "result_schema": {"type": "object"},
            "idempotent": True,
            "timeout_seconds": 30,
            "external_cost_kind": "none",
            "enabled": True,
        }
    ]


@API(
    "invoke_procedure",
    description="执行 LRS 请求的 Procedure",
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
    # scoped_metadata 含 task_id / round_id / branch_id / turn_id / agent_id
    return {
        "success": True,
        "data": {"ok": True},
        "error": None,
        "metadata": {
            "provider_plugin_id": self.plugin_id,
            "duration_ms": 0,
            "provenance": [],
            "external_cost": None,
        },
    }
```

失败时返回结构化结果，例如：

```json
{
  "success": false,
  "data": null,
  "error": {"code": "invalid_arguments", "message": "……"},
  "metadata": {"request_id": "…", "procedure_id": "myplugin.lookup"}
}
```

定义失效或已从 live catalog 移除后，**新调用**得到 `procedure_unavailable`；普通分支不会因此自动终止。

仅当 `idempotent=true` 且错误显式可重试时，LRS 才可能有限重试。超时后无法确认是否执行的非幂等 Procedure **不**自动重试。

## Provider 配置所有权

- 密钥、配额、路径白名单等留在提供方插件配置中。
- LRS `[procedures."<id>"]` / `[agents."<id>"]` 只做启用、timeout、selector、protocol 等 **LRS 侧** override。
- 不要把对方 secrets 写入 LRS config。

## 移除与在途语义

每个 round **冻结** catalog fingerprint。新增/修改默认下一 round 生效；**删除立即影响新边**：

| 场景 | 行为 |
|---|---|
| 在途目标 agent 调用 | 允许完成 |
| 完成后委派给仍存活的 agent | 正常继续 |
| 委派给已移除的自身或其他缺失 agent | 该边以 `agent_unavailable` 终结并分支总结 |
| 同 envelope 其他有效委派 | 不受影响 |
| 已冻结 round 内已开始的 Procedure | 按提供方返回完成 |
| 新 round / 空 catalog 上的调用 | `procedure_unavailable` |

## 推荐示例：fetch-url

`maibot-fetch-url-plugin` 建议在保留原 `fetch_url@1` 的同时暴露 `describe_procedures@1` + `invoke_procedure@1`，procedure id 为 `fetch_url.fetch`。缺失时 LRS 正常运行，health 中 `recommended_fetch.status = recommended_missing`。
