"""内置 `builtin` provider 的九个默认智能体定义。"""

from __future__ import annotations

from lunagentic_research_swarm.agents.bundled.prompts import BUNDLED_CHARACTER_PROMPTS
from lunagentic_research_swarm.extensions.contracts import AgentDefinition

_MEMORY_PROCEDURES = (
    "builtin.chat_streams",
    "builtin.message_recent",
    "builtin.message_by_id",
    "builtin.message_time_range",
    "builtin.person_lookup",
    "builtin.knowledge_search",
)

_PROVENANCE_PROCEDURES = (
    "builtin.normalize_urls",
    "builtin.organize_provenance",
)

_RESEARCHER_PROCEDURES = (
    "builtin.web_search",
    *_PROVENANCE_PROCEDURES,
    "fetch_url.fetch",
)

_PAST_CASE_PROCEDURES = (
    "builtin.past_cases",
    *_PROVENANCE_PROCEDURES,
)

_QUANTITATIVE_PROCEDURES = (
    "builtin.calculate",
    "builtin.statistics",
    "builtin.convert_units",
    *_PROVENANCE_PROCEDURES,
)

_CATALOG: tuple[dict[str, object], ...] = (
    {
        "agent_id": "builtin.quick_thinker",
        "version": "1",
        "display_name": "快速思考者",
        "description": "默认 root：快速拆分问题、动态委派和收敛路线。",
        "model_selector": "task:utils",
        "can_be_root": True,
        "allowed_procedures": ["*"],
    },
    {
        "agent_id": "builtin.deep_thinker",
        "version": "1",
        "display_name": "深度思考者",
        "description": "复杂推理、长期规划与综合约束；建议使用大模型或高思考配置。",
        "model_selector": "task:planner",
        "can_be_root": True,
        "allowed_procedures": ["*"],
    },
    {
        "agent_id": "builtin.debater",
        "version": "1",
        "display_name": "辩手",
        "description": "质疑前提、提供第二意见，寻找反例与风险。",
        "model_selector": "task:replyer",
        "allowed_procedures": ["*"],
    },
    {
        "agent_id": "builtin.researcher",
        "version": "1",
        "display_name": "外部研究员",
        "description": "设计搜索词、选择搜索引擎，并要求外部证据。",
        "model_selector": "task:utils",
        "allowed_procedures": list(_RESEARCHER_PROCEDURES),
    },
    {
        "agent_id": "builtin.memory_researcher",
        "version": "1",
        "display_name": "记忆研究员",
        "description": "设计聊天、消息、人物与知识库查询，不负责过度解释原始结果。",
        "model_selector": "task:utils",
        "allowed_procedures": list(_MEMORY_PROCEDURES),
    },
    {
        "agent_id": "builtin.knowledge_reporter",
        "version": "1",
        "display_name": "知识报告员",
        "description": "低 deliberation 地报告模型已知知识与不确定性。",
        "model_selector": "task:replyer",
        "allowed_procedures": ["*"],
    },
    {
        "agent_id": "builtin.past_case_researcher",
        "version": "1",
        "display_name": "历史案例研究员",
        "description": "查询相似任务、历史决定、反馈和真实结果。",
        "model_selector": "task:utils",
        "allowed_procedures": list(_PAST_CASE_PROCEDURES),
    },
    {
        "agent_id": "builtin.evidence_verifier",
        "version": "1",
        "display_name": "证据核验员",
        "description": "交叉验证引用、识别来源冲突，并区分事实与推断。",
        "model_selector": "task:planner",
        "allowed_procedures": ["*"],
    },
    {
        "agent_id": "builtin.quantitative_analyst",
        "version": "1",
        "display_name": "定量分析员",
        "description": "计算、比较数量级，检查数值假设与成本收益。",
        "model_selector": "task:planner",
        "allowed_procedures": list(_QUANTITATIVE_PROCEDURES),
    },
)


def bundled_agent_definitions() -> list[AgentDefinition]:
    """构造内置默认智能体定义；调用方再走 registry/validator 装入。"""

    definitions: list[AgentDefinition] = []
    for item in _CATALOG:
        agent_id = str(item["agent_id"])
        payload = {
            **item,
            "character_prompt": BUNDLED_CHARACTER_PROMPTS[agent_id],
            "protocol": "json_envelope",
            "enabled": True,
        }
        definitions.append(AgentDefinition.model_validate(payload))
    return definitions
