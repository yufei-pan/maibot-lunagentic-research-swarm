from __future__ import annotations

from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions
from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.llm.protocol_prompt import protocol_system_section
from lunagentic_research_swarm.models import FormalizedTask
from lunagentic_research_swarm.runtime.context import RuntimeHeader, StablePromptBuilder


def test_json_envelope_system_includes_schema_and_examples() -> None:
    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="swarm",
        bot_profile={"nickname": "n"},
        agent_catalog={},
        procedure_catalog={},
        pricing={},
    )
    system = builder.system_message_for_protocol("json_envelope")
    assert builder.catalog_system_message in system
    assert "## 工作方式" in system
    assert "## 输出协议（json_envelope）" in system
    assert "最终输出" in system
    assert "自动自委派" not in system
    assert "并行" in system
    assert '"report"' in system or "`report`" in system
    assert "procedures" in system
    assert "delegations" in system
    assert "core.terminate" in system
    assert system.endswith(protocol_system_section("json_envelope"))


def test_runtime_header_explains_role_and_json_output_contract() -> None:
    header = RuntimeHeader("br_1", 1, "cap", 120, 10.0, 1, 0, protocol="json_envelope")
    content = header.message()["content"]
    assert content.startswith("[LRS runtime]")
    assert "非用户聊天" in content
    assert "只有最后一个有效" in content
    assert "也就是本块" not in content
    assert "protocol=json_envelope" in content
    assert "最终输出" in content
    assert "仅输出一个 JSON object" in content
    assert "procedures" in content
    assert "自委派" in content
    assert "自动自委派" not in content
    assert "并行" in content


def test_native_tools_protocol_section_asks_for_submit_swarm_turn() -> None:
    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="swarm",
        bot_profile={},
        agent_catalog={},
        procedure_catalog={},
        pricing={},
    )
    system = builder.system_message_for_protocol("native_tools")
    assert "submit_swarm_turn" in system
    assert "最终输出" in system
    assert "自动自委派" not in system
    reminder = RuntimeHeader("br_1", 1, "cap", 10, 0.0, 1, 0, protocol="native_tools").message()["content"]
    assert "submit_swarm_turn" in reminder
    assert "自委派" in reminder
    assert "自动自委派" not in reminder


def test_frozen_contract_is_agent_facing_and_omits_noise() -> None:
    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="swarm",
        bot_profile={"nickname": "麦麦"},
        agent_catalog={
            "builtin.researcher": {
                "agent_id": "builtin.researcher",
                "display_name": "研究员",
                "description": "外部检索",
                "character_prompt": "SECRET_PERSONA_SHOULD_NOT_APPEAR",
                "model_selector": "task:utils",
                "protocol": "json_envelope",
                "allowed_procedures": ["builtin.web_search"],
                "can_be_root": False,
            },
            "builtin.quick_thinker": {
                "agent_id": "builtin.quick_thinker",
                "display_name": "快速思考者",
                "description": "拆分与委派",
                "character_prompt": "ANOTHER_SECRET",
                "allowed_procedures": ["*"],
            },
        },
        procedure_catalog={
            "builtin.web_search": {
                "procedure_id": "builtin.web_search",
                "display_name": "网页搜索",
                "description": "按引擎搜索",
                "timeout_seconds": 30.0,
                "arguments_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "builtin.knowledge_search": {
                "procedure_id": "builtin.knowledge_search",
                "display_name": "知识库搜索",
                "description": "搜索知识库",
                "allowed_agents": ["builtin.memory_researcher"],
                "arguments_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        },
        pricing={"fingerprint": "price-1", "models": {}},
    )
    system = builder.system_message
    assert "确定性 Markdown" not in system
    assert "canonical JSON" not in system
    assert "冻结运行契约" not in system
    assert "第三方目录" not in system
    assert "quick_thinker_role" not in system
    assert "external_cost" not in system
    assert "research_credits_charged" not in system
    assert "## 可用 Procedure" in system
    assert "### 控制" in system
    assert "### 研究" in system
    assert "## 可委派智能体" in system
    assert "模型费用参考" not in system  # empty models → omit
    assert "core.compact" in system
    assert "core.checkpoint" in system
    assert "core.terminate" in system
    assert "builtin.web_search" in system
    assert "arguments:" in system
    assert "`builtin.researcher`" in system
    assert "调用限制：仅 `builtin.memory_researcher` 可调用" in system
    assert "可用 Procedure：`builtin.web_search`" not in system
    assert "allowed_procedures" not in system
    assert "can_be_root" not in system
    assert "model_selector" not in system
    assert "protocol=json_envelope" not in system
    assert "SECRET_PERSONA_SHOULD_NOT_APPEAR" not in system
    assert "ANOTHER_SECRET" not in system
    assert "character_prompt" not in system
    assert "### `?`" not in system
    assert "— ?" not in system
    # wildcard agent should not dump ["*"]
    qt_block = system.split("`builtin.quick_thinker`")[1].split("###")[0]
    assert "可用 Procedure" not in qt_block


def test_real_catalog_entries_render_agent_ids() -> None:
    agents = AgentRegistry(root_agent="builtin.quick_thinker")
    agents.replace_provider("builtin", [d.model_dump(mode="json") for d in bundled_agent_definitions()])
    snap = agents.snapshot({})
    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="swarm",
        bot_profile={},
        agent_catalog=snap.entries,
        procedure_catalog=(),
        pricing={},
    )
    system = builder.system_message
    assert "`builtin.quick_thinker`" in system
    assert "### `?`" not in system
    for entry in snap.entries:
        assert entry.definition.character_prompt not in system
