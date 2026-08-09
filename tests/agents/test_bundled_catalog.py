import pytest

from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions
from lunagentic_research_swarm.agents.registry import AgentRegistry, RootAgentUnavailableError
from lunagentic_research_swarm.config import AgentOverride
from lunagentic_research_swarm.extensions.validation import validate_agent_batch


def test_bundled_catalog_contains_nine_valid_agents() -> None:
    definitions = validate_agent_batch("builtin", [item.model_dump() for item in bundled_agent_definitions()])
    assert {item.agent_id for item in definitions} == {
        "builtin.quick_thinker", "builtin.deep_thinker", "builtin.debater",
        "builtin.researcher", "builtin.memory_researcher", "builtin.knowledge_reporter",
        "builtin.past_case_researcher", "builtin.evidence_verifier", "builtin.quantitative_analyst",
    }
    assert all(item.enabled for item in definitions)
    assert next(item for item in definitions if item.agent_id == "builtin.quick_thinker").can_be_root


def test_default_selectors_match_design() -> None:
    selectors = {item.agent_id: item.model_selector for item in bundled_agent_definitions()}
    assert selectors["builtin.quick_thinker"] == "task:utils"
    assert selectors["builtin.deep_thinker"] == "task:planner"
    assert selectors["builtin.debater"] == "task:replyer"
    assert selectors["builtin.researcher"] == "task:utils"
    assert selectors["builtin.memory_researcher"] == "task:mid_memory"
    assert selectors["builtin.knowledge_reporter"] == "task:replyer"
    assert selectors["builtin.past_case_researcher"] == "task:utils"
    assert selectors["builtin.evidence_verifier"] == "task:planner"
    assert selectors["builtin.quantitative_analyst"] == "task:planner"


def test_agent_facing_descriptions_omit_operator_guidance() -> None:
    deep = next(item for item in bundled_agent_definitions() if item.agent_id == "builtin.deep_thinker")
    assert "建议使用" not in deep.description
    assert "大模型" not in deep.description
    assert all(item.allowed_procedures == ["*"] for item in bundled_agent_definitions())


def test_display_names_include_english_in_parentheses() -> None:
    names = {item.agent_id: item.display_name for item in bundled_agent_definitions()}
    assert names == {
        "builtin.quick_thinker": "快速思考者 (Quick Thinker)",
        "builtin.deep_thinker": "深度思考者 (Deep Thinker)",
        "builtin.debater": "辩手 (Debater)",
        "builtin.researcher": "外部研究员 (Researcher)",
        "builtin.memory_researcher": "记忆研究员 (Memory Researcher)",
        "builtin.knowledge_reporter": "知识报告员 (Knowledge Reporter)",
        "builtin.past_case_researcher": "历史案例研究员 (Past-case Researcher)",
        "builtin.evidence_verifier": "证据核验员 (Evidence Verifier)",
        "builtin.quantitative_analyst": "定量分析员 (Quantitative Analyst)",
    }


def test_validate_agent_batch_rejects_foreign_namespace() -> None:
    payloads = [item.model_dump() for item in bundled_agent_definitions()]
    payloads[0]["agent_id"] = "evil.quick_thinker"
    with pytest.raises(ValueError, match="命名空间"):
        validate_agent_batch("builtin", payloads)


def test_registry_loads_builtin_root_and_rejects_disabled_root_without_replacement() -> None:
    registry = AgentRegistry(root_agent="builtin.quick_thinker")
    registry.replace_provider(
        "builtin",
        [definition.model_dump(mode="json") for definition in bundled_agent_definitions()],
    )
    snapshot = registry.snapshot({})
    assert snapshot.get("builtin.quick_thinker") is not None
    assert snapshot.get("builtin.deep_thinker") is not None

    with pytest.raises(RootAgentUnavailableError, match="根智能体"):
        registry.snapshot({"builtin.quick_thinker": AgentOverride(enabled=False)})


def test_root_capable_agent_ids_ignores_non_root_and_disabled() -> None:
    registry = AgentRegistry(root_agent="builtin.quick_thinker")
    registry.replace_provider(
        "builtin",
        [definition.model_dump(mode="json") for definition in bundled_agent_definitions()],
    )
    ids = registry.root_capable_agent_ids({})
    assert ids == ("builtin.deep_thinker", "builtin.quick_thinker")
    trimmed = registry.root_capable_agent_ids({"builtin.deep_thinker": AgentOverride(enabled=False)})
    assert trimmed == ("builtin.quick_thinker",)


def test_procedure_owned_access_restricts_memory_and_past_cases() -> None:
    from types import SimpleNamespace

    from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
    from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

    agents = AgentRegistry(root_agent="builtin.quick_thinker")
    agents.replace_provider(
        "builtin",
        [definition.model_dump(mode="json") for definition in bundled_agent_definitions()],
    )
    procedures = ProcedureRegistry()
    procedures.replace_provider("builtin", BundledProcedureProvider(SimpleNamespace()).describe())
    agent_snap = agents.snapshot({})
    proc_snap = procedures.snapshot({})

    quick = set(agent_snap.resolve_allowed_procedures("builtin.quick_thinker", proc_snap))
    memory = set(agent_snap.resolve_allowed_procedures("builtin.memory_researcher", proc_snap))
    past = set(agent_snap.resolve_allowed_procedures("builtin.past_case_researcher", proc_snap))

    assert "builtin.knowledge_search" not in quick
    assert "builtin.past_cases" not in quick
    assert "builtin.knowledge_search" in memory
    assert "builtin.past_cases" not in memory
    assert "builtin.past_cases" in past
    assert "builtin.knowledge_search" not in past
    # 专职也拿到一般工具（若本轮目录含 web_search）
    if "builtin.normalize_urls" in proc_snap.ids:
        assert "builtin.normalize_urls" in quick & memory & past

