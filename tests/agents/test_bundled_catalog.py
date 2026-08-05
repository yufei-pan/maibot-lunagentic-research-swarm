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
    assert selectors["builtin.memory_researcher"] == "task:utils"
    assert selectors["builtin.knowledge_reporter"] == "task:replyer"
    assert selectors["builtin.past_case_researcher"] == "task:utils"
    assert selectors["builtin.evidence_verifier"] == "task:planner"
    assert selectors["builtin.quantitative_analyst"] == "task:planner"


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
