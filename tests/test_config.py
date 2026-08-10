from __future__ import annotations

import pytest

from maibot_sdk.config import PluginConfigVersionError, extract_plugin_config_version
from lunagentic_research_swarm.config import CURRENT_CONFIG_VERSION, LRSConfig, normalize_config


def test_config_defaults_match_approved_spec() -> None:
    config = LRSConfig()
    assert config.plugin.root_agent == "builtin.quick_thinker"
    assert config.llm.default_selector == ""
    assert config.summarizer.selector == ""
    assert config.summarizer.temperature == 0.2
    assert config.summarizer.max_tokens == 0
    assert config.timing.default_time_budget_seconds == 600
    assert config.timing.grace_period_seconds == 60
    assert config.timing.pause_timeout_seconds == 1200
    assert config.timing.feedback_wait_seconds == 600
    assert config.budget.default_effort_credits == 100.0
    assert config.context.auto_compact_tokens == 258000
    assert config.context.model_context_window is None
    assert config.protocol.default_mode == "json_envelope"
    assert config.protocol.max_correction_turns == 1
    assert not config.storage.store_agent_transcripts
    assert not config.storage.store_raw_procedure_payloads
    assert config.commands.allow_vector_rebuild is False
    assert config.commands.maintenance_allowed_user_ids == []
    assert "reasoning" not in LRSConfig.model_json_schema()["properties"]


def test_normalize_migrates_force_selector_to_default_selector() -> None:
    defaults = LRSConfig().model_dump(mode="python", exclude_none=True)
    raw = {
        "plugin": {"config_version": "1.0.0"},
        "llm": {"force_selector": "model:pinned"},
    }
    normalized, changed, notes = normalize_config(raw, defaults)
    assert normalized["llm"]["default_selector"] == "model:pinned"
    assert "force_selector" not in normalized["llm"]
    assert changed is True
    assert any(CURRENT_CONFIG_VERSION in note for note in notes)


def test_normalize_omits_none_values_for_toml_roundtrip() -> None:
    """Host 用 tomlkit 写回 config.toml；None 会导致整文件写失败并清空配置。"""

    defaults = LRSConfig().model_dump(mode="python", exclude_none=True)
    normalized, _changed, _notes = normalize_config(
        {"plugin": {"config_version": "1.0.0"}, "llm": {"force_selector": ""}},
        defaults,
    )

    def _assert_no_none(value: object, path: str = "") -> None:
        assert value is not None, path
        if isinstance(value, dict):
            for key, child in value.items():
                _assert_no_none(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _assert_no_none(child, f"{path}[{index}]")

    _assert_no_none(normalized)
    assert "model_context_window" not in normalized.get("context", {})


def test_normalize_migrates_legacy_maintenance_person_ids() -> None:
    defaults = LRSConfig().model_dump(mode="python")
    raw = {
        "plugin": {"config_version": CURRENT_CONFIG_VERSION},
        "commands": {"maintenance_allowed_person_ids": ["u1"]},
    }
    normalized, changed, _notes = normalize_config(raw, defaults)
    assert normalized["commands"]["maintenance_allowed_user_ids"] == ["u1"]
    assert "maintenance_allowed_person_ids" not in normalized["commands"]
    assert changed is True


def test_normalize_migrates_duckduckgo_engine_to_ddgs() -> None:
    defaults = LRSConfig().model_dump(mode="python", exclude_none=True)
    raw = {
        "plugin": {"config_version": "1.2.0"},
        "web_search": {"enabled_engines": ["duckduckgo", "searxng", "duckduckgo"]},
    }
    normalized, changed, notes = normalize_config(raw, defaults)
    assert normalized["web_search"]["enabled_engines"] == ["ddgs", "searxng"]
    assert changed is True
    assert any(CURRENT_CONFIG_VERSION in note for note in notes)
    assert LRSConfig().web_search.enabled_engines == ["ddgs"]


def test_web_search_ddgs_defaults() -> None:
    section = LRSConfig().web_search
    assert section.ddgs_region == "us-en"
    assert section.ddgs_safesearch == "moderate"
    assert section.ddgs_backend == "auto"

def test_default_config_exposes_sdk_canonical_plugin_version() -> None:
    defaults = LRSConfig().model_dump(mode="python")

    assert defaults["plugin"]["config_version"] == CURRENT_CONFIG_VERSION
    assert "config_version" not in defaults
    assert extract_plugin_config_version(defaults) == CURRENT_CONFIG_VERSION


def test_explicit_price_override_is_a_complete_profile() -> None:
    config = LRSConfig.model_validate({"pricing": {"models": {"gpt-fast": {"price_in": 1.0}}}})
    profile = config.pricing.models["gpt-fast"]
    assert profile.price_in == 1.0
    assert profile.cache is False
    assert profile.cache_price_in == 0.0
    assert profile.price_out == 0.0


def test_migration_preserves_user_values_and_bumps_version() -> None:
    raw = {
        "plugin": {"config_version": "0.0.1"},
        "timing": {"default_time_budget_seconds": 300},
        "storage": {"store_agent_transcripts": True},
    }
    merged, changed, notes = normalize_config(raw, LRSConfig().model_dump(mode="python"))
    assert changed
    assert merged["plugin"]["config_version"] == CURRENT_CONFIG_VERSION
    assert merged["timing"]["default_time_budget_seconds"] == 300
    assert merged["storage"]["store_agent_transcripts"] is True
    assert notes


@pytest.mark.parametrize("version", [None, [], ""])
def test_migration_rejects_invalid_explicit_plugin_version(version: object) -> None:
    raw = {"plugin": {"config_version": version}}

    with pytest.raises(PluginConfigVersionError):
        normalize_config(raw, LRSConfig().model_dump(mode="python"))


def test_migration_rejects_non_mapping_config_data() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        normalize_config([], LRSConfig().model_dump(mode="python"))
