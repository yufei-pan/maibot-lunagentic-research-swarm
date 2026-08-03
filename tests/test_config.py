from __future__ import annotations

from lunagentic_research_swarm.config import CURRENT_CONFIG_VERSION, LRSConfig, normalize_config


def test_config_defaults_match_approved_spec() -> None:
    config = LRSConfig()
    assert config.plugin.root_agent == "builtin.quick_thinker"
    assert config.summarizer.selector == "task:mid_memory"
    assert config.summarizer.temperature == 0.2
    assert config.summarizer.max_tokens == 0
    assert config.timing.default_time_budget_seconds == 120
    assert config.timing.grace_period_seconds == 60
    assert config.timing.pause_timeout_seconds == 1200
    assert config.timing.feedback_wait_seconds == 600
    assert config.budget.default_effort_credits == 100.0
    assert config.context.auto_compact_tokens == 258000
    assert config.protocol.default_mode == "json_envelope"
    assert config.protocol.max_correction_turns == 1
    assert not config.storage.store_agent_transcripts
    assert not config.storage.store_raw_procedure_payloads
    assert "reasoning" not in LRSConfig.model_json_schema()["properties"]


def test_explicit_price_override_is_a_complete_profile() -> None:
    config = LRSConfig.model_validate({"pricing": {"models": {"gpt-fast": {"price_in": 1.0}}}})
    profile = config.pricing.models["gpt-fast"]
    assert profile.price_in == 1.0
    assert profile.cache is False
    assert profile.cache_price_in == 0.0
    assert profile.price_out == 0.0


def test_migration_preserves_user_values_and_bumps_version() -> None:
    raw = {
        "config_version": "0.0.1",
        "timing": {"default_time_budget_seconds": 300},
        "storage": {"store_agent_transcripts": True},
    }
    merged, changed, notes = normalize_config(raw, LRSConfig().model_dump(mode="python"))
    assert changed
    assert merged["config_version"] == CURRENT_CONFIG_VERSION
    assert merged["timing"]["default_time_budget_seconds"] == 300
    assert merged["storage"]["store_agent_transcripts"] is True
    assert notes
