from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lunagentic_research_swarm.errors import LRSError
from lunagentic_research_swarm.llm.pricing import (
    PriceCatalog,
    PriceProfile,
    TokenUsage,
    charge,
    price_units_to_credits,
)


def test_selector_requires_explicit_supported_scheme_and_nonempty_name() -> None:
    from lunagentic_research_swarm.llm.gateway import ModelSelector

    task = ModelSelector.parse("task:utils")
    physical = ModelSelector.parse("model:gpt")
    assert (task.scheme, task.name, task.raw) == ("task", "utils", "task:utils")
    assert (physical.scheme, physical.name) == ("model", "gpt")
    with pytest.raises(FrozenInstanceError):
        task.name = "planner"  # type: ignore[misc]

    for invalid in ("utils", "task:", "model:   ", "provider:gpt", " task:utils"):
        with pytest.raises(LRSError) as caught:
            ModelSelector.parse(invalid)
        assert caught.value.code == "invalid_selector"
        assert caught.value.message == f"无效模型 selector：{invalid}"


def test_default_selector_priority_over_builtin() -> None:
    from lunagentic_research_swarm.llm.gateway import resolve_generation_selector

    assert resolve_generation_selector("task:utils", "model:pinned").raw == "model:pinned"
    assert resolve_generation_selector("task:utils", "").raw == "task:utils"
    assert (
        resolve_generation_selector("task:utils", "model:default", user_override="model:agent").raw
        == "model:agent"
    )
    assert (
        resolve_generation_selector("task:utils", "model:default", user_override="").raw == "model:default"
    )
    assert resolve_generation_selector("task:utils", "model:default", user_override=None).raw == "model:default"
    with pytest.raises(LRSError, match="无效模型 selector"):
        resolve_generation_selector("task:utils", "bare-name")


def test_plugin_override_ignores_entire_host_profile() -> None:
    catalog = PriceCatalog.from_sources(
        plugin_overrides={"luna": {"price_in": 1.0}},
        host_models={"luna": PriceProfile(9.0, True, 2.0, 12.0)},
        task_models={"utils": ["luna"]},
    )
    resolved = catalog.resolve_model("luna")
    assert resolved.source == "plugin_override"
    assert resolved.profile == PriceProfile(price_in=1.0, cache=False, cache_price_in=0.0, price_out=0.0)


def test_host_unavailable_unpriced_and_actual_unknown_have_distinct_free_sources() -> None:
    unavailable = PriceCatalog.from_host_snapshot(None, {})
    assert unavailable.resolve_model("anything").source == "host_unavailable_free"

    unpriced = PriceCatalog.from_host_snapshot(
        {"models": [{"name": "known"}], "model_task_config": {"utils": {"model_list": ["known"]}}},
        {},
    )
    assert unpriced.resolve_model("known").source == "host_unpriced_free"
    actual = unpriced.charge_actual(
        actual_model_name="other",
        prompt_tokens=10,
        completion_tokens=5,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
    )
    assert actual.price.source == "actual_model_unknown_free"
    assert actual.credits == 0.0


def test_sanitize_skips_host_model_task_config_metadata() -> None:
    """MaiBot PluginConfigBase model_dump 会把 field_docs / suppress_any_warning 混进 task map。

    回归：若再次整包拒绝此类快照，swarm health 会出现
    initial_price_snapshot=host_model_snapshot_invalid，并级联
    root_selector_unavailable / summarizer_selector_unavailable。
    """

    from lunagentic_research_swarm.llm.pricing import sanitize_host_snapshot

    host_like = {
        "models": [{"name": "m", "price_in": 1.0}],
        "model_task_config": {
            "field_docs": {"utils": "组件模型", "mid_memory": "摘要"},
            "suppress_any_warning": False,
            "utils": {
                "field_docs": {"model_list": "使用的模型列表"},
                "suppress_any_warning": True,
                "model_list": ["m"],
                "temperature": 0.7,
            },
            "mid_memory": {"model_list": ["m"]},
        },
    }
    safe = sanitize_host_snapshot(host_like)
    assert set(safe["model_task_config"]) == {"utils", "mid_memory"}
    catalog = PriceCatalog.from_host_snapshot(host_like, {})
    assert catalog.debug_snapshot()["tasks"] == {"utils": ["m"], "mid_memory": ["m"]}
    assert catalog.debug_snapshot()["tasks"].get("field_docs") is None
    assert "suppress_any_warning" not in catalog.debug_snapshot()["tasks"]


def test_partial_host_price_keeps_present_fields_and_defaults_missing_fields() -> None:
    catalog = PriceCatalog.from_host_snapshot(
        {
            "models": [{"name": "known", "price_in": 1, "price_out": 2}],
            "model_task_config": {"utils": {"model_list": ["known"]}},
        },
        {},
    )
    resolved = catalog.resolve_model("known")
    assert resolved.source == "host_config"
    assert resolved.profile == PriceProfile(1.0, False, 0.0, 2.0)
    assert catalog.debug_snapshot() == {
        "models": {"known": {"price_in": 1.0, "cache": False, "cache_price_in": 0.0, "price_out": 2.0}},
        "tasks": {"utils": ["known"]},
    }


def test_missing_host_price_is_free() -> None:
    catalog = PriceCatalog.from_sources({}, {}, {"utils": ["unknown"]})
    assert catalog.estimate_model_for_selector("task:utils").profile.total_is_free


def test_task_estimate_uses_first_model_but_reconcile_uses_actual() -> None:
    catalog = PriceCatalog.from_sources(
        {},
        {
            "first": PriceProfile(1.0, False, 0.0, 2.0),
            "actual": PriceProfile(3.0, True, 0.5, 4.0),
        },
        {"utils": ["first", "actual"]},
    )
    assert catalog.estimate_model_for_selector("task:utils").model_name == "first"
    usage = catalog.charge_actual(
        actual_model_name="actual",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_hit_tokens=600,
        cache_miss_tokens=400,
    )
    assert usage.credits == pytest.approx(((400 * 3.0 + 600 * 0.5 + 500 * 4.0) / 1_000_000) * 100)


def test_cache_accounting_fills_unreported_gap_as_miss_and_rejects_overflow() -> None:
    catalog = PriceCatalog.from_sources(
        {}, {"cached": PriceProfile(2.0, True, 0.5, 3.0)}, {"utils": ["cached"]}
    )
    charged = catalog.charge_actual(
        actual_model_name="cached",
        prompt_tokens=100,
        completion_tokens=10,
        cache_hit_tokens=40,
        cache_miss_tokens=20,
    )
    assert charged.usage == TokenUsage(100, 10, 40, 60, source="actual")

    with pytest.raises(LRSError) as caught:
        catalog.charge_actual(
            actual_model_name="cached",
            prompt_tokens=100,
            completion_tokens=10,
            cache_hit_tokens=70,
            cache_miss_tokens=40,
        )
    assert caught.value.code == "invalid_usage"


def test_direct_charge_bills_unclassified_prompt_tokens_as_cache_miss() -> None:
    credits = charge(
        PriceProfile(price_in=2.0, cache=True, cache_price_in=0.5, price_out=3.0),
        TokenUsage(prompt_tokens=100, completion_tokens=10, cache_hit_tokens=40, cache_miss_tokens=20),
    )
    assert credits == pytest.approx(0.017)


def test_direct_charge_rejects_cache_classification_overflow() -> None:
    with pytest.raises(LRSError) as caught:
        charge(
            PriceProfile(price_in=2.0, cache=True, cache_price_in=0.5, price_out=3.0),
            TokenUsage(prompt_tokens=100, completion_tokens=10, cache_hit_tokens=70, cache_miss_tokens=40),
        )
    assert caught.value.code == "invalid_usage"


@pytest.mark.parametrize(
    "usage",
    [
        TokenUsage(-1, 0, 0, 0),
        TokenUsage(1, -1, 0, 0),
        TokenUsage(1, 0, -1, 0),
        TokenUsage(1, 0, 0, -1),
        TokenUsage(True, 0, 0, 0),
    ],
)
def test_charge_rejects_invalid_usage_values(usage: TokenUsage) -> None:
    with pytest.raises(LRSError) as caught:
        charge(PriceProfile(1.0, False, 0.0, 1.0), usage)
    assert caught.value.code == "invalid_usage"


def test_failed_call_without_usage_keeps_estimated_input_reservation() -> None:
    catalog = PriceCatalog.from_sources({}, {"m": PriceProfile(1.0, False, 0.0, 2.0)}, {"utils": ["m"]})
    result = catalog.reconcile_call(
        estimated_credits=12.5,
        success=False,
        actual_model_name="",
        usage=None,
    )
    assert result.status == "estimated_unreconciled"
    assert result.actual_credits == 12.5
    assert result.adjustment_credits == 0.0


def test_snapshot_is_whitelisted_detached_and_fingerprint_tracks_only_safe_contract_fields() -> None:
    snapshot = {
        "models": [
            {
                "name": "m",
                "price_in": 1.0,
                "cache": True,
                "cache_price_in": 0.2,
                "price_out": 2.0,
                "api_provider": "secret-provider",
                "extra_params": {"headers": {"Authorization": "secret"}},
            }
        ],
        "model_task_config": {"utils": {"model_list": ["m"], "temperature": 1.8}},
        "api_providers": [{"api_key": "secret"}],
    }
    catalog = PriceCatalog.from_host_snapshot(snapshot, {})
    first_fingerprint = catalog.fingerprint
    snapshot["models"][0]["price_in"] = 99.0
    snapshot["models"][0]["extra_params"]["headers"]["Authorization"] = "changed"
    assert catalog.resolve_model("m").profile.price_in == 1.0
    assert "secret" not in repr(catalog)

    secret_changed = {
        **snapshot,
        "models": [{**snapshot["models"][0], "price_in": 1.0, "api_provider": "other-secret"}],
    }
    assert PriceCatalog.from_host_snapshot(secret_changed, {}).fingerprint == first_fingerprint

    priced_changed = {
        **secret_changed,
        "models": [{**secret_changed["models"][0], "price_out": 3.0}],
    }
    assert PriceCatalog.from_host_snapshot(priced_changed, {}).fingerprint != first_fingerprint


def test_replace_host_snapshot_refreshes_task_model_mapping_without_mixing_override() -> None:
    catalog = PriceCatalog.from_host_snapshot(
        {"models": [{"name": "a", "price_in": 9}], "model_task_config": {"utils": {"model_list": ["a"]}}},
        {"a": {"price_in": 1}},
    )
    catalog.replace_host_snapshot(
        {"models": [{"name": "b", "price_out": 4}], "model_task_config": {"utils": {"model_list": ["b"]}}}
    )
    assert catalog.estimate_model_for_selector("task:utils").model_name == "b"
    assert catalog.resolve_model("a").source == "plugin_override"


def test_credit_units_and_low_budget_warning_estimate() -> None:
    profile = PriceProfile(1.0, False, 0.0, 2.0)
    assert price_units_to_credits(1.0) == 100.0
    assert charge(profile, TokenUsage(500_000, 50_000, 0, 500_000)) == pytest.approx(60.0)

    catalog = PriceCatalog.from_sources({}, {"m": profile}, {"utils": ["m"]})
    estimate = catalog.estimate_root_minimum("task:utils")
    assert estimate.credits == pytest.approx(60.0)
    assert catalog.low_budget_warning("task:utils", 59.9) is not None
    assert catalog.low_budget_warning("task:utils", 60.0) is None

    free = PriceCatalog.from_sources({}, {}, {"utils": ["missing"]})
    assert free.low_budget_warning("task:utils", 0.0) is None
