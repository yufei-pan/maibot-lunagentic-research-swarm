from __future__ import annotations

import inspect
import json
from pathlib import Path

from maibot_sdk import MaiBotPlugin, ON_MODEL_CONFIG_RELOAD


def test_manifest_and_factory_contract(plugin_module) -> None:
    repo = Path(plugin_module.__file__).resolve().parent
    manifest = json.loads((repo / "_manifest.json").read_text(encoding="utf-8"))
    instance = plugin_module.create_plugin()

    assert manifest["id"] == "com.0-hz.lunagentic-research-swarm"
    assert manifest["version"] == "0.2.0"
    assert manifest["sdk"]["min_version"] == "2.7.1"
    assert isinstance(instance, MaiBotPlugin)
    assert instance.plugin_id == manifest["id"]
    assert instance.get_config_reload_subscriptions() == [ON_MODEL_CONFIG_RELOAD]
    assert inspect.iscoroutinefunction(type(instance).on_load)
    assert inspect.iscoroutinefunction(type(instance).on_unload)
    assert inspect.iscoroutinefunction(type(instance).on_config_update)


def test_manifest_declares_final_capabilities(plugin_module) -> None:
    repo = Path(plugin_module.__file__).resolve().parent
    manifest = json.loads((repo / "_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["capabilities"]) == {
        "api.call",
        "api.list",
        "chat.get_all_streams",
        "chat.get_group_streams",
        "chat.get_private_streams",
        "config.get",
        "knowledge.search",
        "llm.embed",
        "llm.generate",
        "llm.generate_with_tools",
        "maisaka.context.append",
        "maisaka.proactive.trigger",
        "message.build_readable",
        "message.get_by_id",
        "message.get_by_time",
        "message.get_by_time_in_chat",
        "message.get_recent",
        "person.get_id",
        "person.get_id_by_name",
        "person.get_value",
        "send.text",
    }


def test_refresh_api_component_is_public(plugin_module) -> None:
    component = next(
        item
        for item in plugin_module.create_plugin().get_components()
        if item["name"] == "refresh_extensions" and item["type"] == "API"
    )
    assert component["metadata"]["public"] is True
    assert component["metadata"]["version"] == "1"
