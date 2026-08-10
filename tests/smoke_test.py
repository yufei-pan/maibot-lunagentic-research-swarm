"""离线冒烟测试：不依赖 MaiBot Host。

运行方式（在仓库根目录）：
    PYTHONPATH=../maibot-plugin-sdk python tests/smoke_test.py

期望最后一行：
    ok: Lunagentic Research Swarm 0.1.0 offline smoke test passed
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
SDK_ROOT = PLUGIN_DIR.parent / "maibot-plugin-sdk"
for path in (PLUGIN_DIR, SDK_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import plugin as lrs_plugin  # noqa: E402
from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions  # noqa: E402
from lunagentic_research_swarm.config import CURRENT_CONFIG_VERSION, LRSConfig  # noqa: E402
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider  # noqa: E402
from lunagentic_research_swarm.procedures.core import CORE_PROCEDURE_IDS  # noqa: E402


_RUNTIME_PACKAGES = ("pydantic", "httpx", "ddgs", "lancedb")


def _parse_requirement(line: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)(.+)", line.strip())
    assert match is not None, line
    return match.group(1).lower(), match.group(2)


def check_import_and_factory() -> None:
    instance = lrs_plugin.create_plugin()
    assert instance.plugin_id == "com.0-hz.lunagentic-research-swarm"
    assert hasattr(instance, "on_load")
    assert hasattr(instance, "on_unload")
    assert hasattr(instance, "on_config_update")


def check_components() -> None:
    components = lrs_plugin.create_plugin().get_components()
    tools = {item["name"] for item in components if item["type"] == "TOOL"}
    commands = {item["name"] for item in components if item["type"] == "COMMAND"}
    assert len(tools) == 8, tools
    assert len(commands) == 9, commands
    assert "start_deep_research" in tools
    assert "swarm_health" in commands


def check_config_schema() -> None:
    config = LRSConfig()
    dumped = config.model_dump(mode="python")
    assert "reasoning" not in dumped
    assert "reasoning" not in dumped.get("protocol", {})
    assert dumped["storage"]["store_agent_transcripts"] is False
    assert dumped["plugin"]["root_agent"] == "builtin.quick_thinker"
    text = (PLUGIN_DIR / "config.default.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert "reasoning" not in parsed
    assert parsed["plugin"]["config_version"] == CURRENT_CONFIG_VERSION
    # 模板与模型必须同步：默认下限落在模板里，用户才能看到并改。
    assert parsed["summarizer"]["min_output_chars"] == dumped["summarizer"]["min_output_chars"]
    hint = config.plugin.model_fields["root_agent"].json_schema_extra["hint"]
    assert "源代码" in hint

    plugin = lrs_plugin.create_plugin()
    schema = plugin.get_webui_config_schema(
        plugin_id="com.0-hz.lunagentic-research-swarm",
        plugin_name="lrs",
        plugin_version="0.2.0",
    )
    root_field = schema["sections"]["plugin"]["fields"]["root_agent"]
    assert root_field["ui_type"] == "select"
    assert root_field["choices"] == ["builtin.deep_thinker", "builtin.quick_thinker"]
    assert "源代码" in (root_field.get("hint") or "")


def check_bundled_definitions() -> None:
    agents = bundled_agent_definitions()
    assert len(agents) == 9
    assert {item.agent_id for item in agents} >= {
        "builtin.quick_thinker",
        "builtin.deep_thinker",
        "builtin.researcher",
    }
    provider = BundledProcedureProvider(object())
    procedures = {item["procedure_id"] for item in provider.describe()}
    assert "builtin.web_search" in procedures
    assert "builtin.past_cases" in procedures
    assert "builtin.calculate" in procedures
    assert CORE_PROCEDURE_IDS == frozenset({"core.compact", "core.checkpoint", "core.terminate"})


def check_dependency_sync() -> None:
    pyproject = tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    py_deps = {
        _parse_requirement(item)[0]: _parse_requirement(item)[1]
        for item in pyproject["project"]["dependencies"]
        if not item.startswith("maibot-plugin-sdk")
    }
    req_deps = {
        _parse_requirement(line)[0]: _parse_requirement(line)[1]
        for line in (PLUGIN_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("maibot-plugin-sdk")
    }
    manifest = json.loads((PLUGIN_DIR / "_manifest.json").read_text(encoding="utf-8"))
    man_deps = {item["name"].lower(): item["version_spec"] for item in manifest["dependencies"]}
    assert set(py_deps) == set(_RUNTIME_PACKAGES)
    assert set(req_deps) == set(_RUNTIME_PACKAGES)
    assert set(man_deps) == set(_RUNTIME_PACKAGES)
    assert "fetch-url" not in man_deps and "maibot-fetch-url" not in str(manifest["dependencies"])
    assert manifest["version"] == "0.2.0"


def check_prompts_readable() -> None:
    prompts = PLUGIN_DIR / "i18n" / "zh-CN"
    for name in (
        "swarm_system.txt",
        "formalize_task.txt",
        "finalize_task.txt",
        "finalize_branch.txt",
        "compact_branch.txt",
    ):
        path = prompts / name
        assert path.is_file(), path
        assert path.read_text(encoding="utf-8").strip()


def main() -> None:
    check_import_and_factory()
    check_components()
    check_config_schema()
    check_bundled_definitions()
    check_dependency_sync()
    check_prompts_readable()
    print("ok: Lunagentic Research Swarm 0.2.0 offline smoke test passed")


if __name__ == "__main__":
    main()
