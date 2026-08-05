"""运行期依赖三处声明保持同步。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

_RUNTIME_PACKAGES = ("pydantic", "httpx", "ddgs")


def _parse_requirement(line: str) -> tuple[str, str]:
    text = line.strip()
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)(.+)", text)
    assert match is not None, text
    return match.group(1).lower(), match.group(2)


def _pyproject_runtime_deps() -> dict[str, str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    parsed = dict(_parse_requirement(item) for item in deps if not item.startswith("maibot-plugin-sdk"))
    return parsed


def _requirements_runtime_deps() -> dict[str, str]:
    lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.startswith("maibot-plugin-sdk")
    ]
    return dict(_parse_requirement(line) for line in lines)


def _manifest_runtime_deps() -> dict[str, str]:
    manifest = json.loads((REPO_ROOT / "_manifest.json").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for item in manifest["dependencies"]:
        assert item["type"] == "python_package"
        out[item["name"].lower()] = item["version_spec"]
    return out


def test_runtime_dependencies_are_synced_across_pyproject_requirements_and_manifest() -> None:
    pyproject = _pyproject_runtime_deps()
    requirements = _requirements_runtime_deps()
    manifest = _manifest_runtime_deps()

    assert set(pyproject) == set(_RUNTIME_PACKAGES)
    assert set(requirements) == set(_RUNTIME_PACKAGES)
    assert set(manifest) == set(_RUNTIME_PACKAGES)

    for name in _RUNTIME_PACKAGES:
        py_spec = pyproject[name]
        req_spec = requirements[name]
        man_spec = manifest[name]
        assert py_spec == req_spec, f"{name}: pyproject={py_spec!r} requirements={req_spec!r}"
        # manifest 可省略上界，但下界须与 pyproject/requirements 一致
        lower = re.match(r">=([^,<\s]+)", py_spec)
        assert lower is not None
        assert man_spec.startswith(f">={lower.group(1)}")


def test_ddgs_dependency_pin_matches_brief() -> None:
    pyproject = _pyproject_runtime_deps()
    requirements = _requirements_runtime_deps()
    assert pyproject["ddgs"] == ">=9.14.4,<10.0.0"
    assert requirements["ddgs"] == ">=9.14.4,<10.0.0"
    manifest = _manifest_runtime_deps()
    assert manifest["ddgs"] == ">=9.14.4"
