from __future__ import annotations

from pathlib import Path

import pytest

from live_llm import live_tools_available, load_live_llm_credentials


def test_load_timeouts_and_web_search_defaults(tmp_path: Path) -> None:
    path = tmp_path / "creds.toml"
    path.write_text(
        'base_url = "http://127.0.0.1:9/v1"\n'
        'api_key = "sk-test"\n'
        'model = "m"\n',
        encoding="utf-8",
    )
    creds = load_live_llm_credentials(path)
    assert creds.e2e_timeout_seconds == 180.0
    assert creds.thorough_timeout_seconds == 900.0
    assert creds.web_search_enabled is False
    assert creds.web_search == {}


def test_live_tools_available_requires_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "creds.toml"
    path.write_text(
        'base_url = "http://127.0.0.1:9/v1"\n'
        'api_key = "sk-test"\n'
        'model = "m"\n'
        "web_search_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("live_llm.CREDENTIALS_PATH", path)
    assert live_tools_available() is True
