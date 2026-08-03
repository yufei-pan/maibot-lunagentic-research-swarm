from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SDK_ROOT = REPO_ROOT.parent / "maibot-plugin-sdk"
for path in (REPO_ROOT, SDK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def plugin_module():
    import plugin

    return plugin
