from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fakes import RuntimeHarness

REPO_ROOT = Path(__file__).resolve().parent.parent
SDK_ROOT = REPO_ROOT.parent / "maibot-plugin-sdk"
for path in (REPO_ROOT, SDK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def plugin_module():
    import plugin

    return plugin


@pytest.fixture
async def runtime_harness(tmp_path):
    """为集成测试提供真实 SQLite、可控时钟和无网络 provider。"""

    harness = RuntimeHarness(tmp_path)
    await harness.open()
    try:
        yield harness
    finally:
        await harness.close()
