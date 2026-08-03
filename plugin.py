from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from maibot_sdk import MaiBotPlugin, ON_MODEL_CONFIG_RELOAD

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


class LunagenticResearchSwarmPlugin(MaiBotPlugin):
    plugin_id = "com.0-hz.lunagentic-research-swarm"
    config_reload_subscriptions = {ON_MODEL_CONFIG_RELOAD}

    async def on_load(self) -> None:
        self.ctx.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.logger.info("麦麦深度调查组基础组件已加载")

    async def on_unload(self) -> None:
        self.ctx.logger.info("麦麦深度调查组基础组件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        self.ctx.logger.info("麦麦深度调查组收到配置更新：scope=%s version=%s", scope, version)


def create_plugin() -> LunagenticResearchSwarmPlugin:
    return LunagenticResearchSwarmPlugin()
