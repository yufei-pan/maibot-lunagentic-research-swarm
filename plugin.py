from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF, MaiBotPlugin, ON_MODEL_CONFIG_RELOAD

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from lunagentic_research_swarm.config import LRSConfig, normalize_config


class LunagenticResearchSwarmPlugin(MaiBotPlugin):
    plugin_id = "com.0-hz.lunagentic-research-swarm"
    config_reload_subscriptions = {ON_MODEL_CONFIG_RELOAD}
    config_model = LRSConfig

    def normalize_plugin_config(self, config_data: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
        merged, changed, notes = normalize_config(config_data, LRSConfig().model_dump(mode="python"))
        if notes:
            self.ctx.logger.info("LRS 配置迁移：%s", "；".join(notes))
        return merged, changed

    async def on_load(self) -> None:
        self.ctx.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.logger.info("麦麦深度调查组基础组件已加载")

    async def on_unload(self) -> None:
        self.ctx.logger.info("麦麦深度调查组基础组件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            self.ctx.logger.info("LRS 插件配置已更新：version=%s；新 round 起生效", version)
        elif scope == ON_MODEL_CONFIG_RELOAD:
            self.ctx.logger.info("LRS 模型价格与 task 列表快照已更新：version=%s", version)


def create_plugin() -> LunagenticResearchSwarmPlugin:
    return LunagenticResearchSwarmPlugin()
