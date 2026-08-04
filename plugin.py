from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from maibot_sdk import API, CONFIG_RELOAD_SCOPE_SELF, MaiBotPlugin, ON_MODEL_CONFIG_RELOAD

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from lunagentic_research_swarm.config import LRSConfig, normalize_config  # noqa: E402
from lunagentic_research_swarm.services import LRSServiceContainer  # noqa: E402


class LunagenticResearchSwarmPlugin(MaiBotPlugin):
    plugin_id = "com.0-hz.lunagentic-research-swarm"
    config_reload_subscriptions = {ON_MODEL_CONFIG_RELOAD}
    config_model = LRSConfig

    def __init__(self) -> None:
        super().__init__()
        self._services: LRSServiceContainer | None = None

    def normalize_plugin_config(self, config_data: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
        merged, changed, notes = normalize_config(config_data, LRSConfig().model_dump(mode="python"))
        if notes:
            self.ctx.logger.info("LRS 配置迁移：%s", "；".join(notes))
        return merged, changed

    async def on_load(self) -> None:
        config = self.config
        assert isinstance(config, LRSConfig)
        services = LRSServiceContainer(self.ctx, config)
        self._services = services
        try:
            await services.start()
        except BaseException:
            try:
                await services.close()
            except Exception:
                self.ctx.logger.exception("LRS 加载失败后的基础服务清理也发生错误")
            raise
        self.ctx.logger.info("麦麦深度调查组基础组件已加载")

    async def on_unload(self) -> None:
        services = self._require_services()
        await services.close()
        self.ctx.logger.info("麦麦深度调查组基础组件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        services = self._require_services()
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            detached = LRSConfig.model_validate(config_data)
            await services.update_self_config(detached, version=version)
            self.ctx.logger.info(
                "LRS 插件配置已更新：version=%s；安全限制与刷新间隔已生效，目录、selector 与 prompt 从下一 round 生效",
                version,
            )
        elif scope == ON_MODEL_CONFIG_RELOAD:
            if services.update_model_snapshot(config_data):
                self.ctx.logger.info("LRS 模型价格与 task 列表快照已更新：version=%s", version)
            else:
                self.ctx.logger.warning("LRS 模型价格与 task 列表快照未应用：version=%s", version)

    def _require_services(self) -> LRSServiceContainer:
        if self._services is None:
            raise RuntimeError("LRS 基础服务尚未初始化")
        return self._services

    @API(
        "refresh_extensions",
        description="请求麦麦深度调查组重新扫描智能体与 Procedure provider",
        version="1",
        public=True,
    )
    async def refresh_extensions(self) -> dict[str, Any]:
        return await self._require_services().refresh_extensions(reason="provider_request")


def create_plugin() -> LunagenticResearchSwarmPlugin:
    return LunagenticResearchSwarmPlugin()
