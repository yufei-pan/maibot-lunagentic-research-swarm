"""本地 builtin Procedure provider：describe + invoke 统一入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from lunagentic_research_swarm.config import WebSearchSection
from lunagentic_research_swarm.extensions.contracts import ProcedureResult
from lunagentic_research_swarm.procedures.bundled.analysis import (
    ANALYSIS_HANDLERS,
    analysis_procedure_definitions,
)
from lunagentic_research_swarm.procedures.bundled.memory import (
    MEMORY_HANDLERS,
    memory_procedure_definitions,
)
from lunagentic_research_swarm.procedures.bundled.provenance import (
    PROVENANCE_HANDLERS,
    provenance_procedure_definitions,
)
from lunagentic_research_swarm.procedures.bundled.web_search import (
    WEB_SEARCH_PROCEDURE_ID,
    WebSearchService,
    make_web_search_handler,
    web_search_procedure_definitions,
)


class BundledProcedureProvider:
    """通过与第三方相同的 registry/invoker 路径暴露内置 Procedures。"""

    def __init__(
        self,
        ctx: Any,
        *,
        web_search: WebSearchService | None = None,
        web_search_config: WebSearchSection | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.ctx = ctx
        self._owns_http = False
        if web_search is not None:
            self._web_search = web_search
            self._http = None
        else:
            section = web_search_config or WebSearchSection()
            if http_client is None:
                self._http = httpx.AsyncClient(timeout=section.timeout_seconds)
                self._owns_http = True
            else:
                self._http = http_client
            self._web_search = WebSearchService(section, self._http)
        self._handlers = dict(MEMORY_HANDLERS)
        self._handlers.update(ANALYSIS_HANDLERS)
        self._handlers.update(PROVENANCE_HANDLERS)
        self._handlers[WEB_SEARCH_PROCEDURE_ID] = make_web_search_handler(self._web_search)

    @property
    def web_search(self) -> WebSearchService:
        return self._web_search

    def describe(self) -> list[dict[str, Any]]:
        """返回可交给 ProcedureRegistry.replace_provider 的 model_dump payload。"""

        definitions = (
            memory_procedure_definitions()
            + analysis_procedure_definitions()
            + provenance_procedure_definitions()
            + web_search_procedure_definitions(self._web_search)
        )
        return [item.model_dump(mode="json") for item in definitions]

    async def invoke(self, procedure_id: str, arguments: Mapping[str, Any] | None = None) -> ProcedureResult:
        """按 procedure_id 分发到 handler map；未知 ID 返回 invalid_arguments。"""

        handler = self._handlers.get(procedure_id)
        if handler is None:
            return ProcedureResult(
                success=False,
                data=None,
                error={"code": "invalid_arguments", "message": f"未知 Procedure：{procedure_id}"},
                metadata={},
            )
        return await handler(self.ctx, dict(arguments or {}))

    async def aclose(self) -> None:
        """关闭 provider 自建的 HTTP client（若有）。"""

        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None
            self._owns_http = False
