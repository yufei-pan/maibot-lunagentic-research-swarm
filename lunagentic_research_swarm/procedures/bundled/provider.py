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
from lunagentic_research_swarm.procedures.bundled.past_cases import (
    PAST_CASES_PROCEDURE_ID,
    make_past_cases_handler,
    past_cases_procedure_definitions,
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
        store: Any | None = None,
        vector_index: Any | None = None,
    ) -> None:
        self.ctx = ctx
        self._store = store
        self._vector_index = vector_index
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
        self._past_cases_handler = make_past_cases_handler(store=store, vector_index=vector_index)
        self._handlers = dict(MEMORY_HANDLERS)
        self._handlers.update(ANALYSIS_HANDLERS)
        self._handlers.update(PROVENANCE_HANDLERS)
        self._handlers[WEB_SEARCH_PROCEDURE_ID] = make_web_search_handler(self._web_search)
        self._handlers[PAST_CASES_PROCEDURE_ID] = self._past_cases_handler

    @property
    def web_search(self) -> WebSearchService:
        return self._web_search

    @property
    def store(self) -> Any | None:
        return self._store

    @property
    def vector_index(self) -> Any | None:
        return self._vector_index

    def bind_case_index(self, *, store: Any | None = None, vector_index: Any | None = None) -> None:
        """启动后绑定 SQLite / VectorIndex（向量层可能晚于 builtin provider 就绪）。"""

        if store is not None:
            self._store = store
        if vector_index is not None:
            self._vector_index = vector_index
        deps = getattr(self._past_cases_handler, "_past_cases_deps", None)
        if isinstance(deps, dict):
            if store is not None:
                deps["store"] = store
            if vector_index is not None:
                deps["vector_index"] = vector_index

    def describe(self) -> list[dict[str, Any]]:
        """返回可交给 ProcedureRegistry.replace_provider 的 model_dump payload。"""

        definitions = (
            memory_procedure_definitions()
            + analysis_procedure_definitions()
            + provenance_procedure_definitions()
            + web_search_procedure_definitions(self._web_search)
            + past_cases_procedure_definitions()
        )
        return [item.model_dump(mode="json") for item in definitions]

    async def invoke(
        self,
        procedure_id: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        scoped_metadata: Mapping[str, Any] | None = None,
    ) -> ProcedureResult:
        """按 procedure_id 分发到 handler map；未知 ID 返回 invalid_arguments。"""

        handler = self._handlers.get(procedure_id)
        if handler is None:
            return ProcedureResult(
                success=False,
                data=None,
                error={"code": "invalid_arguments", "message": f"未知 Procedure：{procedure_id}"},
                metadata={},
            )
        args = dict(arguments or {})
        # past_cases：未显式传 exclude_task_id 时，默认排除调用侧 scoped task_id。
        if procedure_id == PAST_CASES_PROCEDURE_ID:
            existing = args.get("exclude_task_id")
            if not (isinstance(existing, str) and existing.strip()):
                meta = scoped_metadata if isinstance(scoped_metadata, Mapping) else {}
                task_id = meta.get("task_id")
                if isinstance(task_id, str) and task_id.strip():
                    args["exclude_task_id"] = task_id.strip()
        return await handler(self.ctx, args)

    async def aclose(self) -> None:
        """关闭 provider 自建的 HTTP client（若有）。"""

        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None
            self._owns_http = False
