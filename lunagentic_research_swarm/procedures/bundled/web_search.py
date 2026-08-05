"""四引擎统一网页搜索 Procedure：DuckDuckGo / SearXNG / Tavily / You。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urljoin

from pydantic import SecretStr

from lunagentic_research_swarm.config import WebSearchSection
from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult

Handler = Callable[[Any, Mapping[str, Any]], Awaitable[ProcedureResult]]

_ENGINE_ORDER = ("duckduckgo", "searxng", "tavily", "you")
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"

DdgsFactory = Callable[[int], Any]


def _failure(code: str, message: str, *, metadata: Mapping[str, Any] | None = None) -> ProcedureResult:
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata=dict(metadata or {}),
    )


def _success(data: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> ProcedureResult:
    return ProcedureResult(
        success=True,
        data=dict(data),
        error=None,
        metadata=dict(metadata or {}),
    )


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_item(
    *,
    url: Any,
    title: Any,
    snippet: Any,
    published_at: Any,
    source: Any,
) -> dict[str, Any]:
    return {
        "url": "" if url is None else str(url),
        "title": "" if title is None else str(title),
        "snippet": "" if snippet is None else str(snippet),
        "published_at": _as_optional_str(published_at),
        "source": "" if source is None else str(source),
    }


def _default_ddgs_factory(timeout: int) -> Any:
    from ddgs import DDGS

    return DDGS(timeout=timeout)


def _http_status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


class WebSearchService:
    """按配置广告可用引擎，并把四引擎结果归一化为统一 provenance。"""

    def __init__(
        self,
        config: WebSearchSection,
        http: Any,
        *,
        ddgs_factory: DdgsFactory | None = None,
        ddgs_text: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self._config = config
        self._http = http
        self._ddgs_factory = ddgs_factory or _default_ddgs_factory
        self._ddgs_text = ddgs_text
        self._tavily_key = SecretStr(config.tavily_api_key) if config.tavily_api_key else None
        self._you_key = SecretStr(config.you_api_key) if config.you_api_key else None

    @property
    def available_engines(self) -> tuple[str, ...]:
        enabled = set(self._config.enabled_engines)
        return tuple(engine for engine in _ENGINE_ORDER if engine in enabled and self._engine_configured(engine))

    def _engine_configured(self, engine: str) -> bool:
        if engine == "duckduckgo":
            return True
        if engine == "searxng":
            return bool(self._config.searxng_url.strip())
        if engine == "tavily":
            return self._tavily_key is not None
        if engine == "you":
            return bool(self._config.you_base_url.strip()) and self._you_key is not None
        return False

    def replace_config(self, config: WebSearchSection) -> None:
        """配置热更新时替换快照；HTTP 请求按当前 timeout_seconds 逐次覆盖 client 超时。"""

        self._config = config
        self._tavily_key = SecretStr(config.tavily_api_key) if config.tavily_api_key else None
        self._you_key = SecretStr(config.you_api_key) if config.you_api_key else None

    async def search(
        self,
        engine: str,
        query: str,
        max_results: int,
        language: str,
        recency: str | None,
    ) -> ProcedureResult:
        if engine not in self.available_engines:
            return _failure("search_engine_unavailable", f"搜索引擎不可用：{engine}")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            return _failure("invalid_arguments", "query 长度必须在 1..2000 范围内")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1 or max_results > 20:
            return _failure("invalid_arguments", "max_results 必须在 1..20 范围内")
        if not isinstance(language, str):
            return _failure("invalid_arguments", "language 必须为字符串")
        if recency is not None and not isinstance(recency, str):
            return _failure("invalid_arguments", "recency 必须为字符串或 null")

        try:
            if engine == "duckduckgo":
                items = await self._search_duckduckgo(query, max_results=max_results, recency=recency)
                external_cost = None
            elif engine == "searxng":
                items = await self._search_searxng(query, max_results=max_results, language=language, recency=recency)
                external_cost = None
            elif engine == "tavily":
                items = await self._search_tavily(query, max_results=max_results)
                external_cost = {"kind": "provider_metered", "provider": "tavily"}
            else:
                items = await self._search_you(query, max_results=max_results)
                external_cost = {"kind": "provider_metered", "provider": "you"}
        except _ProviderContractError as exc:
            return _failure("provider_contract_invalid", str(exc))
        except _HttpSearchError as exc:
            return _failure("http_error", exc.message, metadata={"status_class": exc.status_class})
        except Exception:  # noqa: BLE001 — 引擎适配器边界：任意失败不得泄漏密钥
            return _failure("search_failed", f"{engine} 搜索失败")

        return _success(
            {
                "engine": engine,
                "query": query,
                "results": items[:max_results],
            },
            metadata={"external_cost": external_cost},
        )

    async def _search_duckduckgo(self, query: str, *, max_results: int, recency: str | None) -> list[dict[str, Any]]:
        timeout = int(self._config.timeout_seconds)
        if self._ddgs_text is not None:
            raw = await asyncio.to_thread(
                self._ddgs_text,
                query=query,
                backend="duckduckgo",
                max_results=max_results,
                timeout=timeout,
            )
        else:
            client = self._ddgs_factory(timeout)

            def _call() -> list[dict[str, Any]]:
                return list(
                    client.text(query, backend="duckduckgo", max_results=max_results)
                )

            raw = await asyncio.to_thread(_call)
        items: list[dict[str, Any]] = []
        for row in raw or []:
            if not isinstance(row, Mapping):
                continue
            items.append(
                _normalize_item(
                    url=row.get("href"),
                    title=row.get("title"),
                    snippet=row.get("body"),
                    published_at=None,
                    source="duckduckgo",
                )
            )
        return items

    async def _search_searxng(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        recency: str | None,
    ) -> list[dict[str, Any]]:
        base = self._config.searxng_url.rstrip("/") + "/"
        url = urljoin(base, "search")
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": language or "en",
        }
        if recency:
            params["time_range"] = recency
        payload = await self._http_get_json(url, params=params, headers=None)
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            raise _ProviderContractError("SearXNG 响应缺少 results 数组")
        items: list[dict[str, Any]] = []
        for row in results[:max_results]:
            if not isinstance(row, Mapping):
                continue
            items.append(
                _normalize_item(
                    url=row.get("url"),
                    title=row.get("title"),
                    snippet=row.get("content"),
                    published_at=row.get("publishedDate"),
                    source=row.get("engine") or "searxng",
                )
            )
        return items

    async def _search_tavily(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        assert self._tavily_key is not None
        headers = {
            "Authorization": f"Bearer {self._tavily_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        body = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        payload = await self._http_post_json(_TAVILY_SEARCH_URL, json_body=body, headers=headers)
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            raise _ProviderContractError("Tavily 响应缺少 results 数组")
        items: list[dict[str, Any]] = []
        for row in results[:max_results]:
            if not isinstance(row, Mapping):
                continue
            items.append(
                _normalize_item(
                    url=row.get("url"),
                    title=row.get("title"),
                    snippet=row.get("content"),
                    published_at=row.get("published_date") or row.get("publishedDate"),
                    source="tavily",
                )
            )
        return items

    async def _search_you(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        assert self._you_key is not None
        headers = {"X-API-Key": self._you_key.get_secret_value()}
        params = {"query": query, "num_web_results": max_results}
        payload = await self._http_get_json(self._config.you_base_url, params=params, headers=headers)
        rows = _parse_you_rows(payload)
        items: list[dict[str, Any]] = []
        for row in rows[:max_results]:
            snippet = row.get("snippet")
            if snippet is None:
                snippet = row.get("description")
            if snippet is None:
                snippet = row.get("content")
            published = row.get("published_at")
            if published is None:
                published = row.get("published_date") or row.get("publishedDate")
            items.append(
                _normalize_item(
                    url=row.get("url"),
                    title=row.get("title"),
                    snippet=snippet,
                    published_at=published,
                    source=row.get("source") or "you",
                )
            )
        return items

    async def _http_get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
    ) -> Any:
        try:
            response = await self._http.get(
                url,
                params=params,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
        except Exception as exc:
            raise _HttpSearchError("HTTP 请求失败", status_class="network") from exc
        return self._read_json_response(response)

    async def _http_post_json(
        self,
        url: str,
        *,
        json_body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Any:
        try:
            response = await self._http.post(
                url,
                json=json_body,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
        except Exception as exc:
            raise _HttpSearchError("HTTP 请求失败", status_class="network") from exc
        return self._read_json_response(response)

    def _read_json_response(self, response: Any) -> Any:
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            raise _HttpSearchError(
                f"HTTP {_http_status_class(status)}",
                status_class=_http_status_class(status),
            )
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            try:
                raise_for_status()
            except Exception as exc:
                code = int(getattr(getattr(exc, "response", None), "status_code", status) or status or 0)
                if code >= 400:
                    raise _HttpSearchError(
                        f"HTTP {_http_status_class(code)}",
                        status_class=_http_status_class(code),
                    ) from exc
                raise
        try:
            return response.json()
        except Exception as exc:
            raise _ProviderContractError("响应不是合法 JSON") from exc


def _parse_you_rows(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise _ProviderContractError("You 响应必须为对象")
    hits = payload.get("hits")
    if isinstance(hits, list):
        return [row for row in hits if isinstance(row, Mapping)]
    web = payload.get("web")
    if isinstance(web, Mapping):
        results = web.get("results")
        if isinstance(results, list):
            return [row for row in results if isinstance(row, Mapping)]
    raise _ProviderContractError("You 响应既无 hits 也无 web.results")


class _ProviderContractError(Exception):
    pass


class _HttpSearchError(Exception):
    def __init__(self, message: str, *, status_class: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_class = status_class


def web_search_procedure_definitions(service: WebSearchService) -> list[ProcedureDefinition]:
    """构造 builtin.web_search；无可用引擎时不进入目录。"""

    engines = list(service.available_engines)
    if not engines:
        return []
    payload = {
        "procedure_id": "builtin.web_search",
        "version": "1",
        "display_name": "网页搜索",
        "description": "按指定引擎执行网页搜索，并归一化来源字段。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "engine": {"type": "string", "enum": engines},
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                "language": {"type": "string"},
                "recency": {"type": "string"},
            },
            "required": ["engine", "query"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "engine": {"type": "string"},
                "query": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                            "snippet": {"type": "string"},
                            "published_at": {},
                            "source": {"type": "string"},
                        },
                    },
                },
            },
        },
        "idempotent": True,
        "timeout_seconds": float(service._config.timeout_seconds),
        "external_cost_kind": "provider_metered",
        "enabled": True,
    }
    return [ProcedureDefinition.model_validate(payload)]


def make_web_search_handler(service: WebSearchService) -> Handler:
    async def _web_search(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
        engine = arguments.get("engine")
        if not isinstance(engine, str) or not engine:
            return _failure("invalid_arguments", "engine 必须为非空字符串")
        query = arguments.get("query")
        if not isinstance(query, str):
            return _failure("invalid_arguments", "query 必须为字符串")
        max_results = arguments.get("max_results", service._config.max_results)
        language = arguments.get("language", "en")
        recency = arguments.get("recency")
        if language is None:
            language = "en"
        return await service.search(engine, query, max_results, language, recency)

    return _web_search


WEB_SEARCH_PROCEDURE_ID = "builtin.web_search"
