"""四引擎统一 Web 搜索：可用性、归一化 provenance 与 adapter contract。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from lunagentic_research_swarm.config import LRSConfig, WebSearchSection
from lunagentic_research_swarm.procedures.bundled.web_search import WebSearchService


@dataclass
class _FakeResponse:
    status_code: int
    payload: Any
    text: str = ""

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("GET", "https://example.invalid/")
            response = httpx.Response(self.status_code, request=request, text=self.text or "error")
            raise httpx.HTTPStatusError("HTTP error", request=request, response=response)


@dataclass
class FakeHttp:
    """记录请求并返回预设响应；不含真实网络。"""

    gets: list[dict[str, Any]] = field(default_factory=list)
    posts: list[dict[str, Any]] = field(default_factory=list)
    get_response: _FakeResponse | None = None
    post_response: _FakeResponse | None = None
    get_by_url: dict[str, _FakeResponse] = field(default_factory=dict)

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ):
        self.gets.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        if url in self.get_by_url:
            return self.get_by_url[url]
        if self.get_response is None:
            raise AssertionError(f"未配置 GET 响应：{url}")
        return self.get_response

    async def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ):
        self.posts.append(
            {
                "url": url,
                "json": dict(json or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        if self.post_response is None:
            raise AssertionError(f"未配置 POST 响应：{url}")
        return self.post_response


def fake_http() -> FakeHttp:
    return FakeHttp()


@pytest.fixture
def config() -> LRSConfig:
    return LRSConfig()


@dataclass
class WebHarness:
    http: FakeHttp
    section: WebSearchSection
    ddgs_results: list[dict[str, str]] = field(default_factory=list)

    def configure(self, engine: str) -> None:
        self.section.enabled_engines = ["duckduckgo", "searxng", "tavily", "you"]
        self.section.searxng_url = "https://searx.example"
        self.section.tavily_api_key = "tvly-secret"
        self.section.you_base_url = "https://api.you.example/search"
        self.section.you_api_key = "you-secret"
        if engine == "duckduckgo":
            self.ddgs_results = [
                {
                    "href": "https://ddg.example/a",
                    "title": "DDG Title",
                    "body": "DDG body snippet",
                }
            ]
        elif engine == "searxng":
            self.http.get_response = _FakeResponse(
                200,
                {
                    "results": [
                        {
                            "url": "https://searx.example/a",
                            "title": "SearX Title",
                            "content": "SearX content",
                            "publishedDate": "2026-01-02",
                            "engine": "google",
                        }
                    ]
                },
            )
        elif engine == "tavily":
            self.http.post_response = _FakeResponse(
                200,
                {
                    "results": [
                        {
                            "url": "https://tavily.example/a",
                            "title": "Tavily Title",
                            "content": "Tavily content",
                            "published_date": "2026-03-04",
                        }
                    ]
                },
            )
        elif engine == "you":
            self.http.get_response = _FakeResponse(
                200,
                {
                    "hits": [
                        {
                            "url": "https://you.example/a",
                            "title": "You Title",
                            "snippet": "You snippet",
                            "published_at": "2026-05-06",
                            "source": "you-source",
                        }
                    ]
                },
            )
        else:
            raise AssertionError(engine)

    def service(self) -> WebSearchService:
        return WebSearchService(
            self.section,
            self.http,
            ddgs_text=lambda **kwargs: list(self.ddgs_results),
        )

    async def search(self, engine: str, query: str):
        return await self.service().search(engine, query, max_results=5, language="en", recency=None)


@pytest.fixture
def web_harness() -> WebHarness:
    return WebHarness(http=fake_http(), section=WebSearchSection())


def test_only_correctly_configured_engines_are_advertised(config) -> None:
    config.web_search.enabled_engines = ["duckduckgo", "searxng", "tavily", "you"]
    config.web_search.searxng_url = ""
    config.web_search.tavily_api_key = ""
    config.web_search.you_base_url = "https://example.invalid/search"
    config.web_search.you_api_key = "key"
    service = WebSearchService(config.web_search, fake_http())
    assert service.available_engines == ("duckduckgo", "you")


@pytest.mark.parametrize("engine", ["duckduckgo", "searxng", "tavily", "you"])
@pytest.mark.asyncio
async def test_all_engines_normalize_results(engine, web_harness) -> None:
    web_harness.configure(engine)
    result = await web_harness.search(engine, "query")
    assert result.success
    assert result.data["engine"] == engine
    assert result.data["query"] == "query"
    assert set(result.data["results"][0]) == {"url", "title", "snippet", "published_at", "source"}


@pytest.mark.asyncio
async def test_unavailable_engine_does_not_fallback(web_harness) -> None:
    web_harness.section.enabled_engines = ["duckduckgo"]
    web_harness.section.searxng_url = ""
    result = await web_harness.service().search("searxng", "q", max_results=3, language="en", recency=None)
    assert not result.success
    assert result.error.code == "search_engine_unavailable"
    assert "searxng" in result.error.message


@pytest.mark.asyncio
async def test_searxng_adapter_contract(web_harness) -> None:
    web_harness.configure("searxng")
    result = await web_harness.search("searxng", "hello")
    assert result.success
    assert len(web_harness.http.gets) == 1
    call = web_harness.http.gets[0]
    assert call["url"] == "https://searx.example/search"
    assert call["params"]["q"] == "hello"
    assert call["params"]["format"] == "json"
    assert call["params"]["categories"] == "general"
    assert call["params"]["language"] == "en"
    item = result.data["results"][0]
    assert item["url"] == "https://searx.example/a"
    assert item["title"] == "SearX Title"
    assert item["snippet"] == "SearX content"
    assert item["published_at"] == "2026-01-02"
    assert item["source"] == "google"


@pytest.mark.asyncio
async def test_tavily_adapter_uses_bearer_auth(web_harness) -> None:
    web_harness.configure("tavily")
    result = await web_harness.search("tavily", "leo")
    assert result.success
    assert len(web_harness.http.posts) == 1
    call = web_harness.http.posts[0]
    assert call["url"] == "https://api.tavily.com/search"
    assert call["headers"]["Authorization"] == "Bearer tvly-secret"
    assert call["json"] == {
        "query": "leo",
        "max_results": 5,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    item = result.data["results"][0]
    assert item["url"] == "https://tavily.example/a"
    assert item["snippet"] == "Tavily content"
    assert item["published_at"] == "2026-03-04"
    assert "tvly-secret" not in repr(result)
    assert "tvly-secret" not in str(result.error)


@pytest.mark.asyncio
async def test_you_adapter_uses_x_api_key_and_hits_schema(web_harness) -> None:
    web_harness.configure("you")
    result = await web_harness.search("you", "trends")
    assert result.success
    call = web_harness.http.gets[0]
    assert call["url"] == "https://api.you.example/search"
    assert call["headers"]["X-API-Key"] == "you-secret"
    assert call["params"]["query"] == "trends"
    assert call["params"]["num_web_results"] == 5
    assert result.data["results"][0]["url"] == "https://you.example/a"
    assert "you-secret" not in repr(result)


@pytest.mark.asyncio
async def test_you_accepts_web_results_schema(web_harness) -> None:
    web_harness.configure("you")
    web_harness.http.get_response = _FakeResponse(
        200,
        {
            "web": {
                "results": [
                    {
                        "url": "https://you.example/b",
                        "title": "Web Results Title",
                        "description": "desc",
                        "published_date": "2026-07-08",
                    }
                ]
            }
        },
    )
    result = await web_harness.search("you", "q")
    assert result.success
    item = result.data["results"][0]
    assert item["url"] == "https://you.example/b"
    assert item["title"] == "Web Results Title"
    assert item["snippet"] == "desc"


@pytest.mark.asyncio
async def test_you_rejects_unknown_response_schema(web_harness) -> None:
    web_harness.configure("you")
    web_harness.http.get_response = _FakeResponse(200, {"results": {"web": [{"url": "x"}]}})
    result = await web_harness.search("you", "q")
    assert not result.success
    assert result.error.code == "provider_contract_invalid"
    assert "you-secret" not in result.error.message


@pytest.mark.asyncio
async def test_http_error_reports_status_class_without_secrets(web_harness) -> None:
    web_harness.configure("tavily")
    web_harness.http.post_response = _FakeResponse(401, {"error": "bad key tvly-secret"}, text="bad key tvly-secret")
    result = await web_harness.search("tavily", "q")
    assert not result.success
    assert result.error.code == "http_error"
    assert "4xx" in result.error.message or "401" in result.error.message
    assert "tvly-secret" not in result.error.message
    assert "tvly-secret" not in str(result.metadata)


@pytest.mark.asyncio
async def test_duckduckgo_normalizes_href_title_body(web_harness) -> None:
    web_harness.configure("duckduckgo")
    result = await web_harness.search("duckduckgo", "q")
    assert result.success
    item = result.data["results"][0]
    assert item == {
        "url": "https://ddg.example/a",
        "title": "DDG Title",
        "snippet": "DDG body snippet",
        "published_at": None,
        "source": "duckduckgo",
    }


@pytest.mark.asyncio
async def test_ddgs_invoked_with_fixed_backend_and_timeout(web_harness) -> None:
    calls: list[dict[str, Any]] = []

    def fake_ddgs_text(**kwargs: Any) -> list[dict[str, str]]:
        calls.append(dict(kwargs))
        return [{"href": "https://x", "title": "t", "body": "b"}]

    web_harness.configure("duckduckgo")
    web_harness.section.timeout_seconds = 17.0
    service = WebSearchService(web_harness.section, web_harness.http, ddgs_text=fake_ddgs_text)
    result = await service.search("duckduckgo", "q", max_results=3, language="en", recency=None)
    assert result.success
    assert calls == [{"query": "q", "backend": "duckduckgo", "max_results": 3, "timeout": 17}]


def test_web_search_procedure_definition_is_provider_metered() -> None:
    from lunagentic_research_swarm.procedures.bundled.web_search import web_search_procedure_definitions

    section = WebSearchSection(enabled_engines=["duckduckgo", "you"], you_base_url="https://x", you_api_key="k")
    service = WebSearchService(section, fake_http())
    definitions = web_search_procedure_definitions(service)
    assert len(definitions) == 1
    definition = definitions[0]
    assert definition.procedure_id == "builtin.web_search"
    assert definition.idempotent is True
    assert definition.external_cost_kind == "provider_metered"
    engine_schema = definition.arguments_schema["properties"]["engine"]
    assert engine_schema["enum"] == ["duckduckgo", "you"]


@pytest.mark.asyncio
async def test_provider_exposes_web_search_invoke() -> None:
    from types import SimpleNamespace

    from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider

    section = WebSearchSection(enabled_engines=["duckduckgo"])
    service = WebSearchService(
        section,
        fake_http(),
        ddgs_text=lambda **kwargs: [{"href": "https://a", "title": "t", "body": "b"}],
    )
    provider = BundledProcedureProvider(SimpleNamespace(), web_search=service)
    ids = {item["procedure_id"] for item in provider.describe()}
    assert "builtin.web_search" in ids
    result = await provider.invoke(
        "builtin.web_search",
        {"engine": "duckduckgo", "query": "hello", "max_results": 2},
    )
    assert result.success
    assert result.data["engine"] == "duckduckgo"


@pytest.mark.asyncio
async def test_replace_config_applies_new_timeout_per_http_request(web_harness) -> None:
    web_harness.configure("searxng")
    web_harness.section.timeout_seconds = 12.0
    service = web_harness.service()
    await service.search("searxng", "q", max_results=3, language="en", recency=None)
    assert web_harness.http.gets[-1]["timeout"] == 12.0

    refreshed = WebSearchSection(
        enabled_engines=["duckduckgo", "searxng", "tavily", "you"],
        searxng_url="https://searx.example",
        tavily_api_key="tvly-secret",
        you_base_url="https://api.you.example/search",
        you_api_key="you-secret",
        timeout_seconds=45.0,
    )
    service.replace_config(refreshed)
    await service.search("searxng", "q2", max_results=3, language="en", recency=None)
    assert web_harness.http.gets[-1]["timeout"] == 45.0

    web_harness.configure("tavily")
    await service.search("tavily", "q3", max_results=2, language="en", recency=None)
    assert web_harness.http.posts[-1]["timeout"] == 45.0
