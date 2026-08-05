# Task 3 Report: 实现四引擎统一 Web 搜索

## Status

**DONE**

## Summary

Added `builtin.web_search` through the existing `BundledProcedureProvider` path (no parallel registry). `WebSearchService` advertises only correctly configured engines among DuckDuckGo / SearXNG / Tavily / You, runs the selected engine without fallback, and normalizes every hit to `{url, title, snippet, published_at, source}` under `{engine, query, results}`.

DDGS is pinned in `pyproject.toml`, `requirements.txt`, and `_manifest.json` (`ddgs>=9.14.4,<10.0.0` / manifest lower bound). DuckDuckGo calls use `DDGS(timeout=...).text(..., backend="duckduckgo")` via `asyncio.to_thread`. API keys are held as `SecretStr`; HTTP/error paths report status class only and never echo secrets.

## TDD Evidence

1. **RED** — Wrote `tests/procedures/test_web_search.py` + `tests/test_dependencies.py` first. Collection failed with `ModuleNotFoundError: ...web_search`; dependency tests failed with missing `ddgs` in all three manifests.
2. **GREEN** — Added `ddgs` to the three dependency files, implemented `procedures/bundled/web_search.py`, extended `BundledProcedureProvider`, and wired `web_search_config` through `services.py` (load / config reload / aclose).
3. **VERIFY** — `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/procedures/test_web_search.py tests/test_dependencies.py -v` → **18 passed**.

Regression: bundled provider / memory / executor builtin / lifecycle → **67 passed**.

## Files Changed

| Path | Action |
|---|---|
| `pyproject.toml` | Added `ddgs>=9.14.4,<10.0.0` |
| `requirements.txt` | Same pin |
| `_manifest.json` | Added `ddgs` `>=9.14.4` |
| `lunagentic_research_swarm/procedures/bundled/web_search.py` | Created — service + adapters + procedure definition |
| `lunagentic_research_swarm/procedures/bundled/provider.py` | Merged web search into describe/invoke; optional HTTP client |
| `lunagentic_research_swarm/services.py` | Pass `web_search_config`; refresh on self-reload; aclose provider |
| `tests/procedures/test_web_search.py` | Created |
| `tests/test_dependencies.py` | Created — three-way sync |
| `tests/procedures/test_bundled_provider.py` | Allow web search + `provider_metered` |

## Adapter Contracts (locked by tests)

| Engine | Request | Auth | Parse |
|---|---|---|---|
| DuckDuckGo | `DDGS(timeout).text(query, backend="duckduckgo", max_results=…)` | — | `href/title/body` |
| SearXNG | `GET <base>/search` `q/format=json/categories=general/language` | — | `results[*].url/title/content/publishedDate/engine` |
| Tavily | `POST https://api.tavily.com/search` basic depth, no answer/raw | `Authorization: Bearer` | `results[*].url/title/content/published_date` |
| You | `GET` configured base; `query` + `num_web_results` | `X-API-Key` | `hits` **or** `web.results` only; else `provider_contract_invalid` |

Unavailable engine → `search_engine_unavailable` (no fallback). Procedure: `idempotent=true`, `external_cost_kind=provider_metered`.

## Commits

- `72f285b` — `feat: add configurable web search procedures`

## Concerns

1. You request/response follows the brief’s `num_web_results` + `hits`/`web.results` contract; current You.com docs lean toward `count` + `results.web` — callers using the live You API may need a later schema extension.
2. Empty `available_engines` yields an empty JSON Schema `enum` for `engine` (misconfiguration edge case).
3. Task 4+ not started. `fetch-url` remains recommended-only and is out of this task’s scope.

## Verification Command

```bash
cd /mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm/.worktrees/lrs-runtime-reporting
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/procedures/test_web_search.py \
  tests/test_dependencies.py -v
```

Result: **18 passed**.

## Review-1 P2 Fix Notes

**Findings addressed:**

1. **DDGS upper bound in `_manifest.json`** — Manifest now declares `ddgs>=9.14.4,<10.0.0` (same as pyproject/requirements). Dependency sync asserts exact three-way match for `ddgs`; other packages may still omit the upper bound in the Host manifest.
2. **Live timeout reload** — HTTP GET/POST now pass `timeout=self._config.timeout_seconds` per request, so `replace_config()` after a live reload applies the advertised timeout without rebuilding the shared client.
3. **Startup-failure cleanup** — `_cleanup_start_failure()` now `aclose()`s and clears `_bundled_procedure_provider`, matching normal shutdown and avoiding an owned `httpx.AsyncClient` leak on start rollback.

**Evidence:**
```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/procedures/test_web_search.py \
  tests/test_dependencies.py \
  tests/test_services_startup_cleanup.py \
  tests/procedures/test_bundled_provider.py -v
```
Result: **27 passed**.
