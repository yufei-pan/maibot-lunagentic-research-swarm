# Live LLM E2E Testing Design

Date: 2026-08-06  
Status: approved for planning  
Plugin: `maibot-lunagentic-research-swarm`

## Problem

Offline E2E tests (`RuntimeHarness` + `FakeLLMGateway`) prove reducer/lifecycle contracts with scripted envelopes. They cannot catch the Mein failure modes where a real model ignores the swarm protocol, correction turns are rewritten, or FINAL reports are empty prose.

We already have credentials-gated protocol smoke (`tests/live_llm.py`, `live_llm` marker). We need a tiered live suite: thin vertical slices first, optional thorough research (stub tools and real tools), with LLM-as-judge for report quality—without slowing default offline `pytest`.

## Goals

1. Reuse `RuntimeHarness` (approach: live adapter into existing harness; no second E2E stack).
2. Tiered depth A → B → C, gated by pytest markers only.
3. Light LLM judge on A/B; deep rubric on thorough tiers.
4. Stub procedures by default for thorough; separate marker for real `web_search`.
5. Seed formalized text for A/B; live formalize on thorough / live-tools.
6. Default `pytest` stays offline; missing/placeholder credentials → skip live tests.

## Non-goals

- Deploying to Mein as part of this work.
- Host/SDK changes (prefer test-only adapters; touch plugin code only if a tiny public seam is required).
- Making live tests part of CI without credentials.
- Asserting exact SERP text for live-tools runs.

## Architecture

Keep `FakeLLMGateway` / `FakeProcedureProvider` / `FakeSummarizer` as harness defaults.

Plug live backends for marked tests:

- `LiveLLMGateway` — same `generate(...)` surface as `FakeLLMGateway`; calls OpenAI-compatible `/chat/completions` using `.debug_api_call_credentials` (ignore Host selector model names; use credentials `model` / `temperature`).
- Stub or live procedure providers for `builtin.web_search` / `core.terminate`.
- Live summarizer wrapper for thorough tiers (formalize + branch/task finalize).
- `LiveJudge` helpers via a second `chat_completion` with fixed JSON rubrics.

`FakeScheduler` still records effects. A live driver drains `scheduler.enqueued` through real `RuntimeEffectRunner` + `TurnWorker` (same pattern as scripted E2E), until terminal task status or wall-clock timeout.

```text
credentials (.debug_api_call_credentials)
        │
        ▼
LiveLLMGateway ──► TurnWorker / effect_runner ──► RuntimeHarness / ResearchManager
        │                                              │
        ▼                                              ▼
LiveJudge (light/deep)                    stub or live procedures
```

## Marker tiers

| Marker | Coverage | Formalize | Tools | Judge |
|--------|----------|-----------|-------|-------|
| `live_llm` | Protocol smoke (existing) | n/a | n/a | none |
| `live_llm_e2e` | **A** thin slice + **B** multi-agent | seeded | terminate / none | light yes/no |
| `live_llm_thorough` | **C** stub-tools research | live LLM | deterministic stubs | deep rubric |
| `live_llm_live_tools` | Full live LLM + real `web_search` | live LLM | real network | deep rubric |

Register markers in `pyproject.toml`. All skip unless LLM credentials are usable (`credentials_available()`). `live_llm_live_tools` additionally requires `live_tools_available()`.

## Components

### Extend `tests/live_llm.py`

- Keep: `LiveLLMCredentials`, `load_live_llm_credentials`, `credentials_available`, `chat_completion`.
- Add: `LiveLLMGateway.generate(...)` → `GenerationResult`-compatible object (`text`, usage, optional `tool_calls`).
- Add: `light_judge(objective, report) -> {pass, reason}` and `deep_judge(objective, report, evidence) -> {pass, scores, reason}`.
- Add: `live_tools_available()` and optional web-search config loader from credentials.
- Extend timeouts: per-call `timeout_seconds`; wall-clock `e2e_timeout_seconds` / `thorough_timeout_seconds`.

### `RuntimeHarness` live helpers

- `use_live_llm(creds)` — replace `self.llm`.
- `seed_formalize(text)` — existing FakeSummarizer gate path (A/B).
- `use_live_summarizer(creds)` — thorough / live-tools.
- `use_stub_procedures(fixtures)` — canned `builtin.web_search` by query substring; terminate works.
- `use_live_procedures(web_search_config)` — real `BundledProcedureProvider` / `WebSearchService`.
- `drive_live_until_terminal(...)` — drain scheduler effects until `COMPLETED` / `COMPLETED_WITH_ERRORS` / timeout.

### Test modules

| Module | Marker |
|--------|--------|
| `tests/llm/test_live_json_envelope.py` | `live_llm` (exists) |
| `tests/integration/test_live_e2e_ab.py` | `live_llm_e2e` |
| `tests/integration/test_live_thorough_stub_tools.py` | `live_llm_thorough` |
| `tests/integration/test_live_thorough_live_tools.py` | `live_llm_live_tools` |

## Per-tier flows and assertions

Hard wall-clock from credentials (defaults: 180s A/B, 900s thorough). On failure, dump last report + judge raw + call counts under `tmp_path`.

### A — thin vertical (`live_llm_e2e`)

1. `start` tiny objective → `seed_formalize` fixed text (“用一次 turn 终结并给出简短 report”).
2. `use_live_llm`.
3. Drive root until terminal.
4. Assert: status in `{COMPLETED, COMPLETED_WITH_ERRORS}` but **fail** if FINAL empty/useless; FINAL exists; ≥1 successful agent call; light judge pass (“FINAL addresses formalized objective?”).

### B — multi-agent (`live_llm_e2e`)

1. Seed formalize; objective asks root to delegate ≥1 child then finish.
2. Drive until terminal.
3. Assert: ≥1 child branch materialized; FINAL present; light judge pass.
4. Soft quality: fail on timeout, not on imperfect prose if judge passes.

### C — stub tools (`live_llm_thorough`)

1. Live formalize + live summarizer.
2. Stub `builtin.web_search` returns fixture snippets for known queries.
3. Objective requires search before concluding.
4. Assert: ≥1 stub search invoke; FINAL uses fixture facts (string check + deep judge: relevance / completeness / groundedness 1–5, pass if all ≥3).

### Live-tools (`live_llm_live_tools`)

1. Same as C with real `WebSearchService`; skip if `web_search_enabled` is not true.
2. Deep judge; longer timeout.
3. Assert search was invoked and report judged adequate — **not** exact SERP text.

### Flakiness policy

- One automatic retry **only** for judge JSON parse failures.
- No retry for task timeout or HTTP errors.
- Judge failure never silently treated as pass.

## Credentials schema

File: `.debug_api_call_credentials` (gitignored). Template: `.debug_api_call_credentials.example`.

```toml
base_url = "http://127.0.0.1:8888/v1"
api_key = "YOUR_API_KEY"
model = "your-local-model-id"
temperature = 1.0
timeout_seconds = 300
e2e_timeout_seconds = 180
thorough_timeout_seconds = 900

web_search_enabled = false
# Optional [web_search] mirrors plugin WebSearchSection when live tools are used:
# enabled_engines = ["duckduckgo"]
# timeout_seconds = 30
# max_results = 5
# searxng_url = ""
# tavily_api_key = ""
# you_base_url = ""
# you_api_key = ""
```

## Commands

```bash
pytest -q                                      # offline
pytest -m live_llm -v                          # protocol
pytest -m live_llm_e2e -v                      # A + B
pytest -m live_llm_thorough -v                 # stub-tools thorough
pytest -m live_llm_live_tools -v               # real search thorough
```

## Risks

| Risk | Handling |
|------|----------|
| Model ignores protocol | Correction path; A/B still require FINAL + light judge |
| Local model slow | Wall-clock timeout + artifact dump |
| Live search flaky | Skip unless enabled; no exact SERP asserts |
| Secrets in git | gitignore + placeholder example only |
| CI without credentials | All live markers skip |

## Implementation notes (for planning)

1. Prefer extending `tests/live_llm.py` and `tests/fakes.py` helpers over forking harness.
2. Map live HTTP responses into whatever `TurnWorker` already consumes from `FakeLLMGateway` (`GenerationResult` / text + tool_calls).
3. For live summarizer, wrap existing summarizer prompt construction if feasible; otherwise a minimal formalize/finalize-only live summarizer is acceptable for tests.
4. Stub search fixtures must include distinctive tokens the deep judge / string check can ground on.
5. Bare `pytest` still collects live tests; they **skip** without usable credentials. Use `-m live_llm_…` to run a tier intentionally. No CI job is required to add these markers.
