# Task 5 Report: 实现可重建 LanceDB generation 索引

## Status

**DONE**

## Summary

SQLite remains authoritative; LanceDB is a rebuildable derived case index under `data/vectors/lancedb`.

- Added `lancedb>=0.34.0,<0.35.0` synced across `pyproject.toml` / `requirements.txt` / `_manifest.json`.
- Migration 002: `vector_generations`, `vector_documents`, unique active-generation index.
- `VectorIndex` (`storage/vectors.py`): `start/close/enqueue/rebuild/status/search` with generation fingerprint, dimension validation, atomic active↔retired switch, failed-candidate retention.
- Wired into `LRSServiceContainer` (degrades when `ctx.llm` missing).

Whitelist sources only: formalized task, checkpoint/branch-final summary, intermediate/final report, feedback lesson. Raw chat/transcript/procedure/compact/reasoning never indexed.

## TDD Evidence

1. **RED** — Wrote `tests/storage/test_vectors.py` + `tests/storage/test_vector_rebuild.py` first. Collection failed with `ModuleNotFoundError: lunagentic_research_swarm.storage.vectors`.
2. **GREEN** — Added deps + migration 002 + `vectors.py` + services wiring; privacy/deps tests updated.
3. **VERIFY** — `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/storage/test_vectors.py tests/storage/test_vector_rebuild.py tests/test_dependencies.py -v` → **20 passed**.

Regression: privacy + services startup cleanup → **24 passed** with vector suite.

Covered: selector/model/dimension/schema mismatch, batch inconsistency abort, failed candidate keeps old active, atomic switch, rebuild-time `vector_index_rebuilding`, `model:` → `physical_embedding_selector_unsupported`, already_current / force rebuild.

## Files Changed

| Path | Action |
|---|---|
| `pyproject.toml` / `requirements.txt` / `_manifest.json` | Add lancedb pin |
| `lunagentic_research_swarm/storage/migrations.py` | Migration 002 |
| `lunagentic_research_swarm/storage/vectors.py` | Created |
| `lunagentic_research_swarm/storage/sqlite.py` | `run_locked` for vector metadata |
| `lunagentic_research_swarm/services.py` | Start/close/health for VectorIndex |
| `lunagentic_research_swarm/llm/gateway.py` | `ModelSelector.task_name` |
| `tests/storage/test_vectors.py` | Created |
| `tests/storage/test_vector_rebuild.py` | Created |
| `tests/test_dependencies.py` / `tests/storage/test_privacy.py` | Sync pins + schema expectations |

## Commits

- `8bdf2ee` — `feat: add rebuildable LanceDB case index`

## Concerns

1. Host `llm.embed` still has no physical model pin; `model:` selectors are explicitly rejected (`physical_embedding_selector_unsupported`).
2. Mismatch with `auto_rebuild=True` marks a building candidate and sets `rebuilding` immediately; full rebuild is advanced by explicit `rebuild()` / future worker (avoids test/runtime race on the mismatch window). Task 6 past_cases should treat `status.rebuilding` / `VECTOR_INDEX_REBUILDING` as unavailable.
3. Retired Lance tables are purged only after `retired_generation_retention_seconds` and only when at least one retired generation remains.

## Verification Command

```bash
cd /mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm/.worktrees/lrs-runtime-reporting
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/storage/test_vectors.py \
  tests/storage/test_vector_rebuild.py \
  tests/test_dependencies.py -v
```

Result: **20 passed**.

---

## Review-1 Fixes (post `8bdf2ee`)

**Status:** DONE — addressed Critical/Important findings from `task-5-review-1.md`.

### Fixes

1. **Status whitelist** — `_load_indexable_sources` now accepts `SUCCEEDED`/`FAILED`/`DEGRADED` (`INDEXABLE_CONTENT_STATUSES`), matching `runtime/epochs.py` writers. Fixtures use real statuses (not harness-only `READY`).
2. **Mismatch rebuilds** — `auto_rebuild=True` runs in-process `_run_full_rebuild` (full SQLite re-read + re-embed + atomic activate). Added `ensure_ready()` for Task 6.
3. **Stranded building** — `rebuild(force=False)` / empty rebuild clears stranded `building` candidates when fingerprint is current.
4. **Search fingerprint** — query path recomputes model fingerprint and runs `_detect_mismatch` (same-dimension model swap no longer returns silent wrong hits).
5. **Error codes** — `vector_rebuild_failed` / `vector_index_unavailable` distinct from `embedding_generation_mismatch`.
6. **Upsert** — Lance delete-then-add on `id` for re-index; non-numeric/bool embeddings fail the job (not stuck PENDING); negative privacy test for FORMALIZATION/TASK_FINAL.

### Spec note

Brief mentioned summary/report readiness loosely; preferred actual runtime enum from `epochs.py` over inventing `READY`.

### Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/storage/test_vectors.py \
  tests/storage/test_vector_rebuild.py \
  tests/test_dependencies.py \
  tests/storage/test_privacy.py -v
```

Result: **31 passed**.

### Remaining

- Startup reconcile of orphan Lance tables vs `vector_generations` still deferred.
- `vector_documents` rows for retired/purged generations not yet GC'd.

---

## Review-2 Fixes (post `41fd4c1`)

**Status:** DONE — addressed Important (+ cheap Medium) findings from `task-5-review-2.md`.

### Fixes

1. **Startup stranded building** — `VectorIndex.start()` calls `_fail_stranded_building()` (fail row + drop orphan Lance table). `LRSServiceContainer.start()` also calls `ensure_ready()`. Crash mid-rebuild can no longer leave search stuck on `vector_index_rebuilding`.
2. **Mismatch-only auto-rebuild** — `_enqueue_unlocked` only schedules auto-rebuild for `EmbeddingGenerationMismatch` from `_detect_mismatch` / append schema checks. `VectorIndexUnavailable` (transient `open_table` IO) fails the job and leaves the active generation alone. Invalid probe embeddings also do not rebuild.
3. **Whitelist** — `INDEXABLE_CONTENT_STATUSES = {"SUCCEEDED"}` only. `FAILED`/`DEGRADED` apology report bodies are excluded; formalized tasks + feedback lessons unchanged.
4. **Auto-rebuild lock** — probe/write stay under `_lock`; full rebuild re-acquires after release. `_rebuilding` is set before release so concurrent enqueue fails fast with `vector_index_rebuilding` instead of queueing invisibly for the whole corpus re-embed. Rebuild itself still serializes on `_lock` (documented safe bound).
5. **Medium** — empty-index enqueue returns `VectorOpResult.fail` (no raise); successful auto-rebuild completes the job / returns `indexed`; Lance upsert uses `merge_insert`; `_prepare` reuses `_fail_stranded_building` (drops orphan tables); `ensure_ready` tries `force=False` before force rebuild.

### Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/storage/test_vectors.py \
  tests/storage/test_vector_rebuild.py \
  tests/test_dependencies.py \
  tests/storage/test_privacy.py -v
```

Result: **36 passed**.

---

## Review-3 Fixes (post `e32e53f`)

**Status:** DONE — addressed remaining Important findings from `task-5-review-3.md`.

### Fixes

1. **Non-blocking auto-rebuild** — mismatch path sets `_rebuilding` then schedules a tracked `asyncio.Task` (`_auto_rebuild_task`) that runs `_run_full_rebuild` **without** holding `_lock`. Triggering `enqueue()` returns `vector_index_rebuilding` immediately (option a); concurrent `enqueue()` observes `_rebuilding` and fails fast. Background task completes the original job when the source lands in the new active generation. `rebuild` / `ensure_ready` / `close` await or cancel the task.
2. **Startup `ensure_ready` health** — `LRSServiceContainer.start()` inspects `VectorOpResult`; failures mark `vector_index` `degraded` with the error code. Catches `EmbeddingGenerationMismatch` / `VectorRebuildFailed` / `VectorIndexUnavailable` / `LRSError` so a derived-index failure does not abort LRS start. Never reports healthy when `ensure_ready` failed. `ensure_ready()` itself converts rebuild raises into `VectorOpResult.fail`.

### Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/storage/test_vectors.py \
  tests/storage/test_vector_rebuild.py \
  tests/test_dependencies.py \
  tests/storage/test_privacy.py \
  tests/test_services_startup_cleanup.py -v
```

Result: **40 passed** (includes concurrency fail-fast + ensure_ready degrade coverage).

---

## Review-4 Fixes (post `734b6ee`)

**Status:** DONE — addressed Critical/Important findings from `task-5-review-4.md`.

### Fixes

1. **CRITICAL — suite green** — `LRSServiceContainer.start()` now wraps VectorIndex construction + `start()` + `ensure_ready()` in one `try/except Exception` that records `vector_index` as `unavailable`, closes any partial index, and continues LRS start (no more `sqlite_initialization_failed` mislabel). Lifecycle `FakeStore` gained a `run_locked` stub; `make_context` sets `llm=None` so lifecycle does not open real LanceDB. SQLite migration tests updated for migration 002 (`vector_generations`) / failing fixture version=3.
2. **IMPORTANT — vector start degradation** — production path never aborts the plugin on LanceDB import/IO/`OSError`/broad Exception during vector startup; never reports `healthy` on failure.
3. **IMPORTANT — background rebuild catch-all** — `_auto_rebuild_task` catches broad Exception, logs, fails stranded building, records `_last_rebuild_error` surfaced via `status().last_error_*`, and always clears `_rebuilding`. `_await_background_rebuild` swallows task exceptions so `rebuild()` / `ensure_ready()` keep the `VectorOpResult` contract. Those methods also convert `EmbeddingGenerationMismatch` / infra exceptions into fail results (tests updated accordingly).

### Verification

```bash
timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 542 passed in 1.75s

PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q \
  tests/test_lifecycle.py \
  tests/storage/test_sqlite.py \
  tests/storage/test_vectors.py \
  tests/storage/test_vector_rebuild.py \
  tests/test_services_startup_cleanup.py
# 74 passed in 1.09s
```

Result: **542 passed, 0 failed** (full suite).
