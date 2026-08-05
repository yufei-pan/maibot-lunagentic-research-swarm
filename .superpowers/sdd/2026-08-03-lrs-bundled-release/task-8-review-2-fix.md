# Task 8 Review 2 Fixes — same-tx snapshot, metering degrade, orphan credits

## Status

**DONE**

## Summary

Addressed Important findings from `task-8-review-2.md` (Issues 1–4):

1. **Same-transaction snapshot** — `ReportCoordinator._persist_report_same_tx` computes `compute_task_stats` and inserts report/outbox inside one `store.run_transaction` (`BEGIN IMMEDIATE`). Store gained `run_transaction` + `apply_commands`. Lock-order + concurrent-inject tests cover the old gap.
2. **Metering must not abort** — `_meter_summarizer` and `_formalize` metering wrapped in try/except (log + skip). Tests: `price_catalog=None` synthesize still SUCCEEDED; raising meter during formalize still RUNNING.
3. **Orphan reservations** — task credit totals prefer `credit_ledger` (reservation abs + net actual; unreconciled = reservation-only calls). Usage rows still single-count tokens via non-reserved preference, but lonely `reserved` orphans remain visible for calls/tokens.
4. **Production metering wiring** — `test_synthesize_meters_summarizer_into_saved_stats` drives multi-coverage `synthesize` with usage-bearing summarizer and asserts saved `cost_equivalent_credits` / `summarizer_calls` increase (fails if `_meter_summarizer` deleted).

Deferred (review Issues 5–6 / review-1 carryovers): vacuous vector privacy asserts, unbounded debug failure fingerprints, `credit_debt` semantics.

## Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/test_statistics.py \
  tests/runtime/test_controller_start.py::test_formalize_survives_metering_failure_when_catalog_missing -q
# 12 passed

timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 587 passed in 1.95s
```

## Files

- `lunagentic_research_swarm/statistics.py`
- `lunagentic_research_swarm/storage/sqlite.py`
- `lunagentic_research_swarm/runtime/epochs.py`
- `lunagentic_research_swarm/runtime/manager.py`
- `tests/test_statistics.py`
- `tests/runtime/test_controller_start.py`
