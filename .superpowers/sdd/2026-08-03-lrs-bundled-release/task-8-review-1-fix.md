# Task 8 Review 1 Fixes — stats double-count, report snapshot, summarizer metering

## Status

**DONE**

## Summary

Addressed blocking review findings from `task-8-review-1.md`:

1. **Double-count** — `_compute_task` / `_compute_plugin` skip `reconciliation_status='reserved'` rows so each logical call is counted once (prefer reconciled). New test drives real `reserve_input` + `reconcile_usage`.
2. **Report stats snapshot** — `ReportCoordinator.synthesize` writes `stats_json` from `StatisticsService` / `compute_task_stats` (ledger recompute) after summarizer metering; equality test uses the coordinator production path then independent recompute.
3. **Summarizer metering** — `meter_summarizer_usage` (credits module) records reserve+reconcile usage with non-research roles; wired in `ReportCoordinator` (branch/task) and `ResearchManager` formalize. Stats reuse `is_summarizer_role` from credits.
4. **protocol_correction_count** — counted via distinct `call_id LIKE '%:correction'` (production reducer shape), not fictional lifecycle events.
5. **Debug tests** — failure isolation asserts `extension_fingerprints` + authority write; vector fake matches keyword-only `enqueue(*, source_kind, source_id)`.

## Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/storage/test_debug_storage.py tests/test_statistics.py -q
# 11 passed

timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 581 passed in 1.92s
```

## Files

- `lunagentic_research_swarm/statistics.py`
- `lunagentic_research_swarm/runtime/credits.py`
- `lunagentic_research_swarm/runtime/epochs.py`
- `lunagentic_research_swarm/runtime/manager.py`
- `lunagentic_research_swarm/services.py`
- `tests/test_statistics.py`
- `tests/storage/test_debug_storage.py`
