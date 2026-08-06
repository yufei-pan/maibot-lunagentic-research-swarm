# LRS thin-coverage SDD wave 2

> Plugin: `maibot-lunagentic-research-swarm`  
> Scope: **tests only** (may extend `tests/fakes.py`). **Do not modify** `lunagentic_research_swarm/` unless a task explicitly says production is allowed (none do in this wave). **Do not git commit.**  
> Goal: close residual thin areas from the post–wave-1 agenda (held-release credits, report_id contract, mixed rejects, compact lifecycle, extensions/providers, config/concurrency/privacy depth).

## Global Constraints

- Workdir: `/mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm`
- Pytest: `.venv-tests/bin/pytest …` with `PYTHONPATH=.:../maibot-plugin-sdk` if needed
- No real Host LLM / network; FakeLLM / FakeSummarizer / FakeMaisaka only
- Prefer extending `test_spec_*` discoverability; do not duplicate wave-1 coverage
- If production blocks ideal spec: `pytest.mark.xfail(strict=True, reason="spec §…")` — do not patch production in this wave
- Exclude Lance/vector from mixed full-suite runs (segfault); Task for Lance may add an *isolated* runner only
- Wave-1 grace/continue xfails were fixed in production already — do not reintroduce those xfails

## Already covered (skip)

Wave 1 `test_spec_*` suite + production fixes for §13.1 grace hold-release and §7.5 continue-with-leaves deadline re-arm. Credits §25.3 #1–#8, outbox retry, protocol E2E, feedback STOPPED/INTERRUPTED, etc.

---

## Task 1: Held-release credit conservation E2E

**Files:** `tests/integration/test_spec_e2e_held_release_credits.py` (new); extend `tests/fakes.py` only if needed

**Requirements:**

1. Manual checkpoint holds children; on release (grace/`open_epoch` or in-grace release), drive **real** `_release_held_delegations` / materialize path — **no** launch stub that bypasses credit debit.
2. Assert parent credits decrease by allocated child amounts (or `parent_credits_after` / `pool_return` conservation); no credit invent across parent+children+pool.
3. If live multi-agent catalog cannot be wired offline, pin the deepest real path available and `xfail(strict=True)` any remaining invent gap with §11 cite.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_e2e_held_release_credits.py -v`

---

## Task 2: §17.2 report_id contract pins

**Files:** `tests/runtime/test_spec_report_id_contract.py` (new) or extend `test_spec_pricing_reporting_outbox.py`

**Requirements:**

1. Pin Maisaka delivery surfaces: append `message_id` / trigger `metadata["report_id"]` stable and equal for the same report (current production contract).
2. Spec also wants a consumer-visible stable id suitable for dedupe — if `render_report` body does **not** include `report_id`, add `xfail(strict=True, reason="spec §17.2: report body should carry stable report_id")` asserting body contains it, documenting the mismatch.
3. Do not change production.

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_report_id_contract.py tests/runtime/test_spec_pricing_reporting_outbox.py -v`

---

## Task 3: Mixed deterministic-limit + valid sibling rejects

**Files:** extend `tests/runtime/test_spec_turns_and_edges.py` or new `tests/runtime/test_spec_mixed_rejects.py`

**Requirements:**

1. One ProcedureBatchCompleted / plan_delegations case: sibling A hits deterministic reject (depth or agent_call_limit) → finalize edge; sibling B valid → materialize; credits conserved (no invent).
2. Distinct from existing all-deterministic→parent and unavailable+valid cases.

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_turns_and_edges.py tests/runtime/test_spec_mixed_rejects.py -v` (whichever files exist)

---

## Task 4: Auto-compact / oversize durable lifecycle

**Files:** extend `tests/runtime/test_spec_compact_and_context.py` or new `tests/runtime/test_spec_compact_lifecycle.py`

**Requirements:**

1. Oversize immutable prefix: assert beyond raise — durable branch terminate / no further PerformAgentCall enqueue / formalized User1 unchanged (manager path as deep as tests-only allows).
2. Compact failure: history unchanged and next agent call not fed oversized rewritten history.
3. Prefer real `ResearchManager._maybe_auto_compact` / prepare path; xfail only if TERMINATED lifecycle is unreachable without production edits.

**Verify:** focused compact lifecycle pytest

---

## Task 5: Extensions / missing providers / pinning smoke

**Files:** `tests/integration/test_spec_extensions_and_providers.py` (new)

**Requirements (§14–16, §25.4):**

1. Extension/agent removal mid-task: unavailable edge finalize; valid sibling continues (build on `test_extension_removal.py` patterns with `test_spec_*` name).
2. Missing optional provider (fetch/search) → visible failure / disabled, no silent empty success.
3. Physical pinning / health contract smoke if existing compat helpers allow without Host.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_extensions_and_providers.py -v`

---

## Task 6: Config reload / concurrency / raw-storage privacy depth

**Files:** `tests/runtime/test_spec_config_concurrency_privacy.py` (new) and/or extend existing

**Requirements (§21–22, §25.4 / §18.2):**

1. Config / price reload broadcast path does not mutate in-flight formalized task text.
2. Scheduler fairness or per-task caps under pause (smoke if not already identical to `test_scheduler.py` — add one `test_spec_*` pin).
3. Raw-storage / debug toggles: at least one privacy assert that default config leaves no raw agent transcript in durable summary layer (complement `test_debug_storage` with spec-named case).

**Verify:** focused pytest for new file(s)

---

## Task 7: Lance isolated runner (optional harness)

**Files:** `tests/run_vector_suite.sh` or `tests/README` note — prefer a small shell script that runs **only** `tests/storage/test_vectors.py` + `test_vector_rebuild.py` separately

**Requirements:**

1. Document/isolated command so CI/humans do not mix Lance into the main suite.
2. Do not flip vector tests into the default aggregate.

**Verify:** script exists and `--help`/dry-run or actual run if environment supports Lance without segfault in isolation

---

## Task 8: Aggregate verification + final review package

**Requirements:**

1. Run all `test_spec_*` and full suite excluding vectors; record counts in `.superpowers/sdd/wave2-task-8-report.md`.
2. Confirm no accidental production edits in this wave (wave-1 production fixes may already be dirty in git — do not revert them; do not add new production edits).

---

## Execution notes

- Model: `cursor-grok-4.5-high`
- Status: DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- **Do not commit**
- Report files: `.superpowers/sdd/wave2-task-N-report.md`
