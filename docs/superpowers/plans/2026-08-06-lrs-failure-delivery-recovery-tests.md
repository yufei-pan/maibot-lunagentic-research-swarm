# LRS failure / delivery / recovery coverage wave 3

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close residual high-value FakeLLM/harness gaps: summarizer failure modes, Maisaka deliver gates, crash mid-outbox, live reallocation, multi-task fairness, past-case learning presentation, and a deeper native-tools research round.

**Architecture:** Tests-only wave on `RuntimeHarness` / FakeLLM / FakeSummarizer / FakeMaisaka. Prefer new `test_spec_*` files under `tests/runtime/` or `tests/integration/`. Extend `tests/fakes.py` only when necessary. No Host/network.

**Tech Stack:** pytest-asyncio, plugin SQLite store, existing FakeLLM harness patterns from wave 1–2.

## Global Constraints

- Workdir: `/mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm`
- Pytest: `.venv-tests/bin/pytest …` with `PYTHONPATH=.:../maibot-plugin-sdk` when needed
- **Tests only** — do not modify `lunagentic_research_swarm/` unless a path is unreachable without a one-line harness hook; prefer `xfail(strict=True, reason="spec §…")` over production edits
- **Do not git commit**
- No Lance/vector in default runs; leave `tests/run_vector_suite.sh` alone
- Prefer Chinese only for user-facing strings already present in production; test names/docstrings may stay English with § cites
- Model for implementers/reviewers: `cursor-grok-4.5-high`
- Report each task to `.superpowers/sdd/wave3-task-N-report.md`
- Status: DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT

## Already covered (skip / do not duplicate)

- Credits §25.3 #1–#8, grace hold-release, continue deadline re-arm, report_id in header (§17.2), mixed rejects, compact lifecycle smoke, reminder INTERRUPTED, protocol correction-turn unit/E2E basics, native_tools empty-assistant completion (`test_e2e_e10_*`, `test_spec_turns_and_edges` native path)

---

### Task 1: Summarizer failure E2E (`summary_unavailable` + `COMPLETED_WITH_ERRORS`)

**Files:**
- Create: `tests/integration/test_spec_e2e_summarizer_failures.py`
- Modify (only if needed): `tests/fakes.py` (`FakeSummarizer` fail hooks)

**Requirements:**

1. **Branch finalize fail (§23.2):** Force `FakeSummarizer.finalize_branch` → unsuccessful `SummaryResult`. Assert durable terminal/unavailable record exists (status/text/kind that production actually writes — look for `summary_unavailable` or FAILED BRANCH_FINAL with empty/non-invented body). Sibling branches that succeed must continue; no fabricated prose from the failed summarizer call.
2. **Task finalize fail (§13.4 / §23.2):** Force `finalize_task` fail on a path that reaches FINALIZING. Assert task status `COMPLETED_WITH_ERRORS`, report status FAILED (or equivalent), and prior branch coverage still present in durable report/coverage (no invented task-level synthesis body). Prefer harness/`RuntimeHarness` over copying `test_report_runtime_bridge` verbatim — bridge already pins manager-level; this task needs a `test_spec_*` E2E name and FakeLLM-shaped drive if feasible; deepest real path OK if full fan-out is awkward.
3. Do not invent production APIs; pin observed durable fields.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_e2e_summarizer_failures.py -v`

- [ ] **Step 1:** Write failing/spec-named tests
- [ ] **Step 2:** Wire FakeSummarizer fail hooks if needed
- [ ] **Step 3:** Run verify command; record output in report
- [ ] **Step 4:** Self-review against §23.2; write `wave3-task-1-report.md`

---

### Task 2: `[reporting]` deliver gates

**Files:**
- Create: `tests/runtime/test_spec_deliver_gates.py`
- Modify (only if needed): harness constructor / manager `_runtime_limits` injection in `tests/fakes.py`

**Requirements (§17.2 / config `[reporting]`):**

1. With `deliver_intermediate=False` (and `deliver_final=True` or both false as separate cases): after an intermediate report epoch, durable `reports` row exists with text, but **no** Maisaka append/trigger outbox rows (or deliver_once makes no append calls) for that intermediate report.
2. With `deliver_final=False`: final report still durable; Maisaka not appended for final.
3. Default True path may be a one-liner control if not already identical elsewhere — optional.
4. Wire limits through whatever production path `ReportCoordinator` already reads (`deliver_intermediate` / `deliver_final` on coordinator / manager runtime limits).

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_deliver_gates.py -v`

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Wire config/limits in harness if needed
- [ ] **Step 3:** Verify + report `wave3-task-2-report.md`

---

### Task 3: Crash mid-outbox delivery

**Files:**
- Create: `tests/runtime/test_spec_outbox_crash_recovery.py` (or extend `test_spec_pricing_reporting_outbox.py` if cleaner)
- Use: `MaisakaOutbox`, `FakeMaisaka`, `SQLiteStateStore`

**Requirements (§18.4 / §17.2):**

1. Seed or produce append+trigger outbox pair for one `report_id`.
2. Simulate crash after append succeeds / before trigger: mark append delivered (or stop FakeMaisaka after first append), restart `MaisakaOutbox`, continue delivery.
3. Assert: append not duplicated (same `message_id` / single append call for that report), trigger eventually fires once with matching `metadata["report_id"]`.
4. Optionally assert visible text still contains `report_id：…` when payload came from rendered report.

**Verify:** focused pytest for the new/extended file

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Verify + `wave3-task-3-report.md`

---

### Task 4: Live-tree credit reallocation E2E

**Files:**
- Create: `tests/integration/test_spec_e2e_reallocation.py` and/or extend credits E2E
- Reference: `redistribute_pool` unit pins in `test_spec_coverage_credits.py` — do not re-unit-test math; drive controller/manager path

**Requirements (§11.6):**

1. Running task with ≥1 live child branch; invoke the real reallocation / pool-redistribute control path available to tests (command/controller API used elsewhere).
2. Assert parent/children/pool conservation (no invent); live balances update as production defines.
3. If only pure-function redistribute is reachable without Host command plumbing, pin deepest manager/controller call and note limitation in report; `xfail` only for invent gaps that production cannot exercise offline.

**Verify:** focused pytest

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Verify + `wave3-task-4-report.md`

---

### Task 5: Multi-task fairness / barrier priority smoke

**Files:**
- Create: `tests/runtime/test_spec_scheduler_fairness.py` or extend `test_scheduler.py` with `test_spec_*` names
- Reference existing: `tests/runtime/test_scheduler.py`, `test_spec_config_concurrency_privacy.py`

**Requirements (§22):**

1. Two tasks competing for LLM slots: wide fan-out on task A must not permanently starve task B (fairness smoke — use FakeScheduler or real `ResearchScheduler` as existing tests do).
2. Barrier-class work (stop/pause/report/continue) is preferred over ordinary child launch when both queued — pin with existing scheduler priority APIs if present.
3. Do not duplicate identical assertions already in `test_scheduler.py`; add `test_spec_*` discoverable names that cite §22.

**Verify:** focused pytest

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Verify + `wave3-task-5-report.md`

---

### Task 6: Past-case learning presentation (§20.2)

**Files:**
- Extend: `tests/procedures/test_past_cases.py` and/or create `tests/procedures/test_spec_past_cases_learning.py`

**Requirements:**

1. When retrieval returns accepted/good vs rejected/mixed feedback cases, the procedure result / messages presented to the researcher distinguish them (accepted preferred or rejected surfaced as risk/antipattern — pin actual production formatting).
2. Assert LRS does **not** auto-mutate prompts/config/agent ranking (smoke: no write to agent registry / config from the past_cases invoke path).
3. Keep offline; use existing fake vector/store harness patterns in `test_past_cases.py`.

**Verify:** focused pytest

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Verify + `wave3-task-6-report.md`

---

### Task 7: Native-tools full research round E2E

**Files:**
- Create or extend: `tests/integration/test_spec_e2e_native_tools_round.py`
- Build on: `test_spec_e2e_protocol_and_recovery.py` helpers (`_native_tool_call`)

**Requirements (§9.3):**

1. One FakeLLM research round with `protocol=native_tools`: root delegates (or works), at least one branch checkpoint or terminal, report epoch produces durable report.
2. Distinct from empty-assistant completion E10 — must exercise tool-call shaped turns with non-empty structured tool payloads.
3. Assert protocol stayed `native_tools` on completed turns and report text is non-empty (and preferably embeds `report_id：` if synthesis path used).

**Verify:** focused pytest

- [ ] **Step 1:** Write tests
- [ ] **Step 2:** Verify + `wave3-task-7-report.md`

---

### Task 8: Aggregate verification + final review package

**Requirements:**

1. Run full suite excluding vectors; record pass count in `.superpowers/sdd/wave3-task-8-report.md`.
2. Confirm no accidental production edits (`git diff --stat` should be tests/fakes/docs/plans/sdd only).
3. List any remaining xfails / DONE_WITH_CONCERNS from tasks 1–7.

**Verify:** `.venv-tests/bin/pytest -q --tb=no` (exclude vector files if they auto-collect — currently they do not in default collection)

- [ ] **Step 1:** Aggregate run + report
- [ ] **Step 2:** Ready for whole-branch review

---

## Execution notes

- Fresh implementer per task; task review after each; fix Critical/Important before next task
- Do not pause for human between tasks
- Whole-branch review after Task 8
