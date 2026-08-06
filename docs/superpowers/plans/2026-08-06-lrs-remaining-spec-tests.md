# LRS remaining design-spec test coverage plan

> Plugin: `maibot-lunagentic-research-swarm`  
> Scope: **tests only** (may extend `tests/fakes.py` / test helpers). **Do not modify** `lunagentic_research_swarm/` production code. **Do not git commit.**  
> Goal: close remaining gaps from the design-spec gap analysis so unit/module/E2E FakeLLM coverage approaches full branch/use-case coverage of the design doc.

## Global Constraints

- Work directory: `/mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm`
- Run tests with: `.venv-tests/bin/pytest …` or `PYTHONPATH=.:../maibot-plugin-sdk pytest …`
- Prefer existing patterns in `tests/fakes.py` (`RuntimeHarness`, `FakeLLMGateway`, `FakeSummarizer`, `FakeMaisaka`, `FakeClock`) and specialized harnesses in controller/report/feedback tests
- No real Host LLM / network; enqueue FakeLLM responses
- `RuntimeHarness.root_delegates` skips real credit allocation — for credits E2E use controller materialize / TurnWorker / ChildMaterialized paths
- Exclude Lance vector tests from full-suite runs if they segfault when mixed
- Tests must assert real behavior; no empty asserts
- Comments/docstrings may be English; user-facing strings in production stay zh-CN (tests only here)
- If a production bug blocks a test, **do not fix production** — assert documented current behavior or `pytest.mark.xfail(strict=True, reason=…)` with the spec citation

## Already covered (do not re-implement)

Prior turn already added `tests/runtime/test_spec_*.py` and `tests/integration/test_spec_*.py` for §25.3 #1–#8, message assembly, single-terminal finalize skip, INTERRUPTED no reminder, basic FakeLLM E2E (outbox retry, fan-out, stop, early checkpoint, wall-clock-free FINAL). Extend those files when adding related cases; do not duplicate.

---

## Task 1: Compact / oversize / auto-compact fan-out

**Files:** `tests/runtime/test_spec_compact_and_context.py` (new)

**Requirements (design §12.2, §23.2):**

1. Manual compact runs **before** child clone so all children inherit compacted history (assert message prefix shared / identical post-compact history on children when using ProcedureExecutor / context helpers — follow patterns in `test_context_invariance.py` and `procedures/test_core.py`).
2. Auto-compact is per-child after clone: child A above threshold compacts; sibling B below does not (use `should_auto_compact` + branch-local used_tokens fixtures; if full clone path is unavailable without production edits, unit-pin the decision + document integration limit).
3. Immutable system+catalog+formalized prefix already larger than usable model window → explicit error path / branch terminate; formalized User1 text unchanged.
4. Compact failure (FakeSummarizer returns failure) → branch history unchanged; do not treat as successful compact.

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_compact_and_context.py -v`

---

## Task 2: Grace + continue deadline re-arm + multi-epoch coverage

**Files:** `tests/runtime/test_spec_grace_and_epochs.py` (new); extend `tests/runtime/test_grace_period.py` only if cleaner

**Requirements (§13.1, §7.5, §13.2):**

1. Manual checkpoint requested **during** grace → after summary, branch continues without waiting for held epoch end (not stuck in WAITING_REPORT_WITH_CHECKPOINT forever).
2. After intermediate → RUNNING, continue / next deadline path **re-arms** report deadline (ArmDeadline effect or manager timer state) when applicable.
3. Multi-epoch coverage: branch A terminal early; later epoch coverage includes A's terminal + B checkpoint; kind stays INTERMEDIATE until all terminal.

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_grace_and_epochs.py -v`

---

## Task 3: Stop→reminder + INTERRUPTED continue wiring

**Files:** `tests/integration/test_spec_feedback_control.py` (new) or extend `test_spec_feedback_interrupted.py` / reminder harness

**Requirements (§7.4, §20.3, §18.4):**

1. STOPPED schedules feedback reminder (unlike EXPIRED/INTERRUPTED).
2. Crash/mark INTERRUPTED does not insert reminder; continue into new round does not create reminder for the interrupted round.
3. Reuse ReminderHarness / FeedbackService patterns from `integration/test_feedback_reminders.py`.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_feedback_control.py tests/integration/test_spec_feedback_interrupted.py -v`

---

## Task 4: Protocol / self-delegation / call_id / mixed rejects / native

**Files:** `tests/runtime/test_spec_turns_and_edges.py` (new); optionally extend executor tests

**Requirements (§9.1, §9.3, §10, §23.1):**

1. Procedure `call_id` optional; when present, echoed in result metadata; result order = request order (not sorted by call_id).
2. Successful self-delegation (`agent_id` = parent) materializes child / allows tool-loop under depth/credit limits.
3. Mixed rejected edges: `agent_unavailable` → parent retry once; deterministic depth/call-limit → finalize sibling; credits conserved.
4. Protocol invalid → correction turn via FakeLLM enqueue(invalid, valid) → then terminate; exactly one correction when credits allow.
5. Native `submit_swarm_turn` path (empty assistant text) completes a branch turn via TurnWorker/FakeLLMGateway native mode if harness supports it.

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_turns_and_edges.py -v`

---

## Task 5: Pricing cache=false + report header + outbox report_id

**Files:** `tests/runtime/test_spec_pricing_reporting_outbox.py` (new)

**Requirements (§11.2, §13.3, §17.2):**

1. `cache=false` profile treats prompt tokens as miss for charging.
2. Missing actual `model_name` settlement uses estimated path / flagged reconciliation (assert existing reconcile behavior).
3. Intermediate `render_report` header includes required plugin stats fields (task_id/epoch/running/credits etc.); body is separate from stats.
4. Outbox/Maisaka delivered payload includes stable `report_id` suitable for consumer dedupe.

**Verify:** `.venv-tests/bin/pytest tests/runtime/test_spec_pricing_reporting_outbox.py -v`

---

## Task 6: E2E credits-real path (allocate/materialize/zero-child)

**Files:** `tests/integration/test_spec_e2e_credits.py` (new); extend `tests/fakes.py` **only if needed** with helpers that do not change production

**Requirements (§25.3, §11, E1/E2):**

1. Root → allocate A/B/C 50/25/25 through **real** `allocate_children` + `ChildMaterialized` / controller path (not only `root_delegates` synthetic insert); assert leaf balances and conservation.
2. Zero-credit child + FakeLLM nonzero usage + procedure in envelope → procedures run → finalize → **no** grandchild materialize effect.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_e2e_credits.py -v`

---

## Task 7: E2E report/grace/pause/stop workloads

**Files:** `tests/integration/test_spec_e2e_control_and_grace.py` (new)

**Requirements (§7, §13, E3–E8, E11):**

1. Manual checkpoint holds children → deadline → INTERMEDIATE → release → children run → FINAL.
2. Mid-grace agent return auto-checkpoints clone; new child outside current frontier.
3. Full grace timeout clones **pre-call** stable history; late FakeLLM content not in checkpoint text.
4. Pause → pause timeout → EXPIRED → continue new round from summary layer (no raw restore).
5. Stop with in-flight work discarded → STOPPED (generation bump if applicable).
6. Continue with all-zero active leaves + signed adjustment at barrier.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_e2e_control_and_grace.py -v`

---

## Task 8: E2E protocol correction + native + self-loop + interrupted reopen

**Files:** `tests/integration/test_spec_e2e_protocol_and_recovery.py` (new)

**Requirements (E9, E10, E13, E14):**

1. FakeLLM invalid envelope then valid → one correction → branch can terminate.
2. Native tool-only turn completes without assistant prose body when supported.
3. Self-delegation tool loop then terminate.
4. Seed stranded RUNNING → mark interrupted on “restart” path → continue round 2; no reminder for interrupted round.

**Verify:** `.venv-tests/bin/pytest tests/integration/test_spec_e2e_protocol_and_recovery.py -v`

---

## Task 9: Suite verification harness script (tests-only)

**Files:** `tests/run_spec_suite.sh` (optional) **or** just document in report — prefer running pytest aggregation without new shell if unnecessary.

**Requirements:**

1. Run all `test_spec_*.py` plus related new files; exclude vector tests.
2. Report pass counts in task report.

**Verify:** aggregate pytest command in report.

---

## Execution notes for implementers

- Status values: DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- **Do not commit**
- Write full report to the assigned report file
- Match naming `test_spec_*` for discoverability
- Prefer Chinese-friendly assertion messages only when matching existing style; English ok in tests
