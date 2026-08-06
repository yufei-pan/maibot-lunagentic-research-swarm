# Contractor Coverage Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close high-value test coverage gaps for `builtin.contractor` against `docs/superpowers/specs/2026-08-06-contractor-procedure-design.md`.

**Architecture:** Extend existing `tests/procedures/test_contractor.py` (unit/handler) and add focused integration coverage where RuntimeHarness is needed. Prefer reusing `ContractorHarness` / `FakeLLMGateway` patterns already in the file. Do **not** change production code unless a test reveals a real bug — then fix minimally and note it in the report.

**Tech Stack:** pytest, pytest-asyncio, existing fakes (`FakeLLMGateway`, `RuntimeHarness`), `uv run pytest`.

## Global Constraints

- Work in the feature worktree only; do not modify MaiBot Host/SDK.
- Prefer Chinese for assertion messages / force-return note substrings already used in production.
- Run tests: `PYTHONPATH=.:/mnt/klein/work/maibot-plugins/maibot-plugin-sdk uv run pytest <path> -v`
- Keep tests behavioral (assert nested not called, selector forwarded, balance math) — not mock-theater.
- Spec: `docs/superpowers/specs/2026-08-06-contractor-procedure-design.md` Internal budget machine + Testing checklist.

---

### Task 1: P0 budget-machine tests

**Files:**
- Modify: `tests/procedures/test_contractor.py`
- Test: same file

**Interfaces:**
- Consumes: `ContractorHarness`, `FakeLLMGateway`, `deps.invoke_nested_procedure`
- Produces: three new async tests locking budget machine steps 4–5–7

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_insufficient_funds_skips_same_turn_nested_procedures(contractor_harness):
    """Turn spend goes negative → force-return; nested invoker must NOT run."""
    nested_calls: list[str] = []
    async def _nested(procedure_id: str, arguments=None, **kwargs):
        nested_calls.append(procedure_id)
        return ProcedureResult(success=True, data={}, error=None, metadata={}, research_credits_charged=0.0)
    contractor_harness.deps.invoke_nested_procedure = _nested
    # Queue turn that requests procedures AND costs > credit_budget
    contractor_harness.llm.queue_json({
        "report": "x",
        "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1+1"}}],
        "return": "should-not-matter",
    })
    # Ensure FakeLLM usage meters charge > budget (reuse existing FixedPrice harness pattern)
    result = await contractor_harness.invoke(..., credit_budget=0.0)  # or tiny budget < turn charge
    assert result.metadata["termination_reason"] == "insufficient_funds"
    assert nested_calls == []


@pytest.mark.asyncio
async def test_explicit_return_ignores_sibling_procedures(contractor_harness):
    """Same-turn return + procedures → return wins; nested not invoked."""
    nested_calls: list[str] = []
    ...
    contractor_harness.llm.queue_json({
        "return": "最终答案",
        "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1"}}],
    })
    result = await contractor_harness.invoke(..., credit_budget=100.0)
    assert result.data["result"] == "最终答案"
    assert result.metadata["termination_reason"] == "returned"
    assert nested_calls == []


@pytest.mark.asyncio
async def test_nested_settlement_overspend_force_returns_without_next_turn(contractor_harness):
    """After nested charges push balance negative → insufficient_funds; no second LLM turn."""
    async def _nested(...):
        return ProcedureResult(..., research_credits_charged=5.0)  # large enough
    contractor_harness.llm.queue_json({
        "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1"}}],
    })
    # Do NOT queue a second LLM response — if loop continues, test fails loudly
    result = await contractor_harness.invoke(..., credit_budget=1.0)  # > turn charge, < turn+nested
    assert result.metadata["termination_reason"] == "insufficient_funds"
    assert len(contractor_harness.llm.calls) == 1
    assert float(result.research_credits_charged) >= 5.0  # includes nested
```

Tune budget/meter numbers against existing `_FixedPrices` / FakeLLM usage in the file so assertions are stable.

- [ ] **Step 2: Run RED**

`PYTHONPATH=.:/mnt/klein/work/maibot-plugins/maibot-plugin-sdk uv run pytest tests/procedures/test_contractor.py -k "insufficient_funds_skips or explicit_return_ignores or nested_settlement" -v`

Expected: FAIL (tests missing or behavior wrong).

- [ ] **Step 3: Implement / adjust until GREEN** (tests only unless bug found)

- [ ] **Step 4: Run full contractor unit file**

`... uv run pytest tests/procedures/test_contractor.py -q`

- [ ] **Step 5: Commit**

```bash
git add tests/procedures/test_contractor.py
git commit -m "test: lock contractor budget machine skip/return/nested overspend"
```

---

### Task 2: P1 overrides, native call_procedure, force-return attempted tools

**Files:**
- Modify: `tests/procedures/test_contractor.py`

**Interfaces:**
- Consumes: `CONTRACTOR_NATIVE_TOOLS`, `GenerationRequest` via FakeLLM call capture
- Produces: tests for personality/temperature/selector, native nested tool, attempted-tools in force-return text

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_contractor_uses_selected_agent_model_selector(contractor_harness):
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(agent_id="builtin.quick_thinker", ...)
    assert contractor_harness.llm.calls[0]["selector"] == "task:utils"  # quick_thinker default


@pytest.mark.asyncio
async def test_contractor_personality_and_temperature_overrides(contractor_harness):
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(
        ...,
        personality="【OVERRIDE_PERSONALITY_MARKER】",
        temperature=0.42,
    )
    req = contractor_harness.llm.calls[0]
    assert abs(float(req["temperature"]) - 0.42) < 1e-9
    assert "【OVERRIDE_PERSONALITY_MARKER】" in json.dumps(req["messages"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_contractor_invalid_temperature_rejected(contractor_harness):
    result = await contractor_harness.invoke(..., temperature="hot")
    assert result.success is False
    assert result.error["code"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_native_call_procedure_runs_allowed_nested(contractor_harness):
    nested_calls: list[str] = []
    async def _nested(procedure_id, **kwargs):
        nested_calls.append(procedure_id)
        return ProcedureResult(success=True, data={"v": 1}, error=None, metadata={}, research_credits_charged=0.1)
    contractor_harness.deps.invoke_nested_procedure = _nested
    contractor_harness.llm.enqueue(FakeLLMResponse(
        tool_calls=[{
            "id": "1",
            "type": "function",
            "function": {"name": "call_procedure", "arguments": json.dumps({
                "procedure_id": "builtin.calculate",
                "arguments": {"expression": "1+1"},
            })},
        }],
    ))
    contractor_harness.llm.queue_json  # second turn return via native contractor_return or text
    # ... assert nested_calls == ["builtin.calculate"] and final returned


@pytest.mark.asyncio
async def test_force_return_includes_attempted_procedure_ids(contractor_harness):
    """insufficient_funds force-return text mentions attempted nested procedure_id."""
    # Same setup as skip-nested test: procedures requested, budget 0, charge > 0
    result = await ...
    assert "builtin.calculate" in str(result.data["result"])
```

Match FakeLLMResponse / enqueue API already used by `test_contractor_native_contractor_return`.

- [ ] **Step 2–4: RED → GREEN → full `test_contractor.py`**

- [ ] **Step 5: Commit**

```bash
git commit -m "test: cover contractor overrides, native call_procedure, force-return tools"
```

---

### Task 3: P2 hard timeout, agent-call counter, richer e2e debit

**Files:**
- Modify: `tests/procedures/test_contractor.py` and/or `tests/integration/test_spec_e2e_contractor.py`
- Possibly: `tests/procedures/test_executor.py` only if hard-timeout for contractor id is cleaner there

**Interfaces:**
- Consumes: ProcedureExecutor wait_for path; RuntimeHarness / TurnWorker; controller `agent_calls_started` if accessible
- Produces:

1. Hard definition timeout alone for a slow contractor-like invoke (may reuse executor pattern with `builtin.contractor` id + slow API).
2. Soft + hard coexistence smoke (optional thin).
3. Assert contractor invoke does not increment `max_agent_calls` / `agent_calls_started` when driven through a minimal manager/turn path — if wiring is too heavy, document and assert at the narrowest available seam (e.g. handler path never touches call counter; integration only if harness exposes it).
4. Extend e2e: already-negative caller balance still receives full contractor bill debit via TurnWorker (prior negative + charge).

- [ ] **Step 1: Write failing tests for hard timeout + negative-balance debit**

```python
# negative prior
effect payload credits_after = -2.0
contractor returns research_credits_charged = 3.0
assert completed.credits_after == pytest.approx(-5.0)
```

Hard timeout: follow `test_timeout_seconds_zero_disables_hard_wait` inverse — definition timeout 0.01 + slow sleep API → `procedure_timeout`, using contractor procedure_id in catalog.

For `max_agent_calls`: if RuntimeHarness start + one agent turn that only calls contractor is feasible within this task, assert `agent_calls_started` increases by 1 (the parent turn) not by contractor inner turns. If blocked, implement the negative-balance + hard-timeout pieces and report `DONE_WITH_CONCERNS` with what blocked the counter assert.

- [ ] **Step 2–4: RED → GREEN → run**

`... uv run pytest tests/procedures/test_contractor.py tests/integration/test_spec_e2e_contractor.py tests/procedures/test_executor.py -k "contractor or timeout_seconds" -q`

Then full suite once before commit:

`... uv run pytest -q`

- [ ] **Step 5: Commit**

```bash
git commit -m "test: cover contractor hard timeout and negative-balance billing"
```

---

### Task 4: P3 polish (multi-turn charge sum + disabled invoke path)

**Files:**
- Modify: `tests/procedures/test_contractor.py`

- [ ] **Step 1: Tests**

```python
@pytest.mark.asyncio
async def test_multi_turn_nested_then_return_sums_charges(contractor_harness):
    # turn1: procedure with nested charge 0.5; turn2: return with turn charge 0.3
    # assert research_credits_charged ≈ sum


@pytest.mark.asyncio
async def test_disabled_contractor_unavailable_via_registry_snapshot():
    # Already have catalog omit — add invoke-through-executor path returning unavailable
    # if not already covered elsewhere
```

- [ ] **Step 2–5: RED → GREEN → full suite → commit**

```bash
git commit -m "test: multi-turn contractor charges and disabled invoke path"
```

---

## Spec coverage checklist

| Gap | Task |
|---|---|
| Overspend skips nested | Task 1 |
| Explicit return ignores procedures | Task 1 |
| Nested settlement overspend | Task 1 |
| personality / temperature / selector | Task 2 |
| Native call_procedure | Task 2 |
| Force-return attempted tools text | Task 2 |
| Hard timeout (± soft) | Task 3 |
| Already-negative caller bill | Task 3 |
| max_agent_calls unchanged | Task 3 (best-effort) |
| Multi-turn charge sum | Task 4 |
| Disabled invoke unavailable | Task 4 |
