# Contractor Procedure & Procedure Research Credits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add procedure research-credit billing (`credits` budget hint + `research_credits_charged`) and a bundled `builtin.contractor` outsider LLM procedure with hard/soft timeouts.

**Architecture:** Keep procedures one-shot `invoke → ProcedureResult`. Runtime passes `credit_budget` via `scoped_metadata`, then debits the calling branch once per top-level result. `builtin.contractor` implements its multi-turn loop inside the bundled handler (first-party imports), reporting a single top-level charge. No upfront credit pockets.

**Tech Stack:** Python 3.10+, Pydantic v2, asyncio, existing LRS runtime (`ProcedureExecutor`, `credits`, `LLMGateway`, `FairScheduler`), pytest / pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-06-contractor-procedure-design.md`

## Global Constraints

- Do not modify MaiBot Host or SDK.
- Never persist raw objective, planner context, transcript, reasoning, or raw procedure payloads by default.
- `external_cost*` stays telemetry-only; only `research_credits_charged` debits research branch balances.
- Auto-compact does not debit research credits; agent-requested `core.compact` does.
- Contractor turns do not count toward `max_agent_calls_per_task`.
- Nested `builtin.contractor` is forbidden (exact id); nested `core.checkpoint` / `core.terminate` forbidden; nested `core.compact` allowed.
- Only the calling agent’s procedure-batch results debit the caller; nested invokes inside contractor are internal and must not double-bill.
- `timeout_seconds = 0` means hard executor timeout disabled; contractor soft timeout is `time_budget_seconds` (default `0`).
- Prefer Chinese for user-facing / log strings; keep contract field names in English as in the spec.
- Run tests with: `PYTHONPATH=.:../maibot-plugin-sdk pytest <path> -v`

## File map

| File | Responsibility |
|---|---|
| `lunagentic_research_swarm/extensions/contracts.py` | `timeout_seconds` allow `0`; `ProcedureResult.research_credits_charged` |
| `lunagentic_research_swarm/config.py` | `ProcedureOverride.timeout_seconds` allow `0` |
| `lunagentic_research_swarm/llm/protocol.py` | `ProcedureRequest.credits` |
| `lunagentic_research_swarm/runtime/credits.py` | `charge_procedure_usage(...)` ledger helper |
| `lunagentic_research_swarm/procedures/executor.py` | Skip `wait_for` when timeout `0`; inject `credit_budget`; sum charges into batch `credits_after` |
| `lunagentic_research_swarm/runtime/reducer.py` / manager path | Persist procedure research-credit ledger entries with batch |
| `lunagentic_research_swarm/procedures/core.py` | Agent-requested compact reports charge from summarizer metering |
| `lunagentic_research_swarm/procedures/bundled/contractor.py` | Definition + handler loop |
| `lunagentic_research_swarm/procedures/bundled/provider.py` | Register contractor; inject runtime deps |
| `config.default.toml` | Explicit per-builtin `enabled` / `timeout_seconds` examples |
| Tests under `tests/extensions`, `tests/procedures`, `tests/runtime`, `tests/integration` | Spec coverage |

---

### Task 1: Allow `timeout_seconds = 0` on definitions and overrides

**Files:**
- Modify: `lunagentic_research_swarm/extensions/contracts.py` (`ProcedureDefinition.timeout_seconds`)
- Modify: `lunagentic_research_swarm/config.py` (`ProcedureOverride.timeout_seconds`)
- Modify: `lunagentic_research_swarm/procedures/executor.py` (skip `wait_for` when `0`)
- Modify: `tests/extensions/test_validation.py`
- Test: `tests/procedures/test_executor.py` (add timeout-disabled case)

**Interfaces:**
- Consumes: existing `ProcedureDefinition`, executor `asyncio.wait_for` path
- Produces: `timeout_seconds: float` with `ge=0.0, le=600.0`; `0` disables hard wait

- [ ] **Step 1: Write failing validation tests**

In `tests/extensions/test_validation.py`, change the parametrized invalid timeouts so `0` is **valid**, and add:

```python
def test_procedure_timeout_zero_means_disabled() -> None:
    definition = ProcedureDefinition.model_validate(procedure_payload(timeout_seconds=0.0))
    assert definition.timeout_seconds == 0.0
```

Update any test that currently expects `timeout_seconds=0` to raise.

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/extensions/test_validation.py::test_procedure_timeout_zero_means_disabled -v`

Expected: FAIL (current `gt=0.0` rejects `0`).

- [ ] **Step 3: Implement contract + override + executor**

```python
# contracts.py
timeout_seconds: float = Field(default=30.0, ge=0.0, le=600.0)

# config.py ProcedureOverride
timeout_seconds: float | None = _ui_field(None, label="超时（秒）", hint="0 表示禁用硬超时；留空继承定义。", ge=0.0)
```

In `executor.py`, replace unconditional `wait_for`:

```python
timeout_seconds = float(getattr(definition, "timeout_seconds", 30.0))
call = self._api_call(entry, procedure_id, invocation)
if timeout_seconds > 0:
    raw = await asyncio.wait_for(call, timeout=timeout_seconds)
else:
    raw = await call
```

Keep the existing `TimeoutError` → `procedure_timeout` path for `timeout_seconds > 0` only.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/extensions/test_validation.py tests/procedures/test_executor.py -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/extensions/contracts.py lunagentic_research_swarm/config.py \
  lunagentic_research_swarm/procedures/executor.py tests/extensions/test_validation.py tests/procedures/test_executor.py
git commit -m "$(cat <<'EOF'
feat: allow procedure timeout_seconds=0 to disable hard wait

EOF
)"
```

---

### Task 2: Envelope `credits` + `ProcedureResult.research_credits_charged`

**Files:**
- Modify: `lunagentic_research_swarm/llm/protocol.py` (`ProcedureRequest`)
- Modify: `lunagentic_research_swarm/extensions/contracts.py` (`ProcedureResult`)
- Modify: `tests/extensions/test_validation.py`
- Modify: `tests/llm/test_protocol.py` (or create credits field tests there)
- Test: add helpers if needed in `lunagentic_research_swarm/procedures/billing.py` (optional thin module) — prefer keeping extract helper next to executor to avoid extra file unless reuse demands it

**Interfaces:**
- Consumes: `ProcedureRequest`, `ProcedureResult`
- Produces:
  - `ProcedureRequest.credits: float = 0.0` (`ge=0.0`)
  - `ProcedureResult.research_credits_charged: float = 0.0` (`ge=0.0`)
  - `extract_research_credits_charged(result: ProcedureResult | Mapping) -> float`

- [ ] **Step 1: Write failing tests**

```python
# protocol
def test_procedure_request_credits_default_zero() -> None:
    req = ProcedureRequest(procedure_id="builtin.web_search", arguments={})
    assert req.credits == 0.0

def test_procedure_request_rejects_negative_credits() -> None:
    with pytest.raises(ValidationError):
        ProcedureRequest(procedure_id="builtin.web_search", credits=-1.0, arguments={})

# contracts
def test_procedure_result_research_credits_charged_default() -> None:
    result = ProcedureResult.model_validate(
        {"success": True, "data": {}, "error": None, "metadata": {}}
    )
    assert result.research_credits_charged == 0.0
    assert set(ProcedureResult.model_fields) == {
        "success", "data", "error", "metadata", "research_credits_charged"
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_protocol.py tests/extensions/test_validation.py -k "credits or research_credits or public_fields or fixed_strict" -v`

Expected: FAIL on missing fields / exact field-set assertions.

- [ ] **Step 3: Implement models + extractor**

```python
# protocol.py
class ProcedureRequest(BaseModel):
    ...
    credits: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

# contracts.py
class ProcedureResult(_StrictContract):
    ...
    research_credits_charged: float = Field(default=0.0, ge=0.0)

def extract_research_credits_charged(result: Any) -> float:
    raw = getattr(result, "research_credits_charged", None)
    if raw is None and isinstance(result, Mapping):
        raw = result.get("research_credits_charged", 0.0)
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value
```

Put `extract_research_credits_charged` in `lunagentic_research_swarm/procedures/billing.py` (new small module) and export from `procedures/__init__` only if other packages already re-export helpers; otherwise import the module directly from executor/reducer.

Update all `ProcedureResult(...)` construction sites that break keyword-only/`model_validate` field-set tests (grep `ProcedureResult(`). Default `0.0` should keep most call sites working if they use the model constructor with defaults.

Remove any provisional `metadata["research_credits_charged"]` as the **sole** billing channel in `core.py` failures once the first-class field exists (metadata key may remain for debugging but must not be required for debit).

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/extensions/test_validation.py tests/llm/test_protocol.py tests/procedures/ -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/llm/protocol.py lunagentic_research_swarm/extensions/contracts.py \
  lunagentic_research_swarm/procedures/billing.py tests/extensions/test_validation.py tests/llm/test_protocol.py
git commit -m "$(cat <<'EOF'
feat: add procedure credits budget and research_credits_charged

EOF
)"
```

---

### Task 3: Pass `credit_budget` and debit caller after procedure batch

**Files:**
- Modify: `lunagentic_research_swarm/procedures/executor.py`
- Modify: `lunagentic_research_swarm/runtime/credits.py` (add `charge_procedure_usage`)
- Modify: `lunagentic_research_swarm/runtime/reducer.py` (ledger commands on `ProcedureBatchCompleted`) and/or manager if ledger is assembled outside reducer
- Test: `tests/runtime/test_spec_coverage_credits.py` or new `tests/runtime/test_procedure_credits.py`
- Test: `tests/procedures/test_executor.py`

**Interfaces:**
- Consumes: `ProcedureRequest.credits`, `extract_research_credits_charged`, branch `credits_after` on batch payload
- Produces:
  - `scoped_metadata["credit_budget"] = float`
  - `ProcedureBatchCompleted.credits_after = prior - sum(charges)`
  - ledger entry kind suitable for procedure debit (reuse existing ledger kinds if one fits; otherwise add `procedure_charge` consistently with `CreditLedgerEntry` validators)

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_procedure_batch_debits_research_credits_charged():
    """Fake handler returns research_credits_charged=3.5; payload credits_after=10."""
    # Build PerformProcedureBatch with one ordinary procedure request (credits=0),
    # local invoker returning ProcedureResult(success=True, data={}, error=None,
    #   metadata={}, research_credits_charged=3.5).
    # prior credits_after on effect payload = 10.0
    completed = await worker.perform_procedure_batch(effect)
    assert completed.credits_after == pytest.approx(6.5)


@pytest.mark.asyncio
async def test_procedure_batch_missing_charge_bills_zero():
    """Legacy-shaped result with default charge 0 leaves credits_after unchanged."""
    completed = await worker.perform_procedure_batch(effect_with_prior_7)
    assert completed.credits_after == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_executor_passes_credit_budget_in_scoped_metadata():
    seen = {}

    async def invoker(**kwargs):
        seen["scoped"] = kwargs.get("scoped_metadata") or kwargs.get("invocation")
        return ProcedureResult(success=True, data={}, error=None, metadata={})

    # invoke request with credits=4.0 through executor; assert credit_budget == 4.0
    assert float(seen["scoped"]["credit_budget"]) == pytest.approx(4.0)
```

Negative outer `credits` are rejected by `ProcedureRequest` validation (Task 2); no separate executor debit path for invalid requests.

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_procedure_credits.py tests/procedures/test_executor.py -k credit -v`

Expected: FAIL (no debit / no budget injection).

- [ ] **Step 3: Implement injection + debit + ledger**

When building `ProcedureInvocation` in the executor:

```python
budget = float(getattr(request, "credits", 0.0) or 0.0)
if isinstance(request, Mapping):
    budget = float(request.get("credits", 0.0) or 0.0)
scoped = dict(base_scoped)
scoped["credit_budget"] = budget
```

After `invoke_many` collects results (in executor or `TurnWorker.perform_procedure_batch`):

```python
prior = float(payload.get("credits_after", 0.0))
charged = sum(extract_research_credits_charged(item.result) for item in results)
credits_after = prior - charged
```

Add `charge_procedure_usage(...)` in `credits.py` that returns `StoreCommand`s for `insert_credit_ledger` with metadata `{procedure_id, call_id, budget_hint, ...}` and `amount = -charged` (or positive amount with entry_kind that means debit — **match existing ledger sign conventions** by reading nearby agent reconcile entries before choosing the sign).

In reducer `ProcedureBatchCompleted` handling, append those ledger commands when charged > 0 (or always when charged != 0). Prefer assembling ledger commands in one place (manager/executor → event side-channel, or reducer pure function from result fields) without double-writing.

**Sign convention check (do this before coding):** inspect existing `insert_credit_ledger` amounts for agent usage. Mirror that so procedure bills reduce `balance_after` the same way.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_procedure_credits.py tests/procedures/test_executor.py tests/runtime/test_spec_coverage_credits.py tests/integration/test_spec_e2e_credits.py -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/procedures/executor.py lunagentic_research_swarm/runtime/credits.py \
  lunagentic_research_swarm/runtime/reducer.py lunagentic_research_swarm/runtime/turns.py \
  tests/runtime/test_procedure_credits.py tests/procedures/test_executor.py
git commit -m "$(cat <<'EOF'
feat: bill research credits from procedure results

EOF
)"
```

---

### Task 4: Bill agent-requested `core.compact` (not auto-compact)

**Files:**
- Modify: `lunagentic_research_swarm/procedures/core.py`
- Modify: `lunagentic_research_swarm/runtime/manager.py` (auto-compact path must keep charge `0` / not debit)
- Test: `tests/runtime/test_spec_compact_and_context.py` and/or `tests/procedures/test_core.py`

**Interfaces:**
- Consumes: summarizer usage / `meter_summarizer_usage` or price catalog charge helpers already used for compact
- Produces: `ProcedureResult.research_credits_charged` set for agent-requested compact only

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_agent_requested_compact_charges_caller_even_with_zero_budget(runtime_harness):
    # Start task; force agent turn requesting core.compact with credits=0.
    # Stub summarizer/pricing so compact meters a known non-zero research charge C.
    before = float(runtime_harness.status()["active_leaves"][0]["credits"])
    # ... drive one turn containing compact ...
    after = float(runtime_harness.status()["active_leaves"][0]["credits"])
    assert after == pytest.approx(before - C)


@pytest.mark.asyncio
async def test_auto_compact_does_not_debit_research_credits(runtime_harness):
    # Build a branch context that trips auto_compact_tokens; complete an agent call.
    # Record active_leaves before/after auto-compact path inside prepare_agent_effect.
    # Assert research balance is affected only by the agent reservation/reconcile,
    # not by an extra compact research debit (compare against a run with auto-compact disabled).
```

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_spec_compact_and_context.py tests/procedures/test_core.py -k compact -v`

- [ ] **Step 3: Implement compact charge**

In `execute_core_procedure` for `core.compact`, after summarizer returns, compute research-credit equivalent with the same pricing path used elsewhere for summarizer **but** set `research_credits_charged` on the `ProcedureResult`. Do **not** write the research ledger inside `execute_core_procedure` (keep current comment: caller/batch debit path owns ledger).

Ensure manager `_maybe_auto_compact` / `_record_auto_compact_call` either:
- uses a path that forces `research_credits_charged=0`, or
- bypasses the procedure-batch debit path entirely (today auto-compact is not a branch procedure batch item — keep it that way).

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/runtime/test_spec_compact_and_context.py tests/procedures/test_core.py tests/runtime/test_procedure_credits.py -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/procedures/core.py lunagentic_research_swarm/runtime/manager.py \
  tests/runtime/test_spec_compact_and_context.py tests/procedures/test_core.py
git commit -m "$(cat <<'EOF'
feat: charge research credits for agent-requested compact

EOF
)"
```

---

### Task 5: Config toggles for every builtin + contractor stub registration

**Files:**
- Create: `lunagentic_research_swarm/procedures/bundled/contractor.py` (definition + stub handler)
- Modify: `lunagentic_research_swarm/procedures/bundled/provider.py`
- Modify: `config.default.toml`
- Test: `tests/procedures/test_bundled_provider.py`
- Test: `tests/procedures/test_contractor.py` (definition + disabled)

**Interfaces:**
- Consumes: `ProcedureDefinition`, `BundledProcedureProvider.describe/invoke`
- Produces:
  - `CONTRACTOR_PROCEDURE_ID = "builtin.contractor"`
  - `contractor_procedure_definitions() -> list[ProcedureDefinition]` with `timeout_seconds=0`, `enabled=True`
  - Stub invoke returns structured error `not_implemented` **or** (preferred) a clear `invalid_arguments` until Task 6 — better: stub succeeds with empty loop only after Task 6; for this task, register definition and handler that returns `success=False, error={code: "contractor_runtime_missing"}` if deps missing, so catalog tests pass

- [ ] **Step 1: Write failing tests**

```python
def test_contractor_definition_defaults():
    defs = contractor_procedure_definitions()
    assert len(defs) == 1
    d = defs[0]
    assert d.procedure_id == "builtin.contractor"
    assert d.timeout_seconds == 0.0
    assert d.enabled is True
    assert "agent_id" in d.arguments_schema["properties"]
    assert "question" in d.arguments_schema["properties"]
    assert "time_budget_seconds" in d.arguments_schema["properties"]

def test_bundled_provider_includes_contractor_when_described():
    # describe() includes builtin.contractor
    ...
```

Add commented/default blocks in expectations for `config.default.toml` presence of `[procedures."builtin.contractor"]` and other builtin ids (can be a simple file read test or checklist in this task’s commit without over-testing TOML).

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_contractor.py tests/procedures/test_bundled_provider.py -v`

- [ ] **Step 3: Implement definition, stub handler, config defaults**

`arguments_schema` properties: `agent_id` (string), `question` (string), `temperature` (number, optional), `personality` (string|null, optional), `time_budget_seconds` (number, default 0, minimum 0).

`config.default.toml` — add explicit sections for each bundled procedure id (memory/analysis/provenance/web_search/past_cases/contractor). Contractor:

```toml
[procedures."builtin.contractor"]
enabled = true
timeout_seconds = 0
```

Other builtins: commented examples with `enabled = true` and their current timeouts, matching existing style.

Wire `provider.describe()` / handlers map.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_contractor.py tests/procedures/test_bundled_provider.py tests/test_config.py -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/procedures/bundled/contractor.py \
  lunagentic_research_swarm/procedures/bundled/provider.py config.default.toml \
  tests/procedures/test_contractor.py tests/procedures/test_bundled_provider.py
git commit -m "$(cat <<'EOF'
feat: register builtin.contractor definition and config toggles

EOF
)"
```

---

### Task 6: Contractor loop — context, protocol, return (no nested tools yet)

**Files:**
- Modify: `lunagentic_research_swarm/procedures/bundled/contractor.py`
- Modify: `lunagentic_research_swarm/procedures/bundled/provider.py` / `services.py` to inject `ContractorDeps` (llm gateway, price catalog, agent catalog snapshot accessor, calling protocol, etc.)
- Test: `tests/procedures/test_contractor.py`

**Interfaces:**
- Consumes: `scoped_metadata.credit_budget`, selected agent definition, calling agent protocol from scoped_metadata (pass `caller_protocol`, `caller_agent_id`, frozen catalogs via scoped_metadata or deps)
- Produces: working handler that:
  1. Builds outsider system+user messages (spec § Fresh context)
  2. Calls selected agent’s model with caller protocol
  3. Returns via `contractor_return` / JSON `return` / last text
  4. Sets `research_credits_charged` from metered turn usage
  5. Sets metadata `termination_reason`

**Scoped metadata the runtime must pass into contractor invokes** (add in executor when `procedure_id == builtin.contractor` **or** always pass these generic keys for all procedures):

```python
scoped_metadata.update({
    "credit_budget": budget,
    "caller_protocol": caller_protocol,  # "json_envelope" | "native_tools"
    "caller_agent_id": agent_id,
})
```

Agent catalog / model routing should come from injected deps bound to the frozen round snapshot when possible (do not read live mutable catalog if round freeze exists — follow manager patterns).

- [ ] **Step 1: Write failing tests with FakeLLM**

```python
@pytest.mark.asyncio
async def test_contractor_returns_explicit_json_return(contractor_harness):
    contractor_harness.llm.queue_json({"return": "答案"})
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="1+1?",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert result.success is True
    assert result.data["result"] == "答案"  # use the result_schema field name chosen in Task 5
    assert result.metadata["termination_reason"] == "returned"


@pytest.mark.asyncio
async def test_contractor_last_text_return_without_tool_call(contractor_harness):
    contractor_harness.llm.queue_text("仅正文结论")
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="总结",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert "仅正文结论" in str(result.data)


@pytest.mark.asyncio
async def test_contractor_fresh_context_excludes_parent_task(contractor_harness):
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="旁路问题",
        caller_protocol="json_envelope",
        credit_budget=1.0,
    )
    sent = contractor_harness.llm.calls[0]["messages"]
    blob = json.dumps(sent, ensure_ascii=False)
    assert "旁路问题" in blob
    assert "FORMALIZED_TASK_MARKER" not in blob
    assert "parent transcript marker" not in blob
```

In Task 5, pick a stable `result_schema` field (recommend `{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}`) and use that same key in all contractor tests.

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_contractor.py -v`

- [ ] **Step 3: Implement minimal loop**

Recommended internal structure inside `contractor.py`:

```python
@dataclass
class ContractorDeps:
    llm: Any
    prices: Any
    resolve_agent: Callable[[str], Any]
    invoke_nested_procedure: Callable[..., Awaitable[ProcedureResult]]  # wired in Task 7

async def run_contractor(*, arguments, scoped_metadata, deps: ContractorDeps) -> ProcedureResult:
    ...
```

JSON mode: parse slim envelope `{report?, procedures?, return?}` — **no delegations**. Native mode: tools = `call_procedure` (optional this task — can no-op reject procedures until Task 7) + `contractor_return`.

For this task, if the model requests `procedures`, append a transcript note “nested procedures not yet handled” **or** implement pass-through early — prefer implementing nested in Task 7 and here treating any `procedures` as “continue with error message” only if that keeps tests green; simplest path: Task 6 forbids procedures in tests; if procedures present, return error string into transcript and continue **one** more turn max… Better: Task 6 tests never emit procedures; handler if sees procedures → treat as continue with “procedure execution pending Task 7” is bad. **Implement nested execution stub that rejects all nested ids with a fixed error**, then Task 7 opens allowlist. That matches “forbidden → error in transcript”.

Meter each LLM turn with existing pricing helpers; accumulate `total_charged`; `finally` set `research_credits_charged=total_charged`.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_contractor.py -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/procedures/bundled/contractor.py \
  lunagentic_research_swarm/procedures/bundled/provider.py \
  lunagentic_research_swarm/services.py lunagentic_research_swarm/procedures/executor.py \
  tests/procedures/test_contractor.py
git commit -m "$(cat <<'EOF'
feat: implement contractor outsider LLM loop and return paths

EOF
)"
```

---

### Task 7: Contractor budget force-return, soft timeout, nested allowlist

**Files:**
- Modify: `lunagentic_research_swarm/procedures/bundled/contractor.py`
- Test: `tests/procedures/test_contractor.py`
- Test: `tests/integration/test_spec_e2e_contractor.py` (new)

**Interfaces:**
- Consumes: internal balance machine from spec
- Produces: soft force-return reasons `insufficient_funds` | `timeout`; nested compact allowed; nested contractor/checkpoint/terminate rejected

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_contractor_budget_zero_runs_one_turn_then_force_returns_if_spend():
    # credit_budget=0; first turn costs > 0 → insufficient_funds note; charged > 0

@pytest.mark.asyncio
async def test_contractor_soft_time_budget_force_returns():
    # time_budget_seconds very small; assert termination_reason == "timeout"

@pytest.mark.asyncio
async def test_contractor_rejects_nested_contractor_and_controls():
    # Model asks builtin.contractor / core.terminate / core.checkpoint
    # Assert error text in next turn context; loop can still return

@pytest.mark.asyncio
async def test_contractor_nested_compact_adds_to_bill(monkeypatch):
    # Nested core.compact returns research_credits_charged=2
    # Assert total bill includes 2 and not double-applied to caller beyond contractor total
```

Integration: two parallel contractors in one agent turn → caller balance decreases by sum of both bills.

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_contractor.py tests/integration/test_spec_e2e_contractor.py -v`

- [ ] **Step 3: Implement budget machine + nested invoke**

Follow spec order exactly:

1. `internal_balance = credit_budget`
2. Run model turn
3. Meter; `internal_balance -= usage`; `total_charged += usage`
4. If `internal_balance < 0` → force-return immediately (do not run that turn’s procedures)
5. Else explicit return → return
6. Else no tools → last text return
7. Else run nested procedures (allow compact + ordinary; reject forbidden); apply nested charges to `total_charged` / `internal_balance`; if negative after nested → force-return
8. Soft timeout check around the loop
9. `finally` set charge

Force-return payload: last output + attempted tool call + Chinese termination note.

Nested invoke must call the same procedure stack **without** going through the parent branch batch debit (handler-local). Nested results’ charges only fold into contractor total.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_contractor.py tests/integration/test_spec_e2e_contractor.py tests/runtime/test_procedure_credits.py -q`

- [ ] **Step 5: Commit**

```bash
git add lunagentic_research_swarm/procedures/bundled/contractor.py \
  tests/procedures/test_contractor.py tests/integration/test_spec_e2e_contractor.py
git commit -m "$(cat <<'EOF'
feat: enforce contractor budget, soft timeout, and nested allowlist

EOF
)"
```

---

### Task 8: Disable toggle + full regression sweep

**Files:**
- Test: `tests/procedures/test_contractor.py` (enabled=false)
- Possibly `tests/integration/test_extension_removal.py` patterns
- Docs: optionally add a short pointer in README under procedures (only if README already lists builtins — do not invent a large doc section)

**Interfaces:**
- Consumes: existing registry override `procedures."builtin.contractor".enabled = false`
- Produces: catalog omission / unavailable call behavior unchanged from other builtins

- [ ] **Step 1: Write failing test for disable**

```python
def test_contractor_disabled_by_override_removed_from_snapshot():
    # Build registry with contractor definition enabled, snapshot with
    # ProcedureOverride(enabled=False) → snapshot.get("builtin.contractor") is None
```

- [ ] **Step 2: Run to verify RED/GREEN**

If registry already honors overrides, this may be GREEN immediately — still add the test as regression lock.

- [ ] **Step 3: Full sweep**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest -q`

Fix any fallout from `ProcedureResult` field-set changes, protocol envelope changes, or exact field assertions.

- [ ] **Step 4: Commit**

```bash
git add tests/procedures/test_contractor.py
git commit -m "$(cat <<'EOF'
test: lock contractor disable override and full suite green

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| `timeout_seconds = 0` disables hard wait | Task 1 |
| Outer `credits` budget hint | Task 2–3 |
| `research_credits_charged` bills caller once | Task 2–3 |
| No upfront pocket | Task 3 |
| Parallel bills commutative | Task 3 / 7 integration |
| Agent-requested compact bills; auto does not | Task 4 |
| `external_cost*` unchanged telemetry | (no change; regression in existing tests) |
| `builtin.contractor` definition + config toggles | Task 5, 8 |
| Fresh context outsider pack | Task 6 |
| Protocol from caller; model from selected agent | Task 6 |
| Return tool / last text / force-return | Task 6–7 |
| Budget 0 still runs ≥1 turn | Task 7 |
| Soft `time_budget_seconds` | Task 7 |
| Nested allowlist + no sub-contractor | Task 7 |
| No `max_agent_calls_per_task` counting | Task 6–7 (do not increment); covered by integration assert if harness exposes counter |
| Per-builtin enable examples in config.default.toml | Task 5 |

## Plan self-review notes

- Nested double-billing: Task 7 explicitly uses handler-local nested invoke; Task 3 only sums top-level batch results.
- Hard vs soft timeout: Task 1 (hard) + Task 7 (soft); both may be set.
- `ProcedureResult` field-set tests will break until Task 2 updates them — expected.
- Ledger sign convention must be verified against existing agent entries in Task 3 before choosing amount sign.
