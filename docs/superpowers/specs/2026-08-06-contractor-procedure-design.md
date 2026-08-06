# Contractor Procedure & Procedure Research Credits Design

## Goal

Add a bundled `builtin.contractor` procedure so a calling agent can hand a by-the-way question to a catalog agent as an **outsider tool**: fresh context, no subagent fan-out, optional research-credit budget, explicit or last-message return.

Generalize procedure calls so optional outer `credits` are passed as a **budget hint**, and handlers may bill the caller once via `research_credits_charged`. Keep the public procedure contract one-shot `invoke → ProcedureResult`. Do not build a third-party agentic procedure runtime; improvements to contractor stay in this plugin via issues/PRs.

## Spec deltas vs 2026-08-03

| Prior invariant | New rule |
|---|---|
| Procedure 不扣研究 credits；外部费用仅遥测 | Procedures **may** debit research credits via `research_credits_charged`. `external_cost*` remains separate telemetry and does **not** touch branch balances. |
| `timeout_seconds` on definitions must be `> 0` | Allow `0` = hard executor timeout **disabled**. |
| (none) | Optional outer procedure request field `credits` (default `0`) is a budget hint, not an upfront transfer. |

Auto-compact stays outside research-credit billing (token / cost-equivalent telemetry only). Agent-requested `core.compact` participates in the bill path.

The only balance check that blocks work remains the pre-subagent-launch check. Procedures (including contractors) may run when the caller balance is already negative.

## Architecture

```text
Agent turn
  └─ procedure batch (may be concurrent)
       ├─ for each request:
       │    pass credit_budget (= credits or 0) via scoped_metadata
       │    invoke once (blocking from executor POV)
       │    caller_balance -= research_credits_charged  # missing → 0
       └─ barrier: all results + debits applied
  └─ compact / terminate / balance check / delegations (unchanged order)
```

**Approach:** runtime credit envelope + self-contained builtin contractor (no upfront allocate/settle pocket).

- Parallel procedure calls each receive the budget hint independently and report their own charges; final caller balance is the sum of bills (commutative under allow-negative).
- `builtin.contractor` registers like other bundled procedures (`builtin.invoke_procedure`). Its multi-turn LLM loop lives **inside** the handler and may import LRS internals. Externally it is still one blocking invoke.
- Third-party procedures keep the same one-shot API; they may ignore `credit_budget` and leave `research_credits_charged` at `0`.

## Procedure request surface

Every procedure request (ordinary builtins, third-party, agent-requested `core.compact`, `builtin.contractor`):

```json
{
  "procedure_id": "builtin.contractor",
  "call_id": "optional",
  "credits": 5.0,
  "arguments": {}
}
```

- `credits`: non-negative finite; omit → `0`. Invalid → procedure error result; no debit.
- Runtime does **not** debit `credits` up front.
- Runtime passes `credit_budget` into the handler (via existing `scoped_metadata`).
- Runtime does **not** clamp the reported bill to the budget (so compact with budget `0` can still charge real summarizer cost). Handlers that care about overspend (contractor) enforce policy themselves.

## Billing contract

`ProcedureResult` gains a first-class non-negative finite field `research_credits_charged` (default `0`).

After invoke returns (success or failure):

```text
caller_balance -= research_credits_charged
```

- Missing/invalid charge → treat as `0`.
- One research-credit ledger entry per procedure call: procedure_id, call_id, branch_id, charged, budget_hint.
- No allocate/refund pocket pair.

**Relationship to `external_cost*`**

| Mechanism | Touches research branch balance? | Role |
|---|---|---|
| `external_cost_kind` / `metadata.external_cost` | No | Outside provider/API fee telemetry (already implemented) |
| `research_credits_charged` | Yes | Bill caller after invoke |
| `credits` / `credit_budget` | No (hint only) | Policy/budget for handlers |

Handlers should report partial charge in `finally` when possible. If the invoke errors with no charge field, runtime bills `0`.

### Agent-requested `core.compact`

- Uses the same outer `credits` + bill path.
- Never fails for insufficient funds.
- Charge = metered summarizer research-credit equivalent; debited from the requesting branch after compact returns.
- Budget `0` still runs; cost is retroactive on the caller.

### Auto-compact

Unchanged: no branch research-credit debit.

## Hard vs soft timeouts

### Hard — `ProcedureDefinition.timeout_seconds`

- Allow `0` = disabled (executor skips `asyncio.wait_for`).
- `ProcedureOverride.timeout_seconds` likewise allows `0` (disable hard timeout for that id).
- Still overridable via `procedures."<id>".timeout_seconds`.
- When `> 0`, executor hard-stops the whole invoke → existing `procedure_timeout` error path.
- `builtin.contractor` ships with definition default `0`.

### Soft — contractor argument `time_budget_seconds`

- Per-call only; default `0` = disabled.
- When `> 0`, contractor handler soft force-returns (same UX as insufficient funds).
- Caller may set hard definition/override timeout, soft `time_budget_seconds`, or both. Soft can return a partial answer first; hard remains the outer backstop if the handler never yields.

## `builtin.contractor`

### Arguments

| Field | Required | Default | Meaning |
|---|---|---|---|
| `agent_id` | yes | — | Live catalog agent (same list as subagent routing) |
| `question` | yes | — | Caller-supplied brief; primary user message |
| `temperature` | no | selected agent model default (else plugin default) | Override for this invocation |
| `personality` | no | `null` | `null`/omit → selected agent `character_prompt`; string → overwrite |
| `time_budget_seconds` | no | `0` | Soft wall-clock force-return; `0` = disabled |

Outer envelope still carries `credits` / `call_id`.

### Model and protocol

- LLM route and pricing: **selected** `agent_id` model config.
- Tool-call style (JSON envelope vs native tools): **calling** agent protocol.
- Does **not** count toward `max_agent_calls_per_task` (credits + timeouts only). Revisit later if abuse appears.

### Fresh context (outsider pack)

- **System:** swarm-common contractor rules (no subagents, no sub-contractors) + personality + short catalog of allowed procedures for that agent (excluding contractor recursion) + how to return.
- **User:** `question` only.
- No formalized task text, no parent transcript, no parent runtime header.

### Turn protocol

**Native tools:** procedure invocation tool(s) consistent with existing native patterns, plus contractor-only `contractor_return(result: string)`.

**JSON envelope (slim):**

```json
{
  "report": "optional prose",
  "procedures": [],
  "return": "optional explicit return string"
}
```

No `delegations` field.

**Nested procedure allowlist**

- Allowed: ordinary research procedures + `core.compact`.
- Forbidden: `core.checkpoint`, `core.terminate`, and `builtin.contractor` (exact id for v1; no sub-contractors). Forbidden calls produce an error string appended to the contractor transcript; the loop continues.

Nested procedure requests from inside the contractor may include outer-style `credits` (default `0`). Their `research_credits_charged` adds to the **contractor’s internal spend / total bill**. Those nested invokes are executed **inside** the contractor handler, not as sibling items of the calling agent’s procedure batch, so the outer batch debit path applies **only** the contractor’s top-level `research_credits_charged` (no double billing).

### Internal budget machine

1. `internal_balance = credit_budget` (caller’s outer `credits`).
2. Run turn 1 immediately (even if budget is `0`).
3. After each model turn: meter that turn’s research-credit usage → `internal_balance -= usage`; accumulate `total_charged`.
4. If `internal_balance < 0` → soft force-return (`insufficient_funds`) **immediately**, even if the turn also requested `return` or procedures (do not execute that turn’s nested procedures).
5. Else if explicit `contractor_return` / JSON `return` → normal return with that payload (ignore sibling procedure requests on the same turn).
6. Else if no tool/procedure call → treat last contractor output as return.
7. Else if nested procedures requested → execute, append results or rejection errors; for each nested charge `c`: `total_charged += c`, `internal_balance -= c`. If `internal_balance < 0` after nested settlement → soft force-return (`insufficient_funds`) without another model turn. Otherwise continue the loop.
8. Soft `time_budget_seconds > 0` expired → soft force-return (`timeout`).
9. `finally`: set `research_credits_charged = total_charged`.

### Return payloads

**Force-return** (`insufficient_funds` or soft `timeout`): last contractor output + attempted tool call (if any) + clear termination note. Procedure `success=true` with usable result text; termination reason in structured metadata.

**Normal return:** explicit return tool/field, or last contractor message if they stopped without a tool call. Multi-turn: only the contractor’s last output (plus return tool arg when present).

### Metadata (minimum)

- `termination_reason`: `returned` | `insufficient_funds` | `timeout`
- selected `agent_id`
- turn count
- soft `time_budget_seconds`
- budget hint and charged amount

## Config and disable toggles

Reuse existing `ProcedureOverride` (`enabled`, `timeout_seconds`). No second switch system.

Ship explicit commented defaults in `config.default.toml` for **every** bundled procedure id (memory, analysis, provenance, web_search, past_cases, contractor), for example:

```toml
[procedures."builtin.contractor"]
enabled = true
timeout_seconds = 0

# [procedures."builtin.web_search"]
# enabled = true
# timeout_seconds = 30.0
```

Disabled procedures leave the frozen round catalog (current registry behavior); calls follow the existing unavailable/invalid path.

## Error handling

- Invalid outer `credits` → procedure error; no debit.
- Unknown / disabled / unavailable procedure → existing structured error; charge `0` unless a partial charge was already reported.
- Contractor hard executor timeout (`timeout_seconds > 0`) → `procedure_timeout`; handler should still try/finally bill partial spend when the runtime can observe it; otherwise bill `0`.
- Protocol repair for contractor turns: follow the calling agent’s protocol repair policy where applicable; do not invent a second global retry budget beyond credits/timeouts.

## Testing

1. Envelope `credits` omitted → budget `0`; charge `0` → caller balance unchanged.
2. Handler charges `N` → caller `-= N` (including from already-negative).
3. Agent-requested compact with budget `0` succeeds; summarizer charge debited. Auto-compact does not debit research credits.
4. Contractor with budget `0` runs at least one turn; after turn-1 overspend → soft force-return + note + bill.
5. Soft `time_budget_seconds` vs hard definition timeout — each alone and both together.
6. Nested sub-contractor / checkpoint / terminate rejected into transcript; nested compact allowed and counted in contractor spend.
7. Two parallel contractors → final caller balance equals sum of bills.
8. `procedures."builtin.contractor".enabled = false` → catalog-hidden / not callable.
9. JSON slim envelope and native `contractor_return`; last-text return when no tool call.
10. Fresh context: no parent transcript or formalized task in contractor messages.
11. Definition `timeout_seconds = 0` disables executor `wait_for` for that procedure.

## Non-goals (this design)

- Third-party agentic procedure runtime or Host API for nested LLM loops.
- Sub-contractors (contractor calling contractor).
- Counting contractor turns toward `max_agent_calls_per_task`.
- Upfront credit pockets / allocate-then-refund for procedures.
- Changing auto-compact to consume research credits.
- Letting contractors launch subagents or use checkpoint/terminate branch controls.
