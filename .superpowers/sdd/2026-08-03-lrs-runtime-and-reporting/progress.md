# SDD ledger — plan: docs/superpowers/plans/2026-08-03-lrs-runtime-and-reporting.md

Baseline: foundation merged at `4218ac5`; full suite `240 passed` with pytest-asyncio.

Task 1: fix round 1/5 (1 addressed, 0 open — actual pricing provenance; commits fa8b6d4..3faba31)
Task 1: complete (commits 4218ac5..3faba31, review clean; full suite 259 passed)
Task 2: fix round 1/5 (5 addressed, 0 open — stopped fail-closed, terminal round barrier, PAUSING gate, Outbox whitelist, explicit leaves; commits ea8467f..4d6f841)
Task 2: complete (commits 3faba31..4d6f841, review clean; full suite 279 passed)
Task 3: fix round 1/5 (5 addressed, 0 open — dynamic RR, barrier gating, concurrent close, aggregate stats, error privacy; commits 28cf9dc..97171f8)
Task 3: fix round 2/5 (1 addressed, 0 open — reentrant close deadlock; commits 97171f8..023dc6a)
Task 3: complete (commits 4d6f841..023dc6a, review clean; full suite 294 passed)
Task 4: fix round 1/5 (4 addressed, 0 open — privacy, provenance, retry semantics, immutable core controls; commits b8f8897..058e1c8)
Task 4: fix round 2/5 (1 addressed, 0 open — sensitive procedure/control payload keys; commits 058e1c8..3effce5)
Task 4: complete (commits 023dc6a..3effce5, review clean; full suite 310 passed)
Task 5: fix round 1/5 (4 addressed, 0 open — lifecycle privacy, correction context, failure reconciliation, materialization enforcement; commits d79631f..71716b4)
Task 5: fix round 2/5 (1 addressed, 0 open — missing-agent slot ordering and authoritative task-call counter; commits 71716b4..c03bf0a)
Task 5: complete (commits 3effce5..c03bf0a, review clean; full suite 335 passed)
Task 6: fix round 1/6 (2 addressed, 0 open — controller ownership and duplicate implementation; commits 6428031..a9cc7d1)
Task 6: fix round 2/6 (4 addressed, 0 open — serialized lifecycle, formalization/restart routing, scheduler compatibility, durable continuation; commits a9cc7d1..8dc48a6)
Task 6: fix round 3/6 (3 addressed, 0 open — FairScheduler pause queue accounting, authoritative branch balances, reducer-error isolation; commits 8dc48a6..3ce09a4)
Task 6: fix round 4/6 (1 addressed, 0 open — paused-vs-runnable queue telemetry; commits 3ce09a4..5858b40)
Task 6: fix round 5/6 (1 addressed, 0 open — blocked barrier priority accounting; commits 5858b40..febd453)
Task 6: fix round 6/6 (1 addressed, 0 open — cross-task priority handoff accounting; commits febd453..2a6ff4e)
Task 6: complete (commits c03bf0a..2a6ff4e, review clean; full suite 358 passed)
