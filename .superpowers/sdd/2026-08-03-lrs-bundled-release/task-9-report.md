# Task 9 Report: 实现 `/swarm` 用户命令与 health/status 输出

## Status

**DONE**

## Summary

Wired nine user-facing `/swarm` commands (简体中文) via `SwarmCommandsMixin` and StatisticsService/FeedbackService/VectorIndex/health:

- `/swarm status [task_id]` — plugin+running overview or per-task lifecycle/round/queue/credits/tokens/cache
- `/swarm tasks [status]` — recent tasks
- `/swarm stats [task_id]` — ledger-backed task or plugin aggregate stats
- `/swarm agents` / `procedures` — live provider/enabled/selector/protocol/health (no prompts/secrets)
- `/swarm health` — SQLite/vector/pinning/extension/recommended fetch/queue/outbox/reminder
- `/swarm vectors status` / `rebuild [--force]`
- `/swarm feedback <task_id> <accepted|mixed|rejected> [notes]` — simplified FeedbackService path

Handlers require `stream_id`, reply with `ctx.send.text`, and clip via `commands.max_output_chars` while preserving error summaries.

## TDD Evidence

1. **RED** — Wrote `tests/test_commands.py` first; registration asserted empty COMMAND set; invoke raised `no COMMAND matched`.
2. **GREEN** — Implemented `lunagentic_research_swarm/commands.py` mixin + `plugin.py` inheritance.
3. **VERIFY** — focused **7 passed**; full suite **594 passed**.

## Files Changed

| Path | Action |
|---|---|
| `lunagentic_research_swarm/commands.py` | Created |
| `plugin.py` | Inherit `SwarmCommandsMixin` |
| `tests/test_commands.py` | Created |

## Commits

- `74e3122` — `feat: add swarm status and maintenance commands`

## Concerns

1. Agent/procedure listing walks registry `_providers` (no public list API); acceptable for same-package command layer.
2. Maintenance allowlist checks `person_id`/`user_id` from command kwargs; Host currently passes `user_id` — empty allowlist remains open.
3. Task deadline comes from `report_coordinators[task_id].deadline_at` when present; otherwise omitted.

## Verification Command

```bash
cd /mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm/.worktrees/lrs-runtime-reporting
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/test_commands.py -q
# 7 passed

timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 594 passed in 1.94s
```

Result: **594 passed, 0 failed**.

---

## Review 1 follow-up (Important fixes)

Closed Issues 1–2 from `task-9-review-1.md`:

1. **`/swarm status <task_id>` includes `报告数`** — `swarm_status` now loads report cardinality from live `report_coordinators[task_id].reports` when present, else `store.load_summary_layer(task_id).reports`, and passes `reports=` into `format_task_status`.
2. **`clip_command_output` honors `max_output_chars`** — always `len(out) <= limit`; error footer uses `共 N / 显示 K`; fits as many error lines as the budget allows and never silently drops without the count.

### Tests added

- `test_swarm_status_includes_report_count`
- `test_clip_command_output_honors_limit_with_many_errors`
- `test_clip_command_output_keeps_all_errors_when_budget_allows`

### Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/test_commands.py -q
# 10 passed

timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 597 passed in 1.97s
```
