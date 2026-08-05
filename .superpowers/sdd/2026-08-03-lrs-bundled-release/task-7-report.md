# Task 7 Report: 实现 feedback、lesson 与 600 秒提醒

## Status

**DONE**

## Summary

Added append-only research feedback, deterministic lessons, and one-shot 600s Maisaka reminders:

- `FeedbackService.submit/schedule/cancel_due_to_continue/process_due` in `feedback.py`
- Deterministic lesson renderer (no 5th summarizer); payload stores `lesson` + `source_feedback_id`; VectorIndex `enqueue(feedback_lesson)` via Task 5 API
- Eighth Planner tool `submit_research_feedback` (exactly eight Tool names)
- Terminal `COMPLETED` / `COMPLETED_WITH_ERRORS` / `STOPPED` insert unique(round_id) reminder in the same controller transaction; `EXPIRED` / `INTERRUPTED` / `FAILED` do not
- Due reminder → outbox `trigger` once (`status=triggered`); feedback / continue / new round cancel pending
- Feedback never mutates prompts/selectors/routing — only retrieval/ranking/stats/lessons inputs

## TDD Evidence

1. **RED** — Wrote `tests/test_feedback.py` + `tests/integration/test_feedback_reminders.py` first. Collection failed with `ModuleNotFoundError: lunagentic_research_swarm.feedback`.
2. **GREEN** — Implemented `feedback.py`; wired tools/plugin/controller/services/manager/sqlite cancel command.
3. **VERIFY** — focused suite **19 passed**; full suite **568 passed**.

## Files Changed

| Path | Action |
|---|---|
| `lunagentic_research_swarm/feedback.py` | Created |
| `lunagentic_research_swarm/tools.py` | `FEEDBACK_SCHEMA` |
| `lunagentic_research_swarm/runtime/controller.py` | Same-txn reminder schedule/cancel |
| `lunagentic_research_swarm/runtime/manager.py` | Pass `feedback_service` into controllers |
| `lunagentic_research_swarm/services.py` | Start/close FeedbackService |
| `lunagentic_research_swarm/storage/sqlite.py` | `cancel_pending_feedback_reminders` |
| `plugin.py` | `submit_research_feedback` Tool |
| `tests/test_feedback.py` | Created |
| `tests/integration/test_feedback_reminders.py` | Created |
| `tests/test_planner_tools.py` | Expect eight tools |

## Commits

- `9aa4f77` — `feat: add research feedback and reminders`

## Concerns

1. Reminder due processing cancels (instead of triggering) when the round already has feedback or the task advanced to a newer round — matches brief “无该 round feedback/新 round”.
2. `manager.py` / `sqlite.py` were needed beyond the brief `git add` list so controller can cancel pending reminders and receive the service.
3. Outbox delivery kind is `trigger` (existing MaisakaOutbox path); intent text requires checking the task/report and calling `submit_research_feedback` with the task ID.

## Verification Command

```bash
cd /mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm/.worktrees/lrs-runtime-reporting
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/test_feedback.py tests/integration/test_feedback_reminders.py tests/test_planner_tools.py -q
# 19 passed

timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 568 passed in 1.87s
```

Result: **568 passed, 0 failed**.

---

## Review-1 Fixes (post `9aa4f77`)

**Status:** DONE — Important + cheap Medium from `task-7-review-1.md`.

### Fixes

1. **Controller wiring coverage** — `tests/integration/test_feedback_reminders.py` now builds real
   `TaskController(feedback=service)` and drives `FinalReportCompleted` / `FinalReportFailed` /
   `StopRequested` / `PauseExpired` / `ContinueRequested`. Reminder schedule/cancel goes through
   `_feedback_commands`; harness no longer re-implements `if status in REMINDER_TERMINAL_STATUSES`.
   Asserts `insert_feedback_reminder` shares a `transact` batch with `update_round_status`.
2. **Post-commit enqueue** — `submit()` never raises after durable insert. Lesson indexing reported
   via `FeedbackResult.lesson_indexing` (`indexed`/`pending`/`failed`/`degraded`/`skipped`) +
   `lesson_index_error`; plugin success payload includes these fields. `VectorOpResult` is not
   discarded.
3. **Reminder INSERT** — `ON CONFLICT(round_id) DO NOTHING` so a derived reminder cannot abort the
   terminal transaction.
4. **process_due worker** — `_run` logs `process_due` exceptions with `_LOG.exception` instead of
   bare `pass`.

### Verification

```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/test_feedback.py tests/integration/test_feedback_reminders.py -q
# 13 passed (focused; planner tools separate)

timeout 180s env PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest -q
# 569 passed in 1.84s
```
