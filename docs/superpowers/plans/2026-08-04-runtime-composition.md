# LRS Production Runtime Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the existing runtime components into a production manager/effect runner so Planner Tools can start and advance research tasks.

**Architecture:** Add one scheduler worker that dispatches frozen runtime effects and returns immutable events to `ResearchManager`. `LRSServiceContainer` owns construction and shutdown; `ResearchManager` owns per-task frozen snapshots and durable child registration.

**Tech Stack:** Python 3.10+, asyncio, Pydantic, MaiBot SDK proxies, SQLite, pytest/pytest-asyncio.

## Global Constraints

- Do not modify MaiBot Host or SDK.
- Preserve transaction-before-effect and generation rejection semantics.
- Never persist raw objective, planner context, transcript, reasoning, or raw procedure payload by default.
- Do not add upstream request abortion.
- Keep summarizer outside research credits.

---

### Task 1: Define and test the runtime effect runner

**Files:**
- Create: `lunagentic_research_swarm/runtime/effect_runner.py`
- Create: `tests/runtime/test_effect_runner.py`

**Interfaces:**
- Consumes: `TurnWorker.perform_agent_call(effect)`, `TurnWorker.perform_procedure_batch(effect)`, manager `handle_runtime_event`, `handle_runtime_effect`, `handle_branch_summary_effect`, and `materialize_child_effect`.
- Produces: `RuntimeEffectRunner.bind_manager(manager)`, `RuntimeEffectRunner.run(effect, token=None)`.

- [x] **Step 1: Write failing dispatch tests**

Test `PerformFormalization` no-op; Agent and Procedure completion callback; BranchSummary delegation; OpenReportEpoch delegation; and `NotifyToolWaiter(action="materialize_child")` delegation.

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/runtime/test_effect_runner.py -q`
Expected: import failure because `runtime.effect_runner` does not exist.

- [x] **Step 3: Implement minimal typed dispatch**

Implement a runner with explicit `isinstance` branches. Unknown/no-op effects return `None`; exceptions propagate to `FairScheduler` isolation. Every completion is passed once to manager.

- [x] **Step 4: Verify GREEN**

Run the Task 1 focused test and existing turns/procedure tests.

### Task 2: Add manager adapters for frozen effects and child/summary state

**Files:**
- Modify: `lunagentic_research_swarm/runtime/manager.py`
- Modify: `lunagentic_research_swarm/runtime/reducer.py` only if effect payload lacks required frozen data.
- Modify: `tests/runtime/test_effect_runner.py`
- Modify: `tests/runtime/test_controller_start.py`

**Interfaces:**
- Consumes: frozen `NextRoundSnapshot`, `ReportCoordinator`, `StoreCommand`, controller state/effects.
- Produces: `prepare_agent_effect(effect)`, `materialize_child_effect(effect)`, `handle_branch_summary_effect(effect)`, `shutdown()`.

- [x] **Step 1: Write failing tests for frozen root/child preparation and durable child ordering**

Assert root selector/protocol/messages/live IDs/safety limits come from the task snapshot; child branch is committed before its Agent effect is enqueued; unavailable live agent becomes a summary effect; coordinator and manager/controller leaves observe the child.

- [x] **Step 2: Verify RED**

Run focused manager/effect-runner tests and confirm missing adapter failures.

- [x] **Step 3: Implement minimal manager adapters**

Store the snapshot on start/formalization, build stable root/child prompts, keep branch runtimes synchronized with coordinator, and perform child StoreCommand before enqueuing its Agent effect. Terminal/checkpoint summary effects call the coordinator safe-point API and release finalized raw messages.

- [x] **Step 4: Verify GREEN and refactor**

Run effect-runner, controller start/control, turns, epochs, and integration tests.

### Task 3: Compose and close production services

**Files:**
- Modify: `lunagentic_research_swarm/services.py`
- Modify: `plugin.py`
- Modify: `tests/test_lifecycle.py`
- Modify: `tests/test_planner_tools.py`

**Interfaces:**
- Consumes: `LLMGateway`, `SummarizerService`, `ProcedureExecutor`, `TurnWorker`, `RuntimeEffectRunner`, `FairScheduler`, `ResearchManager`.
- Produces: `LRSServiceContainer.manager` and complete start/close ownership.

- [x] **Step 1: Write failing service composition tests**

Using injected builtin provider and fake SDK context, assert `await services.start()` exposes non-null manager; plugin caches it only after start; Planner start does not return `manager_unavailable`; close shuts scheduler before SQLite.

- [x] **Step 2: Verify RED**

Run lifecycle/plugin focused tests and confirm missing manager.

- [x] **Step 3: Implement service construction and reverse shutdown**

Construct runtime after initial catalog refresh/price load, bind runner to manager, start scheduler, then expose manager. On startup failure or close, close scheduler before outbox/discovery/store. In plugin, assign `_manager = services.manager` after successful `await services.start()`.

- [x] **Step 4: Verify full suite**

Run focused runtime/integration/lifecycle/planner tests, narrow Ruff, `git diff --check`, then complete pytest suite.
