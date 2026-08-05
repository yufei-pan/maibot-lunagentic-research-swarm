"""Tool-facing task lifecycle manager.

Raw tool input is deliberately held only by the formalization coroutine.  The
SQLite command payloads below contain identifiers, formalized text, and public
summary-layer data only.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from lunagentic_research_swarm.llm.gateway import resolve_generation_selector
from lunagentic_research_swarm.llm.summarizer import FormalizationRequest
from lunagentic_research_swarm.models import (
    BranchLifecycle,
    BranchRuntime,
    FormalizedTask,
    ReportKind,
    TaskStatus,
    new_branch_id,
    new_call_id,
    new_round_id,
    new_task_id,
)
from lunagentic_research_swarm.runtime.context import RuntimeHeader, StablePromptBuilder, release_raw_context
from lunagentic_research_swarm.runtime.controller import TaskController
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator, ReportRecord
from lunagentic_research_swarm.runtime.events import (
    AllInflightSettled,
    BranchFinalized,
    ChildMaterialized,
    ContinueRequested,
    FinalReportCompleted,
    FinalReportFailed,
    FormalizationFailed,
    FormalizationSucceeded,
    GraceExpired,
    PauseExpired,
    PauseRequested,
    ReportCompleted,
    ReportDeadlineReached,
    StopRequested,
)
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    OpenReportEpoch,
    PerformAgentCall,
    PerformBranchSummary,
    PerformFormalization,
    RuntimeState,
)
from lunagentic_research_swarm.storage.sqlite import StoreCommand


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _now() -> float:
    return time.time()


def _iso_timestamp(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=UTC).isoformat().replace("+00:00", "Z")


class ResearchManager:
    """Creates and controls independently durable research tasks."""

    def __init__(
        self,
        *,
        ctx: Any,
        store: Any,
        summarizer: Any,
        scheduler: Any,
        snapshot_provider: Any,
        recent_message_limit: int = 20,
        pause_timeout_seconds: int = 1200,
        grace_period_seconds: int = 60,
        report_coordinators: dict[str, Any] | None = None,
        report_coordinator_factory: Any | None = None,
        agent_live_provider: Any | None = None,
        runtime_limits: dict[str, Any] | None = None,
    ) -> None:
        self.ctx, self.store, self.summarizer, self.scheduler = ctx, store, summarizer, scheduler
        self._snapshot_provider = snapshot_provider
        self._recent_message_limit = recent_message_limit
        self._pause_timeout_seconds = pause_timeout_seconds
        self._grace_period_seconds = grace_period_seconds
        self._controllers: dict[str, TaskController] = {}
        self._round_numbers: dict[str, int] = {}
        self._task_streams: dict[str, str] = {}
        self._task_created_at: dict[str, float] = {}
        self._branches: dict[str, dict[str, dict[str, Any]]] = {}
        self._jobs: dict[str, set[asyncio.Task[Any]]] = {}
        self._pause_jobs: dict[str, asyncio.Task[Any]] = {}
        # Effects and worker adapters register one coordinator per task.  The
        # coordinator owns ephemeral report/branch state; TaskController stays
        # the sole authority for durable RuntimeState transitions.
        self.report_coordinators: dict[str, Any] = report_coordinators if report_coordinators is not None else {}
        self._report_coordinator_factory = report_coordinator_factory or ReportCoordinator
        self._agent_live_provider = agent_live_provider
        self._runtime_limits = dict(runtime_limits or {})
        self._round_snapshots: dict[str, Any] = {}
        self._prompt_builders: dict[str, StablePromptBuilder] = {}
        self._bot_profiles: dict[str, dict[str, Any]] = {}
        self._shutting_down = False

    async def start(
        self,
        *,
        objective: str,
        stream_id: str,
        time_budget_seconds: int,
        effort_level: float = 1.0,
        planner_context: str | None = None,
    ) -> dict[str, Any]:
        if self._shutting_down:
            raise RuntimeError("研究运行时正在关闭")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective 不能为空")
        if isinstance(time_budget_seconds, bool) or not isinstance(time_budget_seconds, int) or time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds 必须为正整数")
        if isinstance(effort_level, bool) or not isinstance(effort_level, (int, float)) or effort_level < 0:
            raise ValueError("effort_level 必须为非负数")
        snapshot = await self._snapshot_provider()
        root = snapshot.agent_catalog.get(snapshot.root_agent)
        if root is None or not getattr(root.definition, "enabled", False) or not getattr(root.definition, "can_be_root", False):
            raise ValueError("root agent 不可用")
        if not str(snapshot.summarizer_selector).strip():
            raise ValueError("summarizer selector 不能为空")
        selector = str(snapshot.root_force_selector or root.definition.model_selector)
        credits = float(snapshot.default_effort_credits) * float(effort_level)
        warning = snapshot.price_catalog.low_budget_warning(selector, credits)
        if warning:
            self.ctx.logger.warning("%s（按 500000 cache-miss 输入与 50000 输出 token 重新估算）", warning)

        task_id, round_id, created_at = new_task_id(), new_round_id(), _now()
        state = RuntimeState(task_id, TaskStatus.FORMALIZING, generation=0, active_round_id=round_id)
        controller = TaskController(state, store=self.store, scheduler=self.scheduler)
        initial = (
            StoreCommand("insert_task", {"task_id": task_id, "stream_id": stream_id, "current_round_number": 1, "created_at": created_at}),
            StoreCommand("insert_round", {
                "round_id": round_id, "task_id": task_id, "round_number": 1, "generation": 0,
                "status": TaskStatus.FORMALIZING.value, "time_budget_seconds": time_budget_seconds,
                "grace_period_seconds": self._grace_period_seconds, "credit_pool": 0.0,
                "catalog_fingerprint": snapshot.agent_catalog.fingerprint, "started_at": created_at,
            }),
            self._lifecycle(task_id, round_id, "TaskCreated", None, TaskStatus.FORMALIZING, created_at),
        )
        await self.store.transact(initial)
        self._controllers[task_id] = controller
        self._round_numbers[task_id] = 1
        self._task_streams[task_id] = stream_id
        self._task_created_at[task_id] = created_at
        self._branches[task_id] = {}
        self._round_snapshots[task_id] = snapshot
        await self.scheduler.enqueue(PerformFormalization(task_id, round_id, 0, payload={"stream_id": stream_id}))
        self._track(task_id, self._formalize(task_id, objective, stream_id, planner_context, credits, snapshot, time_budget_seconds))
        return {"task_id": task_id, "status": TaskStatus.FORMALIZING.value, "initial_credits": credits}

    async def _formalize(self, task_id: str, objective: str, stream_id: str, planner_context: str | None, credits: float, snapshot: Any, time_budget_seconds: int) -> None:
        """Collect public context and immediately drop the raw bundle after use."""
        controller = self._controllers[task_id]
        raw_context = ""
        prepared_branch_id: str | None = None
        try:
            recent = await self.ctx.message.get_recent(stream_id, self._recent_message_limit)
            readable = await self.ctx.message.build_readable(recent)
            values = []
            for key in ("bot.nickname", "personality.personality", "personality.behavior_style", "personality.reply_style"):
                values.append((key, await self.ctx.config.get(key)))
            raw_context = json.dumps({"objective": objective, "planner_context": planner_context or "", "persona": values}, ensure_ascii=False)
            result = await self.summarizer.formalize_task(FormalizationRequest(raw_context=raw_context, chat_messages=readable))
            if not result.success or not result.text.strip():
                raise RuntimeError(getattr(getattr(result, "error", None), "message", "formalization_failed"))
            formalized = FormalizedTask.create(result.text)
            branch_id, now = new_branch_id(), _now()
            root_entry = snapshot.agent_catalog.get(snapshot.root_agent)
            if root_entry is None:
                raise RuntimeError("root agent 在 frozen round snapshot 中不存在")
            profile = {key.rsplit(".", 1)[-1]: value for key, value in values}
            self._bot_profiles[task_id] = dict(profile)
            pricing = getattr(snapshot.price_catalog, "debug_snapshot", None)
            pricing_context = pricing() if callable(pricing) else {"fingerprint": getattr(snapshot.price_catalog, "fingerprint", "")}
            builder = StablePromptBuilder(
                formalized_task=formalized,
                swarm_identity="麦麦深度调查组 (Lunagentic Research Swarm)",
                bot_profile=profile,
                agent_catalog=getattr(snapshot.agent_catalog, "entries", ()),
                procedure_catalog=getattr(snapshot.procedure_catalog, "entries", ()),
                pricing=pricing_context,
            )
            root_context = builder.root_context(coordinator=snapshot.root_agent)
            root_messages = builder.messages_for_call(
                root_context,
                RuntimeHeader(
                    branch_id,
                    1,
                    str(getattr(root_entry.definition, "description", snapshot.root_agent)),
                    time_budget_seconds,
                    credits,
                    1,
                    0,
                ),
            )
            self._prompt_builders[task_id] = builder
            prepared_branch_id = branch_id
            self._branches[task_id][branch_id] = {
                "credits": credits,
                "pending_context": [],
                "agent_id": snapshot.root_agent,
                "messages": root_messages,
                "depth": 0,
            }
            self._register_report_coordinator(
                task_id=task_id,
                round_id=controller.state.active_round_id or "",
                formalized_task=formalized,
                root_branch_id=branch_id,
                credits=credits,
                catalog_fingerprint=snapshot.agent_catalog.fingerprint,
                generation=controller.state.generation,
                time_budget_seconds=time_budget_seconds,
                root_messages=root_messages,
            )
            commands = (
                StoreCommand("insert_vector_job", {"job_id": f"vec_{uuid.uuid4().hex}", "source_kind": "formalized_task", "source_id": task_id, "generation": 0, "status": "PENDING", "created_at": now}),
                StoreCommand("insert_branch", {"branch_id": branch_id, "round_id": controller.state.active_round_id, "agent_id": snapshot.root_agent, "lifecycle": BranchLifecycle.READY.value, "depth": 0, "credit_balance": credits, "generation": 0, "created_at": now}),
            )
            accepted = await controller.apply(
                FormalizationSucceeded(
                    _event_id(), task_id, controller.state.active_round_id or "", controller.state.generation,
                    formalized_text=formalized.text, formalized_sha256=formalized.sha256,
                ),
                extra_commands=commands,
                effects=(
                    PerformAgentCall(
                        task_id, controller.state.active_round_id, controller.state.generation,
                        payload={"root": True, "branch_id": branch_id, "formalized_text": formalized.text, "credit_balance": credits},
                    ),
                ),
                state_changes={"active_leaves": {branch_id: credits}},
            )
            if not accepted:
                self._branches[task_id].pop(branch_id, None)
                self.report_coordinators.pop(task_id, None)
                return
        except Exception as exc:
            if prepared_branch_id is not None:
                self._branches.get(task_id, {}).pop(prepared_branch_id, None)
                self.report_coordinators.pop(task_id, None)
            await controller.apply(
                FormalizationFailed(
                    _event_id(), task_id, controller.state.active_round_id or "", controller.state.generation,
                    error_code="formalization_failed", error_message=str(exc),
                )
            )
        finally:
            # objective, readable messages and personality are never retained on the controller.
            raw_context = ""
            objective = ""
            planner_context = None

    async def pause(self, task_id: str, *, stream_id: str | None = None) -> dict[str, Any]:
        await self._assert_stream_owner(task_id, stream_id)
        controller = self._controller(task_id)
        if controller.state.status not in {TaskStatus.RUNNING, TaskStatus.REPORTING}:
            raise ValueError("task 不在可暂停状态")
        await self._submit(
            controller,
            PauseRequested(
                _event_id(),
                task_id,
                controller.state.active_round_id or "",
                controller.state.generation,
                expires_at=_now() + self._pause_timeout_seconds,
            ),
        )
        self.scheduler.pause_task(task_id)
        if self._task_inflight_count(task_id):
            self._track(task_id, self._settle_pause(task_id))
        else:
            await self._mark_paused(task_id)
        return self._status(controller)

    async def _settle_pause(self, task_id: str) -> None:
        wait = getattr(self.scheduler, "wait_task_idle", None)
        if callable(wait):
            result = wait(task_id)
            if hasattr(result, "__await__"):
                await result
        else:
            # FairScheduler exposes telemetry rather than a task-specific wait
            # API.  A task is settled only after both active and queued local
            # work have drained, including an agent→procedure handoff.
            while self._task_inflight_count(task_id):
                await asyncio.sleep(0)
        await self._mark_paused(task_id)

    async def _mark_paused(self, task_id: str) -> None:
        controller = self._controller(task_id)
        if controller.state.status is not TaskStatus.PAUSING:
            return
        await self._submit(
            controller,
            AllInflightSettled(
                _event_id(), task_id, controller.state.active_round_id or "", controller.state.generation
            ),
        )
        self._pause_jobs[task_id] = asyncio.create_task(self._expiry_wait(task_id))

    async def _expiry_wait(self, task_id: str) -> None:
        await asyncio.sleep(self._pause_timeout_seconds)
        await self.expire_pause(task_id)

    async def expire_pause(self, task_id: str) -> None:
        controller = self._controller(task_id)
        if controller.state.status is not TaskStatus.PAUSED:
            return
        await self._submit(
            controller,
            PauseExpired(_event_id(), task_id, controller.state.active_round_id or "", controller.state.generation),
        )
        self._branches[task_id].clear()

    async def stop(self, task_id: str, *, reason: str = "", stream_id: str | None = None) -> dict[str, Any]:
        await self._assert_stream_owner(task_id, stream_id)
        controller = self._controller(task_id)
        state = controller.state
        if state.status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING, TaskStatus.PAUSED}:
            raise ValueError("task 不在可停止状态")
        await self._submit(
            controller,
            StopRequested(_event_id(), task_id, state.active_round_id or "", state.generation, reason=reason),
        )
        self.scheduler.cancel_generation(task_id, state.generation)
        self._branches[task_id].clear()
        return self._status(controller)

    async def add_context(self, task_id: str, context: str, *, stream_id: str | None = None) -> dict[str, Any]:
        await self._assert_stream_owner(task_id, stream_id)
        controller = self._controller(task_id)
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context 不能为空")
        await self.store.transact((self._lifecycle(task_id, controller.state.active_round_id, "ContextSupplied", controller.state.status, controller.state.status, _now(), {"context": context}),))
        for branch in self._branches[task_id].values():
            branch["pending_context"].append(context)
        return self._status(controller)

    async def continue_task(
        self,
        task_id: str,
        *,
        credit_adjustment: float = 0.0,
        time_budget_seconds: int | None = None,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        await self._assert_stream_owner(task_id, stream_id)
        controller = self._controller(task_id)
        if time_budget_seconds is not None and (isinstance(time_budget_seconds, bool) or time_budget_seconds <= 0):
            raise ValueError("time_budget_seconds 必须为正整数")
        state, branches = controller.state, self._branches[task_id]
        effective_time = time_budget_seconds or self._stored_time_budget(task_id) or 120
        if state.status is TaskStatus.PAUSED and branches:
            cancelled = self._pause_jobs.pop(task_id, None)
            if cancelled:
                cancelled.cancel()
            pending_context = {
                branch_id: list(branch["pending_context"])
                for branch_id, branch in branches.items()
            }
            previous_branches = {branch_id: dict(branch) for branch_id, branch in branches.items()}
            await self._submit(
                controller,
                ContinueRequested(
                    _event_id(),
                    task_id,
                    state.active_round_id or "",
                    state.generation,
                    adjustment=float(credit_adjustment),
                    active_leaves={key: item["credits"] for key, item in branches.items()},
                    time_budget_seconds=effective_time,
                    grace_period_seconds=self._grace_period_seconds,
                ),
            )
            branches.clear()
            branches.update(
                {
                    branch_id: {
                        **previous_branches.get(branch_id, {}),
                        "credits": credits,
                        "pending_context": pending_context.get(branch_id, []),
                    }
                    for branch_id, credits in controller.state.active_leaves.items()
                }
            )
            self.scheduler.resume_task(task_id)
            return {**self._status(controller), "effective_time_budget_seconds": effective_time}
        if not branches and state.status in {TaskStatus.STOPPED, TaskStatus.EXPIRED, TaskStatus.COMPLETED, TaskStatus.FAILED}:
            if state.formalized_task is None:
                return {"success": False, "error": {"code": "task_not_formalized", "message": "任务尚未形式化"}}
            restarted = await self._restart_round(task_id, float(credit_adjustment), effective_time)
            if not restarted:
                return {"success": False, "error": {"code": "task_finished_insufficient_funds", "message": "任务没有可用于新 round 的 credits"}}
            return {**self._status(controller), "effective_time_budget_seconds": effective_time}
        raise ValueError("task 不在可继续状态")

    async def _restart_round(self, task_id: str, adjustment: float, time_budget_seconds: int) -> bool:
        controller, state = self._controller(task_id), self._controller(task_id).state
        snapshot = await self._snapshot_provider()
        round_id, branch_id, now = new_round_id(), new_branch_id(), _now()
        number, generation = self._round_numbers[task_id] + 1, state.generation + 1
        layer = await self.store.load_summary_layer(task_id)
        summary_context = {"summaries": list(layer.summaries), "reports": list(layer.reports), "feedback": list(layer.feedback), "supplied_context": list(layer.supplied_context)} if layer else {}
        root_entry = snapshot.agent_catalog.get(snapshot.root_agent)
        if root_entry is None or state.formalized_task is None:
            return False
        credits = state.credit_pool + adjustment
        pricing = getattr(snapshot.price_catalog, "debug_snapshot", None)
        pricing_context = pricing() if callable(pricing) else {
            "fingerprint": getattr(snapshot.price_catalog, "fingerprint", "")
        }
        builder = StablePromptBuilder(
            formalized_task=state.formalized_task,
            swarm_identity="麦麦深度调查组 (Lunagentic Research Swarm)",
            bot_profile=self._bot_profiles.get(task_id, {}),
            agent_catalog=getattr(snapshot.agent_catalog, "entries", ()),
            procedure_catalog=getattr(snapshot.procedure_catalog, "entries", ()),
            pricing=pricing_context,
        )
        summary_layers = [
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for key in ("summaries", "reports", "feedback", "supplied_context")
            for item in summary_context.get(key, ())
        ]
        root_context = builder.restart_context(summary_layers=summary_layers, coordinator=snapshot.root_agent)
        root_messages = builder.messages_for_call(
            root_context,
            RuntimeHeader(
                branch_id,
                1,
                str(getattr(root_entry.definition, "description", snapshot.root_agent)),
                time_budget_seconds,
                credits,
                1,
                0,
            ),
        )
        previous_branches = self._branches.get(task_id, {})
        previous_snapshot = self._round_snapshots.get(task_id)
        previous_builder = self._prompt_builders.get(task_id)
        previous_coordinator = self.report_coordinators.pop(task_id, None)
        self._round_snapshots[task_id] = snapshot
        self._prompt_builders[task_id] = builder
        self._branches[task_id] = {
            branch_id: {
                "credits": credits,
                "pending_context": list(summary_context.get("supplied_context", [])),
                "agent_id": snapshot.root_agent,
                "messages": root_messages,
                "depth": 0,
            }
        }
        self._register_report_coordinator(
            task_id=task_id,
            round_id=round_id,
            formalized_task=state.formalized_task,
            root_branch_id=branch_id,
            credits=credits,
            catalog_fingerprint=snapshot.agent_catalog.fingerprint,
            generation=generation,
            time_budget_seconds=time_budget_seconds,
            root_messages=root_messages,
        )
        event = ContinueRequested(
            _event_id(), task_id, state.active_round_id or "", state.generation,
            adjustment=adjustment, active_leaves={}, next_round_id=round_id,
            next_generation=generation, round_number=number, time_budget_seconds=time_budget_seconds,
            grace_period_seconds=self._grace_period_seconds, catalog_fingerprint=snapshot.agent_catalog.fingerprint,
        )
        try:
            accepted = await controller.apply(
                event,
                extra_commands=(
                    StoreCommand("insert_branch", {"branch_id": branch_id, "round_id": round_id, "agent_id": snapshot.root_agent, "lifecycle": BranchLifecycle.READY.value, "depth": 0, "credit_balance": credits, "generation": generation, "created_at": now}),
                ),
                effects=(
                    PerformAgentCall(task_id, round_id, generation, payload={"root": True, "branch_id": branch_id, "formalized_text": state.formalized_task.text, "summary_layer": summary_context}),
                ),
                state_changes={"active_leaves": {branch_id: credits}, "raw_context_released": False},
            )
        except BaseException:
            self._restore_round_ephemeral(
                task_id,
                previous_branches,
                previous_snapshot,
                previous_builder,
                previous_coordinator,
            )
            raise
        if not accepted or controller.state.status is not TaskStatus.RUNNING:
            self._restore_round_ephemeral(
                task_id,
                previous_branches,
                previous_snapshot,
                previous_builder,
                previous_coordinator,
            )
            return False
        self._round_numbers[task_id] = number
        return True

    async def handle_runtime_event(self, event: Any) -> None:
        if isinstance(event, ReportDeadlineReached):
            await self.handle_report_deadline(event)
            return
        if isinstance(event, GraceExpired):
            await self.handle_grace_expired(event)
            return
        controller = self._controllers.get(event.task_id)
        if controller is None or event.generation != controller.state.generation or event.round_id != controller.state.active_round_id:
            return
        if controller.state.status in {TaskStatus.STOPPED, TaskStatus.EXPIRED, TaskStatus.FAILED}:
            return
        # All current-generation worker outputs follow the same durable reducer
        # path.  In particular, AgentCallCompleted reconciles cost and queues the
        # next procedure/materialization phase after its transaction commits.
        await self._submit(controller, event)
        self._sync_branch_credits(event.task_id, controller)

    async def prepare_agent_effect(self, effect: PerformAgentCall) -> PerformAgentCall:
        """Fill a scheduled agent effect from its task's frozen round snapshot."""

        controller = self._controllers.get(effect.task_id)
        if controller is None or effect.round_id != controller.state.active_round_id or effect.generation != controller.state.generation:
            raise LookupError("agent effect 不属于当前 task round/generation")
        snapshot = self._round_snapshots[effect.task_id]
        branch_id = str(effect.payload.get("branch_id", ""))
        branch = self._branches[effect.task_id].get(branch_id)
        if branch is None:
            raise LookupError(f"branch {branch_id} 不存在")
        agent_id = str(branch.get("agent_id") or snapshot.root_agent)
        entry = snapshot.agent_catalog.get(agent_id)
        if entry is None:
            raise LookupError(f"agent {agent_id} 不存在")
        definition = entry.definition
        selector = resolve_generation_selector(definition.model_selector, snapshot.root_force_selector).raw
        live_ids = tuple(
            sorted(
                item.definition.agent_id
                for item in getattr(snapshot.agent_catalog, "entries", ())
                if item is not None and self._agent_is_live(item.definition.agent_id, snapshot)
            )
        )
        payload = dict(effect.payload)
        payload.update(
            {
                "branch_id": branch_id,
                "agent_id": agent_id,
                "call_id": str(payload.get("call_id") or new_call_id()),
                "selector": selector,
                "protocol": str(getattr(definition, "protocol", "json_envelope")),
                "messages": tuple(dict(item) for item in branch.get("messages", ())),
                "credits_after_reservation": float(branch.get("credits", 0.0)),
                "branch_depth": int(branch.get("depth", 0)),
                "live_agent_ids": live_ids,
                "max_delegations_per_turn": int(self._runtime_limits.get("max_delegations_per_turn", 8)),
                "max_branch_depth": int(self._runtime_limits.get("max_branch_depth", 32)),
                "max_agent_calls_per_task": int(self._runtime_limits.get("max_agent_calls_per_task", 256)),
                "agent_calls_started": int(controller.state.agent_calls_started),
            }
        )
        return replace(effect, payload=payload)

    async def prepare_procedure_effect(self, effect: Any) -> Any:
        """Attach the procedure catalog frozen for this task round."""

        controller = self._controllers.get(effect.task_id)
        if (
            controller is None
            or effect.round_id != controller.state.active_round_id
            or effect.generation != controller.state.generation
        ):
            raise LookupError("procedure effect 不属于当前 task round/generation")
        snapshot = self._round_snapshots.get(effect.task_id)
        if snapshot is None:
            raise LookupError("procedure effect 缺少 frozen round snapshot")
        branch_id = str(effect.payload.get("branch_id", ""))
        if branch_id and branch_id not in self._branches.get(effect.task_id, {}):
            raise LookupError(f"branch {branch_id} 不存在")
        payload = dict(effect.payload)
        payload["procedure_catalog"] = snapshot.procedure_catalog
        payload["procedure_catalog_fingerprint"] = str(
            getattr(snapshot.procedure_catalog, "fingerprint", "")
        )
        return replace(effect, payload=payload)

    def _agent_is_live(self, agent_id: str, snapshot: Any) -> bool:
        if callable(self._agent_live_provider):
            return bool(self._agent_live_provider(agent_id))
        return snapshot.agent_catalog.get(agent_id) is not None

    async def materialize_child_effect(self, effect: NotifyToolWaiter) -> None:
        """Commit one child branch before making its first agent call runnable."""

        controller = self._controllers.get(effect.task_id)
        if controller is None or effect.round_id != controller.state.active_round_id or effect.generation != controller.state.generation:
            return
        payload = dict(effect.payload)
        snapshot = self._round_snapshots[effect.task_id]
        agent_id = str(payload.get("agent_id", ""))
        if not self._agent_is_live(agent_id, snapshot):
            failed = dict(payload)
            failed.update({"reason": "agent_unavailable", "edge_finalization": True})
            await self.scheduler.enqueue(
                PerformBranchSummary(effect.task_id, effect.round_id, effect.generation, priority="barrier", payload=failed)
            )
            return
        branch_id = str(payload["branch_id"])
        parent_id = str(payload["parent_branch_id"])
        credits = float(payload.get("credits", 0.0))
        depth = int(payload.get("depth", 0))
        await self._submit(
            controller,
            ChildMaterialized(
                _event_id(), effect.task_id, str(effect.round_id or ""), effect.generation,
                branch_id=branch_id, parent_branch_id=parent_id, agent_id=agent_id,
                credits=credits, depth=depth, retire_parent=bool(payload.get("retire_parent", False)),
                pool_return=float(payload.get("pool_return", 0.0)),
            ),
        )
        messages = [dict(item) for item in payload.get("messages", ())]
        self._branches[effect.task_id][branch_id] = {
            "credits": credits, "pending_context": [], "agent_id": agent_id,
            "messages": messages, "depth": depth,
        }
        if payload.get("retire_parent"):
            self._branches[effect.task_id].pop(parent_id, None)
        coordinator = self.report_coordinators.get(effect.task_id)
        if coordinator is not None:
            if payload.get("retire_parent") and parent_id in coordinator.branches:
                coordinator.branches[parent_id].lifecycle = BranchLifecycle.FINALIZED
                coordinator.branches[parent_id].messages.clear()
            coordinator.branches[branch_id] = BranchRuntime(
                branch_id, coordinator.formalized_task, snapshot.agent_catalog.fingerprint,
                effect.generation, messages, credits, depth, parent_branch_id=parent_id,
            )
        await self.scheduler.enqueue(
            PerformAgentCall(effect.task_id, effect.round_id, effect.generation, payload={"branch_id": branch_id})
        )

    async def handle_branch_summary_effect(self, effect: PerformBranchSummary) -> None:
        """Finalize/checkpoint a branch through the report coordinator."""

        controller = self._controllers.get(effect.task_id)
        coordinator = self.report_coordinators.get(effect.task_id)
        if (
            controller is None
            or coordinator is None
            or effect.round_id != controller.state.active_round_id
            or effect.generation != controller.state.generation
        ):
            return
        payload = dict(effect.payload)
        branch_id = str(payload.get("branch_id", ""))
        if branch_id not in coordinator.branches:
            snapshot = self._round_snapshots[effect.task_id]
            coordinator.branches[branch_id] = BranchRuntime(
                branch_id=branch_id,
                task=coordinator.formalized_task,
                catalog_fingerprint=snapshot.agent_catalog.fingerprint,
                generation=effect.generation,
                messages=[dict(item) for item in payload.get("messages", ())],
                credits=float(payload.get("credits", 0.0)),
                depth=int(payload.get("depth", 0)),
                parent_branch_id=str(payload.get("parent_branch_id") or "") or None,
            )
        reason = str(payload.get("reason", "no_further_work"))
        checkpoint = reason in {"checkpoint", "compact"}
        await coordinator.on_branch_safe_point(
            branch_id,
            checkpoint=checkpoint,
            terminal=not checkpoint,
            delegations=tuple(payload.get("held_delegations", ())),
        )
        if checkpoint:
            return
        branch = coordinator.branches[branch_id]
        release_raw_context(branch)
        await self.handle_runtime_event(
            BranchFinalized(
                _event_id(), effect.task_id, str(effect.round_id or ""), effect.generation,
                branch_id=branch_id, summary_id=branch.latest_checkpoint_id or "", reason=reason,
            )
        )
        self._branches.get(effect.task_id, {}).pop(branch_id, None)

    async def handle_report_deadline(self, event: ReportDeadlineReached) -> None:
        """Durably enter REPORTING, then open the injected epoch coordinator.

        This is the explicit scheduler/effect bridge for ``OpenReportEpoch``;
        it deliberately invokes the coordinator only after TaskController has
        committed the reducer transition.
        """

        controller = self._controllers.get(event.task_id)
        if controller is None or event.generation != controller.state.generation or event.round_id != controller.state.active_round_id:
            return
        if controller.state.status is not TaskStatus.RUNNING:
            return
        await self._submit(controller, event)
        if controller.state.status is not TaskStatus.REPORTING:
            return
        await self.handle_runtime_effect(
            OpenReportEpoch(
                event.task_id,
                event.round_id,
                event.generation,
                priority="barrier",
                payload={"epoch": controller.state.report_epoch},
            )
        )

    async def handle_grace_expired(self, event: GraceExpired) -> None:
        """Durably process grace expiry, then ask the coordinator for clones."""

        controller = self._controllers.get(event.task_id)
        if controller is None or event.generation != controller.state.generation or event.round_id != controller.state.active_round_id:
            return
        if controller.state.status is not TaskStatus.REPORTING:
            return
        await self._submit(controller, event)
        if controller.state.status is not TaskStatus.REPORTING:
            return
        coordinator = self.report_coordinators.get(event.task_id)
        if coordinator is not None and (event.epoch is None or event.epoch == controller.state.report_epoch):
            await coordinator.on_grace_expired(epoch=controller.state.report_epoch)

    async def handle_runtime_effect(self, effect: Any) -> None:
        """Consume report effects after the controller's durable transition.

        Scheduler integrations may hand report effects here; deadline handling
        uses the same path directly so a default production manager never
        depends on tests manually injecting a coordinator.
        """

        if not isinstance(effect, OpenReportEpoch):
            return
        controller = self._controllers.get(effect.task_id)
        if controller is None or controller.state.status is not TaskStatus.REPORTING:
            return
        coordinator = self.report_coordinators.get(effect.task_id)
        if coordinator is None:
            return
        epoch = int(effect.payload.get("epoch", controller.state.report_epoch))
        current = getattr(coordinator, "current_epoch", None)
        if current is not None and getattr(current, "epoch", None) == epoch:
            return
        await coordinator.open_epoch(epoch=epoch)

    async def _on_report_synthesis_complete(self, task_id: str, record: ReportRecord) -> None:
        """Return a committed coordinator report to the sole state controller."""

        controller = self._controllers.get(task_id)
        if controller is None:
            return
        state = controller.state
        if record.kind is ReportKind.FINAL:
            if state.status is TaskStatus.RUNNING:
                # The intermediate completion returned this task to RUNNING.
                # Commit the new final epoch before publishing its completed
                # event, preserving transaction-before-effect for both steps.
                if record.epoch != state.report_epoch + 1:
                    return
                await self._submit(
                    controller,
                    ReportDeadlineReached(
                        _event_id(), task_id, state.active_round_id or "", state.generation, epoch=record.epoch
                    ),
                )
                state = controller.state
            if state.status not in {TaskStatus.REPORTING, TaskStatus.FINALIZING}:
                return
            if record.status == "SUCCEEDED":
                event = FinalReportCompleted(
                    _event_id(), task_id, state.active_round_id or "", state.generation, report_id=record.report_id
                )
            else:
                # Never derive an error from the rendered report text: it may
                # contain formalized task/coverage prompt data.  The
                # coordinator carries only provider/error metadata separately.
                event = FinalReportFailed(
                    _event_id(), task_id, state.active_round_id or "", state.generation,
                    error_code=getattr(record, "error_code", None) or "final_report_failed",
                    error_message=getattr(record, "error_message", None) or "最终报告生成失败。",
                )
        else:
            if state.status is not TaskStatus.REPORTING:
                return
            event = ReportCompleted(
                _event_id(), task_id, state.active_round_id or "", state.generation, report_id=record.report_id
            )
        await self._submit(controller, event)

    def _register_report_coordinator(
        self,
        *,
        task_id: str,
        round_id: str,
        formalized_task: FormalizedTask,
        root_branch_id: str,
        credits: float,
        catalog_fingerprint: str,
        generation: int,
        time_budget_seconds: int,
        root_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if task_id in self.report_coordinators:
            return
        branch = BranchRuntime(
            branch_id=root_branch_id,
            task=formalized_task,
            catalog_fingerprint=catalog_fingerprint,
            generation=generation,
            messages=[dict(item) for item in (root_messages or ())],
            credits=credits,
            depth=0,
        )

        async def on_synthesis_complete(record: ReportRecord) -> None:
            await self._on_report_synthesis_complete(task_id, record)

        self.report_coordinators[task_id] = self._report_coordinator_factory(
            task_id=task_id,
            round_id=round_id,
            formalized_task=formalized_task,
            branches={root_branch_id: branch},
            store=self.store,
            summarizer=self.summarizer,
            clock=_now,
            on_synthesis_complete=on_synthesis_complete,
            time_budget_seconds=time_budget_seconds,
            grace_period_seconds=self._grace_period_seconds,
            credit_pool=0.0,
        )

    def _restore_round_ephemeral(
        self,
        task_id: str,
        branches: dict[str, dict[str, Any]],
        snapshot: Any | None,
        builder: StablePromptBuilder | None,
        coordinator: Any | None,
    ) -> None:
        """Roll back local routing state when a new-round barrier is rejected."""

        self._branches[task_id] = branches
        if snapshot is None:
            self._round_snapshots.pop(task_id, None)
        else:
            self._round_snapshots[task_id] = snapshot
        if builder is None:
            self._prompt_builders.pop(task_id, None)
        else:
            self._prompt_builders[task_id] = builder
        if coordinator is None:
            self.report_coordinators.pop(task_id, None)
        else:
            self.report_coordinators[task_id] = coordinator

    async def handle_branch_safe_point(
        self,
        task_id: str,
        branch_id: str,
        *,
        checkpoint: bool = False,
        terminal: bool = False,
        delegations: tuple[dict[str, Any], ...] | tuple[Any, ...] = (),
    ) -> Any:
        """Bridge a post-commit summary/checkpoint/terminal effect to reports.

        The caller must invoke this after its normal branch event transaction;
        this method never changes ``TaskController.state`` directly.
        """

        coordinator = self.report_coordinators.get(task_id)
        if coordinator is None:
            return None
        return await coordinator.on_branch_safe_point(
            branch_id, checkpoint=checkpoint, terminal=terminal, delegations=delegations
        )

    async def _submit(self, controller: TaskController, event: Any) -> None:
        """Submit a runtime event through the sole transaction-before-effect driver."""

        if not await controller.submit_and_drain(event):
            raise RuntimeError("task controller 已停止，拒绝处理事件")

    def _task_inflight_count(self, task_id: str) -> int:
        """Read optional scheduler-specific count or FairScheduler public stats."""

        direct = getattr(self.scheduler, "task_inflight_count", None)
        if callable(direct):
            return max(0, int(direct(task_id)))
        stats = getattr(self.scheduler, "stats", None)
        if callable(stats):
            task = dict(stats().get("tasks", {}).get(task_id, {}))
            # A paused FairScheduler deliberately retains queued agent and
            # summarizer work for continue().  It must not make that retained
            # work block PAUSING forever.  Its telemetry separately exposes
            # effects still permitted while paused: procedure plus control and
            # other local handoff effects.
            return max(0, int(task.get("active", 0))) + max(0, int(task.get("pause_runnable_queued", 0)))
        return 0

    def _sync_branch_credits(self, task_id: str, controller: TaskController) -> None:
        """Mirror reducer-owned balances into the status-only branch cache."""

        branches = self._branches.get(task_id)
        if branches is None:
            return
        for branch_id, credits in controller.state.active_leaves.items():
            branch = branches.get(branch_id)
            if branch is not None:
                branch["credits"] = float(credits)

    async def wait_idle(self, task_id: str) -> None:
        while True:
            jobs = [job for job in self._jobs.get(task_id, set()) if not job.done()]
            if not jobs:
                return
            await asyncio.gather(*jobs, return_exceptions=True)

    async def shutdown(self) -> None:
        """Cancel manager-owned coroutines before the scheduler/store close."""

        if self._shutting_down:
            return
        self._shutting_down = True
        for controller in self._controllers.values():
            controller.stopped = True
        tasks: set[asyncio.Task[Any]] = set()
        for bucket in self._jobs.values():
            tasks.update(task for task in bucket if not task.done())
        tasks.update(task for task in self._pause_jobs.values() if not task.done())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._jobs.clear()
        self._pause_jobs.clear()
        self._branches.clear()
        self.report_coordinators.clear()
        self._round_snapshots.clear()
        self._prompt_builders.clear()
        self._bot_profiles.clear()

    async def status(self, task_id: str, *, stream_id: str | None = None) -> dict[str, Any]:
        await self._assert_stream_owner(task_id, stream_id)
        return self._status(self._controller(task_id))

    async def list_tasks(self, *, stream_id: str | None = None) -> list[dict[str, Any]]:
        return [
            self._status(controller)
            for task_id, controller in self._controllers.items()
            if stream_id is None or self._task_streams.get(task_id) == stream_id
        ]

    def _status(self, controller: TaskController) -> dict[str, Any]:
        state = controller.state
        branches = self._branches.get(state.task_id, {})
        leaves = []
        for key, item in branches.items():
            leaf = {"branch_id": key, "credits": item["credits"]}
            if item["pending_context"]:
                leaf["pending_context"] = list(item["pending_context"])
            leaves.append(leaf)
        created_at = self._task_created_at.get(state.task_id)
        return {
            "task_id": state.task_id,
            "status": state.status.value,
            "round_id": state.active_round_id,
            "round_number": self._round_numbers.get(state.task_id, 1),
            "generation": state.generation,
            "active_leaves": leaves,
            "raw_context_released": state.raw_context_released,
            "created_at": _iso_timestamp(created_at) if created_at is not None else None,
        }

    def _stored_time_budget(self, task_id: str) -> int | None:
        return None

    def _controller(self, task_id: str) -> TaskController:
        try:
            return self._controllers[task_id]
        except KeyError as exc:
            raise LookupError(f"task {task_id} 不存在") from exc

    async def _assert_stream_owner(self, task_id: str, stream_id: str | None) -> None:
        if stream_id is None:
            return
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise PermissionError("stream_id 不能为空")
        owner = self._task_streams.get(task_id)
        if owner is None:
            stored = await self.store.load_task(task_id)
            owner = getattr(stored, "stream_id", None)
            if owner is not None:
                self._task_streams[task_id] = str(owner)
            else:
                raise LookupError(f"task {task_id} 不存在")
        if owner != stream_id:
            raise PermissionError("任务不属于当前 stream")

    def _track(self, task_id: str, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        bucket = self._jobs.setdefault(task_id, set())
        bucket.add(task)
        task.add_done_callback(bucket.discard)

    @staticmethod
    def _lifecycle(task_id: str, round_id: str | None, event_type: str, old: TaskStatus | None, new: TaskStatus, now: float, metadata: dict[str, Any] | None = None) -> StoreCommand:
        return StoreCommand("insert_lifecycle_event", {"event_id": _event_id(), "task_id": task_id, "round_id": round_id, "event_type": event_type, "from_status": old.value if old else None, "to_status": new.value, "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True), "created_at": now})
