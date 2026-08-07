"""Task 10 deterministic providers and a small real-SQLite runtime harness."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lunagentic_research_swarm.llm.gateway import GenerationError, GenerationRequest, GenerationResult
from lunagentic_research_swarm.llm.pricing import TokenUsage
from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import (
    BranchLifecycle,
    BranchRuntime,
    FormalizedTask,
    TaskStatus,
)
from lunagentic_research_swarm.runtime.context import release_raw_context
from lunagentic_research_swarm.runtime.epochs import ReportCoordinator
from lunagentic_research_swarm.runtime.manager import ResearchManager
from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


class FakeClock:
    """可由测试显式推进的单调时钟。"""

    def __init__(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("seconds 必须为非负数")
        self.value += float(seconds)
        return self.value


@dataclass(frozen=True, slots=True)
class FakeLLMResponse:
    """Fake LLM 的可复现响应；payload 可为普通文本或 JSON envelope。

    ``tool_calls`` 用于 native_tools 协议（``submit_swarm_turn``）；assistant
    正文可为空字符串（spec §9.3）。
    """

    text: str = ""
    payload: dict[str, Any] | None = None
    model: str = "gpt-5.6-luna-max"
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 10, "completion_tokens": 10})
    tool_calls: list[dict[str, Any]] | None = None
    success: bool = True
    error_code: str = "llm_generation_failed"
    error_message: str = "LLM 调用失败"


class FakeLLMGateway:
    """按队列返回响应，并可阻塞当前调用来制造 late-result 竞态。"""

    def __init__(self) -> None:
        self.responses: list[FakeLLMResponse | Exception] = []
        self.calls: list[dict[str, Any]] = []
        self.gate = asyncio.Event()
        self.gate.set()

    def enqueue(self, *responses: FakeLLMResponse | Exception) -> None:
        self.responses.extend(responses)

    def queue_json(self, payload: dict[str, Any], **kwargs: Any) -> None:
        """Enqueue a JSON-envelope style FakeLLMResponse (payload serialized by generate)."""

        self.enqueue(FakeLLMResponse(payload=dict(payload), **kwargs))

    def queue_text(self, text: str, **kwargs: Any) -> None:
        """Enqueue a plain-text FakeLLMResponse (no payload / tool_calls)."""

        self.enqueue(FakeLLMResponse(text=str(text), **kwargs))

    def block(self) -> None:
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()

    async def generate(
        self,
        request: GenerationRequest | None = None,
        *,
        selector: str | None = None,
        messages: Any = None,
        **kwargs: Any,
    ) -> FakeLLMResponse | GenerationResult:
        if request is not None:
            selector = request.selector.raw
            messages = request.messages
            kwargs = {
                "tools": request.tools,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
        call = {"selector": selector or "", "messages": messages, **kwargs}
        self.calls.append(call)
        await self.gate.wait()
        response: FakeLLMResponse | Exception
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = FakeLLMResponse(payload={"report": "fake response", "procedures": [], "delegations": []})
        if isinstance(response, Exception):
            raise response
        if request is not None:
            rendered = response.text
            if response.payload is not None and response.tool_calls is None:
                rendered = json.dumps(response.payload, ensure_ascii=False, sort_keys=True)
            usage = TokenUsage(
                prompt_tokens=int(response.usage.get("prompt_tokens", 0)),
                completion_tokens=int(response.usage.get("completion_tokens", 0)),
                cache_hit_tokens=int(response.usage.get("cache_hit_tokens", 0)),
                cache_miss_tokens=int(response.usage.get("cache_miss_tokens", 0)),
                source="actual",
            )
            ok = bool(response.success)
            error = None if ok else GenerationError(response.error_code, response.error_message)
            return GenerationResult(rendered, response.tool_calls, response.model, usage, ok, error, 0.0)
        return response


class FakeProcedureProvider:
    """记录 procedure 请求及其 scoped metadata，不访问网络或 Host。"""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}

    async def describe_procedures(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.requests.append({"operation": "describe", **kwargs})
        return [{"procedure_id": "fake.search", "version": "1"}]

    async def invoke_procedure(self, *, request_id: str, metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        record = {"operation": "invoke", "request_id": request_id, "metadata": dict(metadata or {}), **kwargs}
        self.requests.append(record)
        return self.responses.get(request_id, {"ok": True})

    async def call(
        self,
        api_name: str,
        *,
        version: str,
        procedure_id: str,
        request_id: str,
        arguments: dict[str, Any],
        scoped_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(
            {
                "operation": "call",
                "api_name": api_name,
                "version": version,
                "procedure_id": procedure_id,
                "request_id": request_id,
                "arguments": dict(arguments),
                "scoped_metadata": dict(scoped_metadata),
            }
        )
        return self.responses.get(
            request_id,
            {
                "success": True,
                "data": {"ok": True},
                "error": None,
                "metadata": {"request_id": request_id},
            },
        )

    async def invoke_many(self, effect: Any) -> Any:
        """兼容 TurnWorker 的 executor 形状，返回一个确定性的完成事件。"""

        from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted

        payload = dict(getattr(effect, "payload", {}) or {})
        return ProcedureBatchCompleted(
            event_id=f"{getattr(effect, 'event_id', 'effect')}:procedures",
            task_id=str(getattr(effect, "task_id", "task")),
            round_id=str(getattr(effect, "round_id", "round") or "round"),
            generation=int(getattr(effect, "generation", 0)),
            branch_id=str(payload.get("branch_id", "branch")),
            call_id=str(payload.get("call_id", "call")),
            result_id=f"{getattr(effect, 'event_id', 'effect')}:procedure-result",
            results=(),
            controls=None,
        )


class FakeMaisaka:
    """把 append 与 proactive trigger 分开记录，并支持分别注入失败。"""

    class _Context:
        def __init__(self, owner: FakeMaisaka) -> None:
            self.owner = owner

        async def append(self, stream_id: str, segments: list[dict[str, Any]], **kwargs: Any) -> None:
            if self.owner.append_error is not None:
                raise self.owner.append_error
            self.owner.append_calls.append({"stream_id": stream_id, "segments": segments, **kwargs})

    class _Proactive:
        def __init__(self, owner: FakeMaisaka) -> None:
            self.owner = owner

        async def trigger(self, stream_id: str, intent: str, **kwargs: Any) -> None:
            if self.owner.trigger_error is not None:
                raise self.owner.trigger_error
            self.owner.trigger_calls.append({"stream_id": stream_id, "intent": intent, **kwargs})

    def __init__(self) -> None:
        self.append_calls: list[dict[str, Any]] = []
        self.trigger_calls: list[dict[str, Any]] = []
        self.append_error: Exception | None = None
        self.trigger_error: Exception | None = None
        self.context = self._Context(self)
        self.proactive = self._Proactive(self)


class FakeSummarizer:
    """ReportCoordinator 使用的无网络 summarizer。"""

    def __init__(self) -> None:
        self.branch_requests: list[Any] = []
        self.task_requests: list[Any] = []
        self.branch_gate = asyncio.Event()
        self.task_gate = asyncio.Event()
        self.branch_gate.set()
        self.task_gate.set()
        self.formalization_gate = asyncio.Event()
        self.formalization_text = "正式任务"
        # Fail hooks (§23.2): unsuccessful SummaryResult without inventing prose.
        # ``branch_failure`` / ``task_failure`` are error-like objects with .code/.message
        # (GenerationError or SimpleNamespace). ``fail_branch_history_markers`` selects
        # which branches fail when BranchFinalizationRequest has no branch_id — match
        # against seeded history content (RuntimeHarness uses "{id} evidence").
        self.branch_failure: Any | None = None
        self.task_failure: Any | None = None
        self.fail_branch_history_markers: set[str] = set()

    async def formalize_task(self, request: Any) -> SummaryResult:
        await self.formalization_gate.wait()
        return SummaryResult(True, self.formalization_text, "gpt-5.6-luna-max", None, None)

    async def finalize_branch(self, request: Any) -> SummaryResult:
        self.branch_requests.append(request)
        await self.branch_gate.wait()
        history = request.branch_history
        tail = history[-1].get("content", "") if history else "empty"
        marker_hit = any(
            marker in str(item.get("content", ""))
            for marker in self.fail_branch_history_markers
            for item in (history or ())
        )
        if self.branch_failure is not None or marker_hit:
            error = self.branch_failure
            if error is None:
                error = GenerationError("summary_unavailable", "branch summary unavailable")
            return SummaryResult(False, "", "gpt-5.6-luna-max", None, error)
        return SummaryResult(True, f"branch-summary:{tail}", "gpt-5.6-luna-max", None, None)

    async def finalize_task(self, request: Any) -> SummaryResult:
        self.task_requests.append(request)
        await self.task_gate.wait()
        if self.task_failure is not None:
            return SummaryResult(False, "", "gpt-5.6-luna-max", None, self.task_failure)
        # Multi-branch FINAL (live E2E B): join coverage so light_judge sees agent reports.
        coverage = getattr(request, "coverage_summaries", ()) or ()
        parts = [str(item).strip() for item in coverage if str(item).strip()]
        text = "\n".join(parts) if parts else "task-summary"
        return SummaryResult(True, text, "gpt-5.6-luna-max", None, None)


class FakeScheduler:
    """只记录 manager 提交的 effects；测试显式驱动 worker 边界。"""

    def __init__(self) -> None:
        self.enqueued: list[Any] = []
        self.paused: set[str] = set()

    async def enqueue(self, effect: Any) -> bool:
        self.enqueued.append(effect)
        return True

    def task_inflight_count(self, _task_id: str) -> int:
        return 0

    def pause_task(self, task_id: str) -> None:
        self.paused.add(task_id)

    def resume_task(self, task_id: str) -> None:
        self.paused.discard(task_id)

    def cancel_generation(self, _task_id: str, _generation: int) -> int:
        return 0


class _MessageAPI:
    async def get_recent(self, _stream_id: str, _limit: int) -> list[dict[str, str]]:
        return [{"message_id": "message-1", "content": "recent context"}]

    async def build_readable(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": "user", "content": messages[0]["content"]}] if messages else []


class _ConfigAPI:
    async def get(self, key: str) -> str:
        return {
            "bot.nickname": "麦麦",
            "personality.personality": "严谨",
            "personality.behavior_style": "清晰",
            "personality.reply_style": "简洁",
        }[key]


class _AgentCatalog:
    """Integration catalog; ``entries`` feeds held-release ``plan_delegations`` live_ids."""

    fingerprint = "integration-catalog"
    root_agent = "builtin.quick_thinker"

    def __init__(self) -> None:
        self._extra_agents: dict[str, Any] = {}

    def register_agent(self, agent_id: str, *, can_be_root: bool = False) -> None:
        """Make ``agent_id`` visible to catalog.get / entries (held-release credit path)."""

        self._extra_agents[str(agent_id)] = SimpleNamespace(
            definition=SimpleNamespace(
                agent_id=str(agent_id),
                model_selector="model:gpt-5.6-luna-max",
                enabled=True,
                can_be_root=can_be_root,
            )
        )

    def get(self, agent_id: str) -> Any:
        if agent_id == self.root_agent:
            return SimpleNamespace(
                definition=SimpleNamespace(
                    agent_id=self.root_agent,
                    model_selector="model:gpt-5.6-luna-max",
                    enabled=True,
                    can_be_root=True,
                )
            )
        return self._extra_agents.get(agent_id)

    @property
    def entries(self) -> tuple[Any, ...]:
        root = self.get(self.root_agent)
        return (root, *self._extra_agents.values())


class _PriceCatalog:
    fingerprint = "integration-prices"

    def low_budget_warning(self, _selector: str, _credits: float, **_kwargs: object) -> None:
        return None


class RuntimeHarness:
    """以真实 reducer-adjacent report/store APIs 驱动的 deterministic harness。"""

    def __init__(self, root: str | Path, *, runtime_limits: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.clock = FakeClock()
        self.llm = FakeLLMGateway()
        self.procedures = FakeProcedureProvider()
        self.maisaka = FakeMaisaka()
        self.summarizer = FakeSummarizer()
        self.store = SQLiteStateStore(self.root / "lrs-integration.sqlite3")
        self.scheduler = FakeScheduler()
        self.manager: ResearchManager | None = None
        self.task_id = ""
        self.round_id = ""
        self.stream_id = "integration-stream"
        self.formalized_task: FormalizedTask | None = None
        self.coordinator: ReportCoordinator | None = None
        self._root_branch_id = ""
        self._status = TaskStatus.FORMALIZING
        self._started = False
        # Mirrored into ResearchManager; deliver_* must be set before formalize
        # (ReportCoordinator reads them at registration).
        self._runtime_limits: dict[str, Any] = dict(runtime_limits or {})

    def configure_runtime_limits(self, **limits: Any) -> None:
        """Merge into manager ``_runtime_limits`` (call before :meth:`formalize`)."""

        self._runtime_limits.update(limits)
        if self.manager is not None:
            self.manager._runtime_limits.update(limits)

    async def open(self) -> RuntimeHarness:
        await self.store.open()
        ctx = SimpleNamespace(
            message=_MessageAPI(),
            config=_ConfigAPI(),
            logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        )
        snapshot = SimpleNamespace(
            root_agent="builtin.quick_thinker",
            root_force_selector="",
            summarizer_selector="model:gpt-5.6-luna-max",
            default_effort_credits=100.0,
            agent_catalog=_AgentCatalog(),
            procedure_catalog=SimpleNamespace(fingerprint="integration-procedures"),
            price_catalog=_PriceCatalog(),
        )

        async def snapshot_provider() -> Any:
            return snapshot

        def coordinator_factory(**kwargs: Any) -> ReportCoordinator:
            kwargs["clock"] = self.clock
            return ReportCoordinator(**kwargs)

        self.manager = ResearchManager(
            ctx=ctx,
            store=self.store,
            summarizer=self.summarizer,
            scheduler=self.scheduler,
            snapshot_provider=snapshot_provider,
            grace_period_seconds=60,
            report_coordinator_factory=coordinator_factory,
            runtime_limits=dict(self._runtime_limits),
        )
        self._started = True
        return self

    async def close(self) -> None:
        if self._started:
            await self.store.close()
            self._started = False

    async def start(self, objective: str, *, credits: float, time_budget: int) -> dict[str, Any]:
        if not self._started:
            await self.open()
        assert self.manager is not None
        self.summarizer.formalization_gate.clear()
        result = await self.manager.start(
            objective=objective,
            stream_id=self.stream_id,
            time_budget_seconds=time_budget,
            effort_level=float(credits) / 100.0,
        )
        self.task_id = str(result["task_id"])
        self.round_id = str((await self.manager.status(self.task_id))["round_id"])
        return result

    async def formalize(self, text: str) -> None:
        assert self.manager is not None
        self.summarizer.formalization_text = text
        self.summarizer.formalization_gate.set()
        await self.manager.wait_idle(self.task_id)
        self.formalized_task = (await self.store.load_task(self.task_id)).formalized_task
        assert self.formalized_task is not None
        self.round_id = str((await self.manager.status(self.task_id))["round_id"])
        self.coordinator = self.manager.report_coordinators[self.task_id]

    async def root_delegates(self, allocations: dict[str, float]) -> None:
        assert self.coordinator is not None and self.formalized_task is not None
        assert self.manager is not None
        status = await self.manager.status(self.task_id)
        active_leaves = [item["branch_id"] for item in status["active_leaves"]]
        assert len(active_leaves) == 1
        root_branch_id = str(active_leaves[0])
        self._root_branch_id = root_branch_id
        self.coordinator.branches.pop(root_branch_id, None)
        # The harness models the materialization boundary after the real
        # controller has created the root.  Remove that root from the
        # manager's branch cache as well so report coverage reflects the
        # materialized children rather than a stale synthetic leaf.
        self.manager._branches[self.task_id].pop(root_branch_id, None)
        now = self.clock()
        branches: dict[str, BranchRuntime] = {}
        commands: list[StoreCommand] = []
        for name, credits in allocations.items():
            branch_id = str(name)
            branches[branch_id] = BranchRuntime(
                branch_id=branch_id,
                task=self.formalized_task,
                catalog_fingerprint="integration-catalog",
                generation=0,
                messages=[{"role": "assistant", "content": f"{branch_id} evidence"}],
                credits=float(credits),
                depth=1,
            )
            commands.append(
                StoreCommand(
                    "insert_branch",
                    {
                        "branch_id": branch_id,
                        "round_id": self.round_id,
                        "parent_branch_id": root_branch_id,
                        "agent_id": f"agent.{branch_id}",
                        "lifecycle": BranchLifecycle.READY.value,
                        "depth": 1,
                        "credit_balance": float(credits),
                        "generation": 0,
                        "created_at": now,
                    },
                )
            )
        self.coordinator.branches.update(branches)
        await self.store.transact(commands)

    def wire_live_agents(self, *agent_ids: str) -> None:
        """Register catalog agents so held-release ``plan_delegations`` sees them live.

        ``ResearchManager._release_held_delegations`` builds ``live_agent_ids`` from
        ``snapshot.agent_catalog.entries`` (not from ProcedureBatchCompleted). Offline
        multi-agent hold/release therefore needs explicit catalog entries.
        """

        assert self.manager is not None
        snapshot = self.manager._round_snapshots.get(self.task_id)
        if snapshot is None:
            # Before formalize: mutate the shared snapshot provider object.
            # ResearchManager caches the same SimpleNamespace into _round_snapshots.
            raise RuntimeError("wire_live_agents requires an active formalized round snapshot")
        catalog = snapshot.agent_catalog
        if not hasattr(catalog, "register_agent"):
            raise TypeError("agent catalog must support register_agent for held-release E2E")
        for agent_id in agent_ids:
            catalog.register_agent(str(agent_id))
        self.manager._agent_live_provider = lambda _agent_id: True

    async def root_allocates_real(
        self,
        allocations: dict[str, float],
        *,
        mark_agents_live: bool = True,
    ) -> dict[str, float]:
        """Drive root→children through real ``allocate_children`` + ``ChildMaterialized``.

        Unlike :meth:`root_delegates` (synthetic store inserts that skip planning),
        this submits a ``ProcedureBatchCompleted`` with delegations through the
        controller reducer, then runs each ``materialize_child`` effect via
        ``ResearchManager.materialize_child_effect``.

        ``allocations`` maps desired ``branch_id`` → requested credits. Agent ids
        default to ``agent.{branch_id}``. Returns controller ``active_leaves``
        after materialization.
        """

        from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
        from lunagentic_research_swarm.runtime.reducer import NotifyToolWaiter

        assert self.manager is not None
        status = await self.manager.status(self.task_id)
        active = list(status["active_leaves"])
        assert len(active) == 1, "root_allocates_real expects a single root leaf"
        root_branch_id = str(active[0]["branch_id"])
        self._root_branch_id = root_branch_id
        parent_credits = float(active[0]["credits"])
        controller = self.manager._controllers[self.task_id]
        generation = int(status["generation"])
        round_id = str(status["round_id"])
        self.round_id = round_id

        agent_ids = tuple(f"agent.{name}" for name in allocations)
        if mark_agents_live:
            # Also register catalog entries so a later held-release on these
            # leaves (or their children) can resolve the same ids via entries.
            self.wire_live_agents(*agent_ids)
        delegations = tuple(
            {
                "branch_id": str(name),
                "agent_id": f"agent.{name}",
                "task": f"work-{name}",
                "credits": float(credits),
            }
            for name, credits in allocations.items()
        )
        parent_messages = tuple(
            dict(item)
            for item in (self.manager._branches.get(self.task_id, {}).get(root_branch_id) or {}).get(
                "messages", ()
            )
        ) or ({"role": "user", "content": "formalized root"},)

        # Clear prior enqueue noise so we only drain this delegation batch.
        self.scheduler.enqueued.clear()
        await self.manager.handle_runtime_event(
            ProcedureBatchCompleted(
                event_id=f"{self.task_id}:root-allocate",
                task_id=self.task_id,
                round_id=round_id,
                generation=generation,
                branch_id=root_branch_id,
                call_id=f"{root_branch_id}:allocate",
                result_id=f"{root_branch_id}:allocate-result",
                credits_after=parent_credits,
                delegations=delegations,
                parent_messages=parent_messages,
                parent_depth=0,
                live_agent_ids=agent_ids,
                agent_calls_started=int(controller.state.agent_calls_started),
            )
        )

        materialize = [
            item
            for item in self.scheduler.enqueued
            if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
        ]
        if not materialize:
            raise AssertionError(
                "root_allocates_real expected materialize_child effects from real allocate_children path"
            )
        for effect in materialize:
            await self.manager.materialize_child_effect(effect)

        leaves = dict(controller.state.active_leaves)
        # Keep coordinator report graph aligned with materialized leaves.
        if self.coordinator is not None:
            for branch_id, credits in leaves.items():
                runtime = self.coordinator.branches.get(branch_id)
                if runtime is not None:
                    runtime.credits = float(credits)
        return leaves

    async def branch_checkpoint(self, branch_id: str) -> None:
        assert self.coordinator is not None
        await self.coordinator.on_branch_safe_point(branch_id, checkpoint=True)

    async def run_until_idle(self) -> None:
        assert self.coordinator is not None
        if self.coordinator.current_epoch is None:
            assert self.manager is not None
            from lunagentic_research_swarm.runtime.events import ReportDeadlineReached

            status = await self.manager.status(self.task_id)
            await self.manager.handle_runtime_event(
                ReportDeadlineReached(
                    f"{self.task_id}:deadline",
                    self.task_id,
                    str(status["round_id"]),
                    int(status["generation"]),
                    epoch=1,
                )
            )
        epoch = self.coordinator.current_epoch
        if epoch is not None and not epoch.synthesis_finished:
            assert self.manager is not None
            from lunagentic_research_swarm.runtime.events import GraceExpired

            status = await self.manager.status(self.task_id)
            await self.manager.handle_runtime_event(
                GraceExpired(
                    f"{self.task_id}:grace",
                    self.task_id,
                    str(status["round_id"]),
                    int(status["generation"]),
                    epoch=epoch.epoch,
                )
            )
        await self.coordinator.wait_for_synthesis()

    async def finalize_all(self) -> None:
        assert self.coordinator is not None and self.manager is not None
        # 保证真实 SQLite 的 created_at 排序与报告 epoch 的时间顺序一致。
        self.clock.advance(1)
        # Mirror ResearchManager.handle_branch_summary_effect: summarize at the
        # terminal safe point, then release_raw_context and drop the manager
        # activity-graph entry so COMPLETED retains summary layer only.
        for branch_id in list(self.coordinator.active_branch_ids()):
            await self.coordinator.on_branch_safe_point(branch_id, terminal=True)
            branch = self.coordinator.branches.get(branch_id)
            if branch is not None:
                release_raw_context(branch)
            self.manager._branches.get(self.task_id, {}).pop(branch_id, None)
        await self.coordinator.wait_for_synthesis()

    async def persisted_report_kinds(self) -> list[str]:
        layer = await self.store.load_summary_layer(self.task_id)
        return [str(item["kind"]) for item in (layer.reports if layer else ())]

    async def pending_outbox_count(self) -> int:
        return len(await self.store.list_due_outbox(self.clock()))

    async def deliver_outbox(self) -> int:
        from lunagentic_research_swarm.storage.outbox import MaisakaOutbox

        outbox = MaisakaOutbox(self.store, self.maisaka, clock=self.clock)
        attempted = await outbox.deliver_once()
        # append 完成后会在同一 transaction 创建 trigger intent；第二次显式
        # drain 让测试不依赖真实的 periodic timer。
        attempted += await outbox.deliver_once()
        return attempted

    @property
    def reports(self) -> list[Any]:
        return [] if self.coordinator is None else self.coordinator.reports

    @property
    def task_status(self) -> TaskStatus:
        if self.manager is not None and self.task_id:
            controller = self.manager._controllers.get(self.task_id)
            if controller is not None:
                return controller.state.status
        return self._status

    @property
    def raw_context_count(self) -> int:
        """Retained raw transcript messages in coordinator + manager branch caches."""

        count = 0
        if self.coordinator is not None:
            for branch in self.coordinator.branches.values():
                count += len(branch.messages)
        if self.manager is not None and self.task_id:
            for entry in self.manager._branches.get(self.task_id, {}).values():
                messages = entry.get("messages") if isinstance(entry, dict) else None
                if messages:
                    count += len(messages)
                pending = entry.get("pending_context") if isinstance(entry, dict) else None
                if pending:
                    count += len(pending)
        return count

    @property
    def resources_closed(self) -> bool:
        return self.store._connection is None and self.store._executor is None

    def use_live_llm(self, creds: Any) -> None:
        """Swap FakeLLMGateway for LiveLLMGateway (Task 5+ live tiers)."""

        from live_llm import LiveLLMGateway

        self.llm = LiveLLMGateway(creds)

    async def drive_live_until_terminal(
        self,
        *,
        timeout_seconds: float,
        artifact_dir: Any = None,
    ) -> dict[str, Any]:
        """Delegate to ``live_harness.drive_until_terminal``."""

        from live_harness import drive_until_terminal

        return await drive_until_terminal(
            self,
            timeout_seconds=timeout_seconds,
            artifact_dir=artifact_dir,
        )

    async def store_count_child_branches(self) -> int:
        """Count durable branches with a non-null parent (survives coordinator prune)."""

        round_id = str(self.round_id or "")

        def _query(connection: Any) -> int:
            if round_id:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM branches "
                    "WHERE parent_branch_id IS NOT NULL AND parent_branch_id != '' "
                    "AND round_id = ?",
                    (round_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM branches "
                    "WHERE parent_branch_id IS NOT NULL AND parent_branch_id != ''"
                ).fetchone()
            if row is None:
                return 0
            try:
                return int(row["n"])
            except (KeyError, IndexError, TypeError):
                return int(row[0])

        return int(await self.store.run_locked(_query))


__all__ = [
    "FakeClock",
    "FakeLLMGateway",
    "FakeLLMResponse",
    "FakeMaisaka",
    "FakeProcedureProvider",
    "FakeSummarizer",
    "RuntimeHarness",
]
