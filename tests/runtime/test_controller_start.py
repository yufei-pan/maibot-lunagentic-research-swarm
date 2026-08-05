from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.runtime.manager import ResearchManager


@dataclass
class FakeRound:
    round_id: str
    round_number: int
    generation: int
    status: SimpleNamespace
    credit_pool: float = 0.0


@dataclass
class FakeTask:
    task_id: str
    stream_id: str
    formalized_task: object | None
    current_round_number: int
    current_round: FakeRound


class FakeStore:
    def __init__(self) -> None:
        self.commands: list[object] = []
        self.tasks: dict[str, FakeTask] = {}

    async def transact(self, commands) -> None:
        for command in commands:
            self.commands.append(command)
            values = dict(command.values)
            if command.kind == "insert_task":
                self.tasks[values["task_id"]] = FakeTask(
                    values["task_id"], values["stream_id"], None, values["current_round_number"], None
                )
            elif command.kind == "insert_round":
                task = self.tasks[values["task_id"]]
                task.current_round = FakeRound(
                    values["round_id"],
                    values["round_number"],
                    values["generation"],
                    SimpleNamespace(value=values["status"]),
                    float(values.get("credit_pool", 0.0)),
                )
            elif command.kind == "update_round_status":
                for task in self.tasks.values():
                    if task.current_round.round_id == values["round_id"]:
                        task.current_round.status = SimpleNamespace(value=values["status"])
            elif command.kind == "update_task_formalization":
                from lunagentic_research_swarm.models import FormalizedTask

                task = self.tasks[values["task_id"]]
                task.formalized_task = FormalizedTask(values["formalized_text"], values["formalized_sha256"])
            elif command.kind == "set_task_current_round":
                self.tasks[values["task_id"]].current_round_number = values["current_round_number"]

    async def load_task(self, task_id: str):
        return self.tasks.get(task_id)

    async def load_summary_layer(self, task_id: str):
        task = self.tasks.get(task_id)
        if task is None:
            return None
        contexts = tuple(
            json.loads(dict(command.values)["metadata_json"])["context"]
            for command in self.commands
            if command.kind == "insert_lifecycle_event"
            and dict(command.values).get("task_id") == task_id
            and dict(command.values).get("event_type") == "ContextSupplied"
        )
        return SimpleNamespace(
            formalized_task=task.formalized_task,
            summaries=(),
            reports=(),
            feedback=(),
            supplied_context=contexts,
        )


class FakeSummarizer:
    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.gate.set()
        self.error: str | None = None
        self.requests = []

    def block(self) -> None:
        self.gate.clear()

    def fail(self, message: str) -> None:
        self.error = message

    async def formalize_task(self, request):
        self.requests.append(request)
        await self.gate.wait()
        if self.error:
            return SummaryResult(False, "", "", None, SimpleNamespace(code="provider_error", message=self.error))
        return SummaryResult(True, "形式化后的调查任务", "fake-model", None, None)

    async def finalize_branch(self, request):
        return SummaryResult(True, "分支摘要", "fake-model", None, None)

    async def finalize_task(self, request):
        return SummaryResult(True, "任务报告", "fake-model", None, None)


class FakeMessageAPI:
    def __init__(self) -> None:
        self.calls = []

    async def get_recent(self, stream_id: str, limit: int):
        self.calls.append(("get_recent", stream_id, limit))
        return [{"message_id": "m1", "content": "最近消息中的秘密"}]

    async def build_readable(self, messages):
        self.calls.append(("build_readable", tuple(item["message_id"] for item in messages)))
        return [{"role": "user", "content": messages[0]["content"]}]


class FakeConfigAPI:
    def __init__(self) -> None:
        self.calls = []

    async def get(self, key: str):
        self.calls.append(key)
        return {
            "bot.nickname": "麦麦",
            "personality.personality": "好奇",
            "personality.behavior_style": "严谨",
            "personality.reply_style": "简洁",
        }[key]


class FakeScheduler:
    def __init__(self) -> None:
        self.enqueued = []
        self.on_enqueue = None

    async def enqueue(self, effect) -> bool:
        self.enqueued.append(effect)
        if self.on_enqueue is not None:
            result = self.on_enqueue(effect)
            if hasattr(result, "__await__"):
                await result
        return True

    def task_inflight_count(self, task_id: str) -> int:
        return 0

    def pause_task(self, task_id: str) -> None:
        pass

    def resume_task(self, task_id: str) -> None:
        pass

    def cancel_generation(self, task_id: str, generation: int) -> int:
        return 0


class FakePriceCatalog:
    fingerprint = "price-fingerprint"

    def low_budget_warning(self, selector: str, credits: float):
        return "预算偏低" if credits < 25 else None


class FakeCatalog:
    fingerprint = "catalog-fingerprint"

    def get(self, agent_id: str):
        if agent_id not in {"root", "child"}:
            return None
        return SimpleNamespace(
            definition=SimpleNamespace(
                agent_id=agent_id,
                model_selector=f"model:{agent_id}",
                protocol="json_envelope",
                description=f"{agent_id} capability",
                enabled=True,
                can_be_root=agent_id == "root",
            )
        )

    @property
    def entries(self):
        return (self.get("root"), self.get("child"))


@pytest.fixture
def harness():
    store = FakeStore()
    summarizer = FakeSummarizer()
    scheduler = FakeScheduler()
    message = FakeMessageAPI()
    config = FakeConfigAPI()
    ctx = SimpleNamespace(
        message=message,
        config=config,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    snapshot = SimpleNamespace(
        root_agent="root",
        root_force_selector="",
        summarizer_selector="model:summarizer",
        default_effort_credits=100.0,
        agent_catalog=FakeCatalog(),
        procedure_catalog=SimpleNamespace(fingerprint="procedures-fingerprint"),
        price_catalog=FakePriceCatalog(),
    )

    async def snapshot_provider():
        return snapshot

    manager = ResearchManager(
        ctx=ctx,
        store=store,
        summarizer=summarizer,
        scheduler=scheduler,
        snapshot_provider=snapshot_provider,
        recent_message_limit=12,
        pause_timeout_seconds=1200,
        grace_period_seconds=60,
    )
    return manager, store, summarizer, scheduler, message, config


@pytest.mark.asyncio
async def test_manager_injects_feedback_service_into_controller(harness) -> None:
    """Regression: ResearchManager must forward feedback_service into TaskController."""
    _manager, store, summarizer, scheduler, *_ = harness
    feedback = object()
    manager = ResearchManager(
        ctx=_manager.ctx,
        store=store,
        summarizer=summarizer,
        scheduler=scheduler,
        snapshot_provider=_manager._snapshot_provider,
        recent_message_limit=12,
        pause_timeout_seconds=1200,
        grace_period_seconds=60,
        feedback_service=feedback,
    )

    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    controller = manager._controllers[result["task_id"]]
    assert controller.feedback is feedback
    summarizer.gate.set()
    await manager.wait_idle(result["task_id"])


@pytest.mark.asyncio
async def test_start_returns_after_durable_create_without_waiting_for_formalizer(harness) -> None:
    manager, store, summarizer, *_ = harness
    summarizer.block()

    result = await manager.start(
        objective="调查问题", stream_id="stream-1", time_budget_seconds=90, effort_level=1.5
    )

    assert result["task_id"].startswith("lrs_")
    assert result["status"] == "FORMALIZING"
    assert result["initial_credits"] == 150.0
    assert await store.load_task(result["task_id"])
    summarizer.gate.set()
    await manager.wait_idle(result["task_id"])


@pytest.mark.asyncio
async def test_formalize_survives_metering_failure_when_catalog_missing(harness) -> None:
    """计量失败不得中止形式化；raising meter 必须被降级跳过。"""

    manager, store, summarizer, *_ = harness

    from lunagentic_research_swarm.llm.pricing import TokenUsage
    import lunagentic_research_swarm.runtime.manager as manager_mod

    async def formalize_with_usage(request):
        summarizer.requests.append(request)
        await summarizer.gate.wait()
        return SummaryResult(
            True,
            "形式化后的调查任务",
            "fake-model",
            TokenUsage(10, 1, 0, 10, source="actual"),
            None,
        )

    summarizer.formalize_task = formalize_with_usage  # type: ignore[method-assign]
    previous = manager_mod.meter_summarizer_usage

    def boom(**_kwargs):
        raise ValueError("有 usage 时必须提供 catalog 或 actual_charge")

    manager_mod.meter_summarizer_usage = boom  # type: ignore[assignment]
    try:
        result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120, effort_level=1.0)
        await manager.wait_idle(result["task_id"])
    finally:
        manager_mod.meter_summarizer_usage = previous  # type: ignore[assignment]

    stored = await store.load_task(result["task_id"])
    assert stored.formalized_task is not None
    assert stored.formalized_task.text == "形式化后的调查任务"
    assert stored.current_round.status.value == "RUNNING"


@pytest.mark.asyncio
async def test_formalizer_failure_marks_task_failed_without_using_raw_objective(harness) -> None:
    manager, store, summarizer, *_ = harness
    summarizer.fail("provider error")

    result = await manager.start(objective="raw", stream_id="s", time_budget_seconds=120, effort_level=1.0)
    await manager.wait_idle(result["task_id"])

    stored = await store.load_task(result["task_id"])
    assert stored.current_round.status.value == "FAILED"
    assert stored.formalized_task is None


@pytest.mark.asyncio
async def test_formalization_uses_only_public_recent_message_and_config_apis_without_persisting_raw(harness) -> None:
    manager, store, summarizer, _, message, config = harness

    result = await manager.start(
        objective="不要落盘的原始目标",
        stream_id="stream-private",
        planner_context="仅由 Tool 显式提供",
        time_budget_seconds=120,
    )
    await manager.wait_idle(result["task_id"])

    assert message.calls == [
        ("get_recent", "stream-private", 12),
        ("build_readable", ("m1",)),
    ]
    assert config.calls == [
        "bot.nickname",
        "personality.personality",
        "personality.behavior_style",
        "personality.reply_style",
    ]
    request = summarizer.requests[0]
    assert "不要落盘的原始目标" in request.raw_context
    assert "仅由 Tool 显式提供" in request.raw_context
    persisted = json.dumps(
        [{"kind": command.kind, "values": dict(command.values)} for command in store.commands],
        ensure_ascii=False,
        default=str,
    )
    assert "不要落盘的原始目标" not in persisted
    assert "最近消息中的秘密" not in persisted
    assert "好奇" not in persisted


@pytest.mark.asyncio
async def test_success_persists_formalization_vector_job_and_root_before_launch(harness) -> None:
    manager, store, _, scheduler, *_ = harness

    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120, effort_level=2.0)
    await manager.wait_idle(result["task_id"])

    kinds = [command.kind for command in store.commands]
    assert "update_task_formalization" in kinds
    assert "insert_vector_job" in kinds
    assert "insert_branch" in kinds
    assert scheduler.enqueued
    stored = await store.load_task(result["task_id"])
    assert stored.current_round.status.value == "RUNNING"
    assert stored.formalized_task.text == "形式化后的调查任务"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective": "   ", "stream_id": "s", "time_budget_seconds": 1},
        {"objective": "x", "stream_id": "s", "time_budget_seconds": 0},
        {"objective": "x", "stream_id": "s", "time_budget_seconds": 1, "effort_level": -0.1},
    ],
)
async def test_start_rejects_invalid_inputs_before_persistence(harness, kwargs) -> None:
    manager, store, *_ = harness
    with pytest.raises(ValueError):
        await manager.start(**kwargs)
    assert store.commands == []


@pytest.mark.asyncio
async def test_manager_prepares_root_agent_effect_from_frozen_round_snapshot(harness) -> None:
    manager, _, _, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])

    prepared = await manager.prepare_agent_effect(scheduler.enqueued[-1])

    assert prepared.payload["selector"] == "model:root"
    assert prepared.payload["protocol"] == "json_envelope"
    assert prepared.payload["agent_id"] == "root"
    assert prepared.payload["messages"][1]["content"] == "形式化后的调查任务"
    assert prepared.payload["live_agent_ids"] == ("child", "root")


@pytest.mark.asyncio
async def test_manager_prepares_procedure_effect_from_frozen_round_snapshot(harness) -> None:
    from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch

    manager, _, _, _, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    status = await manager.status(result["task_id"])
    branch_id = status["active_leaves"][0]["branch_id"]
    effect = PerformProcedureBatch(
        result["task_id"],
        status["round_id"],
        status["generation"],
        payload={"branch_id": branch_id, "messages": ()},
    )

    prepared = await manager.prepare_procedure_effect(effect)

    assert prepared.payload["procedure_catalog"] is manager._round_snapshots[result["task_id"]].procedure_catalog


@pytest.mark.asyncio
async def test_formalization_prepares_routing_state_before_immediate_agent_dispatch(harness) -> None:
    from lunagentic_research_swarm.runtime.reducer import PerformAgentCall

    manager, _, _, scheduler, *_ = harness
    errors = []

    async def inspect_agent(effect):
        if isinstance(effect, PerformAgentCall):
            try:
                await manager.prepare_agent_effect(effect)
            except Exception as exc:  # pragma: no cover - assertion below records the expected RED path
                errors.append(exc)

    scheduler.on_enqueue = inspect_agent
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])

    assert (await manager.status(result["task_id"]))["status"] == "RUNNING"
    assert errors == []


@pytest.mark.asyncio
async def test_restart_prepares_routing_state_before_immediate_agent_dispatch(harness) -> None:
    from lunagentic_research_swarm.runtime.reducer import PerformAgentCall

    manager, _, _, scheduler, *_ = harness
    errors = []

    async def inspect_agent(effect):
        if isinstance(effect, PerformAgentCall):
            try:
                await manager.prepare_agent_effect(effect)
            except Exception as exc:  # pragma: no cover - assertion below records the expected RED path
                errors.append(exc)

    scheduler.on_enqueue = inspect_agent
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    await manager.stop(result["task_id"], stream_id="s")

    continued = await manager.continue_task(result["task_id"], credit_adjustment=1.0, stream_id="s")

    assert continued["status"] == "RUNNING"
    assert errors == []


@pytest.mark.asyncio
async def test_manager_shutdown_cancels_formalization_and_pause_background_tasks(harness) -> None:
    manager, store, summarizer, _, *_ = harness
    summarizer.block()
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    task_id = result["task_id"]
    await asyncio.sleep(0)
    assert manager._jobs[task_id]

    await manager.shutdown()
    command_count = len(store.commands)
    summarizer.gate.set()
    await asyncio.sleep(0)

    assert manager._jobs == {}
    assert manager._pause_jobs == {}
    assert len(store.commands) == command_count


@pytest.mark.asyncio
async def test_manager_shutdown_cancels_pause_expiry_task(harness) -> None:
    manager, _, _, _, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    await manager.pause(result["task_id"], stream_id="s")
    assert manager._pause_jobs

    await manager.shutdown()

    assert manager._pause_jobs == {}


@pytest.mark.asyncio
async def test_materialize_child_commits_before_enqueuing_agent(harness) -> None:
    from lunagentic_research_swarm.runtime.reducer import NotifyToolWaiter, PerformAgentCall

    manager, store, _, scheduler, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    status = await manager.status(result["task_id"])
    parent_id = status["active_leaves"][0]["branch_id"]
    scheduler.enqueued.clear()
    effect = NotifyToolWaiter(
        result["task_id"],
        status["round_id"],
        status["generation"],
        payload={
            "action": "materialize_child",
            "branch_id": f"{parent_id}:1",
            "parent_branch_id": parent_id,
            "agent_id": "child",
            "assignment": "核对证据",
            "credits": 25.0,
            "depth": 1,
            "messages": ({"role": "user", "content": "assignment: 核对证据"},),
            "retire_parent": True,
        },
    )

    await manager.materialize_child_effect(effect)

    branch_inserts = [command for command in store.commands if command.kind == "insert_branch"]
    assert branch_inserts[-1].values["branch_id"] == f"{parent_id}:1"
    assert isinstance(scheduler.enqueued[-1], PerformAgentCall)
    assert scheduler.enqueued[-1].payload["branch_id"] == f"{parent_id}:1"
    assert (await manager.status(result["task_id"]))["active_leaves"] == [
        {"branch_id": f"{parent_id}:1", "credits": 25.0}
    ]


@pytest.mark.asyncio
async def test_branch_summary_effect_finalizes_coordinator_branch_and_releases_messages(harness) -> None:
    from lunagentic_research_swarm.models import BranchLifecycle
    from lunagentic_research_swarm.runtime.reducer import PerformBranchSummary

    manager, _, _, _, *_ = harness
    result = await manager.start(objective="调查", stream_id="s", time_budget_seconds=120)
    await manager.wait_idle(result["task_id"])
    status = await manager.status(result["task_id"])
    branch_id = status["active_leaves"][0]["branch_id"]

    await manager.handle_branch_summary_effect(
        PerformBranchSummary(
            result["task_id"],
            status["round_id"],
            status["generation"],
            payload={"branch_id": branch_id, "reason": "no_further_work"},
        )
    )

    branch = manager.report_coordinators[result["task_id"]].branches[branch_id]
    assert branch.lifecycle is BranchLifecycle.FINALIZED
    assert branch.messages == []
