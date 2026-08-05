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

    async def enqueue(self, effect) -> bool:
        self.enqueued.append(effect)
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
        if agent_id != "root":
            return None
        return SimpleNamespace(definition=SimpleNamespace(model_selector="model:root", enabled=True, can_be_root=True))


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
