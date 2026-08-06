"""Design §12.2 / §23.2 — manual compact fan-out, per-child auto-compact, oversize prefix, compact failure.

Durable prepare / effect-runner terminate lifecycle lives in ``test_spec_compact_lifecycle``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.llm.gateway import GenerationError
from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.llm.tokens import estimate_prompt_tokens
from lunagentic_research_swarm.models import FormalizedTask, TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_COMPACT_ID, CoreProcedureDecision
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.context import StablePromptBuilder, should_auto_compact
from lunagentic_research_swarm.runtime.delegation import plan_delegations
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.manager import ResearchManager
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformAgentCall,
    PerformProcedureBatch,
    RuntimeState,
    reduce_event,
)


FORMALIZED_TEXT = "正式任务：compact 规格 α\r\n保持空格  与换行。"


class _OkSummarizer:
    def __init__(self, text: str = "压缩后的共享摘要") -> None:
        self.text = text
        self.calls = 0

    async def compact_branch(self, request: Any) -> SummaryResult:
        self.calls += 1
        return SummaryResult(True, self.text, "model:fake", None, None)


class _FailSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def compact_branch(self, request: Any) -> SummaryResult:
        self.calls += 1
        return SummaryResult(
            False,
            "",
            "model:fake",
            None,
            GenerationError(code="provider_error", message="compact unavailable"),
        )


def _parent_messages(*, formalized: str = FORMALIZED_TEXT) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": "swarm catalog prefix"},
        {"role": "user", "content": formalized},
        {"role": "user", "content": "本轮起始协调者：agent.root"},
        {"role": "assistant", "content": "很长的可变证据历史 line-1"},
        {"role": "assistant", "content": "很长的可变证据历史 line-2"},
        {"role": "user", "content": "[LRS runtime]\nbranch=parent; turn=1"},
    )


@pytest.mark.asyncio
async def test_spec_12_2_manual_compact_before_clone_shared_by_all_children() -> None:
    """Manual compact rewrites parent history before child clone; all children inherit it."""

    summarizer = _OkSummarizer("父子共享压缩摘要")
    executor = ProcedureExecutor(catalog=SimpleNamespace(get=lambda _pid: None), summarizer=summarizer)
    original = _parent_messages()
    batch = PerformProcedureBatch(
        "task-1",
        "round-1",
        0,
        payload={
            "branch_id": "parent",
            "call_id": "call-1",
            "formalized_task": FORMALIZED_TEXT,
            "messages": original,
            "requests": ({"procedure_id": CORE_COMPACT_ID, "arguments": {}},),
            "delegations": (
                {"agent_id": "agent.a", "task": "子任务 A", "credits": 4.0},
                {"agent_id": "agent.b", "task": "子任务 B", "credits": 4.0},
            ),
            "credits_after": 8.0,
        },
    )

    completed = await executor.invoke_many(batch)

    assert summarizer.calls == 1
    assert completed.controls.compact is True
    compact_prefix = completed.parent_messages
    assert compact_prefix[0]["role"] == "system"
    assert compact_prefix[1]["content"] == FORMALIZED_TEXT
    assert compact_prefix[1]["content"].encode("utf-8") == FORMALIZED_TEXT.encode("utf-8")
    assert compact_prefix[-1]["content"] == "分支压缩摘要：父子共享压缩摘要"
    assert all("很长的可变证据历史" not in str(item) for item in compact_prefix)

    transition = reduce_event(
        RuntimeState(
            "task-1",
            TaskStatus.RUNNING,
            active_round_id="round-1",
            active_leaves={"parent": 8.0},
            formalized_task=FormalizedTask.create(FORMALIZED_TEXT),
        ),
        ProcedureBatchCompleted(
            "evt-compact-fanout",
            "task-1",
            "round-1",
            0,
            branch_id="parent",
            call_id="call-1",
            credits_after=8.0,
            controls=CoreProcedureDecision(compact=True),
            delegations=(
                {"agent_id": "agent.a", "task": "子任务 A", "credits": 4.0},
                {"agent_id": "agent.b", "task": "子任务 B", "credits": 4.0},
            ),
            parent_messages=completed.parent_messages,
            live_agent_ids=("agent.a", "agent.b"),
        ),
    )
    child_effects = [
        item
        for item in transition.effects
        if isinstance(item, NotifyToolWaiter) and item.payload.get("action") == "materialize_child"
    ]
    assert len(child_effects) == 2
    prefixes = [tuple(dict(m) for m in effect.payload["messages"][:-1]) for effect in child_effects]
    assert prefixes[0] == prefixes[1] == tuple(dict(m) for m in completed.parent_messages)
    assert {effect.payload["messages"][-1]["content"] for effect in child_effects} == {
        "assignment: 子任务 A",
        "assignment: 子任务 B",
    }


def test_spec_12_2_auto_compact_is_per_child_after_clone() -> None:
    """After fan-out clone, each child decides auto-compact from its own used_tokens / override.

    Integration limit: full ResearchManager clone→prepare_agent_effect auto-compact path is not
    driven end-to-end here (would need production-adjacent harness wiring). Decision is unit-pinned
    with branch-local fixtures matching manager._maybe_auto_compact inputs.
    """

    shared_tokens = 5_000
    # Sibling A: agent override 4000 → compact; sibling B: override 8000 → skip.
    assert should_auto_compact(
        shared_tokens,
        agent_override=4_000,
        definition=None,
        global_threshold=258_000,
        model_context_limit=100_000,
        reserved_output_tokens=8_192,
        safety_margin_tokens=8_192,
    )
    assert not should_auto_compact(
        shared_tokens,
        agent_override=8_000,
        definition=None,
        global_threshold=258_000,
        model_context_limit=100_000,
        reserved_output_tokens=8_192,
        safety_margin_tokens=8_192,
    )

    # Clone itself does not force compact: both children inherit the same parent prefix.
    inherited = _parent_messages()
    plan = plan_delegations(
        (
            {"agent_id": "agent.a", "task": "A", "credits": 1.0},
            {"agent_id": "agent.b", "task": "B", "credits": 1.0},
        ),
        parent_branch_id="parent",
        parent_depth=0,
        parent_credits=2.0,
        parent_messages=inherited,
        live_agent_ids=("agent.a", "agent.b"),
    )
    assert len(plan.children) == 2
    assert plan.children[0].messages[:-1] == plan.children[1].messages[:-1] == inherited


@pytest.mark.asyncio
async def test_spec_12_2_immutable_prefix_larger_than_usable_window_errors_without_rewriting_task() -> None:
    """§12.2 — if system/catalog/formalized alone exceeds the safe window, fail explicitly."""

    formalized = FormalizedTask.create("不可变正式任务-" + ("超长前缀" * 4000))
    builder = StablePromptBuilder(
        formalized_task=formalized,
        swarm_identity="swarm",
        bot_profile={"nickname": "n", "personality": "p", "behavior_style": "b", "reply_style": "r"},
        agent_catalog={"agent.root": {"selector": "model:x", "protocol": "json_envelope"}},
        procedure_catalog={"builtin.echo": {"description": "echo"}},
        pricing={"agent.root": {"fingerprint": "p1", "price_in": 1.0, "price_out": 1.0}},
    )
    prefix_only = (
        {"role": "system", "content": builder.system_message},
        {"role": "user", "content": formalized.text},
    )
    prefix_tokens = estimate_prompt_tokens(prefix_only).prompt_tokens
    # Usable window deliberately smaller than the immutable prefix alone.
    model_limit = max(32, prefix_tokens // 2)
    assert should_auto_compact(
        prefix_tokens,
        agent_override=None,
        definition=None,
        global_threshold=10**9,
        model_context_limit=model_limit,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )

    # Compact rewrite keeps User1 byte-identical (cannot mask oversize by rewriting the task).
    rewritten = ProcedureExecutor._rewrite_compacted_messages(
        (*prefix_only, {"role": "assistant", "content": "可变历史"}),
        "短摘要",
    )
    assert rewritten[1]["content"].encode("utf-8") == formalized.text.encode("utf-8")
    post_compact_tokens = estimate_prompt_tokens(rewritten).prompt_tokens
    assert should_auto_compact(
        post_compact_tokens,
        agent_override=None,
        definition=None,
        global_threshold=10**9,
        model_context_limit=model_limit,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )

    summarizer = _OkSummarizer("短摘要")
    store = SimpleNamespace(commands=[], transact=lambda commands: _async_noop(store, commands))
    manager = ResearchManager(
        ctx=SimpleNamespace(logger=SimpleNamespace(warning=lambda *_a, **_k: None, error=lambda *_a, **_k: None)),
        store=store,
        summarizer=summarizer,
        scheduler=SimpleNamespace(enqueue=lambda *_a, **_k: _async_true()),
        snapshot_provider=lambda: SimpleNamespace(price_catalog=None),
        runtime_limits={
            "auto_compact_tokens": 10**9,
            "model_context_window": model_limit,
            "reserved_output_tokens": 0,
            "safety_margin_tokens": 0,
        },
    )
    controller = SimpleNamespace(state=SimpleNamespace(formalized_task=formalized))
    branch: dict[str, Any] = {"messages": [dict(m) for m in (*prefix_only, {"role": "assistant", "content": "可变"})]}
    effect = PerformAgentCall("task-oversize", "round-1", 0, payload={"branch_id": "br"})

    with pytest.raises(RuntimeError, match="仍超过安全窗口|终止该分支"):
        await manager._maybe_auto_compact(
            effect=effect,
            controller=controller,
            snapshot=SimpleNamespace(price_catalog=None),
            definition=SimpleNamespace(auto_compact_tokens=None, agent_id="agent.root"),
            selector="model:x",
            protocol="json_envelope",
            messages=tuple(dict(m) for m in branch["messages"]),
            branch=branch,
            branch_id="br",
            prompt_tokens=prefix_tokens + 100,
            cache_hit=0,
            cache_miss=prefix_tokens + 100,
            estimated_charge=0.0,
            resolved=None,
        )

    assert branch["messages"][0]["content"] == formalized.text
    assert branch["messages"][0]["content"].encode("utf-8") == formalized.text.encode("utf-8")


@pytest.mark.asyncio
async def test_spec_23_2_compact_failure_leaves_history_unchanged() -> None:
    """§23.2 — failed compact must not replace mutable history or count as success."""

    original = _parent_messages()
    executor = ProcedureExecutor(catalog=SimpleNamespace(get=lambda _pid: None), summarizer=_FailSummarizer())
    batch = PerformProcedureBatch(
        "task-1",
        "round-1",
        0,
        payload={
            "branch_id": "parent",
            "call_id": "call-fail",
            "formalized_task": FORMALIZED_TEXT,
            "messages": original,
            "requests": ({"procedure_id": CORE_COMPACT_ID, "arguments": {}},),
            "report": "本 turn 报告仍应保留",
        },
    )
    completed = await executor.invoke_many(batch)

    compact_result = completed.results[-1]
    assert compact_result.procedure_id == CORE_COMPACT_ID
    assert compact_result.result.success is False
    assert compact_result.result.data is None or not compact_result.result.data.get("compacted")

    # History still contains pre-compact mutable content + report; no compact summary rewrite.
    contents = [str(item.get("content", "")) for item in completed.parent_messages]
    assert any("很长的可变证据历史" in text for text in contents)
    assert any("本 turn 报告仍应保留" in text for text in contents)
    assert all("分支压缩摘要：" not in text for text in contents)
    assert completed.parent_messages[1]["content"] == FORMALIZED_TEXT


@pytest.mark.asyncio
async def test_spec_23_2_auto_compact_failure_rejects_oversized_call() -> None:
    """§23.2 auto-compact failure raises; branch messages stay on the pre-compact history."""

    formalized = FormalizedTask.create(FORMALIZED_TEXT)
    original = _parent_messages()
    store = SimpleNamespace(commands=[], transact=lambda commands: _async_noop(store, commands))
    manager = ResearchManager(
        ctx=SimpleNamespace(logger=SimpleNamespace(warning=lambda *_a, **_k: None, error=lambda *_a, **_k: None)),
        store=store,
        summarizer=_FailSummarizer(),
        scheduler=SimpleNamespace(enqueue=lambda *_a, **_k: _async_true()),
        snapshot_provider=lambda: SimpleNamespace(price_catalog=None),
        runtime_limits={
            "auto_compact_tokens": 100,
            "reserved_output_tokens": 0,
            "safety_margin_tokens": 0,
        },
    )
    controller = SimpleNamespace(state=SimpleNamespace(formalized_task=formalized))
    branch: dict[str, Any] = {"messages": [dict(m) for m in original]}
    before = [dict(m) for m in branch["messages"]]
    effect = PerformAgentCall("task-fail-compact", "round-1", 0, payload={"branch_id": "br"})

    with pytest.raises(RuntimeError, match="自动 compact 失败"):
        await manager._maybe_auto_compact(
            effect=effect,
            controller=controller,
            snapshot=SimpleNamespace(price_catalog=None),
            definition=SimpleNamespace(auto_compact_tokens=None, agent_id="agent.root"),
            selector="model:x",
            protocol="json_envelope",
            messages=tuple(dict(m) for m in original),
            branch=branch,
            branch_id="br",
            prompt_tokens=5_000,
            cache_hit=0,
            cache_miss=5_000,
            estimated_charge=0.0,
            resolved=None,
        )

    assert branch["messages"] == before


async def _async_noop(store: Any, commands: Any) -> None:
    store.commands.extend(list(commands))


async def _async_true() -> bool:
    return True


# price_in=10, price_out=20 → (100*10 + 50*20) / 1e6 * 100 = 0.2
_COMPACT_USAGE = TokenUsage(100, 50, 0, 100, source="actual")
_COMPACT_CHARGE = 0.2
_COMPACT_CATALOG = PriceCatalog.from_sources({}, {"model:fake": PriceProfile(price_in=10.0, price_out=20.0)}, {})


class _MeteredSummarizer:
    def __init__(self, text: str = "压缩后的共享摘要") -> None:
        self.text = text
        self.calls = 0

    async def compact_branch(self, request: Any) -> SummaryResult:
        self.calls += 1
        return SummaryResult(True, self.text, "model:fake", _COMPACT_USAGE, None)


@pytest.mark.asyncio
async def test_agent_requested_compact_charges_caller_even_with_zero_budget() -> None:
    """Agent 请求 core.compact（credits=0）仍按 summarizer 计量扣研究余额。"""

    from lunagentic_research_swarm.runtime.turns import TurnWorker

    summarizer = _MeteredSummarizer()
    executor = ProcedureExecutor(catalog=SimpleNamespace(get=lambda _pid: None), summarizer=summarizer)
    prior = 10.0
    effect = PerformProcedureBatch(
        "task-1",
        "round-1",
        0,
        payload={
            "branch_id": "parent",
            "call_id": "call-bill",
            "formalized_task": FORMALIZED_TEXT,
            "messages": _parent_messages(),
            "credits_after": prior,
            "price_catalog": _COMPACT_CATALOG,
            "requests": ({"procedure_id": CORE_COMPACT_ID, "arguments": {}, "credits": 0.0},),
        },
    )

    completed = await TurnWorker(object(), executor).perform_procedure_batch(effect)

    assert summarizer.calls == 1
    assert completed.results[-1].result.research_credits_charged == pytest.approx(_COMPACT_CHARGE)
    assert completed.credits_after == pytest.approx(prior - _COMPACT_CHARGE)


@pytest.mark.asyncio
async def test_auto_compact_does_not_debit_research_credits() -> None:
    """自动 compact 即使 summarizer 有 usage，也不写研究 ledger / 不走 batch 扣费。"""

    formalized = FormalizedTask.create(FORMALIZED_TEXT)
    original = _parent_messages()
    store = SimpleNamespace(commands=[], transact=lambda commands: _async_noop(store, commands))
    manager = ResearchManager(
        ctx=SimpleNamespace(logger=SimpleNamespace(warning=lambda *_a, **_k: None, error=lambda *_a, **_k: None)),
        store=store,
        summarizer=_MeteredSummarizer("短摘要"),
        scheduler=SimpleNamespace(enqueue=lambda *_a, **_k: _async_true()),
        snapshot_provider=lambda: SimpleNamespace(price_catalog=_COMPACT_CATALOG),
        runtime_limits={
            "auto_compact_tokens": 100,
            "reserved_output_tokens": 0,
            "safety_margin_tokens": 0,
        },
    )
    controller = SimpleNamespace(state=SimpleNamespace(formalized_task=formalized))
    branch: dict[str, Any] = {"messages": [dict(m) for m in original], "credits": 10.0}
    credits_before = float(branch["credits"])
    effect = PerformAgentCall("task-auto-compact", "round-1", 0, payload={"branch_id": "br"})

    await manager._maybe_auto_compact(
        effect=effect,
        controller=controller,
        snapshot=SimpleNamespace(price_catalog=_COMPACT_CATALOG),
        definition=SimpleNamespace(auto_compact_tokens=None, agent_id="agent.root"),
        selector="model:x",
        protocol="json_envelope",
        messages=tuple(dict(m) for m in original),
        branch=branch,
        branch_id="br",
        prompt_tokens=5_000,
        cache_hit=0,
        cache_miss=5_000,
        estimated_charge=0.0,
        resolved=None,
    )

    assert float(branch["credits"]) == pytest.approx(credits_before)
    ledger = [cmd for cmd in store.commands if getattr(cmd, "kind", None) == "insert_credit_ledger"]
    assert ledger == []
    assert any(getattr(cmd, "kind", None) == "insert_procedure_call" for cmd in store.commands)
