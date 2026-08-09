"""Design §8.2 / §12.2 message assembly and compact threshold coverage."""

from __future__ import annotations

from lunagentic_research_swarm.models import FormalizedTask
from lunagentic_research_swarm.runtime.context import (
    BranchContext,
    RuntimeHeader,
    StablePromptBuilder,
    render_assignment_section,
    root_assignment_section,
    should_auto_compact,
)


def _builder(formalized: FormalizedTask) -> StablePromptBuilder:
    return StablePromptBuilder(
        formalized_task=formalized,
        swarm_identity="麦麦深度调查组",
        bot_profile={"nickname": "麦麦", "personality": "认真", "behavior_style": "求证", "reply_style": "简洁"},
        agent_catalog={
            "agent.root": {"selector": "task:reasoning", "protocol": "json_envelope", "role": "调查", "api_key": "SECRET"},
            "agent.child": {"selector": "task:utils", "protocol": "json_envelope"},
        },
        procedure_catalog={"builtin.echo": {"description": "回显", "token": "also-secret"}},
        pricing={"agent.root": {"fingerprint": "price-1", "price_in": 1.0, "price_out": 2.0}},
    )


def test_spec_8_2_message_order_and_secrets_stripped_from_system() -> None:
    """system → 正式任务 → …… → `[LRS runtime]`（含根任务分配）。

    根任务分配仍然是一条 user 消息、仍然不进 system（spec §8.2 的实质要求），
    只是与同一 turn 的 runtime 状态合并成了同一个块。
    """

    formalized = FormalizedTask.create("正式任务 α")
    builder = _builder(formalized)
    root = builder.root_context(coordinator="agent.root")
    messages = builder.messages_for_call(
        root,
        RuntimeHeader("root", 1, "调查", 90, 8.0, 1, 0, assignment=root_assignment_section()),
    )
    assert messages[0]["role"] == "system"
    assert "SECRET" not in messages[0]["content"]
    assert "also-secret" not in messages[0]["content"]
    assert "起始协调者" not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == formalized.text
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("[LRS runtime]")
    assert "起始协调者" in messages[-1]["content"]


def test_spec_8_2_runtime_header_distinguishes_zero_and_negative_balances() -> None:
    zero = RuntimeHeader("b", 1, "cap", 10, 0.0, 1, 0).message()["content"]
    negative = RuntimeHeader("b", 1, "cap", 10, -0.1, 1, 0).message()["content"]
    positive = RuntimeHeader("b", 1, "cap", 10, 1.0, 1, 0).message()["content"]
    assert "仍可零 credits 委派" in zero
    assert "不能启动后代" in negative
    assert "可在结构上限内委派" in positive
    assert "不能启动后代" not in zero
    assert "仍可零 credits 委派" not in negative


def test_spec_8_2_child_assignment_appended_without_rewriting_formalized_user1() -> None:
    """子任务分配随尾部 runtime 块下发；User 1 的正式任务描述原样不动。"""

    formalized = FormalizedTask.create("正式任务")
    builder = _builder(formalized)
    inherited = builder.initial_messages(builder.root_context(coordinator="agent.root"))
    child_messages = builder.messages_for_call(
        BranchContext([*inherited[2:], {"role": "assistant", "content": "父分支 report"}]),
        RuntimeHeader(
            "child",
            1,
            "调查",
            90,
            8.0,
            1,
            0,
            assignment=render_assignment_section(character_prompt="核验角色", assignment="核查来源"),
        ),
    )
    assert child_messages[1]["content"] == formalized.text
    assert child_messages[-1]["content"].startswith("[LRS runtime]")
    assert "核查来源" in child_messages[-1]["content"]
    # assignment 不再是一条独立消息
    assert not any(str(item.get("content", "")).startswith("assignment: ") for item in child_messages)


def test_spec_12_2_model_window_beats_high_global_threshold() -> None:
    # usable = 24000 - 8192 - 8192 = 7616; global 258000 alone would not fire.
    assert should_auto_compact(
        7616,
        agent_override=None,
        definition=None,
        global_threshold=258_000,
        model_context_limit=24_000,
        reserved_output_tokens=8192,
        safety_margin_tokens=8192,
    )
    assert not should_auto_compact(
        7615,
        agent_override=None,
        definition=None,
        global_threshold=258_000,
        model_context_limit=24_000,
        reserved_output_tokens=8192,
        safety_margin_tokens=8192,
    )


def test_spec_12_2_agent_override_wins_over_global_and_window() -> None:
    assert should_auto_compact(
        5000,
        agent_override=4000,
        definition=8000,
        global_threshold=258_000,
        model_context_limit=100_000,
        reserved_output_tokens=8192,
        safety_margin_tokens=8192,
    )
    # Override is decisive when window is large enough not to fire first.
    assert not should_auto_compact(
        3999,
        agent_override=4000,
        definition=1000,
        global_threshold=1000,
        model_context_limit=100_000,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )
