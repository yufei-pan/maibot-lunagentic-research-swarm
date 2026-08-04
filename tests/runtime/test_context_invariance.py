from __future__ import annotations

from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask
from lunagentic_research_swarm.runtime.context import (
    RuntimeHeader,
    StablePromptBuilder,
    release_raw_context,
    should_auto_compact,
)


def user_one_bytes(messages: list[dict[str, object]]) -> bytes:
    users = [message for message in messages if message["role"] == "user"]
    return str(users[0]["content"]).encode("utf-8")


def test_formalized_task_is_byte_invariant_across_graph_context_operations() -> None:
    formalized = FormalizedTask.create("逐字保持 α\r\n第二行，空格  不合并。")
    builder = StablePromptBuilder(
        formalized_task=formalized,
        swarm_identity="麦麦深度调查组",
        bot_profile={"nickname": "麦麦", "personality": "认真", "behavior_style": "求证", "reply_style": "简洁"},
        agent_catalog={"agent.root": {"selector": "task:reasoning", "protocol": "json_envelope", "role": "调查"}},
        procedure_catalog={"builtin.echo": {"description": "回显"}},
        pricing={"agent.root": {"fingerprint": "price-1", "price_in": 1.0, "price_out": 2.0}},
    )
    root = builder.root_context(coordinator="agent.root")
    snapshots = [builder.messages_for_call(root, RuntimeHeader("root", 1, "root", 99, 8.0, 1, 0))]
    child = builder.delegate(root, assignment="核查来源", agent_id="agent.child")
    snapshots.append(builder.messages_for_call(child, RuntimeHeader("child", 1, "search", 80, 2.0, 2, 1)))
    builder.compact(child, summary="压缩摘要")
    snapshots.append(builder.messages_for_call(child, RuntimeHeader("child", 2, "search", 60, 1.0, 1, 0)))
    builder.checkpoint(child, summary="checkpoint 摘要")
    snapshots.append(builder.messages_for_call(child, RuntimeHeader("child", 3, "search", 40, 1.0, 1, 0)))
    restarted = builder.restart_context(summary_layers=("branch summary", "task report"), coordinator="agent.root")
    snapshots.append(builder.messages_for_call(restarted, RuntimeHeader("restart", 1, "root", 120, 3.0, 1, 0)))

    assert {user_one_bytes(messages) for messages in snapshots} == {formalized.text.encode("utf-8")}
    assert all(messages[0]["role"] == "system" for messages in snapshots)
    assert "编排器" not in snapshots[0][0]["content"]


def test_stable_system_prefix_is_canonical_and_runtime_header_changes_only_suffix() -> None:
    formalized = FormalizedTask.create("任务")
    kwargs = dict(
        formalized_task=formalized,
        swarm_identity="swarm",
        bot_profile={"reply_style": "x", "nickname": "n"},
        agent_catalog={"b": {"selector": "model:b"}, "a": {"selector": "model:a"}},
        procedure_catalog={"z": {}, "a": {}},
        pricing={"b": {"fingerprint": "2"}, "a": {"fingerprint": "1"}},
    )
    left = StablePromptBuilder(**kwargs)
    right = StablePromptBuilder(**{**kwargs, "agent_catalog": dict(reversed(list(kwargs["agent_catalog"].items())))})
    context = left.root_context(coordinator="a")
    first = left.messages_for_call(context, RuntimeHeader("b", 1, "cap", 10, 0.0, 2, 3))
    second = left.messages_for_call(context, RuntimeHeader("b", 2, "cap", 9, -0.1, 1, 0))

    assert left.system_message == right.system_message
    assert first[:-1] == second[:-1]
    assert "仍可零 credits 委派" in first[-1]["content"]
    assert "本 turn 后不能启动后代" in second[-1]["content"]


def test_auto_compact_honors_override_precedence_and_model_window() -> None:
    assert should_auto_compact(5000, agent_override=4000, definition=8000, global_threshold=9000)
    assert not should_auto_compact(5000, agent_override=None, definition=8000, global_threshold=4000)
    assert should_auto_compact(
        8400,
        agent_override=None,
        definition=None,
        global_threshold=258000,
        model_context_limit=24000,
        reserved_output_tokens=8192,
        safety_margin_tokens=8192,
    )


def test_branch_finalization_releases_raw_messages() -> None:
    branch = BranchRuntime(
        branch_id="branch",
        task=FormalizedTask.create("任务"),
        catalog_fingerprint="catalog",
        generation=0,
        messages=[{"role": "user", "content": "raw assignment"}],
        credits=0.0,
        depth=0,
    )

    release_raw_context(branch)

    assert branch.messages == []
