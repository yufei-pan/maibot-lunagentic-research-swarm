"""智能体必须知道：我是谁、我能调用什么、我要做什么、怎么做。

这些保证分布在两个位置，两边都必须钉住：
- 冻结 system 前缀对全轮所有智能体逐字节相同（design §8.2 的 cache 要求），
  因此逐 agent 的身份只能走 runtime header 与子任务 assignment；
- 角色 `character_prompt` 必须出现在 assignment，且**不得**出现在 system。
"""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions
from lunagentic_research_swarm.agents.bundled.prompts import BUNDLED_CHARACTER_PROMPTS
from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.models import FormalizedTask
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry
from lunagentic_research_swarm.runtime.context import (
    RuntimeHeader,
    StablePromptBuilder,
    render_assignment_section,
    root_assignment_section,
)
from lunagentic_research_swarm.runtime.manager import ResearchManager


def _bundled_snapshot():
    agents = AgentRegistry(root_agent="builtin.quick_thinker")
    agents.replace_provider("builtin", [d.model_dump(mode="json") for d in bundled_agent_definitions()])
    procedures = ProcedureRegistry()
    procedures.replace_provider(
        "builtin",
        [
            {
                "procedure_id": procedure_id,
                "version": "1",
                "display_name": procedure_id,
                "description": "测试用 Procedure",
                "arguments_schema": {"type": "object"},
                "result_schema": {"type": "object"},
                "enabled": True,
            }
            for procedure_id in ("builtin.web_search", "builtin.knowledge_search", "builtin.person_lookup")
        ],
    )
    return agents.snapshot({}), procedures.snapshot({})


class _Snapshot:
    def __init__(self, agent_catalog, procedure_catalog) -> None:
        self.agent_catalog = agent_catalog
        self.procedure_catalog = procedure_catalog


# --- runtime header：我是谁 / 我能调用什么 ---------------------------------


def test_runtime_header_names_the_agent_and_its_callable_procedures() -> None:
    header = RuntimeHeader(
        "br_1",
        3,
        "外部搜索与证据",
        120,
        12.5,
        2,
        1,
        agent_id="builtin.researcher",
        agent_display_name="外部研究员 (Researcher)",
        allowed_procedures=("core.terminate", "builtin.web_search"),
        last_turn_credits=1.25,
    )
    content = header.message()["content"]

    assert "`builtin.researcher`" in content
    assert "外部研究员 (Researcher)" in content
    # 自委派是协议要求的动作，必须能从 header 直接读出用哪个 id。
    assert "自委派时 `agent_id` 就写 `builtin.researcher`" in content
    assert "可调用 Procedure：" in content
    assert "`builtin.web_search`" in content
    assert "上一 turn 本分支实际扣费=1.25 credits" in content


def test_runtime_header_without_identity_does_not_invent_one() -> None:
    content = RuntimeHeader("br_1", 1, "cap", 10, 0.0, 1, 0).message()["content"]

    assert "本分支能力：cap" in content
    assert "自委派时" not in content
    assert "可调用 Procedure" not in content
    assert "实际扣费" not in content


def test_runtime_header_reports_an_empty_allowlist_explicitly() -> None:
    content = RuntimeHeader(
        "br_1", 1, "cap", 10, 0.0, 1, 0, agent_id="a.b", allowed_procedures=()
    ).message()["content"]

    assert "可调用 Procedure：无" in content


# --- assignment：我要做什么 / 以什么角色做 ---------------------------------


def test_assignment_section_carries_role_and_fork_semantics() -> None:
    section = render_assignment_section(
        character_prompt="你负责提供第二意见。",
        assignment="复核根分支关于成本的结论",
    )

    assert "【本分支任务】" in section
    assert "你负责提供第二意见。" in section
    assert "复核根分支关于成本的结论" in section
    assert "不要重做" in section
    # 分叉语义：继承完整历史，兄弟分支不可见。
    assert "完整对话历史" in section
    assert "兄弟分支" in section
    # 身份/额度/allowlist 由同一个 runtime 块的其它行给出，这里不重复。
    assert "credits" not in section
    assert "可调用 Procedure" not in section


def test_root_assignment_tells_the_root_it_is_the_coordinator() -> None:
    section = root_assignment_section()

    assert "你是本轮的起始协调者" in section
    assert "委派" in section


def test_runtime_block_merges_identity_task_and_status() -> None:
    """身份、任务、状态必须在同一个块里，且明确「只有最后一个块有效」。"""

    content = RuntimeHeader(
        "br_1",
        2,
        "证据核验",
        90,
        6.0,
        1,
        0,
        agent_id="builtin.evidence_verifier",
        agent_display_name="证据核验员",
        allowed_procedures=("builtin.web_search",),
        assignment=render_assignment_section(character_prompt="你负责证据核验。", assignment="核查证据"),
    ).message()["content"]

    assert content.startswith("[LRS runtime]")
    assert "只有最后一个" in content
    assert "也就是本块" not in content
    assert "已过期" in content
    assert "`builtin.evidence_verifier`" in content
    assert "【本分支任务】" in content
    assert "你负责证据核验。" in content
    assert "核查证据" in content
    assert "可调用 Procedure：" in content
    assert "branch=br_1; turn=2" in content


def test_manager_lifts_the_bare_assignment_out_of_history() -> None:
    agent_catalog, procedure_catalog = _bundled_snapshot()
    snapshot = _Snapshot(agent_catalog, procedure_catalog)
    manager = object.__new__(ResearchManager)
    inherited = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "formalized task"},
        {"role": "assistant", "content": "父分支的 report"},
    ]
    messages, card = manager._split_role_assignment(
        [*inherited, {"role": "user", "content": "assignment: 核查证据"}],
        snapshot=snapshot,
        agent_id="builtin.evidence_verifier",
        assignment="核查证据",
    )

    # 继承的历史逐字节等于父分支已发送的内容：assignment 不再占一条消息。
    assert messages == inherited
    assert BUNDLED_CHARACTER_PROMPTS["builtin.evidence_verifier"] in card
    assert "核查证据" in card


@pytest.mark.asyncio
async def test_child_history_is_an_exact_prefix_of_what_the_parent_sent(runtime_harness) -> None:
    """append-only：子分支继承的历史就是父分支发过的内容，不增不改。"""

    harness = runtime_harness
    await harness.start("调查一个问题", credits=100.0, time_budget=120)
    await harness.formalize("正式任务：调查一个问题")
    parent_messages = list(harness.manager._branches[harness.task_id][
        (await harness.manager.status(harness.task_id))["active_leaves"][0]["branch_id"]
    ]["messages"])
    leaves = await harness.root_allocates_real({"br_a": 20.0})

    assert "br_a" in leaves
    branch = harness.manager._branches[harness.task_id]["br_a"]
    child_messages = branch["messages"]

    # 角色卡进了 runtime 块，不再是一条独立消息。
    assert not any(str(item.get("content", "")).startswith("【子任务分配】") for item in child_messages)
    assert "agent.br_a 的角色与工作偏好" in branch["assignment"]
    assert "work-br_a" in branch["assignment"]

    # 子分支继承父分支的完整历史（父发过的每条消息都在，且顺序不变）。
    assert child_messages[: len(parent_messages)] == parent_messages
    system_messages = [item for item in child_messages if item["role"] == "system"]
    assert len(system_messages) == 1
    assert str(child_messages[1]["content"]) == harness.formalized_task.text
    # cache lineage 估算用的前缀就是全部继承历史。
    assert tuple(branch["inherited_messages"]) == tuple(dict(item) for item in child_messages)


def test_branch_task_survives_compaction() -> None:
    """compact 会重写可变历史，任务分配必须不在被重写的那部分里。

    分配随每 turn 的 runtime 块下发，而 compact 只保留 system + 正式任务 + 摘要，
    所以被压缩过的分支下一 turn 仍然知道自己是谁、要做什么。
    """

    from lunagentic_research_swarm.procedures.executor import ProcedureExecutor

    assignment = render_assignment_section(character_prompt="你负责证据核验。", assignment="核查证据")
    history = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "formalized task"},
        {
            "role": "user",
            "content": RuntimeHeader(
                "br_1", 1, "核验", 60, 5.0, 1, 0, agent_id="builtin.evidence_verifier", assignment=assignment
            ).message()["content"],
        },
        {"role": "assistant", "content": "一些工作"},
    ]

    compacted = ProcedureExecutor._rewrite_compacted_messages(history, "压缩摘要")

    # 旧块（连同其中的任务）被压掉了——所以下一 turn 必须重新下发，而不是指望它还在历史里。
    assert not any(str(item.get("content", "")).startswith("[LRS runtime]") for item in compacted)
    next_block = RuntimeHeader(
        "br_1", 2, "核验", 50, 4.0, 1, 0, agent_id="builtin.evidence_verifier", assignment=assignment
    ).message()["content"]
    assert "核查证据" in next_block
    assert "你负责证据核验。" in next_block


def test_manager_leaves_non_planner_messages_untouched() -> None:
    agent_catalog, procedure_catalog = _bundled_snapshot()
    manager = object.__new__(ResearchManager)
    original = [{"role": "user", "content": "自定义 payload"}]

    messages, card = manager._split_role_assignment(
        list(original),
        snapshot=_Snapshot(agent_catalog, procedure_catalog),
        agent_id="builtin.debater",
        assignment="核查证据",
    )

    assert messages == original
    assert card == ""


# --- system：共享目录不得泄漏角色，也不得漏掉工作方式 ----------------------


def test_system_prefix_is_identical_for_every_agent_and_hides_roles() -> None:
    agent_catalog, procedure_catalog = _bundled_snapshot()
    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="麦麦深度调查组",
        bot_profile={"nickname": "麦麦"},
        agent_catalog=agent_catalog.entries,
        procedure_catalog=procedure_catalog.entries,
        pricing={"models": {"m1": {"price_in": 1.0, "price_out": 2.0, "cache": False}}},
    )
    system = builder.system_message_for_protocol("json_envelope")

    for role in BUNDLED_CHARACTER_PROMPTS.values():
        assert role not in system
    # 身份/可调用 ID 的指路牌必须在 system 里，否则受限智能体不知道去哪儿看。
    assert "[LRS runtime]" in system
    assert "可调用 Procedure" in system
    assert "调用限制" in system or "全体可调用" in system
    # 历史里会有多个 runtime 块，system 必须说明以最后一个为准。
    assert "永远以最后一个" in system
    assert "也就是本次调用末尾的那个" not in system
    assert "只追加" in system
    assert "`report`" in system and "最终报告" in system
    assert "报告截止前的剩余秒数" in system or "report_seconds_remaining" in system
    assert "builtin.contractor" in system  # 委派 vs 承包商的取舍
    # 价格必须是能用来估算的数字，而不是原始 JSON dump。
    assert "每 100 万 token" in system
    assert "price_in" not in system


@pytest.mark.parametrize("protocol", ["json_envelope", "native_tools"])
def test_prompt_states_delegation_is_a_fork_not_a_call(protocol: str) -> None:
    """委派是分叉：无返回值、父分支退休、兄弟互相不可见。

    没有这段说明，模型会把两条委派当成会依次返回结果的子程序调用，于是把有先后
    依赖的工作拆成兄弟分支，而后者只能基于分叉时的旧信息各自瞎猜。
    """

    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="swarm",
        bot_profile={},
        agent_catalog={},
        procedure_catalog={},
        pricing={},
    )
    system = builder.system_message_for_protocol(protocol)

    assert "分叉" in system
    assert "不会**回到你这里" in system or "不会回到你这里" in system
    assert "退休" in system
    assert "互相看不见" in system
    assert "checkpoint" in system  # 兄弟摘要只在报告边界互通
    assert "兄弟委派" in system
    # 承包商是唯一有返回值的机制，必须与委派对比着讲。
    assert "builtin.contractor" in system
    assert "返回值" in system


@pytest.mark.parametrize("protocol", ["json_envelope", "native_tools"])
def test_protocol_examples_never_teach_a_concrete_invalid_call(protocol: str) -> None:
    """示例里的具体调用一旦不满足真实 schema，模型就会照抄出 invalid_arguments。"""

    builder = StablePromptBuilder(
        formalized_task=FormalizedTask.create("任务"),
        swarm_identity="swarm",
        bot_profile={},
        agent_catalog={},
        procedure_catalog={},
        pricing={},
    )
    section = builder.system_message_for_protocol(protocol)

    # core.terminate 是唯一无参数、永远存在的 Procedure，可以安全示范。
    concrete_ids = [line for line in section.splitlines() if '"procedure_id":"builtin.' in line]
    assert not concrete_ids, concrete_ids
    assert "占位符" in section
    assert "required" in section and "enum" in section
