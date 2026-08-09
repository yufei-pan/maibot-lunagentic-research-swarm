# tests/integration/test_live_agent_identity.py
"""Live 端到端：真实九智能体目录 + 真实总结器 prompt + 真实 web_search。

与 ``test_live_thorough_live_tools`` 的区别：那条用例只跑一个 root 智能体和测试本地
总结器 prompt。这条把**仓库真正发货的东西**放进回路——冻结的 bundled 目录、
``prompts/zh-CN/*.txt`` 四个总结角色——然后检查真实模型收到的消息与它实际给出的
回答：智能体是否知道自己是谁、是否只调用了自己被允许的 Procedure、参数是否符合
schema、`report` 是否有实质内容。
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from live_llm import deep_judge, live_tools_available, load_live_llm_credentials

from lunagentic_research_swarm.models import ReportKind, TaskStatus

LIVE_OBJECTIVE = (
    "调研问题：加州高铁（California High-Speed Rail）Merced–Bakersfield 段"
    "当前规划的开通目标年份是什么？"
    "硬性要求：必须至少一次调用 builtin.web_search 取证（arguments 必须同时包含 engine 与 query，"
    "engine 只能取 schema enum 中列出的值）；"
    "并且必须至少把一个子任务委派给目录中的另一个专职智能体，由它复核或补充证据。"
    "最终结论要点名至少一处来源标题或域名。"
)

JUDGE_OBJECTIVE = (
    "回答加州高铁 Merced–Bakersfield 段的开通/服务目标年份，且结论应体现已使用网页检索（提及来源）。"
)

pytestmark = [
    pytest.mark.live_llm_live_tools,
    pytest.mark.skipif(not live_tools_available(), reason="未启用 web_search_enabled 或缺少 LLM 凭证"),
]

_HEADER_MARKER = "[LRS runtime]"
# 任务分配是 runtime 块内的一节，不再是独立消息。
_ASSIGNMENT_MARKER = "【本分支任务】"
_ROOT_MARKER = "起始协调者"
_IDENTITY_RE = re.compile(r"你是 `([a-z0-9][a-z0-9_.\-]*)`")
_BRANCH_RE = re.compile(r"branch=([^;]+); turn=(\d+)")


def _status_value(status: dict) -> str:
    raw = status.get("status")
    return raw.value if hasattr(raw, "value") else str(raw)


def _text(message: Any) -> str:
    return str((message or {}).get("content", "") or "")


def _agent_turns(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留普通智能体 turn（末条是 runtime 块）；总结器调用没有这个块。"""

    turns: list[dict[str, Any]] = []
    for exchange in exchanges:
        messages = list(exchange.get("messages") or ())
        if not messages or not _text(messages[-1]).startswith(_HEADER_MARKER):
            continue
        block = _text(messages[-1])
        identity = _IDENTITY_RE.search(block)
        index = block.find(_ASSIGNMENT_MARKER)
        assignment = block[index:] if index >= 0 else ""
        turns.append(
            {
                "agent_id": identity.group(1) if identity else "",
                "header": block,
                "assignment": assignment,
                "is_root": _ROOT_MARKER in assignment,
                "response": str(exchange.get("response") or ""),
                "messages": messages,
            }
        )
    return turns


_PROCEDURE_RESULT_MARKER = "【Procedure 结果 · "


def _procedure_errors(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """收集智能体实际看到的 Procedure 失败（去重）。

    直接读回灌进上下文的结果消息，这正是智能体据以决策的那份内容。
    """

    seen: dict[str, dict[str, Any]] = {}
    for turn in turns:
        for message in turn["messages"]:
            content = _text(message)
            if not content.startswith(_PROCEDURE_RESULT_MARKER):
                continue
            _, _, payload = content.partition("\n")
            try:
                summary = json.loads(payload)
            except (TypeError, ValueError):
                continue
            error = summary.get("error")
            if not isinstance(error, dict):
                continue
            record = {
                "procedure_id": str(summary.get("procedure_id") or ""),
                "code": str(error.get("code") or ""),
                "message": str(error.get("message") or "")[:200],
            }
            seen[json.dumps(record, sort_keys=True)] = record
    return list(seen.values())


def _envelope(response: str) -> dict[str, Any] | None:
    from lunagentic_research_swarm.llm.protocol import ProtocolError, parse_json_envelope

    try:
        return parse_json_envelope(response).model_dump(mode="python")
    except ProtocolError:
        return None


def _dump(artifact_dir, name: str, payload: Any) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_live_agents_know_their_identity_tools_and_task(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    harness.use_bundled_agents()
    harness.use_real_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_live_procedures(creds.web_search)
    # Zero pricing in the drain means credits never deplete, so the run has to be
    # bounded structurally.  Every agent call and every branch summary is a real
    # request against a local model, so the ceiling is set by the drain timeout, not
    # by what the prompts allow: 5 agent calls still yields root → delegated child.
    harness.configure_runtime_limits(max_agent_calls_per_task=5, max_delegations_per_turn=2)

    await harness.start(LIVE_OBJECTIVE, credits=150.0, time_budget=900)
    await harness.manager.wait_idle(harness.task_id)
    artifact_dir = tmp_path / "live_identity"
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=artifact_dir,
    )

    turns = _agent_turns(harness.llm.exchanges)
    llm_errors = [
        {"error": item["error"], "last_message": _text((item.get("messages") or [{}])[-1])[:400]}
        for item in harness.llm.exchanges
        if item.get("error")
    ]
    _dump(artifact_dir, "llm_errors.json", llm_errors)
    _dump(
        artifact_dir,
        "agent_turns.json",
        [
            {
                "agent_id": turn["agent_id"],
                "is_root": turn["is_root"],
                "header": turn["header"],
                "assignment": turn["assignment"],
                "response": turn["response"],
            }
            for turn in turns
        ],
    )

    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }
    assert turns, "expected at least one ordinary agent turn"

    # 1. 每个 turn 的 runtime 块都自足：身份 + 任务 + allowlist + 「以最后一个为准」。
    catalog_ids = {entry.definition.agent_id for entry in harness._shared_snapshot.agent_catalog.entries}
    for turn in turns:
        assert turn["agent_id"], f"runtime 块未给出 agent_id：{turn['header'][:200]}"
        assert turn["agent_id"] in catalog_ids, turn["agent_id"]
        assert "可调用 Procedure" in turn["header"]
        assert turn["assignment"], f"runtime 块缺少任务分配：{turn['header'][:200]}"
        assert "只有最后一个" in turn["header"]
        # 块永远是最后一条消息，且历史里只追加不改写。
        assert _text(turn["messages"][-1]).startswith(_HEADER_MARKER)

    # 2. 至少发生了一次向专职智能体的委派，且子分支拿到了自己的角色。
    delegated = [turn for turn in turns if not turn["is_root"]]
    assert delegated, "expected at least one delegated child turn (role card missing?)"
    from lunagentic_research_swarm.agents.bundled.prompts import BUNDLED_CHARACTER_PROMPTS

    for turn in delegated:
        expected_role = BUNDLED_CHARACTER_PROMPTS.get(turn["agent_id"])
        if expected_role is not None:
            assert expected_role in turn["assignment"], turn["agent_id"]

    # 3. append-only：子分支的请求必须以父分支最后一次请求为逐字节前缀。
    #    这正是 DeepSeek 「必须完整匹配某个 cache prefix unit」所要求的形状——
    #    一旦历史被改写，父分支那次请求的缓存单元就永远命不中。
    by_branch: dict[str, list[tuple[int, list[Any]]]] = {}
    for turn in turns:
        matched = _BRANCH_RE.search(turn["header"])
        if matched:
            by_branch.setdefault(matched.group(1), []).append((int(matched.group(2)), turn["messages"]))
    for items in by_branch.values():
        items.sort(key=lambda item: item[0])
    checked_prefixes = 0
    for branch_id, items in by_branch.items():
        # 同一分支内：后一 turn 以前一 turn 为前缀
        for (_t1, first), (_t2, second) in zip(items, items[1:]):
            assert [dict(x) for x in second][: len(first)] == [dict(x) for x in first], branch_id
            checked_prefixes += 1
        # 父子之间：子分支 id 形如 `<parent>:<n>`
        if ":" not in branch_id:
            continue
        parent_items = by_branch.get(branch_id.rsplit(":", 1)[0])
        if not parent_items:
            continue
        parent_last = parent_items[-1][1]
        child_first = items[0][1]
        assert [dict(x) for x in child_first][: len(parent_last)] == [dict(x) for x in parent_last], (
            f"{branch_id} 的首次请求没有以父分支最后一次请求为前缀（append-only 被破坏）"
        )
        checked_prefixes += 1
    assert checked_prefixes, "no parent→child or turn→turn prefix pair observed"

    # 4. 智能体实际给出的 envelope 必须可解析，且 id 来自目录、report 非空。
    procedure_ids: list[str] = []
    parsed = 0
    for turn in turns:
        envelope = _envelope(turn["response"])
        if envelope is None:
            continue  # 协议错误由 runtime 的纠正 turn 负责；这里只统计成功解析的
        parsed += 1
        assert str(envelope["report"]).strip() or envelope["procedures"] or envelope["delegations"]
        for request in envelope["procedures"]:
            procedure_ids.append(str(request["procedure_id"]))
        for edge in envelope["delegations"]:
            assert str(edge["agent_id"]) in catalog_ids, edge
    assert parsed, "no agent turn produced a parsable swarm envelope"

    # 5. 真实检索确实发生；智能体没有调用 allowlist 之外的 Procedure，也没有交出
    #    不满足 schema 的 arguments（这两类错误正是「不知道自己有什么」的症状）。
    assert harness.live_search_invokes >= 1, f"expected real web_search, got {harness.live_search_invokes}"
    procedure_errors = _procedure_errors(turns)
    _dump(artifact_dir, "procedure_errors.json", procedure_errors)
    fatal = [item for item in procedure_errors if item["code"] in {"procedure_not_allowed", "invalid_arguments"}]
    assert not fatal, fatal

    # 6. 最终报告：真实总结器 prompt 产出可用正文。
    assert harness.reports, "expected at least one report"
    final = harness.reports[-1]
    kind = getattr(final, "kind", None)
    if kind is not None:
        assert kind is ReportKind.FINAL, f"last report kind={kind!r}"
    final_text = str(getattr(final, "text", None) or getattr(final, "body", None) or final).strip()
    (artifact_dir / "final_report.txt").write_text(final_text, encoding="utf-8")
    assert final_text

    verdict = await deep_judge(creds, objective=JUDGE_OBJECTIVE, report=final_text, evidence="")
    _dump(artifact_dir, "judge.json", verdict)
    assert verdict["pass"] is True, verdict
