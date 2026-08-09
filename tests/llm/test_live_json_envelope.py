"""真实 LLM：验证 json_envelope 协议段能否让模型输出可解析 envelope。

默认 skip。填写仓库根 `.debug_api_call_credentials`（见 `.example`）后运行：

    pytest tests/llm/test_live_json_envelope.py -v -m live_llm
"""

from __future__ import annotations

import pytest

from lunagentic_research_swarm.llm.protocol import parse_json_envelope
from lunagentic_research_swarm.models import FormalizedTask
from lunagentic_research_swarm.runtime.context import RuntimeHeader, StablePromptBuilder
from live_llm import chat_completion, credentials_available, load_live_llm_credentials

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(not credentials_available(), reason="未配置可用的 .debug_api_call_credentials"),
]


@pytest.mark.asyncio
async def test_live_model_emits_parseable_json_envelope_with_protocol_section() -> None:
    creds = load_live_llm_credentials()
    formalized = FormalizedTask.create(
        "# 正式任务描述：最小协议自测\n\n"
        "## 目标\n用一次 turn 产出可解析的 swarm JSON envelope，不要写 Markdown 报告。\n\n"
        "## 成功标准\n本 turn 最终输出恰好一个含 report/procedures/delegations 的 JSON object。\n"
    )
    builder = StablePromptBuilder(
        formalized_task=formalized,
        swarm_identity="麦麦深度调查组 (Lunagentic Research Swarm)",
        bot_profile={
            "nickname": "测试",
            "personality": "严谨",
            "behavior_style": "按协议输出",
            "reply_style": "JSON only",
        },
        agent_catalog={
            "builtin.quick_thinker": {
                "definition": {
                    "agent_id": "builtin.quick_thinker",
                    "protocol": "json_envelope",
                    "model_selector": "task:utils",
                }
            }
        },
        procedure_catalog={
            "core.terminate": {"procedure_id": "core.terminate", "description": "终结本分支"},
            "builtin.web_search": {"procedure_id": "builtin.web_search", "description": "网页搜索"},
        },
        pricing={"fingerprint": "live-test"},
    )
    messages = builder.messages_for_call(
        builder.root_context(coordinator="builtin.quick_thinker"),
        RuntimeHeader(
            "br_live_test",
            1,
            "快速拆分问题",
            300,
            100.0,
            1,
            0,
            protocol="json_envelope",
        ),
        protocol="json_envelope",
    )
    assert "LRS 输出协议（json_envelope）" in messages[0]["content"]
    assert "仅输出一个 JSON object" in messages[-1]["content"]

    result = await chat_completion(creds, messages)
    assert result["success"]
    raw = str(result["response"])
    envelope = parse_json_envelope(raw)
    assert isinstance(envelope.report, str)
    assert isinstance(envelope.procedures, list)
    assert isinstance(envelope.delegations, list)
