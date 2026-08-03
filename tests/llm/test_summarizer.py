from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from lunagentic_research_swarm.llm.gateway import GenerationError, GenerationRequest, GenerationResult
from lunagentic_research_swarm.llm.pricing import TokenUsage
from lunagentic_research_swarm.llm.summarizer import (
    BranchFinalizationRequest,
    CompactionRequest,
    FormalizationRequest,
    SummarizerService,
    SummaryResult,
    TaskFinalizationRequest,
)
from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask, ReportKind


USAGE = TokenUsage(
    prompt_tokens=20,
    completion_tokens=5,
    cache_hit_tokens=0,
    cache_miss_tokens=20,
    source="actual",
)


def _generation(
    response: str = "摘要完成",
    *,
    success: bool = True,
    error: GenerationError | None = None,
    usage: TokenUsage | None = USAGE,
) -> GenerationResult:
    return GenerationResult(
        response=response,
        tool_calls=[{"forbidden": "总结器不得接收或使用 native tools"}],
        model_name="actual-summary-model",
        usage=usage,
        success=success,
        error=error,
        duration=0.01,
    )


class CapturingGateway:
    """保留真实 GenerationRequest 边界，仅替代外部模型调用。"""

    def __init__(self, *results: GenerationResult) -> None:
        self.results = list(results)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("发生了未预期的隐藏重试")
        return self.results.pop(0)


def _service(gateway: CapturingGateway, *, max_tokens: int = 0) -> SummarizerService:
    return SummarizerService(
        gateway,
        selector="task:mid_memory",
        temperature=0.2,
        max_tokens=max_tokens,
    )


@pytest.mark.asyncio
async def test_formalizer_uses_a_fresh_system_plus_raw_context_and_chat_without_tools() -> None:
    """若 formalizer 复用 swarm prefix、重排原始 ingest 或发送 tools，本测试应失败。"""

    gateway = CapturingGateway(_generation("正式任务原文\n"), _generation("第二次正式任务"))
    service = _service(gateway)
    request = FormalizationRequest(
        raw_context="Maisaka 原始上下文",
        chat_messages=(
            {"role": "user", "content": "用户先问的问题"},
            {"role": "assistant", "content": "此前回答"},
            {"role": "user", "content": "用户最后补充"},
        ),
    )

    first = await service.formalize_task(request)
    first_messages = gateway.requests[0].messages
    second = await service.formalize_task(request)
    second_messages = gateway.requests[1].messages

    assert first.success and first.text.encode("utf-8") == "正式任务原文\n".encode("utf-8")
    assert second.success
    assert first_messages == second_messages
    assert first_messages is not second_messages
    assert isinstance(first_messages, list)
    assert [message["role"] for message in first_messages] == ["system", "user", "user", "assistant", "user"]
    assert first_messages[1:] == [
        {"role": "user", "content": "Maisaka 原始上下文"},
        {"role": "user", "content": "用户先问的问题"},
        {"role": "assistant", "content": "此前回答"},
        {"role": "user", "content": "用户最后补充"},
    ]
    assert "不研究" in first_messages[0]["content"]
    assert "不调用工具" in first_messages[0]["content"]
    assert all(item.tools is None for item in gateway.requests)
    assert all(item.max_tokens is None for item in gateway.requests)


@pytest.mark.asyncio
async def test_branch_finalizer_keeps_task_separate_and_summarizes_branch_evidence() -> None:
    """若分支总结器改写正式任务或遗漏可变 history，本测试应失败。"""

    task = FormalizedTask.create("逐字节保留的正式任务\n")
    history = (
        {"role": "assistant", "content": "发现证据甲"},
        {"role": "user", "content": "Procedure 返回证据乙"},
    )
    gateway = CapturingGateway(_generation("分支结论"))

    result = await _service(gateway, max_tokens=65536).finalize_branch(
        BranchFinalizationRequest(formalized_task=task, branch_history=history, checkpoint=True)
    )

    messages = gateway.requests[0].messages
    assert result == SummaryResult(True, "分支结论", "actual-summary-model", USAGE, None)
    assert isinstance(messages, list)
    assert messages[1] == {"role": "user", "content": task.text}
    assert "发现证据甲" in messages[2]["content"]
    assert "证据" in messages[0]["content"]
    assert "不确定性" in messages[0]["content"]
    assert "建议" in messages[0]["content"]
    assert "原文保留，不得改写" in messages[0]["content"]
    assert gateway.requests[0].tools is None
    assert gateway.requests[0].max_tokens == 65536


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "running_count", "expected_marker"),
    [(ReportKind.INTERMEDIATE, 3, "中间"), (ReportKind.FINAL, 0, "最终")],
)
async def test_task_finalizer_receives_kind_statistics_coverage_and_running_count(
    kind: ReportKind, running_count: int, expected_marker: str
) -> None:
    """若 task finalizer 把 checkpoint 当终局或让模型猜统计，本测试应失败。"""

    task = FormalizedTask.create("正式任务")
    gateway = CapturingGateway(_generation(f"{expected_marker}综合结论"))

    result = await _service(gateway).finalize_task(
        TaskFinalizationRequest(
            formalized_task=task,
            coverage_summaries=("分支一", "分支二"),
            report_kind=kind,
            statistics={"agent_calls": 7, "branches": 2},
            running_branch_count=running_count,
        )
    )

    messages = gateway.requests[0].messages
    assert result.success
    assert isinstance(messages, list)
    assert messages[1]["content"].encode("utf-8") == task.text.encode("utf-8")
    assert expected_marker in messages[2]["content"]
    assert f'"running_branch_count": {running_count}' in messages[2]["content"]
    assert '"agent_calls": 7' in messages[2]["content"]
    assert "分支一" in messages[2]["content"]
    assert "仍在运行" in messages[0]["content"]
    assert "检查点" in messages[0]["content"]
    assert gateway.requests[0].tools is None


@pytest.mark.asyncio
async def test_compactor_only_replaces_mutable_history_and_user_one_stays_byte_identical() -> None:
    """若 compact 输出能覆盖 BranchRuntime.task 或 User 1 字节，本测试应失败。"""

    task = FormalizedTask.create("原始正式任务：标点、空格 与换行。\n")
    runtime = BranchRuntime(
        branch_id="br_1",
        task=task,
        catalog_fingerprint="catalog_1",
        generation=1,
        messages=[{"role": "assistant", "content": "很长的可变历史"}],
        credits=1.0,
        depth=0,
    )
    gateway = CapturingGateway(_generation("完全不同的任务文字，也只能成为压缩历史"))

    result = await _service(gateway).compact_branch(
        CompactionRequest(formalized_task=task, branch_history=tuple(runtime.messages))
    )
    runtime.messages = [{"role": "assistant", "content": result.text}]
    rebuilt = runtime.build_prompt_messages()

    request_messages = gateway.requests[0].messages
    assert isinstance(request_messages, list)
    assert request_messages[1]["content"].encode("utf-8") == task.text.encode("utf-8")
    assert "很长的可变历史" in request_messages[2]["content"]
    assert "只压缩可变历史" in request_messages[0]["content"]
    assert "不复述" in request_messages[0]["content"]
    assert rebuilt[0]["content"].encode("utf-8") == task.text.encode("utf-8")
    assert rebuilt[1]["content"] == "完全不同的任务文字，也只能成为压缩历史"
    assert gateway.requests[0].tools is None


@pytest.mark.asyncio
async def test_empty_branch_history_fails_explicitly_without_call_or_retry() -> None:
    """若无内容分支被伪造成总结或触发隐藏 LLM fallback，本测试应失败。"""

    gateway = CapturingGateway()
    result = await _service(gateway).finalize_branch(
        BranchFinalizationRequest(formalized_task=FormalizedTask.create("正式任务"), branch_history=())
    )

    assert not result.success
    assert result.text == ""
    assert result.error is not None and result.error.code == "summary_input_empty"
    assert gateway.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["finalize_branch", "compact_branch"])
async def test_metadata_cannot_make_whitespace_history_semantically_nonempty(method_name: str) -> None:
    """若 name/timestamp 等元数据让空白 branch/compact 历史伪装成内容，本测试应失败。"""

    gateway = CapturingGateway()
    task = FormalizedTask.create("正式任务")
    history = ({"role": "assistant", "content": " \n", "name": "agent", "timestamp": "now"},)
    requests: dict[str, object] = {
        "finalize_branch": BranchFinalizationRequest(
            formalized_task=task,
            branch_history=history,
        ),
        "compact_branch": CompactionRequest(
            formalized_task=task,
            branch_history=history,
        ),
    }

    result = await getattr(_service(gateway), method_name)(requests[method_name])

    assert not result.success
    assert result.error is not None and result.error.code == "summary_input_empty"
    assert gateway.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["finalize_branch", "compact_branch"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"role": "assistant", "provider": "x"},
        {"role": "assistant", "unknown_metadata": {"nested": "value"}},
        {"role": "tool", "tool_calls": [{"name": "metadata_only"}]},
    ],
)
async def test_mapping_without_content_is_empty_regardless_of_unknown_metadata(
    method_name: str, metadata: dict[str, object]
) -> None:
    """若任意未知 metadata 值能让无 content 的消息伪装成语义内容，本测试应失败。"""

    gateway = CapturingGateway()
    task = FormalizedTask.create("正式任务")
    history = (metadata,)
    requests: dict[str, object] = {
        "finalize_branch": BranchFinalizationRequest(formalized_task=task, branch_history=history),
        "compact_branch": CompactionRequest(formalized_task=task, branch_history=history),
    }

    result = await getattr(_service(gateway), method_name)(requests[method_name])

    assert not result.success
    assert result.error is not None and result.error.code == "summary_input_empty"
    assert gateway.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["finalize_branch", "compact_branch"])
@pytest.mark.parametrize(
    "content",
    [
        "有效证据",
        [" ", {"content": ["", "有效 multipart 证据"]}],
    ],
)
async def test_nonblank_string_and_multipart_content_are_semantically_meaningful(
    method_name: str, content: object
) -> None:
    """若语义判定过度收紧并拒绝 branch/compact 的字符串或 multipart 正文，本测试应失败。"""

    gateway = CapturingGateway(_generation("有效摘要"))
    task = FormalizedTask.create("正式任务")
    history = ({"role": "assistant", "content": content, "provider": "x", "timestamp": "now"},)
    requests: dict[str, object] = {
        "finalize_branch": BranchFinalizationRequest(
            formalized_task=task,
            branch_history=history,
        ),
        "compact_branch": CompactionRequest(
            formalized_task=task,
            branch_history=history,
        ),
    }

    result = await getattr(_service(gateway), method_name)(requests[method_name])

    assert result.success and result.text == "有效摘要"
    assert len(gateway.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["formalize_task", "finalize_branch", "finalize_task", "compact_branch"])
async def test_each_role_rejects_empty_required_summary_without_hidden_retry(method_name: str) -> None:
    """若任一角色把空白 provider 输出当成功或自行二次调用，本测试应失败。"""

    gateway = CapturingGateway(_generation(" \n\t"))
    service = _service(gateway)
    task = FormalizedTask.create("正式任务")
    requests: dict[str, object] = {
        "formalize_task": FormalizationRequest(raw_context="原始请求", chat_messages=()),
        "finalize_branch": BranchFinalizationRequest(
            formalized_task=task,
            branch_history=({"role": "assistant", "content": "内容"},),
        ),
        "finalize_task": TaskFinalizationRequest(
            formalized_task=task,
            coverage_summaries=("覆盖内容",),
            report_kind=ReportKind.FINAL,
            statistics={},
            running_branch_count=0,
        ),
        "compact_branch": CompactionRequest(
            formalized_task=task,
            branch_history=({"role": "assistant", "content": "内容"},),
        ),
    }

    result = await getattr(service, method_name)(requests[method_name])

    assert not result.success
    assert result.text == ""
    assert result.model_name == "actual-summary-model"
    assert result.usage == USAGE
    assert result.error is not None and result.error.code == "summary_empty"
    assert len(gateway.requests) == 1


@pytest.mark.asyncio
async def test_generation_failure_preserves_error_model_and_usage_without_retry() -> None:
    """若总结器吞掉 gateway failure、usage 或偷偷重试，本测试应失败。"""

    error = GenerationError("provider_failed", "上游失败")
    gateway = CapturingGateway(_generation("", success=False, error=error))

    result = await _service(gateway).compact_branch(
        CompactionRequest(
            formalized_task=FormalizedTask.create("正式任务"),
            branch_history=({"role": "assistant", "content": "内容"},),
        )
    )

    assert result == SummaryResult(False, "", "actual-summary-model", USAGE, error)
    assert len(gateway.requests) == 1


def test_summary_result_has_no_reasoning_and_max_tokens_is_strict() -> None:
    """若 summary contract 泄露 reasoning 或越过输出上限，本测试应失败。"""

    summary = SummaryResult(True, "ok", "model", USAGE, None)
    assert "reasoning" not in asdict(summary)
    with pytest.raises(TypeError):
        SummaryResult(**asdict(summary), reasoning="forbidden")  # type: ignore[call-arg]

    gateway = CapturingGateway()
    for invalid in (-1, 65537, True):
        with pytest.raises(ValueError, match="max_tokens"):
            _service(gateway, max_tokens=invalid)  # type: ignore[arg-type]


def test_prompt_assets_are_chinese_role_specific_and_have_no_fifth_summarizer_role() -> None:
    """若角色 prompt 混用、遗漏或新增第五个总结角色，本测试应失败。"""

    prompt_root = Path(__file__).parents[2] / "lunagentic_research_swarm" / "prompts" / "zh-CN"
    expected = {
        "swarm_system.txt",
        "formalize_task.txt",
        "finalize_branch.txt",
        "finalize_task.txt",
        "compact_branch.txt",
    }

    assert {path.name for path in prompt_root.glob("*.txt")} == expected
    for name in expected:
        text = (prompt_root / name).read_text(encoding="utf-8")
        assert text.strip()
        assert any("\u4e00" <= character <= "\u9fff" for character in text)
