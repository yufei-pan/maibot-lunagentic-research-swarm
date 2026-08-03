from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

from lunagentic_research_swarm.llm.gateway import (
    GenerationRequest,
    HostModelSnapshotReader,
    LLMGateway,
)
from lunagentic_research_swarm.llm.tokens import (
    estimate_child_tokens,
    estimate_prompt_tokens,
)


FULL_HOST_RESULT = {
    "success": True,
    "response": "完成",
    "reasoning": "不得保留",
    "model": "actual-model",
    "model_name": "actual-model",
    "tool_calls": [
        {
            "id": "call-1",
            "function": {"name": "submit_swarm_turn", "arguments": {"report": "ok"}},
        }
    ],
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_cache_hit_tokens": 30,
    "prompt_cache_miss_tokens": 70,
    "error": None,
}


class FakeLLM:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = dict(result or FULL_HOST_RESULT)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("generate", kwargs))
        return dict(self.result)

    async def generate_with_tools(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("generate_with_tools", kwargs))
        return dict(self.result)


class FakeContext:
    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm


class FakePinning:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = dict(result or FULL_HOST_RESULT)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return dict(self.result)


@pytest.mark.asyncio
async def test_task_selector_routes_plain_and_native_tool_calls_through_public_capability() -> None:
    public = FakeLLM()
    pinned = FakePinning()
    gateway = LLMGateway(FakeContext(public), physical_pinning=pinned)

    plain = await gateway.generate(
        GenerationRequest(selector="task:utils", messages="hello", temperature=0.4, max_tokens=0)
    )
    tools = [{"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}}]
    native = await gateway.generate(
        GenerationRequest(selector="task:planner", messages=[{"role": "user", "content": "go"}], tools=tools)
    )

    assert [name for name, _ in public.calls] == ["generate", "generate_with_tools"]
    assert public.calls[0][1] == {"prompt": "hello", "model": "utils", "temperature": 0.4, "max_tokens": None}
    assert public.calls[1][1] == {
        "prompt": [{"role": "user", "content": "go"}],
        "tools": tools,
        "model": "planner",
        "temperature": None,
        "max_tokens": None,
    }
    assert not pinned.calls
    assert plain.success and native.success


@pytest.mark.asyncio
async def test_model_selector_never_falls_back_to_public_task_selector() -> None:
    public = FakeLLM()
    pinned = FakePinning(
        {"success": False, "response": "", "error": {"code": "physical_pinning_unsupported", "message": "不兼容"}}
    )
    gateway = LLMGateway(FakeContext(public), physical_pinning=pinned)
    result = await gateway.generate(GenerationRequest(selector="model:exact", messages="hello", max_tokens=4096))

    assert result.success is False
    assert result.error is not None and result.error.code == "physical_pinning_unsupported"
    assert result.usage is None
    assert not public.calls
    assert pinned.calls == [
        {
            "physical_name": "exact",
            "prompt": "hello",
            "tools": None,
            "temperature": None,
            "max_tokens": 4096,
        }
    ]


@pytest.mark.asyncio
async def test_gateway_normalizes_full_host_shape_and_drops_reasoning() -> None:
    gateway = LLMGateway(FakeContext(FakeLLM()), physical_pinning=FakePinning())
    result = await gateway.generate(GenerationRequest(selector="task:utils", messages="hello"))

    assert result.response == "完成"
    assert result.model_name == "actual-model"
    assert result.usage.prompt_tokens == 100
    assert result.usage.cache_hit_tokens == 30
    assert result.usage.cache_miss_tokens == 70
    assert result.tool_calls == FULL_HOST_RESULT["tool_calls"]
    assert result.duration >= 0
    assert "reasoning" not in asdict(result)
    with pytest.raises(TypeError):
        type(result)(**asdict(result), reasoning="forbidden")


@pytest.mark.asyncio
async def test_gateway_rejects_invalid_actual_usage_as_structured_failure() -> None:
    invalid = dict(FULL_HOST_RESULT, prompt_tokens=10, prompt_cache_hit_tokens=8, prompt_cache_miss_tokens=4)
    gateway = LLMGateway(FakeContext(FakeLLM(invalid)), physical_pinning=FakePinning())
    result = await gateway.generate(GenerationRequest(selector="task:utils", messages="hello"))
    assert not result.success
    assert result.error is not None and result.error.code == "invalid_usage"


@pytest.mark.asyncio
async def test_failed_generation_preserves_reported_usage_but_absent_usage_stays_none() -> None:
    reported = {
        "success": False,
        "response": "",
        "error": "provider failed after usage",
        "model_name": "actual-model",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_cache_hit_tokens": 3,
        "prompt_cache_miss_tokens": 7,
    }
    with_usage = await LLMGateway(FakeContext(FakeLLM(reported)), physical_pinning=FakePinning()).generate(
        GenerationRequest(selector="task:utils", messages="hello")
    )
    assert with_usage.success is False
    assert with_usage.usage is not None
    assert with_usage.usage.prompt_tokens == 10
    assert with_usage.model_name == "actual-model"

    absent = {"success": False, "response": "", "error": "provider failed"}
    without_usage = await LLMGateway(FakeContext(FakeLLM(absent)), physical_pinning=FakePinning()).generate(
        GenerationRequest(selector="task:utils", messages="hello")
    )
    assert without_usage.usage is None


def test_generation_request_normalizes_zero_and_rejects_out_of_range_max_tokens() -> None:
    assert GenerationRequest(selector="task:utils", messages="x", max_tokens=0).max_tokens is None
    assert GenerationRequest(selector="task:utils", messages="x", max_tokens=65536).max_tokens == 65536
    for invalid in (-1, 65537, True):
        with pytest.raises(ValueError, match="max_tokens"):
            GenerationRequest(selector="task:utils", messages="x", max_tokens=invalid)


def test_canonical_token_estimator_includes_tools_and_marks_estimated() -> None:
    messages = [{"role": "user", "content": "月亮"}]
    without_tools = estimate_prompt_tokens(messages)
    with_tools = estimate_prompt_tokens(messages, [{"name": "search", "parameters": {"type": "object"}}])
    assert without_tools.source == "estimated"
    assert without_tools.prompt_tokens == 11
    assert without_tools.cache_hit_tokens == 0
    assert without_tools.cache_miss_tokens == without_tools.prompt_tokens
    assert with_tools.prompt_tokens > without_tools.prompt_tokens


def test_child_cache_estimate_requires_same_actual_model_cache_and_byte_exact_prefix() -> None:
    inherited = [{"role": "system", "content": "stable"}, {"role": "user", "content": "task"}]
    appended = {"role": "user", "content": "assignment/runtime/procedure"}
    child = [*inherited, appended]

    proven = estimate_child_tokens(
        child,
        inherited_messages=inherited,
        parent_actual_model_name="m",
        estimated_model_name="m",
        cache_enabled=True,
    )
    assert 0 < proven.cache_hit_tokens < proven.prompt_tokens
    assert proven.cache_miss_tokens == proven.prompt_tokens - proven.cache_hit_tokens

    for kwargs in (
        {"parent_actual_model_name": "other", "estimated_model_name": "m", "cache_enabled": True},
        {"parent_actual_model_name": "m", "estimated_model_name": "m", "cache_enabled": False},
    ):
        unproven = estimate_child_tokens(child, inherited_messages=inherited, **kwargs)
        assert unproven.cache_hit_tokens == 0
        assert unproven.cache_miss_tokens == unproven.prompt_tokens

    changed = [{"role": "system", "content": "changed"}, *child[1:]]
    unproven = estimate_child_tokens(
        changed,
        inherited_messages=inherited,
        parent_actual_model_name="m",
        estimated_model_name="m",
        cache_enabled=True,
    )
    assert unproven.cache_hit_tokens == 0

    empty_prefix = estimate_child_tokens(
        child,
        inherited_messages=[],
        parent_actual_model_name="m",
        estimated_model_name="m",
        cache_enabled=True,
    )
    assert empty_prefix.cache_hit_tokens == 0


def test_host_snapshot_reader_reads_once_and_exposes_health_without_retaining_secrets() -> None:
    calls = 0

    def loader() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "models": [{"name": "m", "price_in": 1, "api_key": "secret"}],
            "model_task_config": {"utils": {"model_list": ["m"]}},
            "api_providers": [{"api_key": "secret"}],
        }

    reader = HostModelSnapshotReader(loader=loader)
    first = reader.read()
    second = reader.read()
    assert calls == 1
    assert first is second
    assert reader.health_issue is None
    assert "secret" not in repr(reader)


def test_host_snapshot_reader_marks_model_dump_or_validation_failure_unavailable() -> None:
    def broken() -> dict[str, Any]:
        raise RuntimeError("sensitive dump failure")

    reader = HostModelSnapshotReader(loader=broken)
    assert reader.read() is None
    assert reader.health_issue == "host_model_snapshot_unavailable"
    assert "sensitive" not in repr(reader)
