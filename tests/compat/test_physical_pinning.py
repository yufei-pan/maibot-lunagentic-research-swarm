from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maibot_sdk.capabilities.llm import LLMCapability

from lunagentic_research_swarm.llm.physical_pinning import PINNED_MODEL_HOST_TASK, PhysicalPinningAdapter


HOST_RESULT = {
    "success": True,
    "response": "ok",
    "model_name": "physical",
    "prompt_tokens": 9,
    "completion_tokens": 3,
    "prompt_cache_hit_tokens": 4,
    "prompt_cache_miss_tokens": 5,
}


class SdkLikeLLM:
    """与 SDK ≥ 2.8.1 ``LLMCapability`` 相同的关键字签名。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate(
        self,
        prompt: Any,
        model: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        task_name: str | None = None,
        model_name: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "generate",
                {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "task_name": task_name,
                    "model_name": model_name,
                    **kwargs,
                },
            )
        )
        return dict(HOST_RESULT)

    async def generate_with_tools(
        self,
        prompt: Any,
        tools: list[dict[str, Any]],
        model: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        task_name: str | None = None,
        model_name: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "generate_with_tools",
                {
                    "prompt": prompt,
                    "tools": tools,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "task_name": task_name,
                    "model_name": model_name,
                    **kwargs,
                },
            )
        )
        return dict(HOST_RESULT)


class LegacyLLM:
    """SDK < 2.8.1：没有 task_name / model_name，也不接受任意关键字。"""

    async def generate(self, prompt: Any, model: str = "", temperature: float | None = None) -> Any:
        raise AssertionError("不支持直选时不应调用")

    async def generate_with_tools(self, prompt: Any, tools: Any, model: str = "") -> Any:
        raise AssertionError("不支持直选时不应调用")


@pytest.mark.asyncio
async def test_physical_pinning_routes_through_public_capability_with_explicit_model_name() -> None:
    llm = SdkLikeLLM()
    adapter = PhysicalPinningAdapter(SimpleNamespace(llm=llm))
    result = await adapter.generate(
        physical_name="physical",
        prompt="hello",
        tools=None,
        temperature=0.6,
        max_tokens=4096,
        rpc_timeout_ms=600_000,
    )

    assert llm.calls == [
        (
            "generate",
            {
                "prompt": "hello",
                "temperature": 0.6,
                "max_tokens": 4096,
                "task_name": PINNED_MODEL_HOST_TASK,
                "model_name": "physical",
                "rpc_timeout_ms": 600_000,
            },
        )
    ]
    assert result == HOST_RESULT


@pytest.mark.asyncio
async def test_physical_pinning_leaves_unset_max_tokens_to_host_model_config() -> None:
    llm = SdkLikeLLM()
    await PhysicalPinningAdapter(SimpleNamespace(llm=llm)).generate(
        physical_name="physical", prompt="hello", tools=None, temperature=None, max_tokens=None
    )

    kwargs = llm.calls[0][1]
    assert kwargs["max_tokens"] is None
    assert kwargs["temperature"] is None
    assert "rpc_timeout_ms" not in kwargs


@pytest.mark.asyncio
async def test_physical_pinning_keeps_message_list_and_tools_on_tool_route() -> None:
    llm = SdkLikeLLM()
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    tools = [{"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}}]
    result = await PhysicalPinningAdapter(SimpleNamespace(llm=llm)).generate(
        physical_name="physical", prompt=messages, tools=tools, temperature=None, max_tokens=None
    )

    assert [name for name, _ in llm.calls] == ["generate_with_tools"]
    kwargs = llm.calls[0][1]
    assert kwargs["prompt"] == messages
    assert kwargs["tools"] == tools
    assert kwargs["model_name"] == "physical"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_physical_pinning_rejects_legacy_sdk_without_attempting_alternate_route() -> None:
    result = await PhysicalPinningAdapter(SimpleNamespace(llm=LegacyLLM())).generate(
        physical_name="physical", prompt="hello", tools=None, temperature=None, max_tokens=None
    )
    assert result["success"] is False
    assert result["error"]["code"] == "physical_pinning_unsupported"


def test_physical_pinning_health_checks_both_prompt_routes() -> None:
    status = PhysicalPinningAdapter(SimpleNamespace(llm=SdkLikeLLM())).check_compatibility()
    assert status.available is True
    assert status.error_code is None

    half = SdkLikeLLM()
    half.generate_with_tools = LegacyLLM().generate_with_tools  # type: ignore[method-assign]
    status = PhysicalPinningAdapter(SimpleNamespace(llm=half)).check_compatibility()
    assert status.available is False
    assert status.error_code == "physical_pinning_unsupported"

    assert PhysicalPinningAdapter(None).check_compatibility().available is False


@pytest.mark.asyncio
async def test_physical_pinning_contract_with_real_sdk_llm_capability_payload() -> None:
    """真实 SDK 必须把 task_name / model_name 原样放进 cap.call payload，Host 才会直选模型。"""

    payloads: list[tuple[str, dict[str, Any]]] = []

    class RecordingContext:
        async def call_capability(self, capability: str, **payload: Any) -> dict[str, Any]:
            payloads.append((capability, payload))
            return dict(HOST_RESULT)

    ctx = SimpleNamespace()
    ctx.llm = LLMCapability(RecordingContext())  # type: ignore[arg-type]
    adapter = PhysicalPinningAdapter(ctx)
    assert adapter.check_compatibility().available is True

    await adapter.generate(
        physical_name="physical",
        prompt=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "submit"}}],
        temperature=0.2,
        max_tokens=None,
        rpc_timeout_ms=600_000,
    )

    capability, payload = payloads[0]
    assert capability == "llm.generate_with_tools"
    assert payload["task_name"] == PINNED_MODEL_HOST_TASK
    assert payload["model_name"] == "physical"
    assert payload["temperature"] == 0.2
    assert "max_tokens" not in payload
    assert payload["rpc_timeout_ms"] == 600_000
