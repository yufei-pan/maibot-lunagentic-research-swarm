from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.llm.physical_pinning import PhysicalPinningAdapter


def _install_host_fakes(monkeypatch: pytest.MonkeyPatch, orchestrator_type: type) -> list[dict[str, Any]]:
    task_configs: list[dict[str, Any]] = []

    class TaskConfig:
        def __init__(self, **kwargs: Any) -> None:
            task_configs.append(kwargs)
            self.model_list = kwargs["model_list"]

    modules = {
        "src": ModuleType("src"),
        "src.config": ModuleType("src.config"),
        "src.config.model_configs": ModuleType("src.config.model_configs"),
        "src.llm_models": ModuleType("src.llm_models"),
        "src.llm_models.utils_model": ModuleType("src.llm_models.utils_model"),
    }
    modules["src.config.model_configs"].TaskConfig = TaskConfig  # type: ignore[attr-defined]
    modules["src.llm_models.utils_model"].LLMOrchestrator = orchestrator_type  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return task_configs


@pytest.mark.asyncio
async def test_physical_pinning_uses_synthetic_single_model_and_host_max_token_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class LLMOrchestrator:
        def __init__(self, task_name: str, request_type: str = "", session_id: str = "") -> None:
            calls.append({"init_task_name": task_name, "request_type": request_type, "session_id": session_id})
            self.model_for_task = self._get_task_config_or_raise()

        def _get_task_config_or_raise(self) -> Any:
            raise AssertionError("必须由 pinned subclass 覆盖")

        async def generate_response_async(
            self,
            prompt: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            model_name: str | None = None,
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            calls.append(
                {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model_name": model_name,
                    "tools": tools,
                    **kwargs,
                }
            )
            return SimpleNamespace(
                response="ok",
                reasoning="secret",
                model_name="physical",
                tool_calls=None,
                prompt_tokens=9,
                completion_tokens=3,
                total_tokens=12,
                prompt_cache_hit_tokens=4,
                prompt_cache_miss_tokens=5,
            )

    task_configs = _install_host_fakes(monkeypatch, LLMOrchestrator)
    adapter = PhysicalPinningAdapter()
    result = await adapter.generate(
        physical_name="physical",
        prompt="hello",
        tools=None,
        temperature=0.6,
        max_tokens=4096,
    )

    assert task_configs == [
        {
            "model_list": ["physical"],
            "temperature": 0.6,
            "max_tokens": 4096,
            "selection_strategy": "random",
            "slow_threshold": 30.0,
        }
    ]
    assert calls[0]["request_type"] == "plugin.lunagentic_research_swarm"
    assert calls[1]["max_tokens"] is None
    assert calls[1]["model_name"] is None
    assert result["success"] is True
    assert result["prompt_cache_hit_tokens"] == 4
    assert "reasoning" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [0, None])
async def test_physical_pinning_zero_max_tokens_uses_65536_synthetic_fallback(
    monkeypatch: pytest.MonkeyPatch,
    max_tokens: int | None,
) -> None:
    class LLMOrchestrator:
        def __init__(self, task_name: str, request_type: str = "", session_id: str = "") -> None:
            self.model_for_task = self._get_task_config_or_raise()

        def _get_task_config_or_raise(self) -> Any:
            raise AssertionError

        async def generate_response_async(
            self,
            prompt: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            model_name: str | None = None,
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            assert max_tokens is None
            return SimpleNamespace(response="ok", model_name="physical")

    task_configs = _install_host_fakes(monkeypatch, LLMOrchestrator)
    result = await PhysicalPinningAdapter().generate(
        physical_name="physical", prompt="hello", tools=None, temperature=None, max_tokens=max_tokens
    )
    assert task_configs[0]["max_tokens"] == 65536
    assert result["success"] is True


@pytest.mark.asyncio
async def test_physical_signature_incompatibility_is_explicit_and_never_attempts_alternate_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompatibleOrchestrator:
        def __init__(self, renamed_task: str) -> None:
            raise AssertionError("签名检查后不应实例化")

        async def generate_response_async(self, renamed_prompt: str) -> Any:
            raise AssertionError("签名检查后不应调用")

    _install_host_fakes(monkeypatch, IncompatibleOrchestrator)
    result = await PhysicalPinningAdapter().generate(
        physical_name="physical", prompt="hello", tools=None, temperature=None, max_tokens=None
    )
    assert result["success"] is False
    assert result["error"]["code"] == "physical_pinning_unsupported"


@pytest.mark.asyncio
async def test_physical_pinning_preserves_message_list_via_host_message_factory_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, Any]] = []

    class LLMOrchestrator:
        def __init__(self, task_name: str, request_type: str = "", session_id: str = "") -> None:
            self.model_for_task = self._get_task_config_or_raise()

        def _get_task_config_or_raise(self) -> Any:
            raise AssertionError

        async def generate_response_with_message_async(
            self,
            message_factory: Any,
            temperature: float | None = None,
            max_tokens: int | None = None,
            model_name: str | None = None,
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            received.extend(await message_factory(None))
            assert max_tokens is None
            return SimpleNamespace(response="ok", model_name="physical")

        async def generate_response_async(
            self,
            prompt: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            model_name: str | None = None,
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            raise AssertionError("消息列表不可扁平化")

    _install_host_fakes(monkeypatch, LLMOrchestrator)
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    result = await PhysicalPinningAdapter(message_factory_builder=lambda raw: lambda _: list(raw)).generate(
        physical_name="physical", prompt=messages, tools=[], temperature=None, max_tokens=None
    )
    assert received == messages
    assert result["success"] is True


def test_physical_pinning_health_checks_both_prompt_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    class CompatibleOrchestrator:
        def __init__(self, task_name: str, request_type: str = "", session_id: str = "") -> None:
            pass

        async def generate_response_async(
            self,
            prompt: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            return None

        async def generate_response_with_message_async(
            self,
            message_factory: Any,
            temperature: float | None = None,
            max_tokens: int | None = None,
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            return None

    _install_host_fakes(monkeypatch, CompatibleOrchestrator)
    status = PhysicalPinningAdapter().check_compatibility()
    assert status.available is True
    assert status.error_code is None

    class MissingMessageRoute(CompatibleOrchestrator):
        generate_response_with_message_async = None  # type: ignore[assignment]

    _install_host_fakes(monkeypatch, MissingMessageRoute)
    status = PhysicalPinningAdapter().check_compatibility()
    assert status.available is False
    assert status.error_code == "physical_pinning_unsupported"
