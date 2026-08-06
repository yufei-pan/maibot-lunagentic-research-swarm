"""builtin.contractor：定义默认值、目录注册、禁用覆写与 outsider 循环。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions
from lunagentic_research_swarm.config import ProcedureOverride
from lunagentic_research_swarm.extensions.contracts import ProcedureResult
from lunagentic_research_swarm.procedures.bundled.contractor import (
    CONTRACTOR_PROCEDURE_ID,
    ContractorDeps,
    contractor_procedure_definitions,
    make_contractor_handler,
    make_nested_procedure_invoker,
    run_contractor,
)
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

_PLUGIN_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DEFAULT = _PLUGIN_ROOT / "config.default.toml"
_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from fakes import FakeLLMGateway, FakeLLMResponse  # noqa: E402

_EXPECTED_BUNDLED_PROCEDURE_IDS = (
    "builtin.chat_streams",
    "builtin.message_recent",
    "builtin.message_by_id",
    "builtin.message_time_range",
    "builtin.person_lookup",
    "builtin.knowledge_search",
    "builtin.calculate",
    "builtin.statistics",
    "builtin.convert_units",
    "builtin.normalize_urls",
    "builtin.organize_provenance",
    "builtin.web_search",
    "builtin.past_cases",
    "builtin.contractor",
)


def test_contractor_definition_defaults() -> None:
    defs = contractor_procedure_definitions()
    assert len(defs) == 1
    d = defs[0]
    assert d.procedure_id == CONTRACTOR_PROCEDURE_ID
    assert d.timeout_seconds == 0.0
    assert d.enabled is True
    assert d.idempotent is False
    props = d.arguments_schema["properties"]
    assert "agent_id" in props
    assert "question" in props
    assert "temperature" in props
    assert "personality" in props
    assert "time_budget_seconds" in props
    assert set(d.arguments_schema["required"]) == {"agent_id", "question"}
    assert d.result_schema == {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    }


def test_bundled_provider_includes_contractor_when_described() -> None:
    payloads = BundledProcedureProvider(SimpleNamespace()).describe()
    ids = {item["procedure_id"] for item in payloads}
    assert CONTRACTOR_PROCEDURE_ID in ids
    contractor = next(item for item in payloads if item["procedure_id"] == CONTRACTOR_PROCEDURE_ID)
    assert contractor["timeout_seconds"] == 0.0
    assert contractor["enabled"] is True
    assert contractor["idempotent"] is False


@pytest.mark.asyncio
async def test_contractor_stub_returns_runtime_missing_without_deps() -> None:
    result = await BundledProcedureProvider(SimpleNamespace()).invoke(
        CONTRACTOR_PROCEDURE_ID,
        {"agent_id": "builtin.quick_thinker", "question": "1+1?"},
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.code == "contractor_runtime_missing"


def test_contractor_disabled_by_override_removed_from_snapshot() -> None:
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", contractor_procedure_definitions())
    enabled = registry.snapshot({})
    assert enabled.get(CONTRACTOR_PROCEDURE_ID) is not None

    disabled = registry.snapshot({CONTRACTOR_PROCEDURE_ID: ProcedureOverride(enabled=False)})
    assert disabled.get(CONTRACTOR_PROCEDURE_ID) is None


def test_config_default_toml_lists_all_bundled_procedure_toggles() -> None:
    text = _CONFIG_DEFAULT.read_text(encoding="utf-8")
    for procedure_id in _EXPECTED_BUNDLED_PROCEDURE_IDS:
        assert f'[procedures."{procedure_id}"]' in text
    assert "timeout_seconds = 0" in text
    contractor_block = text.split('[procedures."builtin.contractor"]', 1)[1]
    assert "enabled = true" in contractor_block.split("[", 1)[0]
    assert "timeout_seconds = 0" in contractor_block.split("[", 1)[0]


@dataclass
class _FakePrices:
    """Deterministic charge_actual for contractor metering tests."""

    per_turn: float = 0.5
    calls: list[dict[str, Any]] = field(default_factory=list)

    def charge_actual(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(credits=float(self.per_turn))


@dataclass
class ContractorHarness:
    llm: FakeLLMGateway
    prices: _FakePrices
    deps: ContractorDeps
    agents: dict[str, Any]

    @classmethod
    def create(cls) -> ContractorHarness:
        agents = {item.agent_id: item for item in bundled_agent_definitions()}
        llm = FakeLLMGateway()
        prices = _FakePrices()
        deps = ContractorDeps(
            llm=llm,
            prices=prices,
            resolve_agent=lambda agent_id: agents.get(agent_id),
            invoke_nested_procedure=None,
        )
        return cls(llm=llm, prices=prices, deps=deps, agents=agents)

    async def invoke(
        self,
        *,
        agent_id: str,
        question: str,
        caller_protocol: str = "json_envelope",
        credit_budget: float = 10.0,
        caller_agent_id: str = "builtin.quick_thinker",
        **arguments: Any,
    ) -> Any:
        handler = make_contractor_handler(self.deps)
        scoped = {
            "credit_budget": float(credit_budget),
            "caller_protocol": caller_protocol,
            "caller_agent_id": caller_agent_id,
            # Markers that must never leak into outsider messages:
            "formalized_task": "FORMALIZED_TASK_MARKER",
            "parent_transcript": "parent transcript marker",
        }
        return await handler(
            None,
            {"agent_id": agent_id, "question": question, **arguments},
            scoped_metadata=scoped,
        )


@pytest.fixture
def contractor_harness() -> ContractorHarness:
    return ContractorHarness.create()


@pytest.mark.asyncio
async def test_contractor_returns_explicit_json_return(contractor_harness: ContractorHarness) -> None:
    contractor_harness.llm.queue_json({"return": "答案"})
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="1+1?",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert result.success is True
    assert result.data["result"] == "答案"
    assert result.metadata["termination_reason"] == "returned"
    assert float(result.research_credits_charged) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_contractor_last_text_return_without_tool_call(contractor_harness: ContractorHarness) -> None:
    contractor_harness.llm.queue_text("仅正文结论")
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="总结",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert result.success is True
    assert "仅正文结论" in str(result.data)
    assert result.metadata["termination_reason"] == "returned"


@pytest.mark.asyncio
async def test_contractor_fresh_context_excludes_parent_task(contractor_harness: ContractorHarness) -> None:
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="旁路问题",
        caller_protocol="json_envelope",
        credit_budget=1.0,
    )
    sent = contractor_harness.llm.calls[0]["messages"]
    blob = json.dumps(sent, ensure_ascii=False)
    assert "旁路问题" in blob
    assert "FORMALIZED_TASK_MARKER" not in blob
    assert "parent transcript marker" not in blob


@pytest.mark.asyncio
async def test_contractor_native_contractor_return(contractor_harness: ContractorHarness) -> None:
    contractor_harness.llm.enqueue(
        FakeLLMResponse(
            text="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "contractor_return",
                        "arguments": json.dumps({"result": "原生返回"}, ensure_ascii=False),
                    },
                }
            ],
        )
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="原生？",
        caller_protocol="native_tools",
        credit_budget=5.0,
    )
    assert result.success is True
    assert result.data["result"] == "原生返回"
    assert result.metadata["termination_reason"] == "returned"
    tools = contractor_harness.llm.calls[0].get("tools")
    assert tools is not None
    names = {item["function"]["name"] for item in tools}
    assert "contractor_return" in names


@pytest.mark.asyncio
async def test_contractor_budget_zero_runs_one_turn_then_force_returns_if_spend(
    contractor_harness: ContractorHarness,
) -> None:
    """credit_budget=0 仍跑 ≥1 轮；turn 花费 >0 → insufficient_funds 强制返回。"""

    contractor_harness.llm.queue_json({"return": "本应返回的答案"})
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="零预算",
        caller_protocol="json_envelope",
        credit_budget=0.0,
    )
    assert result.success is True
    assert result.metadata["termination_reason"] == "insufficient_funds"
    assert float(result.research_credits_charged) > 0
    text = str(result.data["result"])
    assert "额度" in text or "不足" in text or "insufficient" in text.lower()
    assert len(contractor_harness.llm.calls) == 1


@pytest.mark.asyncio
async def test_contractor_soft_time_budget_force_returns(
    contractor_harness: ContractorHarness,
) -> None:
    """极小 time_budget_seconds：第一轮请求工具后软超时强制返回。"""

    contractor_harness.llm.queue_json(
        {
            "report": "还要再想",
            "procedures": [{"procedure_id": "builtin.web_search", "arguments": {"query": "x"}}],
        }
    )
    nested_calls: list[str] = []

    async def _nested(procedure_id: str, arguments: Any, **kwargs: Any) -> ProcedureResult:
        nested_calls.append(str(procedure_id))
        return ProcedureResult(
            success=True,
            data={"ok": True},
            error=None,
            metadata={},
            research_credits_charged=0.0,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="赶时间",
        caller_protocol="json_envelope",
        credit_budget=100.0,
        time_budget_seconds=1e-9,
    )
    assert result.success is True
    assert result.metadata["termination_reason"] == "timeout"
    text = str(result.data["result"])
    assert "超时" in text or "timeout" in text.lower() or "时间" in text


@pytest.mark.asyncio
async def test_contractor_rejects_nested_contractor_and_controls(
    contractor_harness: ContractorHarness,
) -> None:
    """禁止嵌套 builtin.contractor / core.terminate / core.checkpoint；错误写入 transcript。"""

    contractor_harness.llm.queue_json(
        {
            "report": "尝试控制",
            "procedures": [
                {"procedure_id": "builtin.contractor", "arguments": {"agent_id": "x", "question": "y"}},
                {"procedure_id": "core.terminate", "arguments": {}},
                {"procedure_id": "core.checkpoint", "arguments": {}},
            ],
        }
    )
    contractor_harness.llm.queue_json({"return": "在拒绝后给出答案"})
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="控制类",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert result.success is True
    assert result.data["result"] == "在拒绝后给出答案"
    assert result.metadata["termination_reason"] == "returned"
    assert len(contractor_harness.llm.calls) == 2
    second_blob = json.dumps(contractor_harness.llm.calls[1]["messages"], ensure_ascii=False)
    for pid in ("builtin.contractor", "core.terminate", "core.checkpoint"):
        assert pid in second_blob
    assert "不允许" in second_blob or "拒绝" in second_blob or "禁止" in second_blob


@pytest.mark.asyncio
async def test_contractor_nested_compact_adds_to_bill(
    contractor_harness: ContractorHarness,
) -> None:
    """嵌套 core.compact 的 research_credits_charged 折入承包商总账单（不经外层二次扣）。"""

    nested_calls: list[dict[str, Any]] = []

    async def _nested(procedure_id: str, arguments: Any, **kwargs: Any) -> ProcedureResult:
        nested_calls.append({"procedure_id": procedure_id, "arguments": arguments, **kwargs})
        return ProcedureResult(
            success=True,
            data={"summary": "压缩后"},
            error=None,
            metadata={"research_credits_charged": 2.0},
            research_credits_charged=2.0,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    contractor_harness.llm.queue_json(
        {"report": "压缩", "procedures": [{"procedure_id": "core.compact", "arguments": {}, "credits": 0}]}
    )
    contractor_harness.llm.queue_json({"return": "压缩后继续"})
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="需要 compact",
        caller_protocol="json_envelope",
        credit_budget=50.0,
    )
    assert result.success is True
    assert result.metadata["termination_reason"] == "returned"
    assert len(nested_calls) == 1
    assert nested_calls[0]["procedure_id"] == "core.compact"
    # turn1 0.5 + nested 2.0 + turn2 0.5 = 3.0
    assert float(result.research_credits_charged) == pytest.approx(3.0)
    second_blob = json.dumps(contractor_harness.llm.calls[1]["messages"], ensure_ascii=False)
    assert "core.compact" in second_blob or "压缩" in second_blob


@pytest.mark.asyncio
async def test_run_contractor_missing_agent_fails(contractor_harness: ContractorHarness) -> None:
    result = await run_contractor(
        arguments={"agent_id": "missing.agent", "question": "q"},
        scoped_metadata={"credit_budget": 1.0, "caller_protocol": "json_envelope"},
        deps=contractor_harness.deps,
    )
    assert result.success is False
    assert result.error is not None
    assert result.error["code"] in {"invalid_arguments", "agent_unavailable"}


@pytest.mark.asyncio
async def test_contractor_llm_failure_is_not_returned_success(contractor_harness: ContractorHarness) -> None:
    contractor_harness.llm.enqueue(
        FakeLLMResponse(
            text="",
            success=False,
            error_code="upstream_timeout",
            error_message="上游超时",
            usage={"prompt_tokens": 10, "completion_tokens": 0},
        )
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="失败路径",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert result.success is False
    assert result.error is not None
    assert result.error["code"] in {"llm_generation_failed", "upstream_timeout"}
    assert result.metadata.get("termination_reason") != "returned"
    assert float(result.research_credits_charged) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_contractor_prefers_frozen_round_snapshot_over_live_deps() -> None:
    """有 task_id 对应的冻结 round snapshot 时，人设/计价不得读 live registry。"""

    live_agent = bundled_agent_definitions()[0].model_copy(
        update={"character_prompt": "LIVE_PERSONALITY_MUST_NOT_APPEAR"}
    )
    frozen_agent = bundled_agent_definitions()[0].model_copy(
        update={"character_prompt": "FROZEN_ROUND_PERSONALITY"}
    )
    live_prices = _FakePrices(per_turn=99.0)
    frozen_prices = _FakePrices(per_turn=1.25)

    class _FakeEntry:
        def __init__(self, definition: Any) -> None:
            self.definition = definition

    class _FakeAgentCatalog:
        def get(self, agent_id: str) -> Any:
            if agent_id == frozen_agent.agent_id:
                return _FakeEntry(frozen_agent)
            return None

        def resolve_allowed_procedures(self, agent_id: str, procedures: Any) -> tuple[str, ...]:
            del agent_id, procedures
            return ()

    frozen_snapshot = SimpleNamespace(
        agent_catalog=_FakeAgentCatalog(),
        procedure_catalog=SimpleNamespace(ids=(), get=lambda _pid: None),
        price_catalog=frozen_prices,
    )
    llm = FakeLLMGateway()
    llm.queue_json({"return": "ok"})
    deps = ContractorDeps(
        llm=llm,
        prices=live_prices,
        resolve_agent=lambda _aid: live_agent,
        round_snapshot_for_task=lambda task_id: frozen_snapshot if task_id == "task-frozen" else None,
    )
    result = await run_contractor(
        arguments={"agent_id": frozen_agent.agent_id, "question": "旁路"},
        scoped_metadata={
            "task_id": "task-frozen",
            "credit_budget": 5.0,
            "caller_protocol": "json_envelope",
        },
        deps=deps,
    )
    assert result.success is True
    assert float(result.research_credits_charged) == pytest.approx(1.25)
    assert len(frozen_prices.calls) == 1
    assert live_prices.calls == []
    blob = json.dumps(llm.calls[0]["messages"], ensure_ascii=False)
    assert "FROZEN_ROUND_PERSONALITY" in blob
    assert "LIVE_PERSONALITY_MUST_NOT_APPEAR" not in blob


@pytest.mark.asyncio
async def test_contractor_nested_calculate_runs_via_real_provider_invoker() -> None:
    """生产同形 nested invoker：允许的 builtin.calculate 真实执行，结果进 transcript。"""

    from test_memory import FakeCtx

    provider = BundledProcedureProvider(FakeCtx())
    nested = make_nested_procedure_invoker(provider=provider, summarizer=None, price_catalog=None)
    harness = ContractorHarness.create()
    harness.deps.invoke_nested_procedure = nested
    harness.llm.queue_json(
        {
            "report": "算一下",
            "procedures": [
                {"procedure_id": "builtin.calculate", "arguments": {"expression": "6*7"}},
            ],
        }
    )
    harness.llm.queue_json({"return": "四十二"})
    result = await harness.invoke(
        agent_id="builtin.quick_thinker",
        question="6*7=?",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert result.success is True
    assert result.data["result"] == "四十二"
    assert result.metadata["termination_reason"] == "returned"
    second_blob = json.dumps(harness.llm.calls[1]["messages"], ensure_ascii=False)
    assert "builtin.calculate" in second_blob
    assert "42" in second_blob
    assert "未注入" not in second_blob
    assert "拒绝" not in second_blob
