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
from lunagentic_research_swarm.agents.bundled.prompts import BUNDLED_CHARACTER_PROMPTS
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
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureRegistry
from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch

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
    """procedures."builtin.contractor".enabled=false → 目录省略；同批兄弟不受影响。"""
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", BundledProcedureProvider(object()).describe())
    enabled = registry.snapshot({})
    assert enabled.get(CONTRACTOR_PROCEDURE_ID) is not None
    sibling_id = "builtin.calculate"
    assert enabled.get(sibling_id) is not None

    disabled = registry.snapshot({CONTRACTOR_PROCEDURE_ID: ProcedureOverride(enabled=False)})
    assert disabled.get(CONTRACTOR_PROCEDURE_ID) is None
    assert disabled.get(sibling_id) is not None


@pytest.mark.asyncio
async def test_disabled_contractor_unavailable_via_registry_snapshot() -> None:
    """enabled=false 省略目录后，经 executor 调用仍返回 procedure_unavailable（非静默成功）。"""
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", BundledProcedureProvider(object()).describe())
    disabled_catalog = registry.snapshot({CONTRACTOR_PROCEDURE_ID: ProcedureOverride(enabled=False)})
    assert disabled_catalog.get(CONTRACTOR_PROCEDURE_ID) is None
    assert disabled_catalog.get("builtin.calculate") is not None

    executor = ProcedureExecutor(disabled_catalog, api=object())
    completed = await executor.invoke_many(
        PerformProcedureBatch(
            task_id="task-contractor-disabled",
            round_id="round-1",
            generation=0,
            payload={
                "branch_id": "branch-1",
                "call_id": "call-1",
                "turn_id": "turn-1",
                "agent_id": "builtin.quick_thinker",
                "requests": [
                    {
                        "procedure_id": CONTRACTOR_PROCEDURE_ID,
                        "arguments": {
                            "agent_id": "builtin.quick_thinker",
                            "question": "不应执行",
                        },
                    }
                ],
            },
        )
    )
    assert completed.results[0].success is False
    assert completed.results[0].result.error["code"] == "procedure_unavailable"
    assert completed.results[0].result.data in (None, {})


def test_config_default_toml_lists_all_bundled_procedure_toggles() -> None:
    text = _CONFIG_DEFAULT.read_text(encoding="utf-8")
    for procedure_id in _EXPECTED_BUNDLED_PROCEDURE_IDS:
        assert f'[procedures."{procedure_id}"]' in text
    assert "timeout_seconds = 0" in text
    contractor_block = text.split('[procedures."builtin.contractor"]', 1)[1]
    assert "enabled = true" in contractor_block.split("[", 1)[0]
    assert "timeout_seconds = 0" in contractor_block.split("[", 1)[0]


async def _always_ok_nested(
    procedure_id: str,
    arguments: Any = None,
    **_kwargs: Any,
) -> ProcedureResult:
    """嵌套调用永远成功且不扣费；用于驱动承包商跑满轮数上限。"""

    del arguments
    return ProcedureResult(
        success=True,
        data={"procedure_id": str(procedure_id), "results": []},
        error=None,
        metadata={"research_credits_charged": 0.0},
        research_credits_charged=0.0,
    )


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
async def test_contractor_system_prompt_states_identity_budget_and_schemas(
    contractor_harness: ContractorHarness,
) -> None:
    """承包商必须知道：谁在问、还剩多少预算、每个 Procedure 的参数长什么样。"""

    contractor_harness.deps.resolve_procedure_catalog = lambda _agent_id: [
        {
            "procedure_id": "builtin.web_search",
            "description": "按指定引擎执行网页搜索。",
            "arguments_schema": {
                "type": "object",
                "properties": {"engine": {"type": "string", "enum": ["duckduckgo"]}, "query": {"type": "string"}},
                "required": ["engine", "query"],
            },
        }
    ]
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(
        agent_id="builtin.researcher",
        question="旁路问题",
        caller_protocol="json_envelope",
        credit_budget=7.0,
        time_budget_seconds=30,
    )

    system = str(contractor_harness.llm.calls[0]["messages"][0]["content"])
    # 人设仍来自被选中的智能体
    assert BUNDLED_CHARACTER_PROMPTS["builtin.researcher"] in system
    # 提问方 + 预算：没有这些，承包商无法自我调度，只能被强制截断
    assert "`builtin.quick_thinker`" in system
    assert "最多 16 轮" in system
    assert "7 credits" in system
    assert "30 秒" in system
    # arguments schema：只给 id + 描述时，required/enum 的 Procedure 必然被猜错
    assert '"required"' in system and "engine" in system
    assert "duckduckgo" in system


@pytest.mark.asyncio
async def test_contractor_max_turns_is_not_reported_as_a_normal_return(
    contractor_harness: ContractorHarness,
) -> None:
    """轮数耗尽必须可与「答完了」区分，否则调用方会把截断结果当完整答案。"""

    contractor_harness.deps.invoke_nested_procedure = _always_ok_nested
    for _ in range(20):
        contractor_harness.llm.queue_json(
            {"report": "继续查", "procedures": [{"procedure_id": "builtin.web_search", "arguments": {}}]}
        )
    result = await contractor_harness.invoke(
        agent_id="builtin.researcher",
        question="永远查不完的问题",
        caller_protocol="json_envelope",
        credit_budget=10_000.0,
    )

    assert result.success is True
    assert result.metadata["termination_reason"] == "max_turns"
    assert result.metadata["turn_count"] == 16
    assert "max_turns" in str(result.data["result"])


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
async def test_contractor_uses_selected_agent_model_selector(contractor_harness: ContractorHarness) -> None:
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="selector?",
        caller_protocol="json_envelope",
        credit_budget=10.0,
    )
    assert contractor_harness.llm.calls[0]["selector"] == "task:utils"


@pytest.mark.asyncio
async def test_contractor_personality_and_temperature_overrides(
    contractor_harness: ContractorHarness,
) -> None:
    contractor_harness.llm.queue_json({"return": "ok"})
    await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="覆写？",
        caller_protocol="json_envelope",
        credit_budget=10.0,
        personality="【OVERRIDE_PERSONALITY_MARKER】",
        temperature=0.42,
    )
    req = contractor_harness.llm.calls[0]
    assert abs(float(req["temperature"]) - 0.42) < 1e-9
    assert "【OVERRIDE_PERSONALITY_MARKER】" in json.dumps(req["messages"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_contractor_invalid_temperature_rejected(contractor_harness: ContractorHarness) -> None:
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="坏温度",
        caller_protocol="json_envelope",
        credit_budget=10.0,
        temperature="hot",
    )
    assert result.success is False
    assert result.error is not None
    assert result.error["code"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_native_call_procedure_runs_allowed_nested(contractor_harness: ContractorHarness) -> None:
    nested_calls: list[str] = []

    async def _nested(procedure_id: str, arguments: Any = None, **kwargs: Any) -> ProcedureResult:
        del arguments, kwargs
        nested_calls.append(str(procedure_id))
        return ProcedureResult(
            success=True,
            data={"v": 1},
            error=None,
            metadata={},
            research_credits_charged=0.1,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    contractor_harness.llm.enqueue(
        FakeLLMResponse(
            text="",
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "call_procedure",
                        "arguments": json.dumps(
                            {
                                "procedure_id": "builtin.calculate",
                                "arguments": {"expression": "1+1"},
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        )
    )
    contractor_harness.llm.enqueue(
        FakeLLMResponse(
            text="",
            tool_calls=[
                {
                    "id": "2",
                    "type": "function",
                    "function": {
                        "name": "contractor_return",
                        "arguments": json.dumps({"result": "嵌套完成"}, ensure_ascii=False),
                    },
                }
            ],
        )
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="原生嵌套",
        caller_protocol="native_tools",
        credit_budget=10.0,
    )
    assert result.success is True
    assert nested_calls == ["builtin.calculate"]
    assert result.data["result"] == "嵌套完成"
    assert result.metadata["termination_reason"] == "returned"
    tools = contractor_harness.llm.calls[0].get("tools")
    assert tools is not None
    names = {item["function"]["name"] for item in tools}
    assert "call_procedure" in names
    assert "contractor_return" in names
    assert len(contractor_harness.llm.calls) == 2


@pytest.mark.asyncio
async def test_force_return_includes_attempted_procedure_ids(
    contractor_harness: ContractorHarness,
) -> None:
    """insufficient_funds force-return text mentions attempted nested procedure_id."""

    nested_calls: list[str] = []

    async def _nested(procedure_id: str, arguments: Any = None, **kwargs: Any) -> ProcedureResult:
        del arguments, kwargs
        nested_calls.append(str(procedure_id))
        return ProcedureResult(
            success=True,
            data={},
            error=None,
            metadata={},
            research_credits_charged=0.0,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    contractor_harness.llm.queue_json(
        {
            "report": "x",
            "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1+1"}}],
            "return": "should-not-matter",
        }
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="强制返回含尝试工具",
        caller_protocol="json_envelope",
        credit_budget=0.0,
    )
    assert result.success is True
    assert result.metadata["termination_reason"] == "insufficient_funds"
    assert nested_calls == []
    assert "builtin.calculate" in str(result.data["result"])


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
async def test_insufficient_funds_skips_same_turn_nested_procedures(
    contractor_harness: ContractorHarness,
) -> None:
    """Turn spend 使余额 <0 → insufficient_funds；同轮 nested 不得执行（预算机 step 4）。"""

    nested_calls: list[str] = []

    async def _nested(procedure_id: str, arguments: Any = None, **kwargs: Any) -> ProcedureResult:
        del arguments, kwargs
        nested_calls.append(str(procedure_id))
        return ProcedureResult(
            success=True,
            data={},
            error=None,
            metadata={},
            research_credits_charged=0.0,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    # per_turn=0.5 > credit_budget=0 → 扣费后余额为负，即使信封含 procedures
    contractor_harness.llm.queue_json(
        {
            "report": "x",
            "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1+1"}}],
            "return": "should-not-matter",
        }
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="额度不足跳过嵌套",
        caller_protocol="json_envelope",
        credit_budget=0.0,
    )
    assert result.success is True
    assert result.metadata["termination_reason"] == "insufficient_funds"
    assert nested_calls == []
    assert len(contractor_harness.llm.calls) == 1


@pytest.mark.asyncio
async def test_explicit_return_ignores_sibling_procedures(
    contractor_harness: ContractorHarness,
) -> None:
    """同轮显式 return + procedures → return 优先；nested 不调用（预算机 step 5）。"""

    nested_calls: list[str] = []

    async def _nested(procedure_id: str, arguments: Any = None, **kwargs: Any) -> ProcedureResult:
        del arguments, kwargs
        nested_calls.append(str(procedure_id))
        return ProcedureResult(
            success=True,
            data={},
            error=None,
            metadata={},
            research_credits_charged=0.0,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    contractor_harness.llm.queue_json(
        {
            "return": "最终答案",
            "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1"}}],
        }
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="显式返回优先",
        caller_protocol="json_envelope",
        credit_budget=100.0,
    )
    assert result.success is True
    assert result.data["result"] == "最终答案"
    assert result.metadata["termination_reason"] == "returned"
    assert nested_calls == []
    assert len(contractor_harness.llm.calls) == 1


@pytest.mark.asyncio
async def test_nested_settlement_overspend_force_returns_without_next_turn(
    contractor_harness: ContractorHarness,
) -> None:
    """嵌套结算后余额 <0 → insufficient_funds，不再开下一轮 LLM（预算机 step 7）。"""

    nested_charge = 5.0

    async def _nested(procedure_id: str, arguments: Any = None, **kwargs: Any) -> ProcedureResult:
        del procedure_id, arguments, kwargs
        return ProcedureResult(
            success=True,
            data={"ok": True},
            error=None,
            metadata={},
            research_credits_charged=nested_charge,
        )

    contractor_harness.deps.invoke_nested_procedure = _nested
    # turn 0.5 后余额 0.5；嵌套 +5 → 余额 -4.5；勿再排队第二轮 LLM
    contractor_harness.llm.queue_json(
        {
            "report": "算一下",
            "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1"}}],
        }
    )
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="嵌套超额",
        caller_protocol="json_envelope",
        credit_budget=1.0,
    )
    assert result.success is True
    assert result.metadata["termination_reason"] == "insufficient_funds"
    assert len(contractor_harness.llm.calls) == 1
    assert float(result.research_credits_charged) >= nested_charge
    assert float(result.research_credits_charged) == pytest.approx(0.5 + nested_charge)


@pytest.mark.asyncio
async def test_multi_turn_nested_then_return_sums_charges(
    contractor_harness: ContractorHarness,
) -> None:
    """多轮：turn1 嵌套计费 + turn2 return → research_credits_charged 为 turn+nested+turn 之和。"""

    nested_charge = 0.5
    turn_charges = [0.5, 0.3]

    @dataclass
    class _ScheduledPrices:
        remaining: list[float] = field(default_factory=lambda: list(turn_charges))
        calls: list[dict[str, Any]] = field(default_factory=list)

        def charge_actual(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(dict(kwargs))
            return SimpleNamespace(credits=float(self.remaining.pop(0)))

    async def _nested(procedure_id: str, arguments: Any = None, **kwargs: Any) -> ProcedureResult:
        del procedure_id, arguments, kwargs
        return ProcedureResult(
            success=True,
            data={"ok": True},
            error=None,
            metadata={},
            research_credits_charged=nested_charge,
        )

    prices = _ScheduledPrices()
    contractor_harness.deps.prices = prices
    contractor_harness.deps.invoke_nested_procedure = _nested
    contractor_harness.llm.queue_json(
        {
            "report": "先查工具",
            "procedures": [{"procedure_id": "builtin.calculate", "arguments": {"expression": "1"}}],
        }
    )
    contractor_harness.llm.queue_json({"return": "合计完成"})
    result = await contractor_harness.invoke(
        agent_id="builtin.quick_thinker",
        question="多轮计费求和",
        caller_protocol="json_envelope",
        credit_budget=100.0,
    )
    assert result.success is True
    assert result.data["result"] == "合计完成"
    assert result.metadata["termination_reason"] == "returned"
    assert len(contractor_harness.llm.calls) == 2
    assert len(prices.calls) == 2
    expected = turn_charges[0] + nested_charge + turn_charges[1]
    assert float(result.research_credits_charged) == pytest.approx(expected)


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
    """嵌套 core.compact 经真实 invoker：注入 outsider transcript，计费折入总账单。"""

    from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
    from lunagentic_research_swarm.llm.summarizer import SummaryResult
    from test_memory import FakeCtx

    compact_usage = TokenUsage(100, 50, 0, 100, source="actual")
    compact_charge = 0.2
    compact_catalog = PriceCatalog.from_sources(
        {},
        {"model:test": PriceProfile(price_in=10.0, price_out=20.0)},
        {},
    )

    class _CompactSummarizer:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def compact_branch(self, request: Any) -> SummaryResult:
            self.requests.append(request)
            return SummaryResult(True, "压缩摘要", "model:test", compact_usage, None)

    summarizer = _CompactSummarizer()
    provider = BundledProcedureProvider(FakeCtx())
    contractor_harness.deps.invoke_nested_procedure = make_nested_procedure_invoker(
        provider=provider,
        summarizer=summarizer,
        price_catalog=compact_catalog,
    )
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
    assert len(summarizer.requests) == 1
    history = list(summarizer.requests[0].branch_history)
    assert any(item.get("role") == "user" and "需要 compact" in str(item.get("content", "")) for item in history)
    assert any(item.get("role") == "assistant" for item in history)
    # turn1 0.5 + nested compact 0.2 + turn2 0.5 = 1.2
    assert float(result.research_credits_charged) == pytest.approx(0.5 + compact_charge + 0.5)
    second_blob = json.dumps(contractor_harness.llm.calls[1]["messages"], ensure_ascii=False)
    assert "core.compact" in second_blob
    assert "压缩摘要" in second_blob or "compacted" in second_blob
    assert "summary_input_empty" not in second_blob
    assert "未注入" not in second_blob


@pytest.mark.asyncio
async def test_nested_compact_prefers_frozen_round_price_catalog_over_live() -> None:
    """嵌套 compact 与 turn metering 同源：有 round snapshot 时用冻结 catalog，忽略 live。"""

    from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile, TokenUsage
    from lunagentic_research_swarm.llm.summarizer import SummaryResult
    from test_memory import FakeCtx

    compact_usage = TokenUsage(100, 50, 0, 100, source="actual")
    frozen_charge = 0.2
    live_charge_if_used = 2.0  # 10× frozen；若误用 live 会落到此值
    live_catalog = PriceCatalog.from_sources(
        {},
        {"model:test": PriceProfile(price_in=100.0, price_out=200.0)},
        {},
    )
    frozen_catalog = PriceCatalog.from_sources(
        {},
        {"model:test": PriceProfile(price_in=10.0, price_out=20.0)},
        {},
    )
    frozen_snapshot = SimpleNamespace(price_catalog=frozen_catalog)

    class _CompactSummarizer:
        async def compact_branch(self, request: Any) -> SummaryResult:
            del request
            return SummaryResult(True, "冻结价摘要", "model:test", compact_usage, None)

    nested = make_nested_procedure_invoker(
        provider=BundledProcedureProvider(FakeCtx()),
        summarizer=_CompactSummarizer(),
        price_catalog=live_catalog,
        round_snapshot_for_task=lambda task_id: frozen_snapshot if task_id == "task-frozen" else None,
    )
    result = await nested(
        "core.compact",
        {},
        scoped_metadata={
            "task_id": "task-frozen",
            "formalized_task": "旁路问题",
            "branch_history": [{"role": "user", "content": "旁路问题"}],
        },
    )
    assert result.success is True
    assert float(result.research_credits_charged) == pytest.approx(frozen_charge)
    assert float(result.research_credits_charged) != pytest.approx(live_charge_if_used)

    # 无 snapshot / 无 task_id 时仍回落 live
    live_only = make_nested_procedure_invoker(
        provider=BundledProcedureProvider(FakeCtx()),
        summarizer=_CompactSummarizer(),
        price_catalog=live_catalog,
        round_snapshot_for_task=lambda _tid: None,
    )
    live_result = await live_only(
        "core.compact",
        {},
        scoped_metadata={
            "formalized_task": "旁路问题",
            "branch_history": [{"role": "user", "content": "旁路问题"}],
        },
    )
    assert float(live_result.research_credits_charged) == pytest.approx(live_charge_if_used)


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
async def test_contractor_gets_argument_schemas_from_the_frozen_round_snapshot() -> None:
    """生产路径（有 round snapshot）也必须把 arguments schema 交给承包商。"""

    from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

    agent = bundled_agent_definitions()[0]
    procedures = ProcedureRegistry()
    procedures.replace_provider(
        "builtin",
        [
            {
                "procedure_id": "builtin.web_search",
                "version": "1",
                "display_name": "网页搜索",
                "description": "按指定引擎执行网页搜索。",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "engine": {"type": "string", "enum": ["duckduckgo"]},
                        "query": {"type": "string"},
                    },
                    "required": ["engine", "query"],
                },
                "result_schema": {"type": "object"},
                "enabled": True,
            }
        ],
    )

    class _Entry:
        definition = agent

    class _AgentCatalog:
        def get(self, agent_id: str) -> Any:
            return _Entry() if agent_id == agent.agent_id else None

        def resolve_allowed_procedures(self, agent_id: str, _procedures: Any) -> tuple[str, ...]:
            del agent_id
            return ("builtin.web_search",)

    snapshot = SimpleNamespace(
        agent_catalog=_AgentCatalog(),
        procedure_catalog=procedures.snapshot({}),
        price_catalog=_FakePrices(),
    )
    llm = FakeLLMGateway()
    llm.queue_json({"return": "ok"})
    deps = ContractorDeps(llm=llm, round_snapshot_for_task=lambda _task_id: snapshot)

    await run_contractor(
        arguments={"agent_id": agent.agent_id, "question": "旁路问题"},
        scoped_metadata={"task_id": "task-1", "caller_protocol": "json_envelope"},
        deps=deps,
    )

    system = str(llm.calls[0]["messages"][0]["content"])
    assert "builtin.web_search" in system
    assert '"required"' in system
    assert "duckduckgo" in system


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
