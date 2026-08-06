"""E2E：并行承包商账单求和（外层 batch 只扣各承包商 top-level charge）。"""

from __future__ import annotations

from typing import Any

import pytest

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
from lunagentic_research_swarm.llm.protocol import ProcedureRequest
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogEntry, ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.reducer import PerformProcedureBatch
from lunagentic_research_swarm.runtime.turns import TurnWorker


def _contractor_catalog() -> ProcedureCatalogSnapshot:
    definition = ProcedureDefinition.model_validate(
        {
            "procedure_id": "builtin.contractor",
            "version": "1",
            "display_name": "旁路承包商",
            "description": "e2e contractor",
            "arguments_schema": {"type": "object"},
            "result_schema": {"type": "object", "properties": {"result": {"type": "string"}}},
            "idempotent": False,
            "timeout_seconds": 0.0,
        }
    )
    return ProcedureCatalogSnapshot(
        [
            ProcedureCatalogEntry(
                definition,
                "builtin",
                "builtin.invoke_procedure",
                "1",
                "e2e-contractor-fingerprint",
            )
        ]
    )


class _ParallelContractorAPI:
    """两次 builtin.contractor 调用分别申报不同账单。"""

    def __init__(self) -> None:
        self.charges = [3.0, 5.0]
        self.index = 0
        self.calls: list[dict[str, Any]] = []

    async def call(self, name: str, *, version: str = "", **kwargs: Any) -> Any:
        del name, version
        self.calls.append(dict(kwargs))
        charge = float(self.charges[self.index])
        self.index += 1
        return {
            "success": True,
            "data": {"result": f"bill-{charge}"},
            "error": None,
            "metadata": {"termination_reason": "returned", "research_credits_charged": charge},
            "research_credits_charged": charge,
        }


@pytest.mark.asyncio
async def test_parallel_contractors_sum_bills_to_caller_balance() -> None:
    """同一 agent turn 内两个并行承包商 → 调用方余额减少两侧账单之和。"""

    api = _ParallelContractorAPI()
    executor = ProcedureExecutor(_contractor_catalog(), api=api)
    starting_balance = 20.0
    effect = PerformProcedureBatch(
        task_id="task-contractor-e2e",
        round_id="round-1",
        generation=0,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "agent_id": "builtin.quick_thinker",
            "credits_after": starting_balance,
            "requests": [
                ProcedureRequest(
                    procedure_id="builtin.contractor",
                    arguments={"agent_id": "builtin.quick_thinker", "question": "a?"},
                    credits=10.0,
                ),
                ProcedureRequest(
                    procedure_id="builtin.contractor",
                    arguments={"agent_id": "builtin.quick_thinker", "question": "b?"},
                    credits=10.0,
                ),
            ],
        },
    )

    completed = await TurnWorker(object(), executor).perform_procedure_batch(effect)

    assert len(api.calls) == 2
    assert completed.credits_after == pytest.approx(starting_balance - (3.0 + 5.0))
    assert float(completed.results[0].result.research_credits_charged) == pytest.approx(3.0)
    assert float(completed.results[1].result.research_credits_charged) == pytest.approx(5.0)


class _SingleChargeContractorAPI:
    """单次承包商调用申报固定账单。"""

    def __init__(self, charge: float) -> None:
        self.charge = float(charge)
        self.calls: list[dict[str, Any]] = []

    async def call(self, name: str, *, version: str = "", **kwargs: Any) -> Any:
        del name, version
        self.calls.append(dict(kwargs))
        charge = self.charge
        return {
            "success": True,
            "data": {"result": f"bill-{charge}"},
            "error": None,
            "metadata": {"termination_reason": "returned", "research_credits_charged": charge},
            "research_credits_charged": charge,
        }


@pytest.mark.asyncio
async def test_already_negative_caller_balance_still_fully_debited() -> None:
    """调用方余额已为负时，仍按承包商账单全额扣减（prior + charge）。"""

    api = _SingleChargeContractorAPI(3.0)
    executor = ProcedureExecutor(_contractor_catalog(), api=api)
    prior = -2.0
    effect = PerformProcedureBatch(
        task_id="task-contractor-neg",
        round_id="round-1",
        generation=0,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "agent_id": "builtin.quick_thinker",
            "credits_after": prior,
            "requests": [
                ProcedureRequest(
                    procedure_id="builtin.contractor",
                    arguments={"agent_id": "builtin.quick_thinker", "question": "owed?"},
                    credits=10.0,
                ),
            ],
        },
    )

    completed = await TurnWorker(object(), executor).perform_procedure_batch(effect)

    assert len(api.calls) == 1
    assert float(completed.results[0].result.research_credits_charged) == pytest.approx(3.0)
    assert completed.credits_after == pytest.approx(-5.0)


@pytest.mark.asyncio
async def test_contractor_batch_does_not_increment_agent_calls_started() -> None:
    """窄缝：TurnWorker 承包商 batch 透传 agent_calls_started，不因 procedure 内循环加一。

    全量 RuntimeHarness「父 turn +1 / 承包商内轮不加」过重；此处断言 batch 路径不碰计数器。
    """

    api = _SingleChargeContractorAPI(1.0)
    executor = ProcedureExecutor(_contractor_catalog(), api=api)
    started = 7
    effect = PerformProcedureBatch(
        task_id="task-contractor-calls",
        round_id="round-1",
        generation=0,
        payload={
            "branch_id": "branch-1",
            "call_id": "call-1",
            "turn_id": "turn-1",
            "agent_id": "builtin.quick_thinker",
            "credits_after": 10.0,
            "agent_calls_started": started,
            "max_agent_calls_per_task": 256,
            "requests": [
                ProcedureRequest(
                    procedure_id="builtin.contractor",
                    arguments={"agent_id": "builtin.quick_thinker", "question": "count?"},
                    credits=5.0,
                ),
            ],
        },
    )

    completed = await TurnWorker(object(), executor).perform_procedure_batch(effect)

    assert completed.agent_calls_started == started
    assert completed.max_agent_calls_per_task == 256
    assert completed.credits_after == pytest.approx(9.0)
