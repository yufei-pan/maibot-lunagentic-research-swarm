from __future__ import annotations

from lunagentic_research_swarm.agents.registry import AgentRegistry
from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted
from lunagentic_research_swarm.runtime.reducer import (
    NotifyToolWaiter,
    PerformBranchSummary,
    RuntimeState,
    reduce_event,
)


def _procedure_event(*, delegations, live_agent_ids):
    return ProcedureBatchCompleted(
        "procedure", "task", "round", 0,
        branch_id="parent",
        call_id="call",
        result_id="result",
        results=(),
        report="evidence",
        delegations=tuple(delegations),
        credits_after=10.0,
        parent_messages=({"role": "assistant", "content": "parent"},),
        parent_depth=0,
        live_agent_ids=tuple(live_agent_ids),
        max_delegations_per_turn=8,
        max_branch_depth=8,
        max_agent_calls_per_task=32,
        agent_calls_started=1,
    )


def test_removed_extension_edge_is_summarized_while_valid_sibling_is_materialized() -> None:
    state = RuntimeState(
        "task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 10.0}
    )
    transition = reduce_event(
        state,
        _procedure_event(
            delegations=(
                {"agent_id": "extension.removed", "task": "removed work", "credits": 5.0},
                {"agent_id": "builtin.researcher", "task": "valid work", "credits": 5.0},
            ),
            live_agent_ids=("builtin.researcher",),
        ),
    )

    assert any(isinstance(effect, PerformBranchSummary) for effect in transition.effects)
    sibling = [effect for effect in transition.effects if isinstance(effect, NotifyToolWaiter)]
    assert len(sibling) == 1
    assert sibling[0].payload["agent_id"] == "builtin.researcher"
    removed = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    assert removed[0].payload["reason"] == "agent_unavailable"


def test_missing_self_edge_is_terminated_without_blocking_other_branch() -> None:
    state = RuntimeState(
        "task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 0.0}
    )
    transition = reduce_event(
        state,
        _procedure_event(
            delegations=(
                {"agent_id": "extension.self", "task": "retry self", "credits": 0.0},
                {"agent_id": "builtin.knowledge_base", "task": "known facts", "credits": 0.0},
            ),
            live_agent_ids=("builtin.knowledge_base",),
        ),
    )

    removed = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    sibling = [effect for effect in transition.effects if isinstance(effect, NotifyToolWaiter)]
    assert removed and removed[0].payload["reason"] == "agent_unavailable"
    assert sibling and sibling[0].payload["agent_id"] == "builtin.knowledge_base"


def test_registry_removal_is_reflected_in_inflight_edge_resolution() -> None:
    registry = AgentRegistry(root_agent="extension.root")
    registry.replace_provider(
        "provider.extension",
        [
            {
                "agent_id": "extension.root",
                "version": "1",
                "display_name": "root",
                "description": "root",
                "character_prompt": "root",
                "model_selector": "task:utils",
                "can_be_root": True,
            },
            {
                "agent_id": "extension.sibling",
                "version": "1",
                "display_name": "sibling",
                "description": "sibling",
                "character_prompt": "sibling",
                "model_selector": "task:utils",
            },
        ],
    )
    snapshot = registry.snapshot({})
    registry.remove_provider("provider.extension")
    assert snapshot.get("extension.sibling") is not None
    assert not registry.is_live("extension.sibling")

    transition = reduce_event(
        RuntimeState("task", TaskStatus.RUNNING, active_round_id="round", active_leaves={"parent": 1.0}),
        _procedure_event(
            delegations=({"agent_id": "extension.sibling", "task": "in-flight", "credits": 1.0},),
            live_agent_ids=(),
        ),
    )
    removed = [effect for effect in transition.effects if isinstance(effect, PerformBranchSummary)]
    assert removed and removed[0].payload["reason"] == "agent_unavailable"
