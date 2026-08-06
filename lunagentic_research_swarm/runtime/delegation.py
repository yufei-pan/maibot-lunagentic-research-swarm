"""委派计划：结构上限、存活性检查与父子 credits 分配的唯一实现。

一个 turn 的委派可能立即发生（普通 turn），也可能被 ``core.checkpoint`` 暂存到
下一个报告边界再释放。两条路径必须做出完全相同的判定，否则暂存路径会绕过
``max_branch_depth`` / ``max_agent_calls_per_task`` 等安全上限。因此这里只产出
不可变的值对象计划，由 reducer 转成 effect、由 manager 转成 scheduler 入队。

本模块不访问 store、scheduler、时钟或随机数。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lunagentic_research_swarm.runtime.credits import allocate_children

# 只有这些拒绝原因可能在下一 turn 改变；深度/调用上限是确定性的，重试无意义。
RETRYABLE_EDGE_REASONS = frozenset({"agent_unavailable"})


@dataclass(frozen=True, slots=True)
class PlannedChild:
    """一个将要被物化的子分支。"""

    branch_id: str
    agent_id: str
    assignment: str
    credits: float
    depth: int
    messages: tuple[Mapping[str, Any], ...]
    # 委派完成后父节点退休（spec §10 步骤 12）；由第一个可启动的子节点承担。
    retire_parent: bool
    pool_return: float


@dataclass(frozen=True, slots=True)
class RejectedEdge:
    """一条不会成为分支的委派边；credits 从未离开父节点。"""

    branch_id: str
    agent_id: str
    assignment: str
    reason: str
    depth: int
    messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DelegationPlan:
    """一个 turn 的完整委派判定结果。"""

    children: tuple[PlannedChild, ...]
    rejected: tuple[RejectedEdge, ...]
    calls_after_reservation: int
    # 请求了委派但每一条都被拒绝：没有子分支能推进本分支。
    all_rejected: bool
    # 仅在 ``all_rejected`` 时有意义：None 表示可以附错误重试父节点一次，
    # 非 None 表示该原因是确定性的，必须直接终结父分支。
    blocking_reason: str | None = None

    @property
    def parent_retires(self) -> bool:
        return any(child.retire_parent for child in self.children)


def _positive_limit(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须为非负整数")
    return value


def rejected_edge_notice(rejected: Sequence[RejectedEdge]) -> dict[str, str]:
    """把全部被拒绝的委派边汇总成一条可读的 user 消息。"""

    lines = [
        f"- agent_id={edge.agent_id}；assignment={edge.assignment[:200]}；原因={edge.reason}"
        for edge in rejected
    ]
    return {
        "role": "user",
        "content": (
            "本 turn 请求的全部委派都无法启动，没有任何子分支被创建。"
            "分配给这些委派的 credits 已全额退回本分支。\n"
            + "\n".join(lines)
            + "\n请据此改用仍然可用的智能体、改为自行调用 Procedure，或结束本分支。"
        ),
    }


def plan_delegations(
    delegations: Sequence[Mapping[str, Any]],
    *,
    parent_branch_id: str,
    parent_depth: int,
    parent_credits: float,
    parent_messages: Sequence[Mapping[str, Any]] = (),
    live_agent_ids: Sequence[str] | frozenset[str] | None = None,
    agent_calls_started: int = 0,
    max_delegations_per_turn: int = 8,
    max_branch_depth: int = 32,
    max_agent_calls_per_task: int = 256,
) -> DelegationPlan:
    """对一批委派请求做结构判定与 credits 分配。

    被拒绝的边不参与分配：它们不会成为分支，份额留在父分支余额里，并随父节点
    退休一起结算回 dormant pool（spec §11.7 守恒）。
    """

    max_delegations = _positive_limit(max_delegations_per_turn, "max_delegations_per_turn")
    max_depth = _positive_limit(max_branch_depth, "max_branch_depth")
    max_calls = _positive_limit(max_agent_calls_per_task, "max_agent_calls_per_task")
    depth = _nonnegative_int(parent_depth, "parent_depth")
    started = _nonnegative_int(agent_calls_started, "agent_calls_started")
    live_agents = None if live_agent_ids is None else frozenset(str(item) for item in live_agent_ids)
    inherited = tuple(dict(item) for item in parent_messages)

    normalized: list[tuple[int, str, str, str, float, str | None]] = []
    allocatable: list[tuple[str, float]] = []
    reserved_calls = 0
    for index, raw in enumerate(delegations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"delegations[{index}] 必须为对象")
        agent_id = raw.get("agent_id")
        assignment = raw.get("task", raw.get("assignment"))
        requested_credits = raw.get("credits", 0.0)
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError(f"delegations[{index}].agent_id 必须为非空字符串")
        if not isinstance(assignment, str) or not assignment:
            raise ValueError(f"delegations[{index}].task 必须为非空字符串")
        if isinstance(requested_credits, bool) or not isinstance(requested_credits, int | float):
            raise ValueError(f"delegations[{index}].credits 必须为非负有限数")
        branch_id = str(raw.get("branch_id") or f"{parent_branch_id}:{index + 1}")

        reason: str | None = None
        if index >= max_delegations:
            reason = "delegation_limit_exceeded"
        elif depth >= max_depth:
            reason = "branch_depth_exceeded"
        elif live_agents is not None and agent_id not in live_agents:
            reason = "agent_unavailable"
        elif started + reserved_calls >= max_calls:
            reason = "agent_call_limit_exceeded"
        else:
            reserved_calls += 1
        normalized.append((index, branch_id, agent_id, assignment, float(requested_credits), reason))
        if reason is None:
            allocatable.append((str(index), float(requested_credits)))

    allocation = allocate_children(parent_credits, allocatable)
    first_live_index = next((index for index, *_rest, reason in normalized if reason is None), None)

    children: list[PlannedChild] = []
    rejected: list[RejectedEdge] = []
    for index, branch_id, agent_id, assignment, _requested, reason in normalized:
        if reason is not None:
            rejected.append(
                RejectedEdge(
                    branch_id=branch_id,
                    agent_id=agent_id,
                    assignment=assignment,
                    reason=reason,
                    depth=depth + 1,
                    messages=(
                        *inherited,
                        {"role": "user", "content": f"assignment: {assignment}\nedge finalized: {reason}"},
                    ),
                )
            )
            continue
        children.append(
            PlannedChild(
                branch_id=branch_id,
                agent_id=agent_id,
                assignment=assignment,
                credits=float(allocation.allocations.get(str(index), 0.0)),
                depth=depth + 1,
                messages=(*inherited, {"role": "user", "content": f"assignment: {assignment}"}),
                retire_parent=index == first_live_index,
                pool_return=float(allocation.returned_to_pool) if index == first_live_index else 0.0,
            )
        )

    all_rejected = bool(normalized) and first_live_index is None
    blocking_reason: str | None = None
    if all_rejected:
        reasons = {reason for *_rest, reason in normalized if reason is not None}
        blocking_reason = next(
            (reason for reason in sorted(reasons) if reason not in RETRYABLE_EDGE_REASONS), None
        )
        if blocking_reason is None and started + 1 > max_calls:
            blocking_reason = "agent_call_limit_exceeded"
    # 重试父节点也是一次正常 turn，必须占用调用预算。
    calls_after_reservation = started + reserved_calls
    if all_rejected and blocking_reason is None:
        calls_after_reservation = started + 1
    return DelegationPlan(
        children=tuple(children),
        rejected=tuple(rejected),
        calls_after_reservation=calls_after_reservation,
        all_rejected=all_rejected,
        blocking_reason=blocking_reason,
    )


__all__ = [
    "DelegationPlan",
    "PlannedChild",
    "RETRYABLE_EDGE_REASONS",
    "RejectedEdge",
    "plan_delegations",
    "rejected_edge_notice",
]
