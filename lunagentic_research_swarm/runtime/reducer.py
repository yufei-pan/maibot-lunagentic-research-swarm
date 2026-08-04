"""LRS 的单一、纯事件 reducer 与 transaction-before-effect 驱动。

`reduce_event` 只接收一个不可变快照和一个不可变事件，返回下一快照、持久化
commands 以及描述后续工作的 effect。它不会访问时钟、随机数、状态存储或
asyncio；worker 只负责产生事件，权威状态永远由这里产生。

本模块故意把 payload 保留为不可变 mapping。后续 scheduler 可以把 effect 当作
普通值转发，而不会因为 worker 修改了一个共享 dict 而改变已经提交的事件流。
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias

from lunagentic_research_swarm.errors import INVALID_STATE, STORAGE_COMMIT_FAILED, LRSError
from lunagentic_research_swarm.models import FormalizedTask, TaskSnapshot, TaskStatus
from lunagentic_research_swarm.llm.protocol import ProtocolError, build_correction_message
from lunagentic_research_swarm.runtime.credits import reconcile_usage, redistribute_pool, reserve_input
from lunagentic_research_swarm.runtime.events import (
    AgentCallCompleted,
    AgentCallFailed,
    AgentCallRequested,
    AgentCallReserved,
    AllInflightSettled,
    BranchCheckpointed,
    BranchFinalized,
    ContinueRequested,
    FinalReportCompleted,
    FinalReportFailed,
    FormalizationFailed,
    FormalizationSucceeded,
    GraceExpired,
    OutboxDelivered,
    PauseExpired,
    PauseExpiryReached,
    PauseRequested,
    PersistenceFailed,
    ProcedureBatchCompleted,
    ReportCompleted,
    ReportDeadlineReached,
    RuntimeEvent,
    StopRequested,
    SummaryCompleted,
    SummaryFailed,
    TaskCreated,
)
from lunagentic_research_swarm.storage.sqlite import StoreCommand


def _freeze(value: Any) -> Any:
    """递归冻结 effect/state 中会跨异步边界传递的 JSON 值。"""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _json_value(value.value)
    return str(value)


def _metadata_json(value: Mapping[str, Any] | None = None) -> str:
    metadata = _json_value(value or {})
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Reducer 使用的不可变任务快照。

    `TaskSnapshot` 是 foundation 的最小快照；本类型在不改变其字段语义的前提下
    携带 round/branch barrier 所需的运行时信息。`reduce_event` 同时接受两者，
    因而 foundation 消费方可以先从最小快照开始，TaskController 再逐步启用额外
    字段。
    """

    task_id: str
    status: TaskStatus
    formalized_task: FormalizedTask | None = None
    generation: int = 0
    active_round_id: str | None = None
    failure_code: str | None = None
    credit_pool: float = 0.0
    active_leaves: Mapping[str, float] = field(default_factory=dict)
    inflight_count: int = 0
    report_epoch: int = 0
    raw_context_released: bool = False
    continue_barrier: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_leaves", _freeze(self.active_leaves))

    @property
    def round_id(self) -> str | None:
        """`round_id` 是 `active_round_id` 的清晰别名。"""

        return self.active_round_id


# 这些别名让调用方不需要依赖某一个历史命名。
TaskRuntimeState = RuntimeState
ReducerState = RuntimeState


@dataclass(frozen=True, slots=True)
class Effect:
    """可由 scheduler 执行的显式 effect 描述。

    ``kind`` 对基类调用方可显式指定；具体 effect 留空时由类名推导，避免每个
    subclass 重复一份字符串常量。Effect 永远没有 coroutine/task 字段。
    """

    task_id: str
    round_id: str | None
    generation: int
    kind: str = ""
    priority: str = "normal"
    event_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    _KIND_MAP: ClassVar[Mapping[str, str]] = {
        "Effect": "generic",
        "PerformFormalization": "formalization",
        "PerformAgentCall": "agent",
        "PerformProcedureBatch": "procedure",
        "PerformBranchSummary": "branch_summary",
        "PerformTaskSummary": "task_summary",
        "OpenReportEpoch": "report",
        "DeliverOutbox": "outbox",
        "ArmDeadline": "deadline",
        "ArmPauseExpiry": "pause_expiry",
        "ReleaseRawContext": "release_raw_context",
        "NotifyToolWaiter": "notify",
    }

    def __post_init__(self) -> None:
        kind = self.kind or self._KIND_MAP.get(type(self).__name__, "generic")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", _freeze(self.payload))

    @property
    def branch_id(self) -> str | None:
        value = self.payload.get("branch_id")
        return str(value) if value is not None else None

    @property
    def call_id(self) -> str | None:
        value = self.payload.get("call_id")
        return str(value) if value is not None else None

    @property
    def report_id(self) -> str | None:
        value = self.payload.get("report_id")
        return str(value) if value is not None else None


class PerformFormalization(Effect):
    pass


class PerformAgentCall(Effect):
    pass


class PerformProcedureBatch(Effect):
    pass


class PerformBranchSummary(Effect):
    pass


class PerformTaskSummary(Effect):
    pass


class OpenReportEpoch(Effect):
    pass


class DeliverOutbox(Effect):
    pass


class ArmDeadline(Effect):
    pass


class ArmPauseExpiry(Effect):
    pass


class ReleaseRawContext(Effect):
    pass


class NotifyToolWaiter(Effect):
    pass


EffectUnion: TypeAlias = (
    PerformFormalization
    | PerformAgentCall
    | PerformProcedureBatch
    | PerformBranchSummary
    | PerformTaskSummary
    | OpenReportEpoch
    | DeliverOutbox
    | ArmDeadline
    | ArmPauseExpiry
    | ReleaseRawContext
    | NotifyToolWaiter
)


@dataclass(frozen=True, slots=True)
class Transition:
    """一次 reducer 转移的完整描述。"""

    next_state: Any
    commands: tuple[StoreCommand, ...] = ()
    effects: tuple[Effect, ...] = ()
    ignored: bool = False
    reason: str | None = None
    error: LRSError | None = None

    @classmethod
    def from_ignored(cls, state: Any, *, reason: str) -> Transition:
        return cls(state, ignored=True, reason=reason)

    @classmethod
    def ignored_result(cls, state: Any, *, reason: str) -> Transition:
        """兼容 brief 中 ``Transition.ignored(...)`` 的语义命名。"""

        return cls.from_ignored(state, reason=reason)


def _status_value(status: TaskStatus | str) -> str:
    return status.value if isinstance(status, TaskStatus) else str(status)


def _state_status(state: Any) -> TaskStatus:
    status = getattr(state, "status", None)
    if isinstance(status, TaskStatus):
        return status
    return TaskStatus(str(status))


def _state_task_id(state: Any) -> str:
    return str(getattr(state, "task_id"))


def _state_round_id(state: Any) -> str | None:
    value = getattr(state, "active_round_id", None)
    if value is None:
        value = getattr(state, "round_id", None)
    return str(value) if value is not None else None


def _state_generation(state: Any) -> int:
    return int(getattr(state, "generation", 0))


def _state_credit_pool(state: Any) -> float:
    try:
        return float(getattr(state, "credit_pool", getattr(state, "pool", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _state_leaves(state: Any) -> dict[str, float]:
    values = getattr(state, "active_leaves", {})
    if isinstance(values, Mapping):
        return {str(key): float(value) for key, value in values.items()}
    return {}


def _replace_state(state: Any, **changes: Any) -> Any:
    """在保留调用方快照类型的前提下生成新状态。"""

    if is_dataclass(state):
        # `round_id` 只在 RuntimeState 的 property 上存在，不可作为 dataclass 字段。
        valid = {item.name for item in fields(state)}
        return replace(state, **{key: value for key, value in changes.items() if key in valid})

    # 允许简单测试 harness 使用一个带属性的 frozen-ish object；不修改原对象。
    data = dict(getattr(state, "__dict__", {}))
    data.update(changes)
    try:
        return type(state)(**data)
    except Exception:
        # 最后一层兼容不可构造的 named state，仅用于状态显示；不直接修改输入。
        clone = object.__new__(type(state))
        for key, value in data.items():
            try:
                object.__setattr__(clone, key, value)
            except Exception:
                pass
        return clone


def _command(kind: str, values: Mapping[str, Any]) -> StoreCommand:
    return StoreCommand(kind, values)


def _lifecycle_event_command(
    event: RuntimeEvent,
    old: TaskStatus,
    new: TaskStatus,
    *,
    round_id: str | None = None,
) -> StoreCommand:
    """构造 lifecycle event 行；round_id 可指向同 transaction 新建的 round。"""

    event_values = {
        "event_id": event.event_id,
        "task_id": event.task_id,
        "round_id": event.round_id if round_id is None else round_id,
        "event_type": type(event).__name__,
        "from_status": old.value,
        "to_status": new.value,
        "metadata_json": _metadata_json(
            {name: getattr(event, name) for name in event.__dataclass_fields__ if name not in {"event_id", "task_id", "round_id", "generation", "occurred_at"}}
        ),
        "created_at": event.occurred_at.timestamp(),
    }
    return _command("insert_lifecycle_event", event_values)


def _lifecycle_commands(event: RuntimeEvent, old: TaskStatus, new: TaskStatus) -> tuple[StoreCommand, ...]:
    """生成同一 transaction 中的 round 状态与 lifecycle event 行。"""

    occurred_at = event.occurred_at.timestamp()
    ended_at = occurred_at if new in _TERMINAL_STATUSES else None
    return (
        _command(
            "update_round_status",
            {
                "round_id": event.round_id,
                "status": new.value,
                "report_deadline_at": None,
                "ended_at": ended_at,
            },
        ),
        _lifecycle_event_command(event, old, new),
    )


def _effect(
    effect_class: type[Effect],
    event: RuntimeEvent,
    *,
    priority: str = "normal",
    payload: Mapping[str, Any] | None = None,
    round_id: str | None = None,
    generation: int | None = None,
) -> Effect:
    return effect_class(
        task_id=event.task_id,
        round_id=event.round_id if round_id is None else round_id,
        generation=event.generation if generation is None else generation,
        priority=priority,
        event_id=event.event_id,
        payload=payload or {},
    )


_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.COMPLETED_WITH_ERRORS,
        TaskStatus.STOPPED,
        TaskStatus.EXPIRED,
        TaskStatus.INTERRUPTED,
        TaskStatus.FAILED,
    }
)


def _invalid(state: Any, event: RuntimeEvent, message: str) -> Transition:
    error = LRSError(INVALID_STATE, message, {"status": _status_value(_state_status(state)), "event": type(event).__name__})
    effect = _effect(
        NotifyToolWaiter,
        event,
        priority="barrier",
        payload={"error": error.to_result()["error"]},
        round_id=_state_round_id(state),
        generation=_state_generation(state),
    )
    return Transition(state, effects=(effect,), error=error, reason=INVALID_STATE)


def _validate_identity(state: Any, event: RuntimeEvent) -> Transition | None:
    if event.task_id != _state_task_id(state):
        return Transition.from_ignored(state, reason="task_mismatch")
    current_generation = _state_generation(state)
    if event.generation < current_generation:
        return Transition.from_ignored(state, reason="late_generation")
    if event.generation > current_generation:
        return Transition.from_ignored(state, reason="future_generation")
    current_round = _state_round_id(state)
    if current_round is not None and event.round_id != current_round:
        return Transition.from_ignored(state, reason="round_mismatch")
    return None


def _transition_status(
    state: Any,
    event: RuntimeEvent,
    new_status: TaskStatus,
    *,
    effects: Sequence[Effect] = (),
    extra_commands: Sequence[StoreCommand] = (),
    **state_changes: Any,
) -> Transition:
    old_status = _state_status(state)
    next_state = _replace_state(state, status=new_status, **state_changes)
    commands = (*extra_commands, *_lifecycle_commands(event, old_status, new_status))
    return Transition(next_state, tuple(commands), tuple(effects))


def _formalization_succeeded(state: Any, event: FormalizationSucceeded) -> Transition:
    try:
        formalized = FormalizedTask(event.formalized_text, event.formalized_sha256)
    except ValueError as exc:
        error = LRSError(INVALID_STATE, f"正式任务无效：{exc}")
        effect = _effect(NotifyToolWaiter, event, priority="barrier", payload={"error": error.to_result()["error"]})
        return Transition(state, effects=(effect,), error=error, reason=INVALID_STATE)
    commands = (
        _command(
            "update_task_formalization",
            {
                "task_id": event.task_id,
                "formalized_text": formalized.text,
                "formalized_sha256": formalized.sha256,
                "updated_at": event.occurred_at.timestamp(),
            },
        ),
    )
    root_effect = _effect(
        PerformAgentCall,
        event,
        payload={"formalized_text": formalized.text, "root": True},
    )
    return _transition_status(
        state,
        event,
        TaskStatus.RUNNING,
        effects=(root_effect,),
        extra_commands=commands,
        formalized_task=formalized,
        failure_code=None,
    )


def _formalization_failed(state: Any, event: FormalizationFailed) -> Transition:
    return _transition_status(
        state,
        event,
        TaskStatus.FAILED,
        effects=(
            _effect(
                NotifyToolWaiter,
                event,
                priority="barrier",
                payload={"error_code": event.error_code, "error_message": event.error_message},
            ),
        ),
        failure_code=event.error_code or "formalization_failed",
    )


def _continue_terminal(state: Any, event: ContinueRequested) -> Transition:
    """终态 continue 屏障：有叶子再分配，无叶子创建新 round 或明确不足。"""

    # ``None`` means the caller omitted a snapshot; an explicit empty mapping means
    # there are no active leaves and must be allowed to start a new round.
    leaves = dict(event.active_leaves) if event.active_leaves is not None else _state_leaves(state)
    redistribution = redistribute_pool(_state_credit_pool(state), event.adjustment, leaves)
    if leaves:
        # 负叶子在 barrier 处保持终结；非负叶子允许继续运行。
        balances = dict(redistribution.balances)
        running = {key: value for key, value in balances.items() if value >= 0}
        return _transition_status(
            state,
            event,
            TaskStatus.RUNNING,
            active_leaves=running,
            credit_pool=redistribution.pool_after,
            continue_barrier=False,
        )

    restart_balance = redistribution.restart_balance
    if restart_balance is None or not redistribution.can_start_root:
        error = LRSError(
            "task_finished_insufficient_funds",
            "任务没有可用于新 round 的 credits",
            {"balance": restart_balance},
        )
        effect = _effect(
            NotifyToolWaiter,
            event,
            priority="barrier",
            payload={"error": error.to_result()["error"]},
        )
        return Transition(
            _replace_state(state, failure_code="task_finished_insufficient_funds", credit_pool=redistribution.pool_after),
            effects=(effect,),
            error=error,
            reason=error.code,
        )

    # Reducer 不产生 id；调用方必须把新 round 标识放进 ContinueRequested。
    next_round_id = event.next_round_id
    if not next_round_id:
        return _invalid(state, event, "ContinueRequested 缺少 next_round_id")
    next_generation = event.next_generation
    if next_generation is None:
        next_generation = _state_generation(state) + 1
    next_round = _command(
        "insert_round",
        {
            "round_id": next_round_id,
            "task_id": event.task_id,
            "round_number": event.round_number,
            "generation": next_generation,
            "status": TaskStatus.RUNNING.value,
            "time_budget_seconds": event.time_budget_seconds,
            "grace_period_seconds": event.grace_period_seconds,
            "credit_pool": 0.0,
            "catalog_fingerprint": event.catalog_fingerprint,
            "started_at": event.occurred_at.timestamp(),
            "report_deadline_at": None,
            "ended_at": None,
        },
    )
    current_round_number = _command(
        "set_task_current_round",
        {
            "task_id": event.task_id,
            "current_round_number": event.round_number,
            "updated_at": event.occurred_at.timestamp(),
        },
    )
    root_effect = _effect(
        PerformAgentCall,
        event,
        round_id=next_round_id,
        generation=next_generation,
        payload={"root": True, "credit_balance": restart_balance},
    )
    # This is a new round transition.  The old terminal round remains untouched;
    # only the INSERT and the new round's lifecycle row are committed here.
    return Transition(
        _replace_state(
            state,
            status=TaskStatus.RUNNING,
            active_round_id=next_round_id,
            generation=next_generation,
            credit_pool=0.0,
            active_leaves={},
            failure_code=None,
            continue_barrier=False,
        ),
        commands=(
            next_round,
            current_round_number,
            _lifecycle_event_command(
                event,
                _state_status(state),
                TaskStatus.RUNNING,
                round_id=next_round_id,
            ),
        ),
        effects=(root_effect,),
    )


def reduce_event(state: Any, event: RuntimeEvent) -> Transition:
    """对一个事件做纯状态转移。

    任何身份不匹配或迟到 generation 都是显式 ignored transition；只有当前状态
    白名单中的事件才能改变状态，其他组合返回 ``invalid_state``，不会猜测一个
    看似合理的隐式转移。
    """

    identity = _validate_identity(state, event)
    if identity is not None:
        return identity

    status = _state_status(state)

    if isinstance(event, TaskCreated):
        if status is not TaskStatus.FORMALIZING:
            return _invalid(state, event, "TaskCreated 只能用于 FORMALIZING")
        return Transition(state)

    if isinstance(event, FormalizationSucceeded):
        if status is not TaskStatus.FORMALIZING:
            return _invalid(state, event, "FormalizationSucceeded 只能用于 FORMALIZING")
        return _formalization_succeeded(state, event)

    if isinstance(event, FormalizationFailed):
        if status is not TaskStatus.FORMALIZING:
            return _invalid(state, event, "FormalizationFailed 只能用于 FORMALIZING")
        return _formalization_failed(state, event)

    if isinstance(event, PauseRequested):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING}:
            return _invalid(state, event, "PauseRequested 只能用于 RUNNING/REPORTING")
        effects: list[Effect] = []
        if event.expires_at is not None:
            effects.append(_effect(ArmPauseExpiry, event, priority="barrier", payload={"due_at": event.expires_at}))
        return _transition_status(state, event, TaskStatus.PAUSING, effects=effects)

    if isinstance(event, AllInflightSettled):
        if status is not TaskStatus.PAUSING:
            return _invalid(state, event, "AllInflightSettled 只能用于 PAUSING")
        return _transition_status(state, event, TaskStatus.PAUSED, inflight_count=0)

    if isinstance(event, (PauseExpired, PauseExpiryReached)):
        if status is not TaskStatus.PAUSED:
            return _invalid(state, event, "PauseExpired 只能用于 PAUSED")
        return _transition_status(
            state,
            event,
            TaskStatus.EXPIRED,
            effects=(_effect(ReleaseRawContext, event, priority="barrier"),),
            raw_context_released=True,
        )

    if isinstance(event, StopRequested):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING, TaskStatus.PAUSED}:
            return _invalid(state, event, "StopRequested 只能用于活跃 round")
        next_generation = _state_generation(state) + 1
        commands = (
            _command(
                "update_round_generation",
                {"round_id": event.round_id, "generation": next_generation},
            ),
        )
        return _transition_status(
            state,
            event,
            TaskStatus.STOPPED,
            effects=(_effect(ReleaseRawContext, event, priority="barrier", generation=next_generation),),
            extra_commands=commands,
            generation=next_generation,
            raw_context_released=True,
            inflight_count=0,
        )

    if isinstance(event, ReportDeadlineReached):
        if status is not TaskStatus.RUNNING:
            return _invalid(state, event, "ReportDeadlineReached 只能用于 RUNNING")
        epoch = event.epoch if event.epoch is not None else int(getattr(state, "report_epoch", 0)) + 1
        effects = (
            _effect(OpenReportEpoch, event, priority="barrier", payload={"epoch": epoch, "kind": "INTERMEDIATE"}),
        )
        return _transition_status(state, event, TaskStatus.REPORTING, effects=effects, report_epoch=epoch)

    if isinstance(event, ReportCompleted):
        if status is not TaskStatus.REPORTING:
            return _invalid(state, event, "ReportCompleted 只能用于 REPORTING")
        return _transition_status(state, event, TaskStatus.RUNNING)

    if isinstance(event, GraceExpired):
        if status is not TaskStatus.REPORTING:
            return _invalid(state, event, "GraceExpired 只能用于 REPORTING")
        return _transition_status(
            state,
            event,
            TaskStatus.FINALIZING,
            effects=(_effect(PerformTaskSummary, event, priority="barrier", payload={"kind": "FINAL"}),),
        )

    if isinstance(event, FinalReportCompleted):
        if status is not TaskStatus.FINALIZING:
            return _invalid(state, event, "FinalReportCompleted 只能用于 FINALIZING")
        return _transition_status(
            state,
            event,
            TaskStatus.COMPLETED,
            effects=(_effect(DeliverOutbox, event, priority="barrier", payload={"report_id": event.report_id}),),
        )

    if isinstance(event, FinalReportFailed):
        if status is not TaskStatus.FINALIZING:
            return _invalid(state, event, "FinalReportFailed 只能用于 FINALIZING")
        return _transition_status(
            state,
            event,
            TaskStatus.COMPLETED_WITH_ERRORS,
            effects=(
                _effect(
                    NotifyToolWaiter,
                    event,
                    priority="barrier",
                    payload={"error_code": event.error_code, "error_message": event.error_message},
                ),
            ),
            failure_code=event.error_code or "final_report_failed",
        )

    if isinstance(event, ContinueRequested):
        if status is TaskStatus.PAUSED:
            return _transition_status(state, event, TaskStatus.RUNNING, continue_barrier=False)
        if status in _TERMINAL_STATUSES:
            return _continue_terminal(state, event)
        return _invalid(state, event, "ContinueRequested 只能用于 PAUSED 或 round 终态")

    # Worker completion events are deliberately accepted only in active states. They
    # never mutate authoritative state directly; they merely schedule the next explicit
    # phase. A stopped/new-generation event was already filtered above.
    if isinstance(event, AgentCallRequested):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING}:
            return _invalid(state, event, "AgentCallRequested 只能用于 RUNNING/REPORTING")
        try:
            credits_after_reservation = event.balance_before - event.estimated_charge
            reservation = reserve_input(
                event.estimated_charge,
                task_id=event.task_id,
                round_id=event.round_id,
                branch_id=event.branch_id,
                call_id=event.call_id,
                usage_id=event.usage_id or None,
                ledger_id=event.ledger_id or None,
                role="agent",
                selector=event.selector,
                estimated_model_name=event.estimated_model_name or None,
                price_source=event.price_source,
                price_fingerprint=event.price_fingerprint,
                prompt_tokens=event.prompt_tokens,
                cache_hit_tokens=event.cache_hit_tokens,
                cache_miss_tokens=event.cache_miss_tokens,
                balance_after=credits_after_reservation,
                metadata={"event_type": "AgentCallReserved", "agent_id": event.agent_id},
                created_at=event.occurred_at,
            )
        except (TypeError, ValueError) as exc:
            return _invalid(state, event, f"AgentCallRequested reservation 无效：{exc}")
        reserved_event_command = _lifecycle_event_command(event, status, status)
        return Transition(
            state,
            commands=(*reservation.commands, reserved_event_command),
            effects=(
                _effect(
                    PerformAgentCall,
                    event,
                    payload={
                        "branch_id": event.branch_id,
                        "call_id": event.call_id,
                        "agent_id": event.agent_id,
                        "selector": event.selector,
                        "protocol": event.protocol,
                        "messages": event.messages,
                        "estimated_charge": event.estimated_charge,
                        "credits_after_reservation": credits_after_reservation,
                        "correction_count": event.correction_count,
                        "pinning_supported": event.pinning_supported,
                    },
                ),
            ),
        )

    if isinstance(event, AgentCallReserved):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING}:
            return _invalid(state, event, "AgentCallReserved 只能用于活跃 round")
        return Transition(state)

    if isinstance(event, AgentCallCompleted):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING}:
            return _invalid(state, event, "AgentCallCompleted 只能用于活跃 round")
        actual_charge = event.actual_charge
        if actual_charge is None:
            actual_charge = event.estimated_charge
        balance_after = event.balance_before_reconciliation + event.estimated_charge - actual_charge
        try:
            reconciliation = reconcile_usage(
                estimated_charge=event.estimated_charge,
                actual_charge=actual_charge,
                actual_model_name=event.actual_model_name or None,
                usage=event.usage,
                success=True,
                task_id=event.task_id,
                round_id=event.round_id,
                branch_id=event.branch_id,
                call_id=event.call_id,
                role="agent",
                balance_after=balance_after,
                created_at=event.occurred_at,
            )
        except (TypeError, ValueError) as exc:
            return _invalid(state, event, f"AgentCallCompleted reconciliation 无效：{exc}")
        commands: list[StoreCommand] = list(reconciliation.commands)
        if event.protocol_error is not None:
            if balance_after < 0:
                return Transition(
                    state,
                    commands=tuple(commands),
                    effects=(
                        _effect(
                            PerformBranchSummary,
                            event,
                            priority="barrier",
                            payload={"branch_id": event.branch_id, "reason": "negative_credit"},
                        ),
                    ),
                )
            can_correct = (
                event.correction_count == 0
                and event.pinning_supported
                and bool(event.actual_model_name)
            )
            if not can_correct:
                return Transition(
                    state,
                    commands=tuple(commands),
                    effects=(
                        _effect(
                            PerformBranchSummary,
                            event,
                            priority="barrier",
                            payload={"branch_id": event.branch_id, "reason": "protocol_invalid"},
                        ),
                    ),
                )
            raw_errors = event.protocol_error.get("errors", ())
            errors = tuple(item for item in raw_errors if isinstance(item, Mapping))
            protocol_error = ProtocolError(str(event.protocol_error.get("message", "协议无效")), errors)
            correction_message = build_correction_message(protocol_error)
            correction_call_id = event.correction_call_id or f"{event.call_id}:correction"
            correction_estimate = event.correction_estimated_charge
            correction_balance = balance_after - correction_estimate
            correction_reservation = reserve_input(
                correction_estimate,
                task_id=event.task_id,
                round_id=event.round_id,
                branch_id=event.branch_id,
                call_id=correction_call_id,
                usage_id=event.correction_usage_id or None,
                ledger_id=event.correction_ledger_id or None,
                role="agent",
                selector=f"model:{event.actual_model_name}",
                estimated_model_name=event.actual_model_name,
                prompt_tokens=0,
                balance_after=correction_balance,
                metadata={"event_type": "AgentCallReserved", "correction_count": 1},
                created_at=event.occurred_at,
            )
            commands.extend(correction_reservation.commands)
            messages = (*event.messages, correction_message)
            return Transition(
                state,
                commands=tuple(commands),
                effects=(
                    _effect(
                        PerformAgentCall,
                        event,
                        payload={
                            "branch_id": event.branch_id,
                            "call_id": correction_call_id,
                            "selector": f"model:{event.actual_model_name}",
                            "protocol": "json_envelope",
                            "messages": messages,
                            "estimated_charge": correction_estimate,
                            "credits_after_reservation": correction_balance,
                            "correction_count": 1,
                            "pinning_supported": True,
                        },
                    ),
                ),
            )
        result = dict(event.protocol_result or {})
        return Transition(
            state,
            commands=tuple(commands),
            effects=(
                _effect(
                    PerformProcedureBatch,
                    event,
                    payload={
                        "branch_id": event.branch_id,
                        "call_id": event.call_id,
                        "result_id": event.result_id,
                        "report": result.get("report", ""),
                        "requests": tuple(result.get("procedures", ())),
                        "delegations": tuple(result.get("delegations", ())),
                        "credits_after": balance_after,
                    },
                ),
            ),
        )

    if isinstance(event, AgentCallFailed):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING}:
            return _invalid(state, event, "AgentCallFailed 只能用于活跃 round")
        return Transition(
            state,
            effects=(
                _effect(
                    PerformBranchSummary,
                    event,
                    priority="barrier",
                    payload={"branch_id": event.branch_id, "reason": event.error_code or "agent_failed"},
                ),
            ),
        )

    if isinstance(event, ProcedureBatchCompleted):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING}:
            return _invalid(state, event, "ProcedureBatchCompleted 只能用于活跃 round")
        commands: list[StoreCommand] = []
        for item in event.results:
            result = getattr(item, "result", None)
            success = bool(getattr(result, "success", False))
            error = getattr(result, "error", None)
            metadata = getattr(result, "metadata", {})
            error_code = None
            if isinstance(error, Mapping):
                error_code = str(error.get("code") or "procedure_failed")
            external_cost = metadata.get("external_cost") if isinstance(metadata, Mapping) else None
            commands.append(
                _command(
                    "insert_procedure_call",
                    {
                        "request_id": str(getattr(item, "request_id", "")),
                        "task_id": event.task_id,
                        "round_id": event.round_id,
                        "branch_id": event.branch_id,
                        "turn_id": event.call_id,
                        "agent_id": str(metadata.get("agent_id", "lrs.executor"))
                        if isinstance(metadata, Mapping)
                        else "lrs.executor",
                        "procedure_id": str(getattr(item, "procedure_id", "")),
                        "provider_plugin_id": str(getattr(item, "provider_plugin_id", "")),
                        "status": "succeeded" if success else "failed",
                        "duration_ms": int(getattr(item, "duration_ms", 0)),
                        "error_code": error_code,
                        "provenance_json": _metadata_json(
                            {
                                "api_name": str(getattr(item, "api_name", "")),
                                "api_version": str(getattr(item, "api_version", "1")),
                                "attempts": int(getattr(item, "attempts", 1)),
                            }
                        ),
                        "external_cost_json": _metadata_json(external_cost)
                        if isinstance(external_cost, Mapping)
                        else None,
                        "created_at": event.occurred_at.timestamp(),
                    },
                )
            )

        next_state = state
        credits_after = event.credits_after
        if bool(getattr(state, "continue_barrier", False)):
            leaves = _state_leaves(state)
            leaves[event.branch_id] = credits_after
            redistribution = redistribute_pool(_state_credit_pool(state), 0.0, leaves)
            credits_after = redistribution.balances.get(event.branch_id, credits_after)
            next_state = _replace_state(
                state,
                active_leaves={key: value for key, value in redistribution.balances.items() if value >= 0},
                credit_pool=redistribution.pool_after,
                continue_barrier=False,
            )

        controls = event.controls
        terminate = bool(getattr(controls, "terminate", False))
        compact = bool(getattr(controls, "compact", False))
        checkpoint = bool(getattr(controls, "checkpoint", False))
        held = tuple(event.delegations)
        if terminate:
            reason = "terminate"
        elif compact:
            reason = "compact"
        elif credits_after < 0:
            reason = "negative_credit"
        elif checkpoint:
            reason = "checkpoint"
        elif not held:
            reason = "no_further_work"
        else:
            # Child materialization is performed by the transaction-aware controller
            # from this immutable payload; it must submit AgentCallRequested before a
            # scheduler can receive PerformAgentCall.
            return Transition(
                next_state,
                commands=tuple(commands),
                effects=(
                    _effect(
                        NotifyToolWaiter,
                        event,
                        priority="barrier",
                        payload={
                            "action": "materialize_children",
                            "branch_id": event.branch_id,
                            "delegations": held,
                            "credits_after": credits_after,
                        },
                    ),
                ),
            )
        return Transition(
            next_state,
            commands=tuple(commands),
            effects=(
                _effect(
                    PerformBranchSummary,
                    event,
                    priority="barrier",
                    payload={
                        "branch_id": event.branch_id,
                        "reason": reason,
                        "held_delegations": held if reason in {"compact", "checkpoint"} else (),
                    },
                ),
            ),
        )

    if isinstance(event, (SummaryCompleted, SummaryFailed, BranchCheckpointed, BranchFinalized)):
        if status not in {TaskStatus.RUNNING, TaskStatus.REPORTING, TaskStatus.PAUSING, TaskStatus.FINALIZING}:
            return _invalid(state, event, "summary/branch event 只能用于活跃 round")
        if isinstance(event, BranchCheckpointed):
            return Transition(state)
        return Transition(state)

    if isinstance(event, OutboxDelivered):
        if status not in {TaskStatus.COMPLETED, TaskStatus.COMPLETED_WITH_ERRORS}:
            return _invalid(state, event, "OutboxDelivered 只能用于已完成 round")
        # Delivery is idempotent and does not reopen a completed round.
        return Transition(state)

    if isinstance(event, PersistenceFailed):
        return _transition_status(
            state,
            event,
            TaskStatus.FAILED,
            effects=(_effect(NotifyToolWaiter, event, priority="barrier", payload={"error_code": STORAGE_COMMIT_FAILED}),),
            failure_code=event.error_code or STORAGE_COMMIT_FAILED,
        )

    return _invalid(state, event, f"事件 {type(event).__name__} 在 {_status_value(status)} 状态不可用")


class TaskController:
    """把 reducer 接到一个串行 inbox；持久化成功后才暴露状态和 effect。

    这是 Task 2 的最小驱动。Task 6 可以在此基础上加入 manager、scheduler
    生命周期和 Tool-facing API，而不需要复制 transaction ordering。
    """

    def __init__(
        self,
        state: Any,
        *,
        store: Any,
        scheduler: Any | None = None,
        executor: Any | None = None,
        health: Any = None,
    ) -> None:
        self.state = state
        self.store = store
        # ``executor`` is accepted as a compatibility name used by the first
        # persistence harness; both objects expose the same enqueue contract.
        self.scheduler = scheduler if scheduler is not None else executor
        if self.scheduler is None:
            raise TypeError("TaskController 需要 scheduler/executor")
        self.health = health
        self._inbox: deque[RuntimeEvent] = deque()
        self.stopped = False

    async def submit(self, event: RuntimeEvent) -> bool:
        if self.stopped:
            # A failed best-effort FAILED write is a terminal controller failure;
            # accepting another event here could launch work against unknown state.
            return False
        self._inbox.append(event)
        return True

    async def drain_once(self) -> bool:
        if self.stopped:
            self._inbox.clear()
            return False
        if not self._inbox:
            return False
        event = self._inbox.popleft()
        await self._apply(event)
        return True

    async def drain(self) -> None:
        while self._inbox and not self.stopped:
            await self.drain_once()

    async def _apply(self, event: RuntimeEvent) -> None:
        transition = reduce_event(self.state, event)
        if transition.ignored:
            # Late generations are deliberately no-ops: don't even open an empty
            # transaction, and never publish an effect for the old round.
            self.state = transition.next_state
            return
        try:
            await self.store.transact(transition.commands)
        except Exception as exc:
            await self._fail_after_storage_error(event, exc)
            return

        # This assignment is intentionally after transact and before enqueue: an effect
        # can never observe an in-memory state that SQLite rejected.
        self.state = transition.next_state
        for effect in transition.effects:
            await self.scheduler.enqueue(effect)

    async def _fail_after_storage_error(self, event: RuntimeEvent, exc: Exception) -> None:
        failed_state = _replace_state(
            self.state,
            status=TaskStatus.FAILED,
            failure_code=STORAGE_COMMIT_FAILED,
        )
        self.state = failed_state
        round_id = _state_round_id(self.state) or event.round_id
        fallback = (
            _command(
                "update_round_status",
                {
                    "round_id": round_id,
                    "status": TaskStatus.FAILED.value,
                    "report_deadline_at": None,
                    "ended_at": event.occurred_at.timestamp(),
                },
            ),
        )
        try:
            await self.store.transact(fallback)
        except Exception as fallback_exc:
            self.stopped = True
            self._inbox.clear()
            self._record_health(
                {
                    "status": "degraded",
                    "code": STORAGE_COMMIT_FAILED,
                    "message": str(fallback_exc),
                    "original": str(exc),
                }
            )

    def _record_health(self, payload: Mapping[str, Any]) -> None:
        target = self.health
        if target is None:
            return
        if callable(target):
            target(payload)
            return
        if isinstance(target, dict):
            target["runtime"] = dict(payload)
            return
        try:
            setattr(target, "runtime", dict(payload))
        except Exception:
            pass


__all__ = [
    "ArmDeadline",
    "ArmPauseExpiry",
    "DeliverOutbox",
    "Effect",
    "EffectUnion",
    "NotifyToolWaiter",
    "OpenReportEpoch",
    "PerformAgentCall",
    "PerformBranchSummary",
    "PerformFormalization",
    "PerformProcedureBatch",
    "PerformTaskSummary",
    "ReleaseRawContext",
    "ReducerState",
    "RuntimeState",
    "TaskController",
    "TaskRuntimeState",
    "Transition",
    "reduce_event",
]
