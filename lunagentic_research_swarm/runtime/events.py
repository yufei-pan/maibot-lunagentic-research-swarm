"""只携带不可变输入的显式运行期事件与 JSON 编解码。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, TypeAlias

from pydantic import BaseModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    task_id: str
    round_id: str
    generation: int
    occurred_at: datetime = field(default_factory=_now)


EVENT_REGISTRY: dict[str, type[Event]] = {}


def _register(cls: type[Event]) -> type[Event]:
    EVENT_REGISTRY[cls.__name__] = cls
    return cls


def _freeze_value(value: Any) -> Any:
    """递归冻结事件中的 JSON 数据，避免发布后被调用方改写。"""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    """将冻结的 JSON 数据还原为标准 JSON 值。"""

    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_value(value.as_dict())
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@_register
@dataclass(frozen=True, slots=True)
class TaskCreated(Event):
    raw_task_text: str = ""


@_register
@dataclass(frozen=True, slots=True)
class FormalizationSucceeded(Event):
    formalized_text: str = ""
    formalized_sha256: str = ""


@_register
@dataclass(frozen=True, slots=True)
class FormalizationFailed(Event):
    error_code: str = ""
    error_message: str = ""


@_register
@dataclass(frozen=True, slots=True)
class AgentCallRequested(Event):
    branch_id: str = ""
    call_id: str = ""
    agent_id: str = ""
    selector: str = ""
    protocol: str = "json_envelope"
    messages: tuple[Mapping[str, Any], ...] = ()
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    estimated_charge: float = 0.0
    balance_before: float = 0.0
    usage_id: str = ""
    ledger_id: str = ""
    price_source: str = "host_config"
    price_fingerprint: str = ""
    estimated_model_name: str = ""
    correction_count: int = 0
    pinning_supported: bool = True
    branch_depth: int = 0
    live_agent_ids: tuple[str, ...] | None = None
    max_delegations_per_turn: int = 8
    max_branch_depth: int = 32
    max_agent_calls_per_task: int = 256
    agent_calls_started: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(_freeze_value(self.messages)))
        if self.live_agent_ids is not None:
            object.__setattr__(self, "live_agent_ids", tuple(str(item) for item in self.live_agent_ids))


@_register
@dataclass(frozen=True, slots=True)
class AgentCallReserved(Event):
    branch_id: str = ""
    call_id: str = ""
    ledger_id: str = ""
    reserved_credits: float = 0.0


@_register
@dataclass(frozen=True, slots=True)
class AgentCallCompleted(Event):
    branch_id: str = ""
    call_id: str = ""
    result_id: str = ""
    usage: Mapping[str, Any] | None = None
    actual_model_name: str = ""
    actual_charge: float | None = None
    estimated_charge: float = 0.0
    balance_before_reconciliation: float = 0.0
    protocol: str = "json_envelope"
    protocol_result: Mapping[str, Any] | None = None
    protocol_error: Mapping[str, Any] | None = None
    protocol_repairs: tuple[str, ...] = ()
    correction_count: int = 0
    correction_call_id: str = ""
    correction_usage_id: str = ""
    correction_ledger_id: str = ""
    correction_estimated_charge: float = 0.0
    max_correction_turns: int = 1
    pinning_supported: bool = True
    messages: tuple[Mapping[str, Any], ...] = ()
    branch_depth: int = 0
    live_agent_ids: tuple[str, ...] | None = None
    max_delegations_per_turn: int = 8
    max_branch_depth: int = 32
    max_agent_calls_per_task: int = 256
    agent_calls_started: int = 0

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", _freeze_value(self.usage))
        if self.protocol_result is not None:
            object.__setattr__(self, "protocol_result", _freeze_value(self.protocol_result))
        if self.protocol_error is not None:
            object.__setattr__(self, "protocol_error", _freeze_value(self.protocol_error))
        object.__setattr__(self, "protocol_repairs", tuple(str(item) for item in self.protocol_repairs))
        object.__setattr__(self, "messages", tuple(_freeze_value(self.messages)))
        if self.live_agent_ids is not None:
            object.__setattr__(self, "live_agent_ids", tuple(str(item) for item in self.live_agent_ids))


@_register
@dataclass(frozen=True, slots=True)
class AgentCallFailed(Event):
    branch_id: str = ""
    call_id: str = ""
    error_code: str = ""
    error_message: str = ""
    usage: Mapping[str, Any] | None = None
    actual_model_name: str = ""
    actual_charge: float | None = None
    estimated_charge: float = 0.0
    balance_before_reconciliation: float = 0.0
    selector: str = ""

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", _freeze_value(self.usage))


@_register
@dataclass(frozen=True, slots=True)
class ProcedureBatchCompleted(Event):
    branch_id: str = ""
    call_id: str = ""
    result_id: str = ""
    # 实际元素由 procedures.executor 提供；Any 用于解除 events ↔ procedures 的
    # 导入环，executor 完成时已使用 ProcedureResult.model_validate() 校验。
    results: tuple[Any, ...] = ()
    controls: Any | None = None
    report: str = ""
    delegations: tuple[Mapping[str, Any], ...] = ()
    credits_after: float = 0.0
    parent_messages: tuple[Mapping[str, Any], ...] = ()
    parent_depth: int = 0
    live_agent_ids: tuple[str, ...] | None = None
    max_delegations_per_turn: int = 8
    max_branch_depth: int = 32
    max_agent_calls_per_task: int = 256
    agent_calls_started: int = 0

    def __post_init__(self) -> None:
        normalized: list[Any] = []
        for item in self.results:
            if isinstance(item, Mapping) and "result" in item and "procedure_id" in item:
                # 只在解码 persisted event 时走这个兼容分支；执行器本身已经返回
                # frozen ProcedureExecutionResult 值对象。
                try:
                    from lunagentic_research_swarm.extensions.contracts import ProcedureResult
                    from lunagentic_research_swarm.procedures.executor import ProcedureExecutionResult

                    normalized.append(
                        ProcedureExecutionResult(
                            procedure_id=str(item["procedure_id"]),
                            request_id=str(item.get("request_id", "")),
                            result=ProcedureResult.model_validate(item["result"], strict=True),
                            provider_plugin_id=str(item.get("provider_plugin_id", "")),
                            api_name=str(item.get("api_name", "")),
                            api_version=str(item.get("api_version", "1")),
                            attempts=int(item.get("attempts", 1)),
                            duration_ms=int(item.get("duration_ms", 0)),
                        )
                    )
                    continue
                except (ImportError, TypeError, ValueError):
                    pass
            normalized.append(item)
        object.__setattr__(self, "results", tuple(_freeze_value(normalized)))
        object.__setattr__(self, "delegations", tuple(_freeze_value(self.delegations)))
        object.__setattr__(self, "parent_messages", tuple(_freeze_value(self.parent_messages)))
        if self.live_agent_ids is not None:
            object.__setattr__(self, "live_agent_ids", tuple(str(item) for item in self.live_agent_ids))
        if isinstance(self.controls, Mapping):
            try:
                from lunagentic_research_swarm.procedures.core import CoreProcedureDecision

                object.__setattr__(
                    self,
                    "controls",
                    CoreProcedureDecision(
                        compact=bool(self.controls.get("compact", False)),
                        checkpoint=bool(self.controls.get("checkpoint", False)),
                        terminate=bool(self.controls.get("terminate", False)),
                        ignored_controls=self.controls.get("ignored_controls", ()),
                        control_requests=self.controls.get("control_requests", ()),
                    ),
                )
            except (ImportError, TypeError, ValueError):
                object.__setattr__(self, "controls", _freeze_value(self.controls))


@_register
@dataclass(frozen=True, slots=True)
class ChildMaterialized(Event):
    # 暂存 checkpoint 释放的子分支不经过 ProcedureBatchCompleted，因此由本事件
    # 单调推进 Task 级 agent 调用计数，避免绕过 max_agent_calls_per_task。
    agent_calls_started: int = 0
    branch_id: str = ""
    parent_branch_id: str = ""
    agent_id: str = ""
    credits: float = 0.0
    depth: int = 0
    retire_parent: bool = False
    pool_return: float = 0.0
    # When the parent stays alive (checkpoint/grace release), set its new leaf
    # balance so child allocations remain a transfer rather than minting credits.
    parent_credits_after: float | None = None


@_register
@dataclass(frozen=True, slots=True)
class SummaryCompleted(Event):
    branch_id: str = ""
    summary_id: str = ""


@_register
@dataclass(frozen=True, slots=True)
class SummaryFailed(Event):
    branch_id: str = ""
    error_code: str = ""
    error_message: str = ""


@_register
@dataclass(frozen=True, slots=True)
class BranchCheckpointed(Event):
    branch_id: str = ""
    checkpoint_id: str = ""


@_register
@dataclass(frozen=True, slots=True)
class BranchFinalized(Event):
    branch_id: str = ""
    summary_id: str = ""
    reason: str = ""


@_register
@dataclass(frozen=True, slots=True)
class ReportDeadlineReached(Event):
    """Wall-clock report budget elapsed; opens the next report epoch."""

    epoch: int | None = None


@_register
@dataclass(frozen=True, slots=True)
class FinalEpochCommitted(Event):
    """Commit FINALIZING + optional report_epoch after final synthesis (not a deadline).

    ``epoch == current``: same-epoch FINAL freeze (status only).
    ``epoch == current + 1``: bump durable report_epoch without reopening a frontier.
    """

    epoch: int | None = None


@_register
@dataclass(frozen=True, slots=True)
class GraceExpired(Event):
    # ``None`` preserves persisted events emitted before epochs were carried
    # by the grace timer.  New timers must include the epoch they armed for.
    epoch: int | None = None


@_register
@dataclass(frozen=True, slots=True)
class ReportCompleted(Event):
    report_id: str = ""


@_register
@dataclass(frozen=True, slots=True)
class FinalReportCompleted(Event):
    report_id: str = ""


@_register
@dataclass(frozen=True, slots=True)
class FinalReportFailed(Event):
    error_code: str = ""
    error_message: str = ""


@_register
@dataclass(frozen=True, slots=True)
class AllInflightSettled(Event):
    pass


@_register
@dataclass(frozen=True, slots=True)
class PauseRequested(Event):
    # due_at 必须由 controller/clock 在构造事件时携带；reducer 不读取时间。
    expires_at: float | None = None


@_register
@dataclass(frozen=True, slots=True)
class PauseExpired(Event):
    """暂停超时；只允许 PAUSED -> EXPIRED，不触发总结或反馈。"""

    pass


@_register
@dataclass(frozen=True, slots=True)
class PauseExpiryReached(Event):
    """PauseExpired 的显式兼容名称，便于外部 clock adapter 直译事件。"""

    pass


@_register
@dataclass(frozen=True, slots=True)
class ContinueRequested(Event):
    adjustment: float = 0.0
    active_leaves: Mapping[str, float] | None = None
    next_round_id: str | None = None
    next_generation: int | None = None
    round_number: int = 1
    time_budget_seconds: int = 120
    grace_period_seconds: int = 60
    catalog_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.active_leaves is not None:
            object.__setattr__(self, "active_leaves", _freeze_value(self.active_leaves))


@_register
@dataclass(frozen=True, slots=True)
class StopRequested(Event):
    reason: str = ""


@_register
@dataclass(frozen=True, slots=True)
class ContextSupplied(Event):
    context: str = ""


@_register
@dataclass(frozen=True, slots=True)
class FeedbackSubmitted(Event):
    feedback: str = ""


@_register
@dataclass(frozen=True, slots=True)
class OutboxDelivered(Event):
    outbox_id: str = ""


@_register
@dataclass(frozen=True, slots=True)
class PersistenceFailed(Event):
    error_code: str = ""
    error_message: str = ""


RuntimeEvent: TypeAlias = (
    TaskCreated | FormalizationSucceeded | FormalizationFailed | AgentCallRequested | AgentCallReserved | AgentCallCompleted
    | AgentCallFailed | ProcedureBatchCompleted | ChildMaterialized | SummaryCompleted | SummaryFailed | BranchCheckpointed | BranchFinalized
    | ReportDeadlineReached | FinalEpochCommitted | GraceExpired | ReportCompleted | FinalReportCompleted | FinalReportFailed | AllInflightSettled
    | PauseRequested | PauseExpired | PauseExpiryReached | ContinueRequested | StopRequested | ContextSupplied | FeedbackSubmitted | OutboxDelivered | PersistenceFailed
)


def event_to_json(event: RuntimeEvent) -> str:
    """编码事件类型与字段；datetime 一律使用 ISO 8601。"""

    event_type = type(event).__name__
    if EVENT_REGISTRY.get(event_type) is not type(event):
        raise ValueError(f"未注册事件类型：{event_type}")
    payload = {name: _json_value(getattr(event, name)) for name in event.__dataclass_fields__}
    payload["event_type"] = event_type
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def event_from_json(data: str) -> RuntimeEvent:
    """只从显式注册表解码，绝不为未知事件猜测兼容路径。"""

    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("事件 JSON 必须是对象")
    event_type = payload.pop("event_type", None)
    event_class = EVENT_REGISTRY.get(event_type)
    if event_class is None:
        raise ValueError(f"未知事件类型：{event_type}")
    occurred_at = payload.get("occurred_at")
    if not isinstance(occurred_at, str):
        raise ValueError("事件 occurred_at 必须是 ISO 8601 字符串")
    payload["occurred_at"] = datetime.fromisoformat(occurred_at)
    try:
        return event_class(**payload)
    except TypeError as exc:
        raise ValueError(f"事件字段无效：{event_type}") from exc
