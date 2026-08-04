"""只携带不可变输入的显式运行期事件与 JSON 编解码。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, TypeAlias


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

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", _freeze_value(self.usage))


@_register
@dataclass(frozen=True, slots=True)
class AgentCallFailed(Event):
    branch_id: str = ""
    call_id: str = ""
    error_code: str = ""
    error_message: str = ""


@_register
@dataclass(frozen=True, slots=True)
class ProcedureBatchCompleted(Event):
    branch_id: str = ""
    call_id: str = ""
    result_id: str = ""


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
    epoch: int | None = None


@_register
@dataclass(frozen=True, slots=True)
class GraceExpired(Event):
    pass


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
    | AgentCallFailed | ProcedureBatchCompleted | SummaryCompleted | SummaryFailed | BranchCheckpointed | BranchFinalized
    | ReportDeadlineReached | GraceExpired | ReportCompleted | FinalReportCompleted | FinalReportFailed | AllInflightSettled
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
