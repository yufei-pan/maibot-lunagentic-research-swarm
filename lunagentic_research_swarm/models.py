"""LRS 的领域模型；持久状态不可变，分支叶子仅在 reducer 内可变。"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_task_id() -> str:
    return _new_id("lrs")


def new_round_id() -> str:
    return _new_id("rnd")


def new_branch_id() -> str:
    return _new_id("br")


def new_turn_id() -> str:
    return _new_id("turn")


def new_call_id() -> str:
    return _new_id("call")


def new_summary_id() -> str:
    return _new_id("sum")


def new_report_id() -> str:
    return _new_id("rpt")


def new_ledger_id() -> str:
    return _new_id("led")


def new_outbox_id() -> str:
    return _new_id("out")


class TaskStatus(str, Enum):
    FORMALIZING = "FORMALIZING"
    RUNNING = "RUNNING"
    REPORTING = "REPORTING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    STOPPED = "STOPPED"
    EXPIRED = "EXPIRED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


class BranchLifecycle(str, Enum):
    READY = "READY"
    IN_FLIGHT = "IN_FLIGHT"
    WAITING_PROCEDURES = "WAITING_PROCEDURES"
    WAITING_REPORT_WITH_CHECKPOINT = "WAITING_REPORT_WITH_CHECKPOINT"
    PAUSED = "PAUSED"
    FINALIZING = "FINALIZING"
    FINALIZED = "FINALIZED"


class ReportKind(str, Enum):
    INTERMEDIATE = "INTERMEDIATE"
    FINAL = "FINAL"


class SummaryKind(str, Enum):
    FORMALIZATION = "FORMALIZATION"
    CHECKPOINT = "CHECKPOINT"
    BRANCH_FINAL = "BRANCH_FINAL"
    TASK_FINAL = "TASK_FINAL"


def _credit_number(value: Any, field_name: str, *, nonnegative: bool = False) -> float:
    """校验 credits 数字；账本不得接受 NaN、无穷或布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        qualifier = "非负有限数" if nonnegative else "有限数"
        raise ValueError(f"{field_name} 必须为{qualifier}")
    normalized = float(value)
    if not math.isfinite(normalized) or (nonnegative and normalized < 0):
        qualifier = "非负有限数" if nonnegative else "有限数"
        raise ValueError(f"{field_name} 必须为{qualifier}")
    return normalized


def _metadata_contains_reasoning(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key == "reasoning" or _metadata_contains_reasoning(item) for key, item in value.items())
    if isinstance(value, list | tuple):
        return any(_metadata_contains_reasoning(item) for item in value)
    return False


@dataclass(frozen=True, slots=True, init=False)
class CreditBalance:
    """不可变的 branch/task credits 快照。

    ``balance`` 是当前可用于该 branch 的有符号余额，``pool`` 是 dormant task
    pool。两者保持独立，只有 continue 屏障才会把 pool 分配给活动叶子。
    ``available``/``dormant_pool`` 是便于调用方阅读的只读别名，不是额外状态。
    """

    balance: float
    pool: float

    def __init__(
        self,
        balance: float = 0.0,
        pool: float = 0.0,
        *,
        available: float | None = None,
        dormant_pool: float | None = None,
    ) -> None:
        if available is not None:
            if balance != 0.0 and float(balance) != float(available):
                raise ValueError("balance 与 available 不一致")
            balance = available
        if dormant_pool is not None:
            if pool != 0.0 and float(pool) != float(dormant_pool):
                raise ValueError("pool 与 dormant_pool 不一致")
            pool = dormant_pool
        object.__setattr__(self, "balance", _credit_number(balance, "balance"))
        object.__setattr__(self, "pool", _credit_number(pool, "pool"))

    @property
    def available(self) -> float:
        return self.balance

    @property
    def dormant_pool(self) -> float:
        return self.pool

    @property
    def amount(self) -> float:
        """兼容账本调用方使用的 amount 只读别名。"""

        return self.balance

    @property
    def credits(self) -> float:
        return self.balance


@dataclass(frozen=True, slots=True)
class CreditLedgerEntry:
    """与 SQLite ``credit_ledger`` 行一一对应的不可变账本条目。"""

    ledger_id: str = ""
    task_id: str = ""
    round_id: str = ""
    branch_id: str | None = None
    call_id: str | None = None
    entry_kind: str = ""
    amount: float = 0.0
    balance_after: float | None = None
    metadata_json: str = "{}"
    created_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _credit_number(self.amount, "amount"))
        if self.balance_after is not None:
            object.__setattr__(self, "balance_after", _credit_number(self.balance_after, "balance_after"))
        if not isinstance(self.metadata_json, str):
            raise ValueError("metadata_json 必须为 JSON 字符串")
        try:
            metadata = json.loads(self.metadata_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata_json 必须为有效 JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json 必须编码 JSON 对象")
        if _metadata_contains_reasoning(metadata):
            raise ValueError("账本 metadata 不得包含 reasoning")
        object.__setattr__(self, "created_at", _credit_number(self.created_at, "created_at"))

    @property
    def metadata(self) -> dict[str, Any]:
        """返回新对象，调用方无法通过它改写账本条目。"""

        value = json.loads(self.metadata_json)
        return dict(value)


@dataclass(frozen=True, slots=True)
class FormalizedTask:
    text: str
    sha256: str

    @classmethod
    def create(cls, text: str) -> FormalizedTask:
        if not text.strip():
            raise ValueError("正式任务描述不能为空")
        return cls(text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task_id: str
    status: TaskStatus
    formalized_task: FormalizedTask | None = None
    generation: int = 0
    active_round_id: str | None = None
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class RoundSnapshot:
    round_id: str
    task_id: str
    generation: int
    catalog_fingerprint: str
    root_branch_id: str | None = None


@dataclass
class BranchRuntime:
    """单个活跃分支的可变叶子，不可变正式任务不存入可压缩 messages。"""

    branch_id: str
    task: FormalizedTask
    catalog_fingerprint: str
    generation: int
    messages: list[dict[str, Any]]
    credits: float
    depth: int
    parent_branch_id: str | None = None
    child_branch_ids: list[str] = field(default_factory=list)
    pending_delegations: list[dict[str, Any]] = field(default_factory=list)
    procedure_results: list[dict[str, Any]] = field(default_factory=list)
    latest_checkpoint_id: str | None = None
    lifecycle: BranchLifecycle = BranchLifecycle.READY

    def build_prompt_messages(self) -> list[dict[str, Any]]:
        """每次构建都把不可变任务原文作为 User 1 重新写入。"""

        task_message = {"role": "user", "content": self.task.text}
        return [task_message, *[dict(message) for message in self.messages]]
