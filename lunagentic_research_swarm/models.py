"""LRS 的领域模型；持久状态不可变，分支叶子仅在 reducer 内可变。"""

from __future__ import annotations

import hashlib
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
