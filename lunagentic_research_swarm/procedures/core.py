"""本地 core Procedure 与控制请求解析。

``core.compact``、``core.checkpoint`` 和 ``core.terminate`` 是 runtime 的控制
边界，不属于任何可热更新的第三方 Procedure provider。这个模块只负责识别和
执行本地控制；余额核销、委派优先级以及 branch 生命周期仍由 reducer 负责。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from lunagentic_research_swarm.extensions.contracts import ProcedureResult
from lunagentic_research_swarm.models import FormalizedTask
from lunagentic_research_swarm.llm.summarizer import (
    BranchFinalizationRequest,
    CompactionRequest,
)


CORE_COMPACT_ID = "core.compact"
CORE_CHECKPOINT_ID = "core.checkpoint"
CORE_TERMINATE_ID = "core.terminate"
CORE_PROCEDURE_IDS = frozenset({CORE_COMPACT_ID, CORE_CHECKPOINT_ID, CORE_TERMINATE_ID})


def is_core_procedure(procedure_id: str) -> bool:
    """返回 ID 是否属于固定的本地控制 Procedure。"""

    return procedure_id in CORE_PROCEDURE_IDS


def _request_procedure_id(request: Any) -> str:
    if isinstance(request, Mapping):
        value = request.get("procedure_id", "")
    else:
        value = getattr(request, "procedure_id", "")
    return str(value) if value is not None else ""


_SENSITIVE_CONTROL_KEYS = frozenset(
    {
        "reasoning",
        "raw_payload",
        "raw_provenance",
        "raw_payloads",
        "raw_result",
        "raw_arguments",
        "provenance",
        "payload",
        "transcript",
        "messages",
    }
)


def _freeze_control_json(value: Any) -> Any:
    """递归冻结 control request 的 JSON 值，避免事件发布后被改写。"""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_control_json(item)
                for key, item in value.items()
                if str(key).lower() not in _SENSITIVE_CONTROL_KEYS
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_control_json(item) for item in value)
    return value


def _thaw_control_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_control_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_control_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrozenControlRequest:
    """可安全放入 CoreProcedureDecision/event 的控制请求值。"""

    procedure_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "procedure_id", str(self.procedure_id))
        object.__setattr__(self, "arguments", _freeze_control_json(dict(self.arguments)))

    @classmethod
    def from_request(cls, request: Any) -> "FrozenControlRequest":
        if isinstance(request, cls):
            return request
        if isinstance(request, Mapping):
            return cls(
                procedure_id=str(request.get("procedure_id", "")),
                arguments=dict(request.get("arguments", {}) or {}),
            )
        model_dump = getattr(request, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="python")
            return cls(
                procedure_id=str(payload.get("procedure_id", "")),
                arguments=dict(payload.get("arguments", {}) or {}),
            )
        return cls(
            procedure_id=str(getattr(request, "procedure_id", "")),
            arguments=dict(getattr(request, "arguments", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"procedure_id": self.procedure_id, "arguments": _thaw_control_json(self.arguments)}


@dataclass(frozen=True, slots=True, init=False)
class CoreProcedureDecision:
    """一批 turn 请求中的 core 控制决定。

    ``ignored_controls`` 对外返回普通 list，便于 Tool/JSON 层直接消费；内部用
    tuple 保存，避免发布后调用方修改 reducer 输入。terminate 一旦出现就压过
    compact/checkpoint，并把其它 core 控制记录为 ignored。
    """

    compact: bool
    checkpoint: bool
    terminate: bool
    _ignored_controls: tuple[str, ...] = field(default=(), repr=False, compare=False)
    control_requests: tuple[FrozenControlRequest, ...] = field(default=(), repr=False, compare=False)

    def __init__(
        self,
        compact: bool = False,
        checkpoint: bool = False,
        terminate: bool = False,
        *,
        ignored_controls: Sequence[str] = (),
        control_requests: Sequence[Any] = (),
    ) -> None:
        object.__setattr__(self, "compact", bool(compact))
        object.__setattr__(self, "checkpoint", bool(checkpoint))
        object.__setattr__(self, "terminate", bool(terminate))
        object.__setattr__(self, "_ignored_controls", tuple(str(item) for item in ignored_controls))
        object.__setattr__(
            self,
            "control_requests",
            tuple(FrozenControlRequest.from_request(item) for item in control_requests),
        )

    @property
    def ignored_controls(self) -> list[str]:
        """返回副本，保留 brief 约定的 list 形状。"""

        return list(self._ignored_controls)

    @property
    def has_control(self) -> bool:
        return self.compact or self.checkpoint or self.terminate

    def as_dict(self) -> dict[str, Any]:
        return {
            "compact": self.compact,
            "checkpoint": self.checkpoint,
            "terminate": self.terminate,
            "ignored_controls": self.ignored_controls,
            "control_requests": [item.as_dict() for item in self.control_requests],
        }


def split_procedure_requests(requests: Sequence[Any]) -> tuple[list[Any], CoreProcedureDecision]:
    """把普通请求与本地控制请求分开，并保持普通请求原顺序。

    该函数不修改输入、不校验第三方参数 schema，也不做任何网络/总结器调用，
    因而可在 reducer 之外安全复用。terminate 会忽略同一批中的其它 core
    控制；普通 Procedure 仍会保留，实际是否启动后代由 reducer 在持久化后决定。
    """

    ordinary: list[Any] = []
    controls: list[Any] = []
    compact = False
    checkpoint = False
    terminate = False
    for request in requests:
        procedure_id = _request_procedure_id(request)
        if procedure_id == CORE_COMPACT_ID:
            compact = True
            controls.append(request)
        elif procedure_id == CORE_CHECKPOINT_ID:
            checkpoint = True
            controls.append(request)
        elif procedure_id == CORE_TERMINATE_ID:
            terminate = True
            controls.append(request)
        else:
            ordinary.append(request)

    ignored: list[str] = []
    if terminate:
        # 只记录实际出现过的其它控制，并保持请求顺序；重复控制不被静默去重。
        for request in controls:
            procedure_id = _request_procedure_id(request)
            if procedure_id != CORE_TERMINATE_ID:
                ignored.append(procedure_id)
        compact = False
        checkpoint = False

    return ordinary, CoreProcedureDecision(
        compact=compact,
        checkpoint=checkpoint,
        terminate=terminate,
        ignored_controls=ignored,
        control_requests=controls,
    )


@dataclass(frozen=True, slots=True)
class CoreProcedureContext:
    """总结器 core Procedure 所需的最小、显式输入。"""

    formalized_task: FormalizedTask | str | None = None
    branch_history: tuple[Mapping[str, Any], ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_history", tuple(dict(item) for item in self.branch_history))

    def task(self) -> FormalizedTask | None:
        if isinstance(self.formalized_task, FormalizedTask):
            return self.formalized_task
        if isinstance(self.formalized_task, str) and self.formalized_task.strip():
            return FormalizedTask.create(self.formalized_task)
        return None


def _failure(procedure_id: str, code: str, message: str) -> ProcedureResult:
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata={"core": True, "procedure_id": procedure_id, "research_credits_charged": 0.0},
    )


async def execute_core_procedure(
    procedure_id: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    summarizer: Any | None = None,
    context: CoreProcedureContext | None = None,
    formalized_task: FormalizedTask | str | None = None,
    branch_history: Sequence[Mapping[str, Any]] = (),
) -> ProcedureResult:
    """执行一个本地 core Procedure。

    core.terminate 不需要总结器；compact/checkpoint 只通过明确的
    :class:`SummarizerService` 方法生成摘要，绝不会绕到 ``ctx.api.call``。
    该函数不写 credits ledger，调用方应把结果作为普通 Procedure event 交给
    reducer 持久化。
    """

    if procedure_id not in CORE_PROCEDURE_IDS:
        return _failure(procedure_id, "procedure_unavailable", "不是 core Procedure")
    args = dict(arguments or {})
    supplied_task = formalized_task if formalized_task is not None else args.get("formalized_task")
    supplied_history: Sequence[Mapping[str, Any]] = branch_history
    if not supplied_history and isinstance(args.get("branch_history"), Sequence) and not isinstance(
        args["branch_history"], (str, bytes, bytearray)
    ):
        supplied_history = args["branch_history"]
    ctx = context or CoreProcedureContext(
        formalized_task=supplied_task,
        branch_history=tuple(supplied_history),
        reason=str(args.get("reason") or ""),
    )
    if procedure_id == CORE_TERMINATE_ID:
        return ProcedureResult(
            success=True,
            data={"terminated": True, "reason": str(args.get("reason") or ctx.reason or "")},
            error=None,
            metadata={"core": True, "procedure_id": procedure_id, "research_credits_charged": 0.0},
        )
    if summarizer is None:
        return _failure(procedure_id, "core_summary_unavailable", "core Procedure 缺少总结器")
    task = ctx.task()
    if task is None:
        return _failure(procedure_id, "summary_input_empty", "core Procedure 缺少正式任务")
    if not ctx.branch_history:
        return _failure(procedure_id, "summary_input_empty", "core Procedure 缺少分支历史")

    if procedure_id == CORE_COMPACT_ID:
        summary = await summarizer.compact_branch(
            CompactionRequest(formalized_task=task, branch_history=ctx.branch_history)
        )
        success_data = {"summary": summary.text, "compacted": True}
    else:
        summary = await summarizer.finalize_branch(
            BranchFinalizationRequest(formalized_task=task, branch_history=ctx.branch_history, checkpoint=True)
        )
        success_data = {"summary": summary.text, "checkpoint": True}
    if not summary.success:
        error = summary.error
        return _failure(
            procedure_id,
            error.code if error is not None else "summary_failed",
            "core 总结器调用失败",
        )
    return ProcedureResult(
        success=True,
        data=success_data,
        error=None,
        metadata={
            "core": True,
            "procedure_id": procedure_id,
            "research_credits_charged": 0.0,
            "model_name": summary.model_name,
        },
    )


class CoreProcedureExecutor:
    """对 ``execute_core_procedure`` 的小型面向对象适配器。"""

    def __init__(self, summarizer: Any | None = None) -> None:
        self.summarizer = summarizer

    async def invoke(
        self,
        procedure_id: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ProcedureResult:
        return await execute_core_procedure(procedure_id, arguments, summarizer=self.summarizer, **kwargs)


# 兼容调用方偏好的命名；所有入口仍共享同一实现。
run_core_procedure = execute_core_procedure


__all__ = [
    "CORE_CHECKPOINT_ID",
    "CORE_COMPACT_ID",
    "CORE_PROCEDURE_IDS",
    "CORE_TERMINATE_ID",
    "CoreProcedureContext",
    "CoreProcedureDecision",
    "CoreProcedureExecutor",
    "FrozenControlRequest",
    "execute_core_procedure",
    "is_core_procedure",
    "run_core_procedure",
    "split_procedure_requests",
]
