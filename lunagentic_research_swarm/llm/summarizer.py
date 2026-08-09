"""不可委派的核心总结器四角色服务。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunagentic_research_swarm.models import FormalizedTask, ReportKind

from .gateway import GenerationError, GenerationRequest, GenerationResult, LLMGateway
from .pricing import TokenUsage


@dataclass(frozen=True, slots=True)
class FormalizationRequest:
    raw_context: str
    # Host ``message.build_readable`` 返回字符串；测试/旧假实现可能仍给 message dict 列表。
    chat_messages: str | Sequence[Mapping[str, Any]] = ()


def _chat_history_as_messages(chat_messages: str | Sequence[Mapping[str, Any]] | Any) -> list[dict[str, Any]]:
    """把 formalizer 的聊天输入规范成 LLM message dict 列表。

    Host SDK：``build_readable`` → ``str``。若误对字符串做 ``dict(ch)`` / 按字符迭代，
    会触发 ``dictionary update sequence element #0 has length 1; 2 is required``。
    """

    if chat_messages is None:
        return []
    if isinstance(chat_messages, str):
        text = chat_messages.strip()
        return [{"role": "user", "content": text}] if text else []
    if isinstance(chat_messages, Mapping):
        return [dict(chat_messages)]
    if isinstance(chat_messages, Sequence) and not isinstance(chat_messages, (bytes, bytearray)):
        normalized: list[dict[str, Any]] = []
        for item in chat_messages:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    normalized.append({"role": "user", "content": text})
            else:
                raise TypeError(f"chat_messages 条目类型无效：{type(item).__name__}")
        return normalized
    raise TypeError(f"chat_messages 类型无效：{type(chat_messages).__name__}")


@dataclass(frozen=True, slots=True)
class BranchFinalizationRequest:
    formalized_task: FormalizedTask
    branch_history: Sequence[Mapping[str, Any]]
    checkpoint: bool = False


@dataclass(frozen=True, slots=True)
class TaskFinalizationRequest:
    formalized_task: FormalizedTask
    coverage_summaries: Sequence[str]
    report_kind: ReportKind
    statistics: Mapping[str, Any]
    running_branch_count: int

    def __post_init__(self) -> None:
        if isinstance(self.running_branch_count, bool) or not isinstance(self.running_branch_count, int):
            raise ValueError("running_branch_count 必须是非负整数")
        if self.running_branch_count < 0:
            raise ValueError("running_branch_count 必须是非负整数")
        if not isinstance(self.report_kind, ReportKind):
            try:
                object.__setattr__(self, "report_kind", ReportKind(self.report_kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("report_kind 必须是 INTERMEDIATE 或 FINAL") from exc


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    formalized_task: FormalizedTask
    branch_history: Sequence[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SummaryResult:
    success: bool
    text: str
    model_name: str
    usage: TokenUsage | None
    error: GenerationError | None


_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "zh-CN"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ": "), default=str)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return "content" in value and _has_content(value["content"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_content(item) for item in value)
    return True


class SummarizerService:
    """固定四套 prompt 的 LLM 服务；不接收目录、Procedure schema 或 native tools。"""

    _ROLE_FILES = {
        "formalize_task": "formalize_task.txt",
        "finalize_branch": "finalize_branch.txt",
        "finalize_task": "finalize_task.txt",
        "compact_branch": "compact_branch.txt",
    }

    def __init__(
        self,
        gateway: LLMGateway | Any,
        *,
        selector: str = "task:mid_memory",
        temperature: float = 0.2,
        max_tokens: int = 0,
        prompt_root: Path | None = None,
        min_output_chars: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0 or max_tokens > 65536:
            raise ValueError("max_tokens 必须为 0..65536 的整数")
        self._gateway = gateway
        self._selector = selector
        self._temperature = temperature
        self._max_tokens = None if max_tokens == 0 else max_tokens
        root = prompt_root or _PROMPT_ROOT
        self._prompts = {
            role: (root / filename).read_text(encoding="utf-8") for role, filename in self._ROLE_FILES.items()
        }
        self._min_output_chars = self._normalize_min_output_chars(min_output_chars)

    @classmethod
    def _normalize_min_output_chars(cls, value: Mapping[str, Any] | None) -> dict[str, int]:
        if value is None:
            return {role: 0 for role in cls._ROLE_FILES}
        source = value.model_dump() if hasattr(value, "model_dump") else dict(value)
        floors: dict[str, int] = {}
        for role in cls._ROLE_FILES:
            raw = source.get(role, 0)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0 or raw > 20000:
                raise ValueError(f"min_output_chars.{role} 必须为 0..20000 的整数")
            floors[role] = raw
        return floors

    def _system_prompt(self, role: str) -> str:
        """角色 prompt + 可选的输出下限。

        只加下限、不加上限：给出长度区间会让模型去凑区间而不是照材料该写多少写多少；
        过长由 ``max_tokens`` 截断，并且本身就是值得排查的异常信号。
        """

        prompt = self._prompts[role]
        floor = self._min_output_chars.get(role, 0)
        if floor <= 0:
            return prompt
        return (
            f"{prompt}\n"
            f"输出不应少于约 {floor} 字符。这只是防止内容退化的下限，不是目标长度，"
            "也没有上限：该写多少由材料本身决定，不要为了凑数注水，也不要为了简短省略事实。"
        )

    @staticmethod
    def _input_failure(message: str) -> SummaryResult:
        return SummaryResult(
            success=False,
            text="",
            model_name="",
            usage=None,
            error=GenerationError("summary_input_empty", message),
        )

    async def _generate(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        validate_formalized_task: bool = False,
    ) -> SummaryResult:
        result: GenerationResult = await self._gateway.generate(
            GenerationRequest(
                selector=self._selector,
                messages=messages,
                tools=None,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )
        if not result.success:
            return SummaryResult(False, "", result.model_name, result.usage, result.error)
        if not result.response.strip():
            return SummaryResult(
                False,
                "",
                result.model_name,
                result.usage,
                GenerationError("summary_empty", f"{role} 返回了空摘要"),
            )
        if validate_formalized_task:
            try:
                FormalizedTask.create(result.response)
            except ValueError as exc:
                return SummaryResult(
                    False,
                    "",
                    result.model_name,
                    result.usage,
                    GenerationError("summary_empty", str(exc)),
                )
        return SummaryResult(True, result.response, result.model_name, result.usage, None)

    async def formalize_task(self, request: FormalizationRequest) -> SummaryResult:
        """一次性 ingest 原始 context/chat；每次新建消息列表且不做 prefix 优化。"""

        if not _has_content(request.raw_context) and not _has_content(request.chat_messages):
            return self._input_failure("任务形式化缺少原始上下文和聊天内容")
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt("formalize_task")}]
        if request.raw_context.strip():
            messages.append({"role": "user", "content": request.raw_context})
        try:
            messages.extend(_chat_history_as_messages(request.chat_messages))
        except TypeError as exc:
            return self._input_failure(str(exc))
        return await self._generate("formalize_task", messages, validate_formalized_task=True)

    async def finalize_branch(self, request: BranchFinalizationRequest) -> SummaryResult:
        if not _has_content(request.branch_history):
            return self._input_failure("分支总结缺少可总结的分支历史")
        mutable_input = {
            "summary_scope": "检查点" if request.checkpoint else "终结分支",
            "branch_history": [dict(message) for message in request.branch_history],
        }
        messages = [
            {"role": "system", "content": self._system_prompt("finalize_branch")},
            {"role": "user", "content": request.formalized_task.text},
            {"role": "user", "content": _json_text(mutable_input)},
        ]
        return await self._generate("finalize_branch", messages)

    async def finalize_task(self, request: TaskFinalizationRequest) -> SummaryResult:
        if not _has_content(request.coverage_summaries):
            return self._input_failure("任务总结缺少分支覆盖摘要")
        mutable_input = {
            "report_kind": "中间" if request.report_kind is ReportKind.INTERMEDIATE else "最终",
            "running_branch_count": request.running_branch_count,
            "statistics": dict(request.statistics),
            "coverage_summaries": list(request.coverage_summaries),
        }
        messages = [
            {"role": "system", "content": self._system_prompt("finalize_task")},
            {"role": "user", "content": request.formalized_task.text},
            {"role": "user", "content": _json_text(mutable_input)},
        ]
        return await self._generate("finalize_task", messages)

    async def compact_branch(self, request: CompactionRequest) -> SummaryResult:
        if not _has_content(request.branch_history):
            return self._input_failure("分支压缩缺少可压缩的可变历史")
        messages = [
            {"role": "system", "content": self._system_prompt("compact_branch")},
            {"role": "user", "content": request.formalized_task.text},
            {
                "role": "user",
                "content": _json_text({"branch_history": [dict(message) for message in request.branch_history]}),
            },
        ]
        return await self._generate("compact_branch", messages)


__all__ = [
    "BranchFinalizationRequest",
    "CompactionRequest",
    "FormalizationRequest",
    "SummarizerService",
    "SummaryResult",
    "TaskFinalizationRequest",
]
