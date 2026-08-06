"""普通智能体的稳定 prompt prefix 与可释放分支上下文。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from pydantic import BaseModel

from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask


_PROFILE_FIELDS = ("nickname", "personality", "behavior_style", "reply_style")
_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "authorization"})


def _safe_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_json(item)
            for key, item in value.items()
            if str(key).casefold() not in _SECRET_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """序列化冻结 prompt catalog；顺序稳定且不允许非有限数。"""

    return json.dumps(
        _safe_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class RuntimeHeader:
    branch_id: str
    turn_number: int
    agent_capability: str
    report_seconds_remaining: int
    credits_after_input_reservation: float
    active_count: int
    queued_count: int

    def message(self) -> dict[str, str]:
        if self.credits_after_input_reservation < 0:
            credit_rule = "余额为负；本 turn 后不能启动后代。"
        elif self.credits_after_input_reservation == 0:
            credit_rule = "余额为零；仍可零 credits 委派。"
        else:
            credit_rule = "余额非负；可在结构上限内委派。"
        content = (
            "[LRS runtime]\n"
            f"branch={self.branch_id}; turn={self.turn_number}; capability={self.agent_capability}; "
            f"report_seconds_remaining={max(0, self.report_seconds_remaining)}; "
            f"credits_after_input_reservation={self.credits_after_input_reservation:g}; "
            f"active={max(0, self.active_count)}; queued={max(0, self.queued_count)}\n"
            f"{credit_rule}"
        )
        return {"role": "user", "content": content}


@dataclass(slots=True)
class BranchContext:
    """只保存可变 history；system 与正式任务由 builder 每次重新插入。"""

    history: list[dict[str, Any]]


class StablePromptBuilder:
    """构建 stable system + immutable User 1 + mutable history + runtime suffix。"""

    def __init__(
        self,
        *,
        formalized_task: FormalizedTask,
        swarm_identity: str,
        bot_profile: Mapping[str, Any],
        agent_catalog: Any,
        procedure_catalog: Any,
        pricing: Any,
    ) -> None:
        self.formalized_task = formalized_task
        profile = {field: str(bot_profile.get(field, "")) for field in _PROFILE_FIELDS}
        prefix = {
            "swarm_identity": str(swarm_identity),
            "bot_profile": profile,
            "architecture_rules": {
                "quick_thinker_role": "本轮起始协调者与普通调查智能体",
                "delegation": "子分支继承上下文并收到明确 assignment",
                "credits": "LLM 调用消耗研究 credits；Procedure 不消耗研究 credits；零余额仍可零额委派，负余额不得启动后代",
                "protocol": "每个 turn 必须提交唯一 swarm envelope",
            },
            "agents": agent_catalog,
            "procedures": procedure_catalog,
            "pricing": pricing,
        }
        self.system_message = "你属于以下调查 swarm。冻结运行契约（canonical JSON）：\n" + canonical_json(prefix)

    def root_context(self, *, coordinator: str) -> BranchContext:
        return BranchContext(
            [{"role": "user", "content": f"本轮起始协调者：{coordinator}"}]
        )

    def restart_context(self, *, summary_layers: Sequence[str], coordinator: str) -> BranchContext:
        history = [
            {"role": "assistant", "content": f"既有 summary layer：{summary}"}
            for summary in summary_layers
        ]
        history.append({"role": "user", "content": f"本轮起始协调者：{coordinator}"})
        return BranchContext(history)

    def messages_for_call(self, context: BranchContext, header: RuntimeHeader) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.formalized_task.text},
            *[dict(message) for message in context.history],
            header.message(),
        ]


def should_auto_compact(
    used_tokens: int,
    *,
    agent_override: int | None,
    definition: int | None,
    global_threshold: int = 258_000,
    model_context_limit: int | None = None,
    reserved_output_tokens: int = 8192,
    safety_margin_tokens: int = 8192,
) -> bool:
    """任一配置阈值或已知物理 context window 阈值触发即压缩。"""

    if used_tokens < 0:
        raise ValueError("used_tokens 必须非负")
    configured = agent_override if agent_override is not None else definition
    if configured is None:
        configured = global_threshold
    if used_tokens >= configured:
        return True
    if model_context_limit is None:
        return False
    usable = max(0, model_context_limit - reserved_output_tokens - safety_margin_tokens)
    return used_tokens >= usable


def release_raw_context(branch: BranchRuntime | BranchContext) -> None:
    """分支终结后同步释放 raw message graph。"""

    if isinstance(branch, BranchRuntime):
        branch.messages.clear()
    else:
        branch.history.clear()


__all__ = [
    "BranchContext",
    "RuntimeHeader",
    "StablePromptBuilder",
    "canonical_json",
    "release_raw_context",
    "should_auto_compact",
]
