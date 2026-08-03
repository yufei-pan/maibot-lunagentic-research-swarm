"""可复现、可审计的保守 token 与 inherited-prefix 估算。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    prompt_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    source: Literal["estimated"] = "estimated"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tokens_for_bytes(byte_count: int) -> int:
    return math.ceil(byte_count / 3.5)


def estimate_prompt_tokens(
    messages: str | Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None
) -> TokenEstimate:
    """按 canonical JSON UTF-8 bytes / 3.5 向上取整，全部视为 miss。"""

    byte_count = len(_canonical_bytes(messages))
    if tools is not None:
        byte_count += len(_canonical_bytes(tools))
    prompt_tokens = _tokens_for_bytes(byte_count)
    return TokenEstimate(prompt_tokens, 0, prompt_tokens)


def estimate_child_tokens(
    child_messages: Sequence[Mapping[str, Any]],
    *,
    inherited_messages: Sequence[Mapping[str, Any]],
    parent_actual_model_name: str,
    estimated_model_name: str,
    cache_enabled: bool,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> TokenEstimate:
    """只有模型相同、cache 开启且前缀逐字节相同时才估算 inherited hit。"""

    total = estimate_prompt_tokens(child_messages, tools)
    prefix_length = len(inherited_messages)
    candidate_prefix = list(child_messages[:prefix_length])
    proven = (
        prefix_length > 0
        and bool(inherited_messages)
        and bool(parent_actual_model_name)
        and parent_actual_model_name == estimated_model_name
        and cache_enabled
        and len(child_messages) >= prefix_length
        and _canonical_bytes(candidate_prefix) == _canonical_bytes(list(inherited_messages))
    )
    if not proven:
        return total
    hit = min(_tokens_for_bytes(len(_canonical_bytes(list(inherited_messages)))), total.prompt_tokens)
    return TokenEstimate(total.prompt_tokens, hit, total.prompt_tokens - hit)
