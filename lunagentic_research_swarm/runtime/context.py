"""普通智能体的稳定 prompt prefix 与可释放分支上下文。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lunagentic_research_swarm.llm.protocol_prompt import (
    JSON_ENVELOPE_PROTOCOL,
    normalize_protocol,
    protocol_runtime_reminder,
    protocol_system_section,
)
from lunagentic_research_swarm.models import BranchRuntime, FormalizedTask
from lunagentic_research_swarm.procedures.core import CORE_PROCEDURE_PROMPTS


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
    """稳定 JSON（指纹/估算等）；prompt 展示请用 render_frozen_system_contract。"""

    return json.dumps(
        _safe_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(
        _safe_json(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return dict(value.model_dump(mode="json"))
    if is_dataclass(value):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    return {}


def _definition_payload(entry: Any) -> dict[str, Any]:
    """从 catalog entry / 裸 definition / dict 抽出 definition 字段。"""

    if isinstance(entry, str):
        return {}
    if isinstance(entry, Mapping):
        nested = entry.get("definition")
        if isinstance(nested, (Mapping, BaseModel)) or is_dataclass(nested):
            return _as_mapping(nested)
        return {str(key): entry[key] for key in entry}
    definition = getattr(entry, "definition", None)
    if definition is not None:
        return _as_mapping(definition)
    return _as_mapping(entry)


def _attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _collect_agent_cards(agent_catalog: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if isinstance(agent_catalog, Mapping):
        items = [(str(key), agent_catalog[key]) for key in sorted(agent_catalog.keys(), key=str)]
    elif isinstance(agent_catalog, Sequence) and not isinstance(agent_catalog, (str, bytes, bytearray)):
        items = [("", item) for item in agent_catalog]
    else:
        entries = getattr(agent_catalog, "entries", None)
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
            items = [("", item) for item in entries]
        elif agent_catalog is None:
            items = []
        else:
            items = [("", agent_catalog)]

    for fallback_id, item in items:
        payload = _definition_payload(item)
        agent_id = str(
            payload.get("agent_id")
            or _attr_or_key(item, "agent_id")
            or fallback_id
            or ""
        ).strip()
        if not agent_id:
            continue
        display = str(payload.get("display_name") or agent_id).strip() or agent_id
        description = str(payload.get("description") or payload.get("role") or "").strip()
        cards.append(
            {
                "agent_id": agent_id,
                "display_name": display,
                "description": description,
            }
        )
    cards.sort(key=lambda card: str(card["agent_id"]))
    return cards


def _collect_procedure_cards(procedure_catalog: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if isinstance(procedure_catalog, Mapping):
        items = [(str(key), procedure_catalog[key]) for key in sorted(procedure_catalog.keys(), key=str)]
    elif isinstance(procedure_catalog, Sequence) and not isinstance(procedure_catalog, (str, bytes, bytearray)):
        items = [("", item) for item in procedure_catalog]
    else:
        entries = getattr(procedure_catalog, "entries", None)
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
            items = [("", item) for item in entries]
        elif procedure_catalog is None:
            items = []
        else:
            items = [("", procedure_catalog)]

    for fallback_id, item in items:
        payload = _definition_payload(item)
        procedure_id = str(
            payload.get("procedure_id")
            or _attr_or_key(item, "procedure_id")
            or fallback_id
            or ""
        ).strip()
        if not procedure_id:
            continue
        display = str(payload.get("display_name") or procedure_id).strip() or procedure_id
        description = str(payload.get("description") or payload.get("summary") or "").strip()
        card: dict[str, Any] = {
            "procedure_id": procedure_id,
            "display_name": display,
            "description": description,
        }
        allowed = payload.get("allowed_agents")
        if allowed is None:
            allowed = _attr_or_key(_attr_or_key(item, "definition") or {}, "allowed_agents")
        if isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes, bytearray)):
            normalized = [str(x) for x in allowed]
            if normalized and normalized != ["*"]:
                card["allowed_agents"] = normalized
        schema = payload.get("arguments_schema")
        if isinstance(schema, Mapping):
            card["arguments_schema"] = _safe_json(schema)
        cards.append(card)
    cards.sort(key=lambda card: str(card["procedure_id"]))
    return cards


_IDENTITY_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "zh-CN" / "swarm_system.txt"
_IDENTITY_PLACEHOLDER = "{swarm_identity}"


@lru_cache(maxsize=1)
def _identity_template() -> str:
    """读取共享身份说明；文件缺失时回退到等价的内置文本。

    身份段是运维可能想改而又不该动 Python 的部分，因此和四个总结器角色一样存成
    prompt 文件。缓存保证渲染仍是纯函数式的（同一进程内逐字节稳定，cache 前缀不变）。
    """

    try:
        return _IDENTITY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return (
            f"你属于调查 swarm「{_IDENTITY_PLACEHOLDER}」。"
            "你是其中的执行智能体，不是面向用户的闲聊回复器。"
        )


def _format_identity_section(*, swarm_identity: str, bot_profile: Mapping[str, str]) -> str:
    lines = [_identity_template().replace(_IDENTITY_PLACEHOLDER, str(swarm_identity)), ""]
    profile_bits = [f"{field}={bot_profile.get(field, '')}" for field in _PROFILE_FIELDS if bot_profile.get(field, "")]
    if profile_bits:
        lines.append("宿主风格：" + "；".join(profile_bits))
        lines.append("（仅供最终报告的措辞参考，不得降低取证与推理的严格程度。）")
        lines.append("")
    return "\n".join(lines)


def _format_procedure_section(procedure_catalog: Any) -> str:
    lines = ["## 可用 Procedure", ""]
    lines.append("调用时把 `procedure_id` 与 `arguments` 写入本 turn 的 `procedures`。")
    lines.append(
        "本节是**本轮 round 的完整目录**（对所有智能体相同）。"
        "多数 Procedure 默认全体可调用；若某项写了「调用限制」，则只有列出的智能体可调用——"
        "其他智能体应委派该角色，不要自行调用。"
        "你本人本 turn 实际可调用的 ID 见末尾 `[LRS runtime]` 的「可调用 Procedure」一行。"
    )
    lines.append("`arguments` 必须满足下方 arguments schema：`required` 字段不可缺省，`enum` 只能取列出的值。")
    lines.append("")
    lines.append("### 控制")
    lines.append("")
    for item in CORE_PROCEDURE_PROMPTS:
        lines.append(f"- `{item['procedure_id']}`（{item['display_name']}）：{item['description']}")
    lines.append("")
    research = _collect_procedure_cards(procedure_catalog)
    lines.append("### 研究")
    lines.append("")
    if not research:
        lines.append("（本 round 无额外研究 Procedure。）")
        lines.append("")
        return "\n".join(lines)
    for card in research:
        pid = card["procedure_id"]
        display = card["display_name"]
        lines.append(f"#### `{pid}` — {display}")
        if card.get("description"):
            lines.append(str(card["description"]))
        restricted = card.get("allowed_agents")
        if restricted:
            agents = "、".join(f"`{item}`" for item in restricted)
            lines.append(f"调用限制：仅 {agents} 可调用；其他智能体请委派该角色。")
        schema = card.get("arguments_schema")
        if schema is not None:
            lines.append("arguments:")
            lines.append("```json")
            lines.append(_pretty_json(schema))
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _format_agent_section(agent_catalog: Any) -> str:
    lines = ["## 可委派智能体", ""]
    lines.append("委派时使用下方 `agent_id`，并给出明确 `task` 与非负 `credits`。")
    lines.append(
        "把子任务交给角色最匹配的智能体；要自己接着读本 turn 的 Procedure 结果，"
        "就把**你自己的 agent_id**（见 `[LRS runtime]`）写进 `delegations`。"
        "记忆族与历史案例等受限 Procedure 请委派对应专职智能体。"
    )
    lines.append("")
    cards = _collect_agent_cards(agent_catalog)
    if not cards:
        lines.append("（本 round 无可用智能体。）")
        lines.append("")
        return "\n".join(lines)
    for card in cards:
        agent_id = card["agent_id"]
        display = card["display_name"]
        lines.append(f"### `{agent_id}` — {display}")
        if card.get("description"):
            lines.append(str(card["description"]))
        lines.append("")
    return "\n".join(lines)


def _pricing_is_useful(pricing: Any) -> bool:
    data = _safe_json(pricing if pricing is not None else {})
    if not data:
        return False
    if isinstance(data, Mapping):
        models = data.get("models")
        if isinstance(models, Mapping) and models:
            return True
        # 有实质字段才展示；单独 fingerprint / 空 models 对 agent 无决策价值
        useful_keys = {key for key in data if key not in {"fingerprint", "models", "tasks"}}
        if useful_keys:
            return True
        tasks = data.get("tasks")
        return isinstance(tasks, Mapping) and bool(tasks)
    return True


def _model_price_line(name: str, profile: Any) -> str | None:
    """把一个价格 profile 渲染成「每 100 万 token 多少 credits」的可读行。

    Host 价格数值 1.0 == 100 credits（``pricing.price_units_to_credits``），价格按
    token/1e6 计价，因此每 100 万 token 的 credits 就是 ``price * 100``。原始 JSON
    dump 对 agent 没有决策价值，这里只保留它真正能用来估算的数字。
    """

    if not isinstance(profile, Mapping):
        return None
    try:
        price_in = float(profile.get("price_in", 0.0) or 0.0)
        price_out = float(profile.get("price_out", 0.0) or 0.0)
        cache_price_in = float(profile.get("cache_price_in", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    parts = [f"输入 {price_in * 100.0:g}", f"输出 {price_out * 100.0:g}"]
    if profile.get("cache"):
        parts.insert(1, f"缓存命中输入 {cache_price_in * 100.0:g}")
    return f"- `{name}`：{'；'.join(parts)}"


def _format_pricing_section(pricing: Any) -> str:
    if not _pricing_is_useful(pricing):
        return ""
    data = _safe_json(pricing if pricing is not None else {})
    models = data.get("models") if isinstance(data, Mapping) else None
    lines = ["## 模型费用参考", ""]
    if isinstance(models, Mapping) and models:
        lines.append("每 100 万 token 消耗的研究 credits，用于粗略估计委派与工具调用的成本：")
        lines.append("")
        rendered = [
            line
            for line in (_model_price_line(str(name), models[name]) for name in sorted(models, key=str))
            if line is not None
        ]
        if rendered:
            lines.extend(rendered)
            lines.append("")
            return "\n".join(lines)
    # 非标准形状：保底给出原始数据，好过丢失信息。
    lines.append("用于粗略估计委派成本（研究 credits）：")
    lines.append("```json")
    lines.append(_pretty_json(pricing))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_frozen_system_contract(
    *,
    swarm_identity: str,
    bot_profile: Mapping[str, Any],
    agent_catalog: Any,
    procedure_catalog: Any,
    pricing: Any,
    architecture_rules: Mapping[str, str] | None = None,
) -> str:
    """渲染 agent 需要的冻结目录与身份（工作方式见末尾协议段）。"""

    del architecture_rules  # 兼容旧调用方；规则改由 protocol_prompt 统一说明
    profile = {field: str(bot_profile.get(field, "")) for field in _PROFILE_FIELDS}
    parts = [
        _format_identity_section(swarm_identity=str(swarm_identity), bot_profile=profile),
        _format_procedure_section(procedure_catalog),
        _format_agent_section(agent_catalog),
        _format_pricing_section(pricing),
    ]
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def root_assignment_section() -> str:
    """design §8.2 步骤 3：只在根调用里给出「你是本轮起始协调者」。

    与子任务分配一样，这段是 ``[LRS runtime]`` 块内的一节，而不是独立消息：
    历史只追加、不改写，任何一 turn 需要的东西都集中在最后那个块里。
    """

    return (
        "【本分支任务】你是本轮的起始协调者。\n"
        "请先判断这个任务的关键未知点和可并行的路线，再决定：自己调用 Procedure 取证，"
        "还是把子任务委派给更合适的智能体。你的委派决定本轮的调查结构。"
    )


def render_assignment_section(
    *,
    character_prompt: str = "",
    assignment: str,
) -> str:
    """渲染子分支的任务分配段（design §8.2：角色/personality 放这里，不进 system）。

    冻结 system 目录对全体智能体逐字节相同，``character_prompt`` 不能进 system；
    子分支的角色只能随任务分配送达，否则专职智能体与默认智能体的行为完全一致。
    身份、额度与可用 Procedure 由同一个块的其它行给出，这里不重复。
    """

    lines = [
        "【本分支任务】上文是本分支自根节点以来的完整对话历史；从这里开始由你继续推进这条路线。"
    ]
    role = str(character_prompt or "").strip()
    if role:
        lines.append(f"角色与工作偏好：{role}")
    lines.append(f"本次子任务：{assignment}")
    lines.append(
        "上文中父分支已完成的步骤不要重做，直接从那里往前推进，"
        "并把结论、证据与仍不确定的点写进 `report`。"
        "可能有兄弟分支正在并行处理其它路线：你看不到它们的过程，也不能假定它们的结论——"
        "本子任务需要的证据要自己取。"
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RuntimeHeader:
    """每次调用重新计算的运行时状态；不进入稳定 system 前缀。

    这里也是**唯一**告诉智能体「我是谁、我能调用什么」的地方：冻结的 system 目录
    对本轮全部智能体逐字节相同（design §8.2 的 cache 前缀要求），因此逐 agent 的身份
    与 allowlist 只能放在这条每 turn 重建的消息里。
    """

    branch_id: str
    turn_number: int
    agent_capability: str
    report_seconds_remaining: int
    credits_after_input_reservation: float
    active_count: int
    queued_count: int
    protocol: str = JSON_ENVELOPE_PROTOCOL
    agent_id: str = ""
    agent_display_name: str = ""
    allowed_procedures: tuple[str, ...] | None = None
    last_turn_credits: float | None = None
    # 本分支的任务分配（根分支为协调者说明）。与身份/状态合并成同一个块，使
    # 「最后一个 [LRS runtime] 块」自足：读它就够了，不必回溯更早的消息。
    assignment: str = ""

    def _identity_line(self) -> str:
        if not self.agent_id:
            # 身份未注入（旧调用方/测试）：至少保留能力描述，不要谎称 agent_id。
            return f"本分支能力：{self.agent_capability}"
        display = f"（{self.agent_display_name}）" if self.agent_display_name else ""
        capability = f"职责：{self.agent_capability}" if self.agent_capability else ""
        return (
            f"你是 `{self.agent_id}`{display}。{capability}\n"
            f"自委派时 `agent_id` 就写 `{self.agent_id}`。"
        )

    def _procedure_line(self) -> str:
        if self.allowed_procedures is None:
            return ""
        if not self.allowed_procedures:
            return "可调用 Procedure：无（本 turn 只能提交 report 或委派）。"
        return "可调用 Procedure：" + "、".join(f"`{item}`" for item in self.allowed_procedures)

    def message(self) -> dict[str, str]:
        if self.credits_after_input_reservation < 0:
            credit_rule = "余额为负；本 turn 后不能启动后代。"
        elif self.credits_after_input_reservation == 0:
            credit_rule = "余额为零；仍可零 credits 委派。"
        else:
            credit_rule = "余额非负；可在结构上限内委派。"
        protocol = normalize_protocol(self.protocol)
        lines = [
            "[LRS runtime]",
            "本块是你的任务分配与本 turn 的运行时状态（非用户聊天）。"
            "对话历史里可能有多个 `[LRS runtime]` 块，**只有最后一个有效**；"
            "更早的块属于历史记录，其中的身份、余额与剩余时间都已过期。",
            "同一 turn 可并行发出多条 procedures，以及多条**互相独立**的委派；"
            "有先后依赖的工作要串成一条链，不要拆成互相看不见的兄弟委派。"
            "要自己读 Procedure 结果请自委派。",
            self._identity_line(),
        ]
        procedure_line = self._procedure_line()
        if procedure_line:
            lines.append(procedure_line)
        assignment = str(self.assignment or "").strip()
        if assignment:
            lines.append(assignment)
        lines.append(
            f"branch={self.branch_id}; turn={self.turn_number}; capability={self.agent_capability}; "
            f"protocol={protocol}; "
            f"report_seconds_remaining={max(0, self.report_seconds_remaining)}; "
            f"credits_after_input_reservation={self.credits_after_input_reservation:g}; "
            f"active={max(0, self.active_count)}; queued={max(0, self.queued_count)}"
        )
        if self.last_turn_credits is not None:
            lines.append(
                f"上一 turn 本分支实际扣费={self.last_turn_credits:g} credits（用于估算后续委派额度）。"
            )
        lines.append(credit_rule)
        lines.append(protocol_runtime_reminder(protocol))
        return {"role": "user", "content": "\n".join(lines)}


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
        self.catalog_system_message = render_frozen_system_contract(
            swarm_identity=str(swarm_identity),
            bot_profile=bot_profile,
            agent_catalog=agent_catalog,
            procedure_catalog=procedure_catalog,
            pricing=pricing,
        )
        # 兼容旧测试：默认 system_message 为仅 catalog（无协议段）；实际调用走 system_message_for_protocol。
        self.system_message = self.catalog_system_message

    def system_message_for_protocol(self, protocol: str | None) -> str:
        """同一条 system：冻结 catalog + 本 agent 协议说明。"""

        return f"{self.catalog_system_message}\n\n{protocol_system_section(protocol)}"

    def root_context(self, *, coordinator: str) -> BranchContext:
        """根分支起始历史为空：协调者说明随 ``[LRS runtime]`` 块下发。"""

        del coordinator  # 身份由 runtime 块给出，不再单独占一条消息
        return BranchContext([])

    def restart_context(self, *, summary_layers: Sequence[str], coordinator: str) -> BranchContext:
        del coordinator
        history: list[dict[str, Any]] = []
        for index, summary in enumerate(summary_layers, start=1):
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"【上一轮已结束的结论摘要 {index}/{len(summary_layers)}】"
                        "以下是之前几轮留下的 summary layer；原始过程上下文已释放，"
                        "只有这些摘要可用。本轮是**续跑**：在这些摘要基础上继续推进未决问题，"
                        "不要从零重做。\n"
                        f"{summary}"
                    ),
                }
            )
        return BranchContext(history)

    def initial_messages(
        self,
        context: BranchContext,
        *,
        protocol: str | None = None,
    ) -> list[dict[str, Any]]:
        """A branch's stored history **without** the trailing runtime block.

        The block is appended per call by the runtime; storing one here too would
        make the branch start with two of them on its very first turn.
        """

        resolved = normalize_protocol(protocol)
        return [
            {"role": "system", "content": self.system_message_for_protocol(resolved)},
            {"role": "user", "content": self.formalized_task.text},
            *[dict(message) for message in context.history],
        ]

    def messages_for_call(
        self,
        context: BranchContext,
        header: RuntimeHeader,
        *,
        protocol: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved = normalize_protocol(protocol if protocol is not None else header.protocol)
        header_message = (
            header.message()
            if normalize_protocol(header.protocol) == resolved
            else replace(header, protocol=resolved).message()
        )
        return [
            {"role": "system", "content": self.system_message_for_protocol(resolved)},
            {"role": "user", "content": self.formalized_task.text},
            *[dict(message) for message in context.history],
            header_message,
        ]


def replace_leading_system_message(
    messages: Sequence[Mapping[str, Any]], system_content: str
) -> tuple[dict[str, Any], ...]:
    """用协议感知的 system 替换首条 system；若无 system 则插入到开头。"""

    out: list[dict[str, Any]] = []
    replaced = False
    for item in messages:
        if not replaced and str(item.get("role", "")) == "system":
            out.append({"role": "system", "content": system_content})
            replaced = True
            continue
        out.append(dict(item))
    if not replaced:
        out.insert(0, {"role": "system", "content": system_content})
    return tuple(out)


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
    "render_assignment_section",
    "render_frozen_system_contract",
    "replace_leading_system_message",
    "root_assignment_section",
    "should_auto_compact",
]
