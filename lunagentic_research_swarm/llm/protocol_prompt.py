"""按 agent protocol 注入的系统协议段与 runtime 提醒（单一 system 消息内拼接）。"""

from __future__ import annotations

JSON_ENVELOPE_PROTOCOL = "json_envelope"
NATIVE_TOOLS_PROTOCOL = "native_tools"

# 三段共用的工作方式说明：两种协议只是最终输出的载体不同，分支语义完全一致。
_WORKING_RULES = """\
## 工作方式

你在一个分支（branch）里工作。每 turn 提交一个结果信封；runtime 据此执行 Procedure 并物化子分支。

- `report`：本分支**唯一会被保留**的产物。分支结束时，总结器读取你的 `report` 与 Procedure 结果写成分支总结，
  再综合成用户看到的最终报告。请写「已确认的结论、支撑证据与来源、仍不确定的点」，
  不要只写「我将要……」这类计划陈述。
- 语言：`report` 与 `delegations[].task` 一律使用**与正式任务描述相同的语言**；
  引用来源标题、原文片段与检索关键词时保留其原本语言。分支之间语言不一致会让最终报告失真。
- `procedures`：本 turn 要执行的工具。结果会写入本分支上下文，供**随后启动的子分支**阅读；
  它们**不会**像聊天 tool-call 那样自动回到同一气泡让你接着推理。
  若要自己读结果再决策，必须在 `delegations` 里写上**你自己的 agent_id**（自委派）。
- `delegations`：本 turn 要启动的子分支。**这是分叉（fork），不是函数调用。**
  每个子分支继承你当前的完整对话历史（含本 turn 的 Procedure 结果），然后各自独立往下走；
  从分叉那一刻起，它们的推理、取到的证据和结论就会分道扬镳。
  委派之后本分支即退休：子分支的结果**不会**回到你这里，你也没有机会再综合它们——
  综合由总结器在各分支结束后统一完成。`task` 要写清目标、已知前提和期望产出，
  因为这是你唯一一次向那条路线交代意图的机会。
- 并行与依赖：同一 turn 的多个委派**并发执行且互相看不见**，
  只有到 checkpoint / 报告边界才会收到彼此的分支摘要。
  所以只把**真正互相独立、可以同时推进**的路线拆成兄弟委派。
  若 B 必须先看到 A 的结果，就不要写成两条兄弟委派——那样 B 只会基于分叉时的旧信息瞎猜；
  应让同一条链路依次完成（先做 A，再自委派读结果做 B，或让一个子分支承担 A→B 整段）。
- 想让一个答案**回到你自己这一 turn 的上下文**里，用 `builtin.contractor`：
  它让目录中的某个角色以全新上下文独立作答，并把结果直接交回本分支，
  是唯一「有返回值」的方式；`delegations` 没有返回值。
- Credits：每次 LLM 调用都会消耗研究 credits，部分 Procedure 也会扣费。
  委派边的 `credits` 是**分给**子分支的余额（非负，可为 0）：给多少，子分支就只能花多少。
  负余额不能再启动后代；零余额仍可零额委派。
- 时间：runtime header 的 `report_seconds_remaining` 是本轮报告截止前的剩余秒数。
  剩余时间不足以完成「调用工具 + 解读结果」两步时，不要再开新分支，直接整理已有结论并 `core.terminate`。
- 结束本分支：调用 `core.terminate`。若既无普通 Procedure、也无委派，本分支也会自然结束。
"""

_ID_RULES = """\
`procedure_id` / `agent_id` 必须逐字取自上方目录；你本人本 turn 可调用的 Procedure ID 见末尾
`[LRS runtime]` 消息（含 `core.*`）。目录中标注「调用限制」的 Procedure 仅所列智能体可调用，
其他智能体应委派该角色。`arguments` 必须满足该 Procedure 的 arguments schema：`required` 字段一个都不能少，
`enum` 只能取其中列出的值。下面示例中的 `<…>` 是占位符，必须替换成目录里的真实值，不要原样照抄。\
"""

_JSON_ENVELOPE_SECTION = f"""\
{_WORKING_RULES}
## 输出协议（json_envelope）

本 turn 的最终输出必须是**恰好一个** JSON object（禁止 Markdown、代码围栏、前言、后记）。
若底层另有内部工具调用，那些不是本协议的最终输出；最终仍须提交此 envelope。

字段（禁止额外字段）：
- `report` (string)：本 turn 的结论/进展
- `procedures` (array)：`{{"procedure_id":"…","arguments":{{…}},"credits":0,"call_id":null}}`
- `delegations` (array)：`{{"agent_id":"…","task":"…","credits":0}}`

{_ID_RULES}

示例 A（先调用工具，再自委派以阅读结果）：
{{"report":"先取官方时间线，下一 turn 解读检索结果。","procedures":[{{"procedure_id":"<目录中的 procedure_id>","arguments":{{"<schema required 字段>":"…"}},"credits":0}}],"delegations":[{{"agent_id":"<你自己的 agent_id>","task":"根据上一步 Procedure 结果提炼时间线要点","credits":8}}]}}

示例 B（拆分两条**互不依赖**的路线；若 B 需要 A 的结果就不能这样写）：
{{"report":"A、B 两条线互不依赖，可同时推进。","procedures":[],"delegations":[{{"agent_id":"<agent_id>","task":"核查 A：……（含已知前提与期望产出）","credits":5}},{{"agent_id":"<另一个 agent_id>","task":"核查 B：……","credits":5}}]}}

示例 C（终结本分支）：
{{"report":"关键未知点已覆盖。结论：……；证据：……；仍不确定：……","procedures":[{{"procedure_id":"core.terminate","arguments":{{}}}}],"delegations":[]}}
"""

_NATIVE_TOOLS_SECTION = f"""\
{_WORKING_RULES}
## 输出协议（native_tools）

本 turn 的最终输出必须**恰好调用一次** `submit_swarm_turn`，参数为 `report`、`procedures`、`delegations`
（字段语义与上文一致）。不要输出顶层裸 JSON；分析写进 `report`。

{_ID_RULES}
"""

_JSON_ENVELOPE_RUNTIME = (
    "【本 turn 最终输出】仅输出一个 JSON object："
    '{"report":"…","procedures":[…],"delegations":[…]}；'
    "禁止 Markdown；并行提交的委派必须互相独立（委派是分叉，无返回值）；"
    "要自己读 Procedure 结果请用上面的 agent_id 自委派；结束用 core.terminate。"
)

_NATIVE_TOOLS_RUNTIME = (
    "【本 turn 最终输出】恰好调用一次 submit_swarm_turn；"
    "参数含 report/procedures/delegations；并行委派必须互相独立（委派是分叉，无返回值）；"
    "要自己读 Procedure 结果请用上面的 agent_id 自委派；结束用 core.terminate。"
)


def normalize_protocol(protocol: str | None) -> str:
    value = str(protocol or JSON_ENVELOPE_PROTOCOL).strip() or JSON_ENVELOPE_PROTOCOL
    if value not in {JSON_ENVELOPE_PROTOCOL, NATIVE_TOOLS_PROTOCOL}:
        return JSON_ENVELOPE_PROTOCOL
    return value


def protocol_system_section(protocol: str | None) -> str:
    """追加到冻结 catalog 之后的协议说明（仍在同一条 system 消息内）。"""

    if normalize_protocol(protocol) == NATIVE_TOOLS_PROTOCOL:
        return _NATIVE_TOOLS_SECTION.strip()
    return _JSON_ENVELOPE_SECTION.strip()


def protocol_runtime_reminder(protocol: str | None) -> str:
    """写入 `[LRS runtime]` user 消息末尾的短提醒。"""

    if normalize_protocol(protocol) == NATIVE_TOOLS_PROTOCOL:
        return _NATIVE_TOOLS_RUNTIME
    return _JSON_ENVELOPE_RUNTIME


__all__ = [
    "JSON_ENVELOPE_PROTOCOL",
    "NATIVE_TOOLS_PROTOCOL",
    "normalize_protocol",
    "protocol_runtime_reminder",
    "protocol_system_section",
]
