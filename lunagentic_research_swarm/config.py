"""麦麦深度调查组的强类型配置与兼容迁移。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from maibot_sdk import Field, PluginConfigBase
from maibot_sdk.config import (
    extract_plugin_config_version,
    merge_plugin_config_data,
    rebuild_plugin_config_data,
    validate_plugin_config,
)

CURRENT_CONFIG_VERSION = "1.4.0"


def _ui_field(
    default: Any = ..., *, label: str, hint: str, placeholder: str = "", **kwargs: Any
) -> Any:
    """构造带中文 WebUI 元数据的配置字段。"""

    return Field(
        default,
        json_schema_extra={"label": label, "hint": hint, "placeholder": placeholder},
        **kwargs,
    )


def _ui_factory(*, factory: Any, label: str, hint: str, placeholder: str = "", **kwargs: Any) -> Any:
    """构造带中文 WebUI 元数据的可变默认配置字段。"""

    return Field(
        default_factory=factory,
        json_schema_extra={"label": label, "hint": hint, "placeholder": placeholder},
        **kwargs,
    )


class PluginSection(PluginConfigBase):
    __ui_label__ = "插件"
    config_version: str = _ui_field(CURRENT_CONFIG_VERSION, label="配置版本", hint="由插件用于配置迁移。")
    enabled: bool = _ui_field(True, label="启用插件", hint="是否启用麦麦深度调查组。")
    root_agent: str = _ui_field(
        "builtin.quick_thinker",
        label="根智能体",
        hint=(
            "默认启动的根智能体（下拉为当前可用且 can_be_root 的条目）。"
            "列表可能因扩展未刷新而不完整；已选但已禁用/移除的 ID 仍会保留在选项中。"
            "改动仅对新 Task/new round 生效。"
            "若需要填入列表外的 ID，请用插件配置页的「源代码」模式直接编辑 TOML。"
        ),
    )


class LLMSection(PluginConfigBase):
    __ui_label__ = "语言模型"
    default_selector: str = _ui_field(
        "",
        label="默认模型选择器",
        hint=(
            "非空时覆盖各智能体/摘要器的内置默认 selector（不含 embedding）。"
            "若某智能体在「智能体覆盖」中或摘要器单独指定了 selector，则以该指定为准。"
            "格式：task:名称 或 model:物理名。"
        ),
    )


class SummarizerMinCharsSection(PluginConfigBase):
    """各总结角色的输出下限（字符）。

    这里只有下限、没有上限，而且下限刻意压得很低：给出长度**区间**会让模型去凑那个
    区间，而不是照材料本身该写多少写多少。上限交给 ``summarizer.max_tokens``——
    异常冗长本身就是需要被截断并排查的信号，不是提示词该管的事。
    ``0`` 表示完全不在提示词里提长度。
    """

    __ui_label__ = "摘要输出下限"
    formalize_task: int = _ui_field(
        80,
        label="任务形式化下限（字符）",
        hint="低于此长度通常意味着约束或未知点被丢掉了。0 = 不提及长度。",
        ge=0,
        le=20000,
    )
    finalize_branch: int = _ui_field(
        80,
        label="分支总结下限（字符）",
        hint="防止退化成「已检索到相关资料」这类无信息量摘要。0 = 不提及长度。",
        ge=0,
        le=20000,
    )
    finalize_task: int = _ui_field(
        120,
        label="任务综合下限（字符）",
        hint="用户实际读到的报告正文下限。0 = 不提及长度。",
        ge=0,
        le=20000,
    )
    compact_branch: int = _ui_field(
        80,
        label="分支压缩下限（字符）",
        hint="防止把可继续工作的上下文压成一句话。0 = 不提及长度。",
        ge=0,
        le=20000,
    )


class SummarizerSection(PluginConfigBase):
    __ui_label__ = "摘要"
    selector: str = _ui_field(
        "",
        label="摘要模型",
        hint="留空则使用 llm.default_selector，再回退内置 task:mid_memory；非空优先于全局默认。",
    )
    temperature: float = _ui_field(0.2, label="摘要温度", hint="摘要生成的随机性。", ge=0.0, le=2.0)
    max_tokens: int = _ui_field(0, label="摘要最大 token", hint="0 表示由 Host 决定。", ge=0, le=65536)
    min_output_chars: SummarizerMinCharsSection = _ui_factory(
        factory=SummarizerMinCharsSection,
        label="输出下限",
        hint="各总结角色的输出字符下限；只设下限不设上限，上限由 max_tokens 负责。",
    )


class EmbeddingSection(PluginConfigBase):
    __ui_label__ = "向量化"
    selector: str = _ui_field("task:embedding", label="嵌入模型", hint="用于知识向量化的模型选择器。")
    batch_size: int = _ui_field(32, label="批大小", hint="单批处理的文本数。", ge=1, le=256)
    max_concurrent: int = _ui_field(4, label="最大并发", hint="允许同时进行的向量化请求数。", ge=1, le=32)
    auto_rebuild: bool = _ui_field(True, label="自动重建", hint="嵌入模型变更后自动重建向量。")
    retired_generation_retention_seconds: int = _ui_field(
        86400, label="旧代保留秒数", hint="重建后保留旧向量代的时间。", ge=0
    )


class TimingSection(PluginConfigBase):
    __ui_label__ = "时间"
    default_time_budget_seconds: int = _ui_field(600, label="默认时间预算（秒）", hint="每项研究任务的默认执行时间。", ge=1)
    grace_period_seconds: int = _ui_field(60, label="宽限期（秒）", hint="任务超时后的收尾时间。", ge=0)
    pause_timeout_seconds: int = _ui_field(1200, label="暂停超时（秒）", hint="等待恢复前的最长暂停时间。", ge=1)
    feedback_wait_seconds: int = _ui_field(600, label="反馈等待（秒）", hint="等待用户反馈的最长时间。", ge=1)


class BudgetSection(PluginConfigBase):
    __ui_label__ = "预算"
    default_effort_credits: float = _ui_field(100.0, label="默认工作额度", hint="单个任务可使用的默认工作额度。", ge=0.0)
    warning_miss_input_tokens: int = _ui_field(500000, label="输入 token 警告阈值", hint="输入 token 达到此值时提示。", ge=0)
    warning_output_tokens: int = _ui_field(50000, label="输出 token 警告阈值", hint="输出 token 达到此值时提示。", ge=0)


class SchedulerSection(PluginConfigBase):
    __ui_label__ = "调度"
    max_task_llm_concurrency: int = _ui_field(8, label="任务 LLM 并发", hint="单任务的最大 LLM 并发数。", ge=1)
    max_global_llm_concurrency: int = _ui_field(16, label="全局 LLM 并发", hint="插件全局最大 LLM 并发数。", ge=1)
    max_task_procedure_concurrency: int = _ui_field(16, label="任务流程并发", hint="单任务最大流程并发数。", ge=1)
    max_delegations_per_turn: int = _ui_field(8, label="每轮委派数", hint="每个智能体轮次最多委派数。", ge=1)
    max_branch_depth: int = _ui_field(32, label="最大分支深度", hint="智能体委派树允许的最大深度。", ge=1)
    max_agent_calls_per_task: int = _ui_field(256, label="每任务智能体调用数", hint="单任务允许的智能体调用上限。", ge=1)


class ContextSection(PluginConfigBase):
    __ui_label__ = "上下文"
    auto_compact_tokens: int = _ui_field(258000, label="自动压缩阈值", hint="上下文达到此 token 数时自动压缩。", ge=1)
    reserved_output_tokens: int = _ui_field(8192, label="预留输出 token", hint="为最终输出保留的 token 数。", ge=0)
    safety_margin_tokens: int = _ui_field(8192, label="安全余量 token", hint="上下文窗口的安全余量。", ge=0)
    model_context_window: int | None = _ui_field(
        None,
        label="模型上下文窗口",
        hint="可选。已知物理 context window（token）时与自动压缩阈值一并生效；留空则仅用阈值。",
        ge=1,
    )


class ProtocolSection(PluginConfigBase):
    __ui_label__ = "协议"
    default_mode: Literal["json_envelope", "native_tools"] = _ui_field(
        "json_envelope", label="默认协议", hint="智能体与流程之间的调用协议。"
    )
    max_correction_turns: int = _ui_field(1, label="最大修正轮次", hint="结构化输出解析失败后的修正次数。", ge=0, le=1)


class StorageSection(PluginConfigBase):
    __ui_label__ = "存储"
    store_agent_transcripts: bool = _ui_field(False, label="保存智能体记录", hint="是否保存智能体完整对话记录。")
    store_raw_procedure_payloads: bool = _ui_field(False, label="保存原始流程载荷", hint="是否保存流程的原始请求和响应。")


class ExtensionsSection(PluginConfigBase):
    __ui_label__ = "扩展"
    refresh_interval_seconds: int = _ui_field(60, label="刷新间隔（秒）", hint="扩展目录的轮询刷新间隔。", ge=10)


class PriceProfileConfig(PluginConfigBase):
    price_in: float = _ui_field(0.0, label="输入价格", hint="每单位输入 token 的价格。", ge=0.0)
    cache: bool = _ui_field(False, label="支持缓存价格", hint="该模型是否提供缓存输入价格。")
    cache_price_in: float = _ui_field(0.0, label="缓存输入价格", hint="每单位缓存输入 token 的价格。", ge=0.0)
    price_out: float = _ui_field(0.0, label="输出价格", hint="每单位输出 token 的价格。", ge=0.0)


class PricingSection(PluginConfigBase):
    __ui_label__ = "价格"
    models: dict[str, PriceProfileConfig] = _ui_factory(
        factory=dict,
        label="模型价格覆盖",
        hint="只要配置该模型条目，就会忽略 Host 中该模型的全部实际价格；未写字段按 0 计。",
    )


class ReportingSection(PluginConfigBase):
    __ui_label__ = "报告"
    max_report_chars: int = _ui_field(60000, label="报告最大字符数", hint="最终报告允许的最大字符数。", ge=1000, le=200000)
    max_stats_chars: int = _ui_field(12000, label="统计最大字符数", hint="统计摘要允许的最大字符数。", ge=1000, le=50000)
    deliver_intermediate: bool = _ui_field(True, label="发送中间结果", hint="是否向用户发送中间研究结果。")
    deliver_final: bool = _ui_field(True, label="发送最终结果", hint="是否向用户发送最终报告。")
    outbox_poll_seconds: float = _ui_field(2.0, label="发件箱轮询（秒）", hint="报告发件箱的轮询间隔。", ge=0.1, le=60.0)


class FeedbackSection(PluginConfigBase):
    __ui_label__ = "反馈"
    reminders_enabled: bool = _ui_field(True, label="启用提醒", hint="是否提醒用户补充反馈。")
    index_lessons: bool = _ui_field(True, label="索引经验", hint="是否将反馈经验建立索引。")
    max_lesson_chars: int = _ui_field(8000, label="经验最大字符数", hint="单条经验的最大字符数。", ge=500, le=30000)


class CommandsSection(PluginConfigBase):
    __ui_label__ = "命令"
    enabled: bool = _ui_field(True, label="启用命令", hint="是否注册插件命令。")
    max_output_chars: int = _ui_field(12000, label="命令最大输出字符数", hint="命令输出的最大字符数。", ge=1000, le=50000)
    maintenance_allowed_user_ids: list[str] = _ui_factory(
        factory=list,
        label="维护人员用户 ID",
        hint=(
            "允许执行维护命令的身份列表。每项可填平台 user_id（与命令 RPC 的 user_id 对齐），"
            "也可填 MaiBot person_id（md5(平台_用户ID)，跨适配器唯一）。空列表表示不限制。"
        ),
    )
    allow_vector_rebuild: bool = _ui_field(
        False,
        label="允许向量重建",
        hint="是否允许 `/swarm vectors rebuild`。默认关闭；开启后昂贵重建仍受维护白名单约束（空白名单=不限制）。",
    )


class WebSearchSection(PluginConfigBase):
    __ui_label__ = "网页搜索"
    enabled_engines: list[Literal["ddgs", "searxng", "tavily", "you"]] = _ui_factory(
        factory=lambda: ["ddgs"],
        label="启用的搜索引擎",
        hint="按顺序尝试的网页搜索引擎。ddgs 使用 ddgs 库的自动后端聚合。",
    )
    timeout_seconds: float = _ui_field(30.0, label="搜索超时（秒）", hint="单次网页搜索最长等待时间。", gt=0.0, le=120.0)
    max_results: int = _ui_field(10, label="最大结果数", hint="每次搜索返回的最大结果数。", ge=1, le=20)
    ddgs_region: str = _ui_field(
        "us-en",
        label="ddgs 区域",
        hint="ddgs.text 的 region（如 us-en、wt-wt、cn-zh）。",
        placeholder="us-en",
    )
    ddgs_safesearch: Literal["on", "moderate", "off"] = _ui_field(
        "moderate",
        label="ddgs 安全搜索",
        hint="ddgs.text 的 safesearch：on / moderate / off。",
    )
    ddgs_backend: str = _ui_field(
        "auto",
        label="ddgs 后端",
        hint='ddgs.text 的 backend：auto、all，或逗号分隔列表（如 brave,wikipedia）。',
        placeholder="auto",
    )
    searxng_url: str = _ui_field("", label="SearXNG 地址", hint="自建 SearXNG 服务地址。", placeholder="https://searxng.example")
    tavily_api_key: str = _ui_field("", label="Tavily API 密钥", hint="Tavily 搜索 API 密钥。", repr=False)
    you_base_url: str = _ui_field("", label="You.com 地址", hint="You.com API 服务地址。", placeholder="https://api.you.com")
    you_api_key: str = _ui_field("", label="You.com API 密钥", hint="You.com 搜索 API 密钥。", repr=False)


class AgentOverride(PluginConfigBase):
    enabled: bool | None = _ui_field(None, label="启用状态", hint="留空时继承全局设置。")
    selector: str | None = _ui_field(None, label="模型选择器", hint="留空时继承默认选择器。")
    protocol: Literal["json_envelope", "native_tools"] | None = _ui_field(None, label="协议", hint="留空时继承默认协议。")
    auto_compact_tokens: int | None = _ui_field(None, label="压缩阈值", hint="留空时继承默认阈值。", ge=1)


class ProcedureOverride(PluginConfigBase):
    enabled: bool | None = _ui_field(None, label="启用状态", hint="留空时继承全局设置。")
    timeout_seconds: float | None = _ui_field(
        None, label="超时（秒）", hint="0 表示禁用硬超时；留空继承定义。", ge=0.0
    )


class LRSConfig(PluginConfigBase):
    """LRS 插件完整配置根模型。"""

    plugin: PluginSection = _ui_factory(factory=PluginSection, label="插件", hint="插件基础设置。")
    llm: LLMSection = _ui_factory(factory=LLMSection, label="语言模型", hint="语言模型选择设置。")
    summarizer: SummarizerSection = _ui_factory(factory=SummarizerSection, label="摘要", hint="摘要生成设置。")
    embedding: EmbeddingSection = _ui_factory(factory=EmbeddingSection, label="向量化", hint="向量索引设置。")
    timing: TimingSection = _ui_factory(factory=TimingSection, label="时间", hint="时间预算设置。")
    budget: BudgetSection = _ui_factory(factory=BudgetSection, label="预算", hint="工作额度设置。")
    scheduler: SchedulerSection = _ui_factory(factory=SchedulerSection, label="调度", hint="并发与委派设置。")
    context: ContextSection = _ui_factory(factory=ContextSection, label="上下文", hint="上下文压缩设置。")
    protocol: ProtocolSection = _ui_factory(factory=ProtocolSection, label="协议", hint="调用协议设置。")
    storage: StorageSection = _ui_factory(factory=StorageSection, label="存储", hint="本地存储设置。")
    extensions: ExtensionsSection = _ui_factory(factory=ExtensionsSection, label="扩展", hint="扩展发现设置。")
    pricing: PricingSection = _ui_factory(factory=PricingSection, label="价格", hint="模型价格覆盖设置。")
    reporting: ReportingSection = _ui_factory(factory=ReportingSection, label="报告", hint="报告投递设置。")
    feedback: FeedbackSection = _ui_factory(factory=FeedbackSection, label="反馈", hint="用户反馈设置。")
    commands: CommandsSection = _ui_factory(factory=CommandsSection, label="命令", hint="插件命令设置。")
    web_search: WebSearchSection = _ui_factory(factory=WebSearchSection, label="网页搜索", hint="网页搜索引擎设置。")
    agents: dict[str, AgentOverride] = _ui_factory(factory=dict, label="智能体覆盖", hint="按智能体名称覆盖默认配置。")
    procedures: dict[str, ProcedureOverride] = _ui_factory(factory=dict, label="流程覆盖", hint="按流程名称覆盖默认配置。")

    def resolve_agent_overrides(self, agent_name: str = "") -> AgentOverride | dict[str, AgentOverride] | None:
        """按名称取得智能体覆盖；空名称返回全部覆盖。"""

        return self.agents if not agent_name else self.agents.get(agent_name)

    def resolve_procedure_overrides(self, procedure_name: str = "") -> ProcedureOverride | dict[str, ProcedureOverride] | None:
        """按名称取得流程覆盖；空名称返回全部覆盖。"""

        return self.procedures if not procedure_name else self.procedures.get(procedure_name)


def normalize_config(
    config_data: Mapping[str, Any] | None, defaults: Mapping[str, Any]
) -> tuple[dict[str, Any], bool, list[str]]:
    """补齐首版配置字段，保留用户值并让显式非法值直接失败。"""

    default_data = dict(defaults)
    extract_plugin_config_version(default_data)
    if config_data is None:
        normalized = validate_plugin_config(LRSConfig, default_data).model_dump(mode="python", exclude_none=True)
        return normalized, True, ["已生成默认配置"]
    if not isinstance(config_data, Mapping):
        raise TypeError("config_data 必须为 Mapping 或 None")

    raw = dict(config_data)
    # 兼容旧字段 maintenance_allowed_person_ids → maintenance_allowed_user_ids
    commands_raw = dict(raw.get("commands") or {}) if isinstance(raw.get("commands"), Mapping) else {}
    if "maintenance_allowed_person_ids" in commands_raw:
        if "maintenance_allowed_user_ids" not in commands_raw:
            commands_raw["maintenance_allowed_user_ids"] = commands_raw["maintenance_allowed_person_ids"]
        del commands_raw["maintenance_allowed_person_ids"]
        raw["commands"] = commands_raw
    # 兼容 llm.force_selector → llm.default_selector（语义从强制覆盖改为默认回退）
    llm_raw = dict(raw.get("llm") or {}) if isinstance(raw.get("llm"), Mapping) else {}
    if "force_selector" in llm_raw:
        if "default_selector" not in llm_raw or not str(llm_raw.get("default_selector") or "").strip():
            llm_raw["default_selector"] = llm_raw.get("force_selector") or ""
        del llm_raw["force_selector"]
        raw["llm"] = llm_raw
    # 兼容 web_search.enabled_engines: duckduckgo → ddgs（自动后端）
    web_search_raw = dict(raw.get("web_search") or {}) if isinstance(raw.get("web_search"), Mapping) else {}
    engines_raw = web_search_raw.get("enabled_engines")
    if isinstance(engines_raw, list):
        migrated: list[str] = []
        seen: set[str] = set()
        for engine in engines_raw:
            name = "ddgs" if engine == "duckduckgo" else engine
            if not isinstance(name, str) or name in seen:
                continue
            seen.add(name)
            migrated.append(name)
        if migrated != engines_raw:
            web_search_raw["enabled_engines"] = migrated
            raw["web_search"] = web_search_raw
    raw_version = extract_plugin_config_version(raw)
    rebuilt = rebuild_plugin_config_data(default_data, raw)
    merged, merged_changed = merge_plugin_config_data(default_data, rebuilt)
    plugin = dict(merged["plugin"])
    plugin["config_version"] = CURRENT_CONFIG_VERSION
    merged["plugin"] = plugin
    validated = validate_plugin_config(LRSConfig, merged)
    # Host 用 tomlkit 写回 config.toml，不能包含 None（如 model_context_window 默认未设置）
    normalized = validated.model_dump(mode="python", exclude_none=True)
    changed = merged_changed or raw != normalized
    notes: list[str] = []
    if raw_version != CURRENT_CONFIG_VERSION:
        notes.append(f"配置版本已从 {raw_version} 迁移至 {CURRENT_CONFIG_VERSION}")
    elif changed:
        notes.append("已补齐缺失的配置字段")
    return normalized, changed, notes
