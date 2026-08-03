# 麦麦深度调查组：基础与契约 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可被 Host 加载的 LRS 插件骨架、强类型配置、稳定领域模型、SQLite 权威存储、扩展契约、模型选择/计费适配、普通智能体协议与不可委派的核心总结器服务。

**Architecture:** `plugin.py` 只负责 SDK 组件与生命周期接线，复杂逻辑放在 `lunagentic_research_swarm/` 小模块中。所有权威状态以 SQLite transaction 保存；外部 agent/Procedure 与内置定义共用同一组 Pydantic 契约和 validator；任务别名走公开 `ctx.llm`，物理模型走单一隔离适配器。

**Tech Stack:** Python 3.10+、MaiBot Plugin SDK 2.7.1、Pydantic 2、stdlib `sqlite3`、`asyncio`、`httpx`、pytest、pytest-asyncio、Hypothesis。

## Global Constraints

- 在独立仓库 `maibot-lunagentic-research-swarm/` 内工作；不要修改 MaiBot Host、SDK 或 workspace 根目录。
- 插件 ID 固定为 `com.0-hz.lunagentic-research-swarm`，首发版本固定为 `0.1.0`，Python 包固定为 `lunagentic_research_swarm`。
- Host 兼容范围 `>=1.0.0, <=1.99.99`；SDK 兼容范围 `>=2.7.1, <=2.99.99`；Python `>=3.10`。
- `pyproject.toml` 是依赖真源，`requirements.txt` 必须与运行期依赖同步；测试依赖只放 `project.optional-dependencies.dev`。
- 源码注释、日志、WebUI 文案和 README 默认使用简体中文；本阶段只提供中文 prompt，避免出现未同步的英/日模板。
- 不定义、不保存、不转发 provider `reasoning`；任何内部结果归一化都主动丢弃该字段。
- 不做静默模型 fallback：selector、root、总结器或物理 pinning 不可用时返回结构化错误并进入 health。
- 原始 agent transcript 与原始 Procedure payload 默认不持久化，且永不进入向量索引。
- 所有文件编辑使用 `apply_patch`；每个 Task 按红—绿—重构顺序执行并独立提交。
- 本计划只建立基础与契约；完整调度、credits、报告、默认扩展与向量/反馈分别由后续计划实现，不能把本阶段标记为可发布版本。

## 计划套件与稳定边界

按顺序执行：

1. 本计划：基础、配置、存储、扩展与 LLM 契约。
2. `docs/superpowers/plans/2026-08-03-lrs-runtime-and-reporting.md`：credits、reducer、调度器、turn、生命周期、checkpoint/grace/report、Planner tools 与 Maisaka outbox。
3. `docs/superpowers/plans/2026-08-03-lrs-bundled-release.md`：全部默认 agent/Procedure、LanceDB、历史案例、feedback、命令、隐私验收与首发文档。
4. `maibot-fetch-url-plugin/docs/superpowers/plans/2026-08-03-lrs-procedure-provider.md`：推荐但非必需的网页全文抓取 provider。

本计划交付并冻结下列跨计划接口：

| Owner | 冻结签名 |
|---|---|
| `SQLiteStateStore` | `open() -> None`、`close() -> None`、`transact(commands: Sequence[StoreCommand]) -> None`、`load_task(task_id: str) -> StoredTask | None`（均为 async） |
| `AgentRegistry` | `replace_provider(provider_id: str, definitions: Sequence[AgentDefinition]) -> CatalogDelta`、`snapshot(overrides: Mapping[str, AgentOverride]) -> AgentCatalogSnapshot` |
| `ProcedureRegistry` | `replace_provider(provider_id: str, definitions: Sequence[ProcedureDefinition]) -> CatalogDelta`、`snapshot(overrides: Mapping[str, ProcedureOverride]) -> ProcedureCatalogSnapshot` |
| `LLMGateway` | async `generate(request: GenerationRequest) -> GenerationResult` |
| `SummarizerService` | async `formalize_task(FormalizationRequest)`、`finalize_branch(BranchFinalizationRequest)`、`finalize_task(TaskFinalizationRequest)`、`compact_branch(CompactionRequest)`，全部返回 `SummaryResult` |

---

### Task 1: 创建可加载的独立插件骨架

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `_manifest.json`
- Create: `config.default.toml`
- Create: `plugin.py`
- Create: `lunagentic_research_swarm/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_plugin_contract.py`

**Interfaces:**
- Consumes: MaiBot SDK `MaiBotPlugin`、`ON_MODEL_CONFIG_RELOAD`、`PluginContext.paths`。
- Produces: `LunagenticResearchSwarmPlugin`、模块级 `create_plugin()`、最终 manifest capability/dependency 声明，以及所有后续测试共享的 `plugin_module` fixture。

- [ ] **Step 1: 写插件装载契约失败测试**

```python
# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SDK_ROOT = REPO_ROOT.parent / "maibot-plugin-sdk"
for path in (REPO_ROOT, SDK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def plugin_module():
    import plugin

    return plugin
```

```python
# tests/test_plugin_contract.py
from __future__ import annotations

import inspect
import json
from pathlib import Path

from maibot_sdk import MaiBotPlugin, ON_MODEL_CONFIG_RELOAD


def test_manifest_and_factory_contract(plugin_module) -> None:
    repo = Path(plugin_module.__file__).resolve().parent
    manifest = json.loads((repo / "_manifest.json").read_text(encoding="utf-8"))
    instance = plugin_module.create_plugin()

    assert manifest["id"] == "com.0-hz.lunagentic-research-swarm"
    assert manifest["version"] == "0.1.0"
    assert manifest["sdk"]["min_version"] == "2.7.1"
    assert isinstance(instance, MaiBotPlugin)
    assert instance.plugin_id == manifest["id"]
    assert instance.get_config_reload_subscriptions() == [ON_MODEL_CONFIG_RELOAD]
    assert inspect.iscoroutinefunction(type(instance).on_load)
    assert inspect.iscoroutinefunction(type(instance).on_unload)
    assert inspect.iscoroutinefunction(type(instance).on_config_update)


def test_manifest_declares_final_capabilities(plugin_module) -> None:
    repo = Path(plugin_module.__file__).resolve().parent
    manifest = json.loads((repo / "_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["capabilities"]) == {
        "api.call",
        "api.list",
        "chat.get_all_streams",
        "chat.get_group_streams",
        "chat.get_private_streams",
        "config.get",
        "knowledge.search",
        "llm.embed",
        "llm.generate",
        "llm.generate_with_tools",
        "maisaka.context.append",
        "maisaka.proactive.trigger",
        "message.build_readable",
        "message.get_by_id",
        "message.get_by_time",
        "message.get_by_time_in_chat",
        "message.get_recent",
        "person.get_id",
        "person.get_id_by_name",
        "person.get_value",
        "send.text",
    }
```

- [ ] **Step 2: 运行装载测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_plugin_contract.py -v`

Expected: FAIL，原因是 `plugin.py`、manifest 或 factory 尚不存在。

- [ ] **Step 3: 写 packaging、manifest 与薄入口**

`pyproject.toml` 使用以下依赖边界；LanceDB 留到第三阶段加入，避免基础阶段下载大型 wheel：

```toml
[project]
name = "maibot-lunagentic-research-swarm"
version = "0.1.0"
description = "麦麦深度调查组：可扩展的多智能体深度研究蜂群"
requires-python = ">=3.10"
dependencies = [
    "maibot-plugin-sdk>=2.7.1,<3.0.0",
    "pydantic>=2.10.0,<3.0.0",
    "httpx>=0.28.0,<0.29.0",
]

[project.optional-dependencies]
dev = [
    "hypothesis>=6.100.0,<7.0.0",
    "pytest>=8.0.0,<9.0.0",
    "pytest-asyncio>=1.4.0,<2.0.0",
    "ruff>=0.4.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["lunagentic_research_swarm*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py310"
line-length = 120
```

`requirements.txt` 必须逐项同步三个运行期依赖。`_manifest.json` 使用作者 `kes`、仓库 URL `https://github.com/yufei-pan/maibot-lunagentic-research-swarm`，并把测试断言中的 capability 全部声明；dependencies 声明 `pydantic>=2.10.0` 与 `httpx>=0.28.0`，SDK 不重复列入 Python package dependency。

`plugin.py` 使用下列入口方式，兼容 Host 的 package spec 和离线 `import plugin`：

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from maibot_sdk import MaiBotPlugin, ON_MODEL_CONFIG_RELOAD

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


class LunagenticResearchSwarmPlugin(MaiBotPlugin):
    plugin_id = "com.0-hz.lunagentic-research-swarm"
    config_reload_subscriptions = {ON_MODEL_CONFIG_RELOAD}

    async def on_load(self) -> None:
        self.ctx.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.logger.info("麦麦深度调查组基础组件已加载")

    async def on_unload(self) -> None:
        self.ctx.logger.info("麦麦深度调查组基础组件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        self.ctx.logger.info("麦麦深度调查组收到配置更新：scope=%s version=%s", scope, version)


def create_plugin() -> LunagenticResearchSwarmPlugin:
    return LunagenticResearchSwarmPlugin()
```

`.gitignore` 至少忽略 `.venv/`、`__pycache__/`、`.pytest_cache/`、`*.pyc`、`config.toml`、`config.local.toml`、`data/`、`.superpowers/`。`config.default.toml` 暂只放 `[plugin]` 内的 `config_version = "1.0.0"` 与 `enabled/root_agent`，完整 schema 在 Task 2 一次补齐。

- [ ] **Step 4: 运行装载测试并确认通过**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_plugin_contract.py -v`

Expected: 2 passed。

- [ ] **Step 5: 提交插件骨架**

```bash
git add .gitignore pyproject.toml requirements.txt _manifest.json config.default.toml plugin.py lunagentic_research_swarm/__init__.py tests/conftest.py tests/test_plugin_contract.py
git commit -m "feat: scaffold Lunagentic Research Swarm plugin"
```

### Task 2: 实现完整强类型配置与迁移

**Files:**
- Create: `lunagentic_research_swarm/config.py`
- Modify: `plugin.py`
- Modify: `config.default.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: SDK `PluginConfigBase`、`Field`、`merge_plugin_config_data`、`rebuild_plugin_config_data`、`extract_plugin_config_version`、`validate_plugin_config`。
- Produces: `CURRENT_CONFIG_VERSION`、`LRSConfig`、`normalize_config(config_data, defaults) -> tuple[dict[str, Any], bool, list[str]]`、`resolve_agent_overrides()`、`resolve_procedure_overrides()`。

- [ ] **Step 1: 写默认值、override 与迁移失败测试**

```python
from __future__ import annotations

from lunagentic_research_swarm.config import CURRENT_CONFIG_VERSION, LRSConfig, normalize_config


def test_config_defaults_match_approved_spec() -> None:
    config = LRSConfig()
    assert config.plugin.root_agent == "builtin.quick_thinker"
    assert config.summarizer.selector == "task:mid_memory"
    assert config.summarizer.temperature == 0.2
    assert config.summarizer.max_tokens == 0
    assert config.timing.default_time_budget_seconds == 120
    assert config.timing.grace_period_seconds == 60
    assert config.timing.pause_timeout_seconds == 1200
    assert config.timing.feedback_wait_seconds == 600
    assert config.budget.default_effort_credits == 100.0
    assert config.context.auto_compact_tokens == 258000
    assert config.protocol.default_mode == "json_envelope"
    assert config.protocol.max_correction_turns == 1
    assert not config.storage.store_agent_transcripts
    assert not config.storage.store_raw_procedure_payloads
    assert "reasoning" not in LRSConfig.model_json_schema()["properties"]


def test_explicit_price_override_is_a_complete_profile() -> None:
    config = LRSConfig.model_validate(
        {"pricing": {"models": {"gpt-fast": {"price_in": 1.0}}}}
    )
    profile = config.pricing.models["gpt-fast"]
    assert profile.price_in == 1.0
    assert profile.cache is False
    assert profile.cache_price_in == 0.0
    assert profile.price_out == 0.0


def test_migration_preserves_user_values_and_bumps_version() -> None:
    raw = {
        "plugin": {"config_version": "0.0.1"},
        "timing": {"default_time_budget_seconds": 300},
        "storage": {"store_agent_transcripts": True},
    }
    merged, changed, notes = normalize_config(raw, LRSConfig().model_dump(mode="python"))
    assert changed
    assert merged["plugin"]["config_version"] == CURRENT_CONFIG_VERSION
    assert merged["timing"]["default_time_budget_seconds"] == 300
    assert merged["storage"]["store_agent_transcripts"] is True
    assert notes
```

- [ ] **Step 2: 运行配置测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: lunagentic_research_swarm.config`。

- [ ] **Step 3: 实现配置模型**

在 `config.py` 定义以下 section，所有数值上下限由 Pydantic 明确拒绝非法值，不使用 `or` 链掩盖错误：

```python
CURRENT_CONFIG_VERSION = "1.0.0"

class PluginSection(PluginConfigBase):
    enabled: bool = True
    root_agent: str = "builtin.quick_thinker"

class LLMSection(PluginConfigBase):
    force_selector: str = ""

class SummarizerSection(PluginConfigBase):
    selector: str = "task:mid_memory"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=0, ge=0, le=65536)

class EmbeddingSection(PluginConfigBase):
    selector: str = "task:embedding"
    batch_size: int = Field(default=32, ge=1, le=256)
    max_concurrent: int = Field(default=4, ge=1, le=32)
    auto_rebuild: bool = True
    retired_generation_retention_seconds: int = Field(default=86400, ge=0)

class TimingSection(PluginConfigBase):
    default_time_budget_seconds: int = Field(default=120, ge=1)
    grace_period_seconds: int = Field(default=60, ge=0)
    pause_timeout_seconds: int = Field(default=1200, ge=1)
    feedback_wait_seconds: int = Field(default=600, ge=1)

class BudgetSection(PluginConfigBase):
    default_effort_credits: float = Field(default=100.0, ge=0.0)
    warning_miss_input_tokens: int = Field(default=500000, ge=0)
    warning_output_tokens: int = Field(default=50000, ge=0)

class SchedulerSection(PluginConfigBase):
    max_task_llm_concurrency: int = Field(default=8, ge=1)
    max_global_llm_concurrency: int = Field(default=16, ge=1)
    max_task_procedure_concurrency: int = Field(default=16, ge=1)
    max_delegations_per_turn: int = Field(default=8, ge=1)
    max_branch_depth: int = Field(default=32, ge=1)
    max_agent_calls_per_task: int = Field(default=256, ge=1)

class ContextSection(PluginConfigBase):
    auto_compact_tokens: int = Field(default=258000, ge=1)
    reserved_output_tokens: int = Field(default=8192, ge=0)
    safety_margin_tokens: int = Field(default=8192, ge=0)

class ProtocolSection(PluginConfigBase):
    default_mode: Literal["json_envelope", "native_tools"] = "json_envelope"
    max_correction_turns: int = Field(default=1, ge=0, le=1)

class StorageSection(PluginConfigBase):
    store_agent_transcripts: bool = False
    store_raw_procedure_payloads: bool = False

class ExtensionsSection(PluginConfigBase):
    refresh_interval_seconds: int = Field(default=60, ge=10)

class PriceProfileConfig(PluginConfigBase):
    price_in: float = Field(default=0.0, ge=0.0)
    cache: bool = False
    cache_price_in: float = Field(default=0.0, ge=0.0)
    price_out: float = Field(default=0.0, ge=0.0)

class PricingSection(PluginConfigBase):
    models: dict[str, PriceProfileConfig] = Field(default_factory=dict)

class ReportingSection(PluginConfigBase):
    max_report_chars: int = Field(default=60000, ge=1000, le=200000)
    max_stats_chars: int = Field(default=12000, ge=1000, le=50000)
    deliver_intermediate: bool = True
    deliver_final: bool = True
    outbox_poll_seconds: float = Field(default=2.0, ge=0.1, le=60.0)

class FeedbackSection(PluginConfigBase):
    reminders_enabled: bool = True
    index_lessons: bool = True
    max_lesson_chars: int = Field(default=8000, ge=500, le=30000)

class CommandsSection(PluginConfigBase):
    enabled: bool = True
    max_output_chars: int = Field(default=12000, ge=1000, le=50000)
    maintenance_allowed_person_ids: list[str] = Field(default_factory=list)
    allow_vector_rebuild: bool = True
```

同时定义 `AgentOverride(enabled: bool | None, selector: str | None, protocol: Literal["json_envelope", "native_tools"] | None, auto_compact_tokens: int | None)`、`ProcedureOverride(enabled: bool | None, timeout_seconds: float | None)` 与下列搜索配置。root `LRSConfig` 另含 `agents: dict[str, AgentOverride]`、`procedures: dict[str, ProcedureOverride]`。secret 字段必须 `repr=False`，不写入日志：

```python
class WebSearchSection(PluginConfigBase):
    enabled_engines: list[Literal["duckduckgo", "searxng", "tavily", "you"]] = Field(
        default_factory=lambda: ["duckduckgo"]
    )
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    max_results: int = Field(default=10, ge=1, le=20)
    searxng_url: str = ""
    tavily_api_key: str = Field(default="", repr=False)
    you_base_url: str = ""
    you_api_key: str = Field(default="", repr=False)
```

`LRSConfig` 汇总全部 section，并通过 `json_schema_extra` 为每个配置项提供中文 `label`、`hint`、`placeholder`；`pricing.models` 的 hint 必须明确写出：“只要配置该模型条目，就会忽略 Host 中该模型的全部实际价格；未写字段按 0 计。”

`normalize_config()` 先取得版本，再用 SDK merge/rebuild helper 合并，最后 `validate_plugin_config(LRSConfig, merged)`；显式非法值让验证异常向上暴露，不把它换成默认值。首版迁移只允许从缺失/旧版本补字段并保留用户值。

- [ ] **Step 4: 补齐默认 TOML 并接入 plugin config_model**

把规格第 21 节的全部静态 section 写入 `config.default.toml`，包括默认 DuckDuckGo、reporting/feedback/commands；不写示例 API key。更新入口：

```python
from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF, MaiBotPlugin, ON_MODEL_CONFIG_RELOAD
from lunagentic_research_swarm.config import LRSConfig, normalize_config

class LunagenticResearchSwarmPlugin(MaiBotPlugin):
    config_model = LRSConfig

    def normalize_plugin_config(self, config_data: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
        merged, changed, notes = normalize_config(config_data or {}, LRSConfig().model_dump(mode="python"))
        if notes:
            self.ctx.logger.info("LRS 配置迁移：%s", "；".join(notes))
        return merged, changed

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            self.ctx.logger.info("LRS 插件配置已更新：version=%s；新 round 起生效", version)
        elif scope == ON_MODEL_CONFIG_RELOAD:
            self.ctx.logger.info("LRS 模型价格与 task 列表快照已更新：version=%s", version)
```

- [ ] **Step 5: 运行配置与装载测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_config.py tests/test_plugin_contract.py -v`

Expected: 5 passed。

- [ ] **Step 6: 提交配置契约**

```bash
git add lunagentic_research_swarm/config.py plugin.py config.default.toml tests/test_config.py
git commit -m "feat: add typed LRS configuration"
```

### Task 3: 定义领域模型、ID 与不可变任务描述

**Files:**
- Create: `lunagentic_research_swarm/errors.py`
- Create: `lunagentic_research_swarm/models.py`
- Create: `lunagentic_research_swarm/runtime/__init__.py`
- Create: `lunagentic_research_swarm/runtime/events.py`
- Test: `tests/test_models.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: Python `dataclasses`、`enum.StrEnum` 兼容写法（Python 3.10 使用 `class X(str, Enum)`）。
- Produces: `new_task_id() -> str`、`FormalizedTask.create(text) -> FormalizedTask`、`TaskStatus`、`BranchLifecycle`、`ReportKind`、`SummaryKind`、`TaskSnapshot`、`RoundSnapshot`、`BranchRuntime`、带 `event_id/generation/occurred_at` 的事件 union。

- [ ] **Step 1: 写 ID、摘要 hash 与序列化失败测试**

```python
from lunagentic_research_swarm.models import FormalizedTask, TaskStatus, new_task_id


def test_task_id_and_formalized_task_are_stable() -> None:
    task_id = new_task_id()
    formalized = FormalizedTask.create("目标：逐字节保持。\n")
    assert task_id.startswith("lrs_")
    assert len(task_id) == 36
    assert formalized.text.encode("utf-8") == b"\xe7\x9b\xae\xe6\xa0\x87\xef\xbc\x9a\xe9\x80\x90\xe5\xad\x97\xe8\x8a\x82\xe4\xbf\x9d\xe6\x8c\x81\xe3\x80\x82\n"
    assert FormalizedTask.create(formalized.text).sha256 == formalized.sha256
    assert TaskStatus.RUNNING.value == "RUNNING"


def test_formalized_task_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="正式任务描述不能为空"):
        FormalizedTask.create("  ")
```

```python
from lunagentic_research_swarm.runtime.events import AgentCallCompleted, event_to_json, event_from_json


def test_event_round_trip_keeps_generation_and_usage() -> None:
    event = AgentCallCompleted(
        event_id="evt_1",
        task_id="lrs_1",
        round_id="rnd_1",
        generation=3,
        branch_id="br_1",
        call_id="call_1",
        result_id="result_1",
    )
    assert event_from_json(event_to_json(event)) == event
```

- [ ] **Step 2: 运行领域测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_models.py tests/test_events.py -v`

Expected: FAIL，因为模型与事件模块尚不存在。

- [ ] **Step 3: 实现稳定模型和错误码**

`errors.py` 定义 `LRSError(code: str, message: str, metadata: Mapping[str, Any] | None)`，`to_result()` 固定返回 `{"success": False, "error": {"code", "message", "metadata"}}`。预先定义后续计划要使用的 code 常量：`task_not_found`、`invalid_state`、`invalid_selector`、`root_agent_unavailable`、`summarizer_unavailable`、`agent_unavailable`、`procedure_unavailable`、`protocol_invalid`、`storage_commit_failed`、`task_finished_insufficient_funds`、`embedding_generation_mismatch`、`vector_index_rebuilding`。

`models.py` 用 frozen/slots dataclass 表达不可变持久模型，用普通 dataclass 表达运行期可变叶子：

```python
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

@dataclass(frozen=True, slots=True)
class FormalizedTask:
    text: str
    sha256: str

    @classmethod
    def create(cls, text: str) -> "FormalizedTask":
        if not text.strip():
            raise ValueError("正式任务描述不能为空")
        return cls(text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
```

`new_task_id()` 返回 `lrs_` + `uuid.uuid4().hex`；round、branch、turn、call、summary、report、ledger、outbox 同样用显式 prefix helper。`BranchRuntime` 至少包含 immutable task 引用、frozen catalog fingerprint、generation、messages、credits、depth、parent/children、pending delegations、procedure results、latest checkpoint ID 与 lifecycle；其 `build_prompt_messages()` 每次都从原 `FormalizedTask.text` 重新插入 User 1，不接受 compact summary 覆盖该字段。

- [ ] **Step 4: 实现显式事件 union**

`runtime/events.py` 为每一种外部完成/控制输入建立 dataclass，禁止 worker 传入可变 `TaskSnapshot`。至少定义：`TaskCreated`、`FormalizationSucceeded`、`FormalizationFailed`、`AgentCallRequested`、`AgentCallReserved`、`AgentCallCompleted`、`AgentCallFailed`、`ProcedureBatchCompleted`、`SummaryCompleted`、`SummaryFailed`、`BranchCheckpointed`、`BranchFinalized`、`ReportDeadlineReached`、`GraceExpired`、`ReportCompleted`、`FinalReportCompleted`、`FinalReportFailed`、`AllInflightSettled`、`PauseRequested`、`ContinueRequested`、`StopRequested`、`ContextSupplied`、`FeedbackSubmitted`、`OutboxDelivered`、`PersistenceFailed`。所有事件含 `event_id/task_id/round_id/generation/occurred_at`；`event_to_json` 写入 `event_type`，`event_from_json` 只接受注册表里的已知类型，未知类型抛 `ValueError`。

- [ ] **Step 5: 运行模型与事件测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_models.py tests/test_events.py -v`

Expected: 全部 PASS。

- [ ] **Step 6: 提交领域契约**

```bash
git add lunagentic_research_swarm/errors.py lunagentic_research_swarm/models.py lunagentic_research_swarm/runtime tests/test_models.py tests/test_events.py
git commit -m "feat: define LRS domain events and models"
```

### Task 4: 建立 SQLite 权威存储与迁移

**Files:**
- Create: `lunagentic_research_swarm/storage/__init__.py`
- Create: `lunagentic_research_swarm/storage/migrations.py`
- Create: `lunagentic_research_swarm/storage/sqlite.py`
- Test: `tests/storage/test_sqlite.py`
- Test: `tests/storage/test_privacy.py`

**Interfaces:**
- Consumes: `FormalizedTask`、`TaskStatus`、`ctx.paths.data_dir / "state.sqlite3"`。
- Produces: `SQLiteStateStore.open/close/transact/load_task/list_active_rounds/load_summary_layer`，以及 immutable `StoreCommand(kind, values)`。

- [ ] **Step 1: 写 transaction、crash 与默认隐私失败测试**

```python
import sqlite3

import pytest

from lunagentic_research_swarm.storage.sqlite import SQLiteStateStore, StoreCommand


@pytest.mark.asyncio
async def test_commands_commit_atomically(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    commands = [
        StoreCommand("insert_task", {"task_id": "lrs_a", "stream_id": "s", "created_at": 1.0}),
        StoreCommand(
            "insert_round",
            {
                "round_id": "rnd_a", "task_id": "lrs_a", "round_number": 1,
                "status": "FORMALIZING", "generation": 1, "time_budget_seconds": 120,
                "credit_pool": 0.0, "started_at": 1.0,
            },
        ),
    ]
    await store.transact(commands)
    assert (await store.load_task("lrs_a")).current_round.status.value == "FORMALIZING"
    await store.close()


@pytest.mark.asyncio
async def test_failed_command_rolls_back_whole_transition(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    with pytest.raises(sqlite3.IntegrityError):
        await store.transact([
            StoreCommand("insert_task", {"task_id": "lrs_b", "stream_id": "s", "created_at": 1.0}),
            StoreCommand("insert_task", {"task_id": "lrs_b", "stream_id": "s", "created_at": 1.0}),
        ])
    assert await store.load_task("lrs_b") is None
    await store.close()
```

```python
@pytest.mark.asyncio
async def test_default_storage_has_no_raw_transcript_or_payload(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.open()
    names = await store.table_names()
    assert "agent_transcripts" not in names
    assert "procedure_raw_payloads" not in names
    await store.close()
```

- [ ] **Step 2: 运行存储测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_sqlite.py tests/storage/test_privacy.py -v`

Expected: FAIL because storage modules do not exist。

- [ ] **Step 3: 写 migration 001 的完整权威表结构**

`migrations.py` 暴露有序 `MIGRATIONS: Sequence[Migration]`，migration 001 在一个 transaction 中创建以下表和关键约束：

```sql
CREATE TABLE tasks (
  task_id TEXT PRIMARY KEY,
  stream_id TEXT NOT NULL,
  formalized_text TEXT,
  formalized_sha256 TEXT,
  current_round_number INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE investigation_rounds (
  round_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  round_number INTEGER NOT NULL,
  generation INTEGER NOT NULL,
  status TEXT NOT NULL,
  time_budget_seconds INTEGER NOT NULL,
  grace_period_seconds INTEGER NOT NULL DEFAULT 60,
  credit_pool REAL NOT NULL,
  catalog_fingerprint TEXT,
  started_at REAL NOT NULL,
  report_deadline_at REAL,
  ended_at REAL,
  UNIQUE(task_id, round_number)
);
CREATE TABLE lifecycle_events (
  event_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
  event_type TEXT NOT NULL,
  from_status TEXT,
  to_status TEXT,
  metadata_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE branches (
  branch_id TEXT PRIMARY KEY,
  round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
  parent_branch_id TEXT,
  agent_id TEXT NOT NULL,
  lifecycle TEXT NOT NULL,
  depth INTEGER NOT NULL,
  credit_balance REAL NOT NULL,
  generation INTEGER NOT NULL,
  latest_checkpoint_summary_id TEXT,
  terminal_summary_id TEXT,
  created_at REAL NOT NULL,
  finalized_at REAL
);
CREATE TABLE summaries (
  summary_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
  branch_id TEXT,
  kind TEXT NOT NULL,
  report_epoch INTEGER,
  text TEXT,
  status TEXT NOT NULL,
  error_code TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE reports (
  report_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  round_id TEXT NOT NULL REFERENCES investigation_rounds(round_id),
  epoch INTEGER NOT NULL,
  kind TEXT NOT NULL,
  text TEXT,
  status TEXT NOT NULL,
  running_branch_count INTEGER NOT NULL,
  stats_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(round_id, epoch)
);
CREATE TABLE credit_ledger (
  ledger_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  branch_id TEXT,
  call_id TEXT,
  entry_kind TEXT NOT NULL,
  amount REAL NOT NULL,
  balance_after REAL,
  metadata_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE llm_usage (
  usage_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  branch_id TEXT,
  call_id TEXT NOT NULL,
  role TEXT NOT NULL,
  selector TEXT NOT NULL,
  estimated_model_name TEXT,
  actual_model_name TEXT,
  price_source TEXT NOT NULL,
  price_fingerprint TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  cache_hit_tokens INTEGER NOT NULL,
  cache_miss_tokens INTEGER NOT NULL,
  estimated_charge REAL NOT NULL,
  actual_charge REAL,
  adjustment REAL,
  reconciliation_status TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE procedure_calls (
  request_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  procedure_id TEXT NOT NULL,
  provider_plugin_id TEXT NOT NULL,
  status TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  error_code TEXT,
  provenance_json TEXT NOT NULL,
  external_cost_json TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE extension_fingerprints (
  event_id TEXT PRIMARY KEY,
  provider_plugin_id TEXT NOT NULL,
  extension_kind TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  availability TEXT NOT NULL,
  error_json TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE feedback_events (
  feedback_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  disposition TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE feedback_reminders (
  reminder_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  due_at REAL NOT NULL,
  status TEXT NOT NULL,
  triggered_at REAL,
  UNIQUE(round_id)
);
CREATE TABLE maisaka_outbox (
  outbox_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  round_id TEXT NOT NULL,
  report_id TEXT,
  delivery_kind TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  next_attempt_at REAL NOT NULL,
  last_error TEXT,
  created_at REAL NOT NULL,
  delivered_at REAL
);
CREATE TABLE vector_jobs (
  job_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  generation INTEGER,
  status TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  error_code TEXT,
  error_json TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at REAL NOT NULL
);
```

为 `round_id/status`、`lifecycle_events(task_id, created_at)`、`task_id/created_at`、`branch_id/kind/created_at`、`vector_jobs/status`、`outbox/status/next_attempt_at` 建索引。debug 原始表不在 migration 中；第三阶段只在相应开关开启时创建独立 debug 数据库，确保默认权威库没有 raw columns。

- [ ] **Step 4: 实现串行异步 SQLite store**

`SQLiteStateStore` 使用单个 `sqlite3.Connection(check_same_thread=False)`、`PRAGMA foreign_keys=ON`、`journal_mode=WAL`、`synchronous=FULL`，并用 `asyncio.Lock` 串行访问。每个 public async 方法通过 `asyncio.to_thread()` 执行同步 helper；`transact()` 明确 `BEGIN IMMEDIATE`，逐个 dispatch `StoreCommand.kind` 到固定 handler map，成功 commit，任何异常 rollback 后原样抛出。不要动态拼接表名或 SQL 字段。

`load_summary_layer(task_id)` 只返回 formalized task、summaries、reports、feedback 与新 supplied context，不读取 debug 数据。`list_active_rounds()` 只选择六个非终态与 `FINALIZING`。`mark_active_rounds_interrupted(now)` 在一个 transaction 内设置 `INTERRUPTED/ended_at` 并保留 usage reservation 状态。

- [ ] **Step 5: 运行存储测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_sqlite.py tests/storage/test_privacy.py -v`

Expected: 全部 PASS。

- [ ] **Step 6: 提交 SQLite 权威层**

```bash
git add lunagentic_research_swarm/storage tests/storage
git commit -m "feat: add transactional SQLite state store"
```

### Task 5: 实现 agent/Procedure 扩展契约与注册表

**Files:**
- Create: `lunagentic_research_swarm/extensions/__init__.py`
- Create: `lunagentic_research_swarm/extensions/contracts.py`
- Create: `lunagentic_research_swarm/extensions/validation.py`
- Create: `lunagentic_research_swarm/extensions/discovery.py`
- Create: `lunagentic_research_swarm/agents/__init__.py`
- Create: `lunagentic_research_swarm/agents/registry.py`
- Create: `lunagentic_research_swarm/procedures/__init__.py`
- Create: `lunagentic_research_swarm/procedures/registry.py`
- Test: `tests/extensions/test_validation.py`
- Test: `tests/extensions/test_discovery.py`
- Test: `tests/extensions/test_registry.py`

**Interfaces:**
- Consumes: `ctx.api.list/get/call`、Task 2 override models。
- Produces: `AgentDefinition`、`ProcedureDefinition`、`ProcedureResult`、catalog snapshot/fingerprint、`ExtensionDiscovery.refresh()`、`refresh_extensions@1` 后续调用入口。

- [ ] **Step 1: 写严格验证与默认值测试**

```python
import pytest
from pydantic import ValidationError

from lunagentic_research_swarm.extensions.contracts import AgentDefinition, ProcedureDefinition


def test_missing_optional_agent_fields_use_documented_defaults() -> None:
    definition = AgentDefinition.model_validate({
        "agent_id": "example.reader",
        "version": "1",
        "display_name": "阅读者",
        "description": "读取材料",
        "character_prompt": "只基于材料报告。",
        "model_selector": "task:utils",
    })
    assert definition.protocol == "json_envelope"
    assert definition.allowed_procedures == ["*"]
    assert definition.can_be_root is False
    assert definition.enabled is True


@pytest.mark.parametrize("agent_id", ["core.summarizer", "summarizer", "bad id", "a" * 129])
def test_agent_id_cannot_impersonate_core_or_be_invalid(agent_id: str) -> None:
    with pytest.raises(ValidationError):
        AgentDefinition.model_validate({
            "agent_id": agent_id,
            "version": "1",
            "display_name": "x",
            "description": "x",
            "character_prompt": "x",
            "model_selector": "task:utils",
        })


def test_explicit_invalid_selector_is_not_replaced() -> None:
    with pytest.raises(ValidationError, match="task:|model:"):
        AgentDefinition.model_validate({
            "agent_id": "example.reader", "version": "1", "display_name": "x",
            "description": "x", "character_prompt": "x", "model_selector": "utils",
        })
```

- [ ] **Step 2: 运行扩展测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/extensions -v`

Expected: FAIL because extension modules do not exist。

- [ ] **Step 3: 实现 public contract models 与 canonical fingerprint**

`AgentDefinition` 精确包含：`agent_id/version/display_name/description/character_prompt/model_selector/protocol/allowed_procedures/can_be_root/auto_compact_tokens/enabled`。限制：ID regex `^[a-z0-9][a-z0-9_.-]{2,127}$`，version 1–32 字符，display 1–80，description 1–2000，prompt 1–20000，selector 必须 `task:` 或 `model:` 非空后缀，protocol 两值，compact 为 `None` 或 `>=1024`。禁止 `core.*`、`summarizer`、`*.summarizer`。`allowed_procedures` 只验证 `*` 或合法 Procedure ID语法，不要求可选 provider此刻存在；round snapshot把 allowlist与实际 live Procedure求交集，未安装的推荐 provider不会让 agent definition失效或被错误 advertised。

`ProcedureDefinition` 精确包含规格字段并把 `arguments_schema/result_schema` 验证为 object schema；timeout `(0, 600]`；ID/版本同类限制。`ProcedureInvocation` 含 request/task/round/branch/turn/agent 与 arguments/scoped metadata；`ProcedureResult` 固定 success/data/error/metadata。`canonical_fingerprint()` 对 `model_dump(mode="json")` 做 keys 排序、UTF-8 JSON 后 SHA-256，任何 catalog 顺序变化先按 ID 排序。

- [ ] **Step 4: 实现 replace-provider registry 与冻结 snapshot**

`AgentRegistry.replace_provider()` 先验证整批 ID 唯一且 definition namespace 不伪造其他 provider，再原子替换该 provider 全集；一项非法则整批拒绝并记录 `ProviderHealth(status="invalid", errors=validation_errors)`，不能保留半批。`snapshot(overrides)` 复制定义并只应用显式 override；override selector 也走同一 selector validator。最终要求配置 root 存在、enabled 且 `can_be_root=True`。

`ProcedureRegistry` 使用同一替换语义。snapshot 只含 enabled 定义，并保存 `provider_plugin_id/api_name/api_version/fingerprint`。round 只持有 snapshot，不读取 live registry；但是调度新边前另外询问 `is_live(agent_id)`，实现“在途完成、移除后新边失效”。

- [ ] **Step 5: 实现 tagged API discovery**

`ExtensionDiscovery.refresh()` 调用 `ctx.api.list()`，只接受 metadata 完整匹配：

```python
def _is_agent_descriptor(info: Mapping[str, Any]) -> bool:
    metadata = info.get("metadata")
    return (
        info.get("name") == "describe_agents"
        and info.get("version") == "1"
        and info.get("public") is True
        and isinstance(metadata, Mapping)
        and metadata.get("lunagentic_extension") == "agents"
        and metadata.get("lunagentic_contract") == "1"
    )
```

descriptor 返回 envelope 固定为 `{"contract_version": "1", "agents": agent_payloads}` 或 `{"contract_version": "1", "procedures": procedure_payloads}`，其中 payloads 必须是 list；provider identity 只取 Host API metadata 中的 plugin ID，不信任 payload 自报身份。缺 envelope、contract_version 不匹配、items 非 list 或存在未知顶层字段都整批拒绝。

用 API 返回的完整 plugin ID 构造 `f"{plugin_id}.describe_agents"` 调用，绝不使用可能冲突的短名；Procedure descriptor 同理。刷新开始时记录本轮可见 provider 集，完成后移除不再可见 provider。某个 provider 调用/验证失败只把该 provider 标 invalid，不阻止其他 provider 更新；错误必须进入 health 与 `extension_fingerprints`，不能静默吞掉。

新增 `request_refresh()` 使用 `asyncio.Event` 合并并发 refresh 请求；周期扫描间隔来自配置。卸载时取消周期 task 并 await 结束。

- [ ] **Step 6: 运行扩展测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/extensions -v`

Expected: 全部 PASS，包括 provider 移除、整批拒绝、fingerprint 顺序稳定与 live/snapshot 区分。

- [ ] **Step 7: 提交扩展契约**

```bash
git add lunagentic_research_swarm/extensions lunagentic_research_swarm/agents lunagentic_research_swarm/procedures tests/extensions
git commit -m "feat: add validated extension registries"
```

### Task 6: 实现 selector、价格快照、token 估算与调用核销

**Files:**
- Create: `lunagentic_research_swarm/llm/__init__.py`
- Create: `lunagentic_research_swarm/llm/pricing.py`
- Create: `lunagentic_research_swarm/llm/tokens.py`
- Create: `lunagentic_research_swarm/llm/physical_pinning.py`
- Create: `lunagentic_research_swarm/llm/gateway.py`
- Test: `tests/llm/test_pricing.py`
- Test: `tests/llm/test_gateway.py`
- Test: `tests/compat/test_physical_pinning.py`

**Interfaces:**
- Consumes: `ctx.llm.generate/generate_with_tools`、Host reload model config dict、isolated Host internals only in `physical_pinning.py`。
- Produces: `ModelSelector.parse()`、`PriceCatalog`、`TokenEstimate`、`GenerationRequest`、`GenerationResult`、`LLMGateway.generate()`、统一 credits charge helper。

- [ ] **Step 1: 写价格优先级与实际模型核销失败测试**

```python
from lunagentic_research_swarm.llm.pricing import PriceCatalog, PriceProfile


def test_plugin_override_ignores_entire_host_profile() -> None:
    catalog = PriceCatalog.from_sources(
        plugin_overrides={"luna": {"price_in": 1.0}},
        host_models={"luna": PriceProfile(9.0, True, 2.0, 12.0)},
        task_models={"utils": ["luna"]},
    )
    resolved = catalog.resolve_model("luna")
    assert resolved.source == "plugin_override"
    assert resolved.profile == PriceProfile(price_in=1.0, cache=False, cache_price_in=0.0, price_out=0.0)


def test_missing_host_price_is_free() -> None:
    catalog = PriceCatalog.from_sources({}, {}, {"utils": ["unknown"]})
    assert catalog.estimate_model_for_selector("task:utils").profile.total_is_free


def test_task_estimate_uses_first_model_but_reconcile_uses_actual() -> None:
    catalog = PriceCatalog.from_sources(
        {},
        {
            "first": PriceProfile(1.0, False, 0.0, 2.0),
            "actual": PriceProfile(3.0, True, 0.5, 4.0),
        },
        {"utils": ["first", "actual"]},
    )
    assert catalog.estimate_model_for_selector("task:utils").model_name == "first"
    usage = catalog.charge_actual(
        actual_model_name="actual", prompt_tokens=1000, completion_tokens=500,
        cache_hit_tokens=600, cache_miss_tokens=400,
    )
    assert usage.credits == pytest.approx(((400 * 3.0 + 600 * 0.5 + 500 * 4.0) / 1_000_000) * 100)
```

- [ ] **Step 2: 运行 LLM 测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm tests/compat/test_physical_pinning.py -v`

Expected: FAIL because LLM modules do not exist。

- [ ] **Step 3: 实现 selector、价格目录与 credits 单位**

`ModelSelector.parse("task:utils")` / `parse("model:gpt")` 返回 frozen object；空后缀、未知 scheme、裸名字抛 `LRSError(code="invalid_selector", message=f"无效模型 selector：{raw_selector}")`。`force_selector` 非空时先解析并统一覆盖所有普通 agent 和核心总结器的生成式 LLM selector；embedding 保持独立 `[embedding].selector`，因为它不是生成式 LLM。

`PriceCatalog.from_host_snapshot()` 只提取 `models[*].name/price_in/cache/cache_price_in/price_out` 与 `model_task_config.*.model_list`，立即丢弃原 dict 引用；禁止日志输出 snapshot。插件 override 的条目存在即创建完整 profile，缺字段按 0/false，完全忽略 Host 同模型。Host 未读到、字段未定价或实际模型找不到时 profile 为 0，source 分别标 `host_unavailable_free`、`host_unpriced_free`、`actual_model_unknown_free`。

价格单位换算固定为：

```python
def price_units_to_credits(price_units: float) -> float:
    return price_units * 100.0

def charge(profile: PriceProfile, usage: TokenUsage) -> float:
    if profile.cache:
        input_units = (
            usage.cache_miss_tokens * profile.price_in
            + usage.cache_hit_tokens * profile.cache_price_in
        ) / 1_000_000
    else:
        input_units = usage.prompt_tokens * profile.price_in / 1_000_000
    output_units = usage.completion_tokens * profile.price_out / 1_000_000
    return price_units_to_credits(input_units + output_units)
```

如果 Host 的 hit+miss 小于 prompt，差额按 miss 计；大于 prompt 则拒绝 usage 并记录 `invalid_usage`。失败且没有 usage 的调用保留 input 预估，状态 `estimated_unreconciled`。

- [ ] **Step 4: 实现初始 Host 快照与模型热更新适配**

在 `gateway.py` 的 `HostModelSnapshotReader` 只做一次隔离内部 import：

```python
def read_initial_host_model_snapshot() -> dict[str, Any] | None:
    try:
        from src.config.config import model_config
    except (ImportError, ModuleNotFoundError):
        return None
    return model_config.model_dump(mode="json")
```

如果 `model_dump` 或结构验证失败，捕获异常并向 health 写入 `host_model_snapshot_unavailable`，价格按 0；不能尝试自行猜 TOML 路径。`on_config_update(scope="model")` 通过 `PriceCatalog.replace_host_snapshot()` 使用公开广播刷新。只保留白名单字段，provider/api_key/headers/query/auth 等字段不落内存、不记录、不持久化。

- [ ] **Step 5: 实现 task 公共调用与物理 pinning**

`LLMGateway.generate()`：task selector 调 `ctx.llm.generate` 或 `generate_with_tools`；model selector 调 `PhysicalPinningAdapter`。`GenerationResult` 只含 response/tool_calls/model_name/token usage/success/error/duration，构造时不接收 reasoning。

`physical_pinning.py` 是唯一允许 import `src.*` 的 package 模块。沿用 maibook 的 `_PinnedOrchestrator` 方案，但传消息列表时使用当前 Host `LLMServiceClient`/`MessageFactory` 可接受的接口；synthetic `TaskConfig` 固定 `model_list=[physical_name]`、fallback `max_tokens=65536`，而调用 `generate_response_async(max_tokens=None)`，因此 Host仍优先使用目标 `ModelInfo.max_tokens` / `extra_params.max_tokens`，两者都没有时才用65536。contract test 要 monkeypatch `src.config.model_configs.TaskConfig` 与 `src.llm_models.utils_model.LLMOrchestrator`，断言 `request_type="plugin.lunagentic_research_swarm"`、`max_tokens=None` 没有显式覆盖。任何签名不兼容返回 `physical_pinning_unsupported`，不能退回 task selector。

`max_tokens=0` 在 gateway 边界转换为 `None`；任何正数原样传递，最大允许 65536。不传 reasoning 参数。

- [ ] **Step 6: 实现确定性 token estimator 与低预算警告计算**

`tokens.py` 对 JSON canonical message UTF-8 byte length 使用 `ceil(bytes/3.5)`，工具 schema 同样计入；该值明确标 `estimated`，实际 usage 优先。预启动 cache 估算规则必须可审计：root/formalizer/correction无已确认共享前缀时全部按 miss；child 只有在其 parent 上一次实际 `model_name` 与本次 estimated model相同且 profile `cache=true` 时，才把逐字节继承的稳定 prefix token估为 hit，新追加 assignment/runtime/procedure内容估为 miss；模型不同或无法证明前缀相同时全部按 miss。实际 Host hit/miss返回后整体核销。

`estimate_root_minimum(selector)` 使用 task 首模型、500000 cache-miss input 与 50000 output；加载时和每次 start 都比较默认/有效预算，低于该成本就写 warning，但不拒绝任务。零价格时不 warning。

- [ ] **Step 7: 运行 LLM 与兼容测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm tests/compat/test_physical_pinning.py -v`

Expected: 全部 PASS；测试还应断言 normalized result 不包含 reasoning，价格 fingerprint 对字段变化敏感。

- [ ] **Step 8: 提交 LLM 与定价适配层**

```bash
git add lunagentic_research_swarm/llm tests/llm tests/compat/test_physical_pinning.py
git commit -m "feat: add model routing and credit pricing adapters"
```

### Task 7: 实现普通智能体协议与核心总结器四角色

**Files:**
- Create: `lunagentic_research_swarm/llm/protocol.py`
- Create: `lunagentic_research_swarm/llm/summarizer.py`
- Create: `lunagentic_research_swarm/prompts/zh-CN/swarm_system.txt`
- Create: `lunagentic_research_swarm/prompts/zh-CN/formalize_task.txt`
- Create: `lunagentic_research_swarm/prompts/zh-CN/finalize_branch.txt`
- Create: `lunagentic_research_swarm/prompts/zh-CN/finalize_task.txt`
- Create: `lunagentic_research_swarm/prompts/zh-CN/compact_branch.txt`
- Test: `tests/llm/test_protocol.py`
- Test: `tests/llm/test_summarizer.py`

**Interfaces:**
- Consumes: `LLMGateway`、agent protocol mode、immutable formalized task。
- Produces: `SwarmTurnEnvelope`、`parse_json_envelope()`、`parse_native_tool_result()`、`build_correction_message()`、四个 summarizer request/result types。

- [ ] **Step 1: 写 JSON 修复、单次纠正与 native 无正文测试**

```python
from lunagentic_research_swarm.llm.protocol import parse_json_envelope, parse_native_tool_result


def test_local_repair_handles_fenced_json_trailing_comma() -> None:
    parsed = parse_json_envelope(
        '```json\n{"report":"ok","procedures":[],"delegations":[],}\n```'
    )
    assert parsed.report == "ok"


def test_native_mode_accepts_tool_call_without_assistant_text() -> None:
    result = parse_native_tool_result(
        response="",
        tool_calls=[{
            "id": "call_1",
            "function": {
                "name": "submit_swarm_turn",
                "arguments": {"report": "", "procedures": [], "delegations": []},
            },
        }],
    )
    assert result.report == ""


def test_native_mode_rejects_any_other_tool() -> None:
    with pytest.raises(ProtocolError, match="submit_swarm_turn"):
        parse_native_tool_result("", [{"function": {"name": "compact", "arguments": {}}}])
```

- [ ] **Step 2: 运行协议/总结器测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_protocol.py tests/llm/test_summarizer.py -v`

Expected: FAIL because modules do not exist。

- [ ] **Step 3: 实现唯一 turn envelope 与保守 JSON repair**

Envelope 固定为：

```python
class ProcedureRequest(BaseModel):
    procedure_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class DelegationRequest(BaseModel):
    agent_id: str
    task: str = Field(min_length=1, max_length=12000)
    credits: float = Field(ge=0.0)

class SwarmTurnEnvelope(BaseModel):
    report: str = Field(default="", max_length=30000)
    procedures: list[ProcedureRequest] = Field(default_factory=list)
    delegations: list[DelegationRequest] = Field(default_factory=list)
```

本地 repair 只允许：去 BOM/空白、剥一层 Markdown JSON fence、提取首个括号平衡的 JSON object、移除 `}`/`]` 前 trailing comma、若完整内容被 JSON string 双重编码则 decode 一次。禁止补造缺失字段含义、改 agent/procedure ID、把自然语言猜成 JSON。repair 后仍不合法，由 caller 最多追加一次 correction user message 并递归调用同一 agent/selector；第二次非法终结分支。第一次调用结算后余额已经负数时不发 correction。

native mode tools 数组只包含一个 synthetic `submit_swarm_turn`，parameters schema 等于 envelope；response 可以为空，tool call 必须恰好一个且名字正确。若同时有正文，正文只作为可选 report 补充，不覆盖 arguments 中非空 report。

- [ ] **Step 4: 写固定总结器 prompt 与四角色服务**

`SummarizerService` 永不把 agent catalog、Procedure tools 或 native tools 传给 LLM。`formalize_task` 每次构造新的 system + raw context/user messages，不复用普通 swarm stable prefix，也不为了 cache 重排初始输入；这是一次性 ingest 调用。四个 prompt 明确：

- `formalize_task`：整理目标/约束/事实/未知/成功标准，不研究、不调用工具，返回独立任务描述。
- `finalize_branch`：基于 branch history 给分支结论、证据、不确定性、建议；缺内容返回 failure，不发第二次 LRS 重试。
- `finalize_task`：输入 immutable task + coverage summaries，标 intermediate/final、统计与仍运行 branch 数；不得把 checkpoint 当 terminal。
- `compact_branch`：只压缩可变历史，不复述/改写正式任务描述；返回 compacted history 文本。

`SummaryResult` 含 `success/text/model_name/usage/error`，不含 reasoning。`formalize_task` 输出用 `FormalizedTask.create()` 保存 byte-stable text；其他角色都接受 formalized task 为独立字段并在 prompt 中标“原文保留，不得改写”。max_tokens 0 转 None；正数最高 65536。

- [ ] **Step 5: 测试 compact 后正式任务原文保持**

新增 fake gateway 测试：让 compactor 返回完全不同的任务文本，随后调用 `BranchRuntime.build_prompt_messages()`，断言 User 1 仍与 `FormalizedTask.text.encode()` 逐字节相同；summarizer request 的 tools 为 `None`，native tool API 从未调用。

- [ ] **Step 6: 运行协议与总结器测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_protocol.py tests/llm/test_summarizer.py tests/test_models.py -v`

Expected: 全部 PASS。

- [ ] **Step 7: 提交协议与总结器**

```bash
git add lunagentic_research_swarm/llm/protocol.py lunagentic_research_swarm/llm/summarizer.py lunagentic_research_swarm/prompts tests/llm/test_protocol.py tests/llm/test_summarizer.py
git commit -m "feat: add swarm protocol and core summarizer"
```

### Task 8: 接线基础服务、模型热更新与扩展刷新 API

**Files:**
- Create: `lunagentic_research_swarm/services.py`
- Modify: `plugin.py`
- Modify: `tests/test_plugin_contract.py`
- Create: `tests/test_lifecycle.py`

**Interfaces:**
- Consumes: Tasks 2–7 的 config/store/registry/discovery/gateway/summarizer。
- Produces: `LRSServiceContainer.start/close/refresh_extensions/update_model_snapshot/health`，公开 `refresh_extensions@1` API；为下一计划提供 `self._services`。

- [ ] **Step 1: 写生命周期、敏感字段清除与 refresh API 失败测试**

```python
@pytest.mark.asyncio
async def test_model_reload_keeps_only_safe_pricing_fields(fake_plugin) -> None:
    snapshot = {
        "api_providers": [{"name": "secret", "api_key": "never-store"}],
        "models": [{"name": "m", "price_in": 1, "price_out": 2, "api_provider": "secret"}],
        "model_task_config": {"utils": {"model_list": ["m"]}},
    }
    await fake_plugin.on_config_update("model", snapshot, "v2")
    debug = fake_plugin._services.price_catalog.debug_snapshot()
    assert debug == {
        "models": {"m": {"price_in": 1.0, "cache": False, "cache_price_in": 0.0, "price_out": 2.0}},
        "tasks": {"utils": ["m"]},
    }
    assert "never-store" not in repr(fake_plugin._services)


def test_refresh_api_component_is_public(plugin_module) -> None:
    component = next(
        item for item in plugin_module.create_plugin().get_components()
        if item["name"] == "refresh_extensions" and item["type"] == "API"
    )
    assert component["metadata"]["public"] is True
    assert component["metadata"]["version"] == "1"
```

- [ ] **Step 2: 运行生命周期测试并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_lifecycle.py tests/test_plugin_contract.py -v`

Expected: FAIL because service container/API is missing。

- [ ] **Step 3: 实现 service container 与生命周期顺序**

`LRSServiceContainer.start()` 严格按顺序：创建目录 → open SQLite/migrate → 把遗留 active round 标 INTERRUPTED → 读取 initial Host model snapshot/构造价格目录 → 注册内置 provider hook（本阶段为空）→ refresh 外部 extensions → 启动周期 refresh。任何 SQLite 关键失败向上抛，Host 看见 plugin load failure；模型 snapshot/某个 extension 失败写 health 但插件可加载。

`close()`：设置 closing → 停止 refresh loop → 等待已启动基础 background task → close SQLite。多次调用幂等。

入口加入：

```python
from maibot_sdk import API

@API(
    "refresh_extensions",
    description="请求麦麦深度调查组重新扫描智能体与 Procedure provider",
    version="1",
    public=True,
)
async def refresh_extensions(self) -> dict[str, Any]:
    return await self._services.refresh_extensions(reason="provider_request")
```

`on_config_update("self")` 只更新 live safety limits/refresh interval/health，并记录“catalog/selectors/prompt 从下一 round 生效”；`on_config_update("model")` 调安全 extractor。不要持有原 `config_data` 引用。

- [ ] **Step 4: 加入基础 health 与启动预算 warning**

`health()` 返回 SQLite、initial price snapshot、physical pinning、extension providers、root/summarizer selector 的显式状态。没有任何 root agent 时是 `degraded/root_agent_unavailable`，这是本阶段预期；不把 quick thinker 硬编码进 core。加载时用 Task 6 的 500k/50k 估算检查 `default_effort_credits`，低则每次加载 warning。

- [ ] **Step 5: 运行本阶段全部测试**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest -v`

Expected: 全部 PASS；无网络、无 Host 也可用 fakes 完成。

- [ ] **Step 6: 运行窄范围 lint**

Run: `python -m ruff check plugin.py lunagentic_research_swarm tests`

Expected: exit 0。不运行 workspace-wide format/build。

- [ ] **Step 7: 提交基础接线**

```bash
git add lunagentic_research_swarm/services.py plugin.py tests/test_lifecycle.py tests/test_plugin_contract.py
git commit -m "feat: wire LRS foundation services"
```

## 本计划完成门槛

- `plugin.py` 可由离线 import 与当前 Host loader package 方式装载。
- config schema/default/migration 一致，且没有 reasoning 字段。
- SQLite 事务、默认隐私和 crash 标记测试通过。
- 内置/外部定义共用同一 validator/registry，provider 移除可见。
- task/model selector、价格优先级、实际 `model_name` 核销、免费缺省与 pinning health 测试通过。
- JSON/native 协议及四角色总结器测试通过；native 无正文 tool call 可运行。
- `git status --short` 只显示本计划尚未提交的预期文件，完成最后提交后为空。
