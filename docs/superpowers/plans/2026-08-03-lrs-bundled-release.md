# 麦麦深度调查组：首发扩展与发布 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LRS 自己的 extension contracts 装入九个默认 agent 与全部非核心 Procedure，完成 LanceDB 历史案例、feedback/提醒、调试存储、统计命令、中文文档和 0.1.0 发布验收。

**Architecture:** 内置 agent/Procedure 是名为 `builtin` 的普通 provider，先构造 public contract payload，再走与第三方相同的 validation/registry；scheduler 不出现角色特例。SQLite 继续是权威源，LanceDB 仅保存可重建向量；反馈以 immutable event 和显式 lesson 参与历史案例排序，不自动改 prompt 或训练模型。

**Tech Stack:** 前两阶段依赖、LanceDB 0.34.x、DDGS 9.14.x、MaiBot chat/message/person/knowledge/embed APIs、httpx、stdlib `ast/statistics/urllib`、pytest。

## Global Constraints

- 必须先完成基础计划与运行时计划；继续使用已冻结的 catalog、Procedure、storage、controller 与 report 接口。
- 九个默认 agent、所有 memory/web/past-case/analysis/provenance Procedure 都必须通过 `AgentDefinition`/`ProcedureDefinition` validator 和 registry 加载；禁止 scheduler 按具体角色 ID 写分支逻辑。
- 只有 scheduler/reducer/summarizer/credits ledger 与 `core.compact/checkpoint/terminate` 是 core；总结器绝不注册成 agent/Procedure，不可 root、不可委派。
- `fetch-url` 只作为推荐 provider；缺失时插件正常加载，目录不出现 fetch Procedure，health 明确提示。
- 文件仓库仍是独立可选插件方向；LRS 不提供 shell、`sed`、`awk` 或任意路径文件访问。
- 默认只索引 formalized task、branch/checkpoint terminal summary、intermediate/final report、feedback lesson；raw chat/transcript/procedure/compact/reasoning 永不进入 LanceDB。
- Embedding selector/实际模型 fingerprint/维度/schema 任一变化必须显式 rebuild；不截断、不 padding、不把 mismatch 当空结果。
- Feedback 不静默改 prompt、agent selector 或路由；只用于透明检索、排序、统计和人工可见 lesson。
- COMPLETED、COMPLETED_WITH_ERRORS、手动 STOPPED 在 600 秒无 feedback 时提醒一次；continue/new round 和 feedback 取消 pending reminder；EXPIRED/INTERRUPTED/FAILED 不提醒。
- 0.1.0 发布前八个 Planner tools、全部 `/swarm` 命令、两个 raw storage 开关四组合与依赖存在/缺失场景必须通过。
- 本计划文档、README、配置说明使用简体中文；模型名示例不构成依赖或默认固定路由。

---

### Task 1: 用 registry 装入九个默认智能体

**Files:**
- Create: `lunagentic_research_swarm/agents/bundled/__init__.py`
- Create: `lunagentic_research_swarm/agents/bundled/catalog.py`
- Create: `lunagentic_research_swarm/agents/bundled/prompts.py`
- Modify: `lunagentic_research_swarm/services.py`
- Test: `tests/agents/test_bundled_catalog.py`

**Interfaces:**
- Consumes: `AgentRegistry.replace_provider()`、Task 2 config overrides。
- Produces: `bundled_agent_definitions() -> list[AgentDefinition]` 与可用默认 root `builtin.quick_thinker`。

- [ ] **Step 1: 写 catalog 完整性与同一 validator 失败测试**

```python
from lunagentic_research_swarm.agents.bundled.catalog import bundled_agent_definitions
from lunagentic_research_swarm.extensions.validation import validate_agent_batch


def test_bundled_catalog_contains_nine_valid_agents() -> None:
    definitions = validate_agent_batch("builtin", [item.model_dump() for item in bundled_agent_definitions()])
    assert {item.agent_id for item in definitions} == {
        "builtin.quick_thinker", "builtin.deep_thinker", "builtin.debater",
        "builtin.researcher", "builtin.memory_researcher", "builtin.knowledge_reporter",
        "builtin.past_case_researcher", "builtin.evidence_verifier", "builtin.quantitative_analyst",
    }
    assert all(item.enabled for item in definitions)
    assert next(item for item in definitions if item.agent_id == "builtin.quick_thinker").can_be_root


def test_default_selectors_match_design() -> None:
    selectors = {item.agent_id: item.model_selector for item in bundled_agent_definitions()}
    assert selectors["builtin.quick_thinker"] == "task:utils"
    assert selectors["builtin.deep_thinker"] == "task:planner"
    assert selectors["builtin.debater"] == "task:replyer"
    assert selectors["builtin.researcher"] == "task:utils"
    assert selectors["builtin.memory_researcher"] == "task:utils"
    assert selectors["builtin.knowledge_reporter"] == "task:replyer"
    assert selectors["builtin.past_case_researcher"] == "task:utils"
    assert selectors["builtin.evidence_verifier"] == "task:planner"
    assert selectors["builtin.quantitative_analyst"] == "task:planner"
```

- [ ] **Step 2: 运行 agent tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/agents/test_bundled_catalog.py -v`

Expected: FAIL because bundled catalog is missing。

- [ ] **Step 3: 写九个明确 character prompts**

`prompts.py` 用 immutable dict 保存以下完整角色意图；通用 envelope/credits 规则仍只由 common system prompt 提供：

```python
BUNDLED_CHARACTER_PROMPTS = {
    "builtin.quick_thinker": (
        "你偏向快速建立问题地图：先识别最关键未知点、可并行路线和收敛条件，再把工作分给最合适的专员。"
        "优先用便宜、快速、信息增益高的步骤验证方向；发现证据已足够时及时结束，不为了消耗预算而扩张分支。"
    ),
    "builtin.deep_thinker": (
        "你负责复杂推理和长期约束整合。耐心拆解因果链、隐含假设、相互制约的目标与边界条件，"
        "在给出结论前检查推导是否跳步；需要事实时委派研究，不把未经验证的记忆伪装成证据。"
    ),
    "builtin.debater": (
        "你负责提供第二意见。先以最强形式复述当前主张，再主动寻找反例、替代解释、失败条件、"
        "利益冲突和被忽略的风险；质疑应指向可验证问题，而不是为了唱反调而唱反调。"
    ),
    "builtin.researcher": (
        "你以外部证据为中心。把研究问题转成清晰搜索词，选择已配置的搜索引擎，记录来源、时间与查询，"
        "区分搜索摘要和网页原文；需要全文时调用可用的 fetch provider，否则明确证据粒度有限。"
    ),
    "builtin.memory_researcher": (
        "你负责把问题转成针对聊天、消息、人物和知识库的查询。优先设计高召回、可复查的查询词和范围，"
        "如实转交结果与缺口，不对零散记忆做超出材料的解释。"
    ),
    "builtin.knowledge_reporter": (
        "你只报告模型已经知道的相关知识、常见框架和明显不确定点。避免长时间推演，避免把记忆当最新事实；"
        "内容可能过时或需要证据时明确标注，并把验证工作交给研究或核验专员。"
    ),
    "builtin.past_case_researcher": (
        "你负责检索相似历史调查、过去决定、反馈和后来结果。区分 accepted、mixed、rejected 与无反馈案例，"
        "提炼可迁移模式和反模式，不把相似度本身当成正确性。"
    ),
    "builtin.evidence_verifier": (
        "你负责证据核验。检查来源是否真正支持主张、是否独立、是否过时，比较来源冲突，"
        "明确区分原始事实、二手摘要、计算结果和推断；无法验证时给出缺失的具体证据。"
    ),
    "builtin.quantitative_analyst": (
        "你负责数值与量级检查。明确单位、基准、区间和假设，使用受限计算/统计/换算 Procedure 复核，"
        "检查百分比、样本量、敏感性和成本收益；数据不足时报告可计算范围而不伪造精度。"
    ),
}
```

- [ ] **Step 4: 构造 definitions 与允许 Procedure 范围**

quick/deep `can_be_root=True`，其他默认 false；protocol 全部 `json_envelope` 并可被 per-agent override 改为 native。researcher allow `builtin.web_search/builtin.normalize_urls/builtin.organize_provenance/fetch_url.fetch`；memory researcher allow memory family；past-case only past/provenance；quantitative only calculator/statistics/unit/provenance；其他为 `[*]`。core procedures 对所有 agent 固有可用，不通过 allowlist移除。

服务启动时执行 `agent_registry.replace_provider("builtin", [definition.model_dump(mode="json") for definition in bundled_agent_definitions()])`，和外部 provider 完全相同；应用 config override 后 root health 由 degraded 变 healthy。禁用 quick thinker且没有有效替代 root 时 start 明确失败。

- [ ] **Step 5: 运行 agent tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/agents/test_bundled_catalog.py tests/test_lifecycle.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/agents/bundled lunagentic_research_swarm/services.py tests/agents/test_bundled_catalog.py
git commit -m "feat: load bundled research agents through registry"
```

### Task 2: 实现聊天、消息、人物与知识库 Procedures

**Files:**
- Create: `lunagentic_research_swarm/procedures/bundled/__init__.py`
- Create: `lunagentic_research_swarm/procedures/bundled/memory.py`
- Create: `lunagentic_research_swarm/procedures/bundled/provider.py`
- Modify: `lunagentic_research_swarm/services.py`
- Test: `tests/procedures/test_memory.py`
- Test: `tests/procedures/test_bundled_provider.py`

**Interfaces:**
- Consumes: SDK `ctx.chat/message/person/knowledge`。
- Produces: 本地 `builtin` Procedure provider definitions/handlers，通过 `ProcedureRegistry` 和 `ProcedureExecutor` 调用。

- [ ] **Step 1: 写安全参数与结果归一化失败测试**

```python
@pytest.mark.asyncio
async def test_recent_messages_never_requests_binary_data(memory_provider) -> None:
    result = await memory_provider.invoke("builtin.message_recent", {"stream_id": "s", "limit": 8})
    assert result.success
    assert memory_provider.ctx.message.recent_calls == [("s", 8)]
    assert "image_base64" not in repr(result.data)


@pytest.mark.asyncio
async def test_memory_limits_are_explicit(memory_provider) -> None:
    result = await memory_provider.invoke("builtin.message_recent", {"stream_id": "s", "limit": 500})
    assert not result.success
    assert result.error.code == "invalid_arguments"
```

- [ ] **Step 2: 运行 memory tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_memory.py tests/procedures/test_bundled_provider.py -v`

Expected: FAIL because bundled provider is missing。

- [ ] **Step 3: 定义并实现六个 memory Procedures**

通过同一 `ProcedureDefinition` validator 注册：

- `builtin.chat_streams(kind=all|group|private, platform="qq")` → 对应 `ctx.chat.get_*_streams`。
- `builtin.message_recent(stream_id, limit 1..50)` → `get_recent` 后 `build_readable`，同时保留最小 message IDs/timestamps。
- `builtin.message_by_id(message_id, stream_id="")` → `include_binary_data=False`。
- `builtin.message_time_range(stream_id, start_time, end_time, limit 1..100)` → `get_by_time_in_chat`，结果再裁到 limit并 build_readable。
- `builtin.person_lookup(mode=id|name|field, platform/user_id/person_name/person_id/field_name)` → 只调用对应公开 capability。
- `builtin.knowledge_search(query 1..2000, limit 1..20)`。

所有定义 idempotent=true，timeout 30，external_cost_kind=none。返回 data 明确 `query/stream_id/items/readable/truncated`，不在日志打印 message body。SDK 调用失败返回 `host_capability_failed`，不能假装空数组。

- [ ] **Step 4: 通过 bundled provider 注册/调用**

`BundledProcedureProvider.describe()` 返回 model_dump payload，`invoke()` 在 handler map 里按 ID dispatch；服务用 `procedure_registry.replace_provider("builtin", provider.describe())`，executor 仍通过统一 provider invoker interface，不为 builtin 写 scheduler shortcut。

- [ ] **Step 5: 运行 memory tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_memory.py tests/procedures/test_bundled_provider.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/procedures/bundled lunagentic_research_swarm/services.py tests/procedures/test_memory.py tests/procedures/test_bundled_provider.py
git commit -m "feat: add memory research procedures"
```

### Task 3: 实现四引擎统一 Web 搜索

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `_manifest.json`
- Create: `lunagentic_research_swarm/procedures/bundled/web_search.py`
- Modify: `lunagentic_research_swarm/procedures/bundled/provider.py`
- Test: `tests/procedures/test_web_search.py`
- Test: `tests/test_dependencies.py`

**Interfaces:**
- Consumes: config `enabled_engines/searxng_url/tavily_api_key/you_base_url/you_api_key`、httpx、DDGS。
- Produces: `builtin.web_search` definition 与 `WebSearchService.search(engine: str, query: str, max_results: int, language: str, recency: str | None) -> ProcedureResult`。

**Primary references:** [SearXNG Search API](https://docs.searxng.org/dev/search_api.html)、[Tavily API authentication/search](https://docs.tavily.com/documentation/api-reference/introduction)、[You Search API guide](https://documentation.you.com/api-modes/search-api)、[DDGS 9.x package/API](https://pypi.org/project/ddgs/)；实现时以这些 contract tests 锁定所用字段，不从博客复制 endpoint。

- [ ] **Step 1: 写 engine availability 与统一 provenance 失败测试**

```python
def test_only_correctly_configured_engines_are_advertised(config) -> None:
    config.web_search.enabled_engines = ["duckduckgo", "searxng", "tavily", "you"]
    config.web_search.searxng_url = ""
    config.web_search.tavily_api_key = ""
    config.web_search.you_base_url = "https://example.invalid/search"
    config.web_search.you_api_key = "key"
    service = WebSearchService(config.web_search, fake_http())
    assert service.available_engines == ("duckduckgo", "you")


@pytest.mark.parametrize("engine", ["duckduckgo", "searxng", "tavily", "you"])
@pytest.mark.asyncio
async def test_all_engines_normalize_results(engine, web_harness) -> None:
    web_harness.configure(engine)
    result = await web_harness.search(engine, "query")
    assert result.success
    assert result.data["engine"] == engine
    assert result.data["query"] == "query"
    assert set(result.data["results"][0]) == {"url", "title", "snippet", "published_at", "source"}
```

- [ ] **Step 2: 运行 web tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_web_search.py tests/test_dependencies.py -v`

Expected: FAIL because web service/dependency is missing。

- [ ] **Step 3: 加入 DDGS 依赖并保持三处同步**

在 `pyproject.toml` dependencies、`requirements.txt`、manifest dependencies 同时加入 `ddgs>=9.14.4,<10.0.0`。测试解析三份并断言 name/spec 同步。DDGS 调用固定 `DDGS(timeout=int(config.timeout_seconds)).text(query, backend="duckduckgo", max_results=max_results)`，通过 `asyncio.to_thread` 执行；不启用 DHT/API server，不在插件中启动子进程。

- [ ] **Step 4: 实现四个 adapters**

- DuckDuckGo：把 `href/title/body` 归一化。
- SearXNG：GET `<base>/search`，params `q/format=json/categories=general/language`；解析 `results[*].url/title/content/publishedDate/engine`。
- Tavily：POST `https://api.tavily.com/search`，`Authorization: Bearer <key>`，JSON `query/max_results/search_depth="basic"/include_answer=false/include_raw_content=false`；解析 results。
- You：只有 base URL 与 key 都存在才可用；GET configured base URL，header `X-API-Key`，params `query/num_web_results`；parser 接受官方 response 中 `hits` 或 `web.results` 两种明确 schema，其他结构返回 `provider_contract_invalid`。

以上 endpoint/鉴权以实现时记录的官方文档为 contract；SearXNG JSON 格式、Tavily Bearer、You X-API-Key 都写入 adapter contract tests。API key 使用 `SecretStr`/repr false，异常和 health 不含 key。

- [ ] **Step 5: 实现统一 Procedure definition 与错误**

arguments：`engine` 必须是当前 snapshot advertised enum、query 1..2000、max_results 1..20、language optional、recency optional。未配置 engine 返回 `search_engine_unavailable`，不换其他引擎。result 每项保留 engine/query/url/title/snippet/published/source；HTTP error含 status class但不含 response secrets。此 Procedure idempotent=true，external_cost_kind=`provider_metered`（DDG/SearXNG可报告 null external cost）。

- [ ] **Step 6: 运行 web tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_web_search.py tests/test_dependencies.py -v`

Expected: 全部 PASS。

```bash
git add pyproject.toml requirements.txt _manifest.json lunagentic_research_swarm/procedures/bundled/web_search.py lunagentic_research_swarm/procedures/bundled/provider.py tests/procedures/test_web_search.py tests/test_dependencies.py
git commit -m "feat: add configurable web search procedures"
```

### Task 4: 实现受限计算、统计、单位和来源处理

**Files:**
- Create: `lunagentic_research_swarm/procedures/bundled/analysis.py`
- Create: `lunagentic_research_swarm/procedures/bundled/provenance.py`
- Modify: `lunagentic_research_swarm/procedures/bundled/provider.py`
- Test: `tests/procedures/test_analysis.py`
- Test: `tests/procedures/test_provenance.py`

**Interfaces:**
- Consumes: stdlib `ast`、`math`、`statistics`、`urllib.parse`。
- Produces: `builtin.calculate/statistics/convert_units/normalize_urls/organize_provenance`。

- [ ] **Step 1: 写拒绝 arbitrary code 与 URL 保真失败测试**

```python
@pytest.mark.parametrize("expression", ["__import__('os')", "(1).__class__", "[x for x in range(3)]", "2 ** 100000"])
def test_calculator_rejects_unsafe_expression(expression) -> None:
    result = calculate(expression)
    assert not result.success
    assert result.error.code == "unsafe_expression"


def test_url_normalizer_does_not_reorder_semantic_query() -> None:
    value = normalize_url("HTTPS://Example.COM:443/a?b=2&b=1#fragment")
    assert value == "https://example.com/a?b=2&b=1"
```

- [ ] **Step 2: 运行 analysis tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_analysis.py tests/procedures/test_provenance.py -v`

Expected: FAIL because analysis modules are missing。

- [ ] **Step 3: 实现 AST calculator 与统计**

calculator 只允许 numeric constants、`+ - * / // % **`、unary `+/-`、括号（AST 自然表达），最多 128 nodes，abs(number)<=1e100，exponent abs<=100，division by zero明确错误。statistics operations 固定 `mean/median/stdev/pstdev/min/max/quantiles`，输入最多 10000 finite numbers；quantiles `n` 2..100。

unit conversion 用显式 factor/offset tables 支持 length、mass、time、temperature、data size；温度按 C/K/F 专用函数，未知/跨维度拒绝。不得引入 `eval`、第三方 unit expression parser。

- [ ] **Step 4: 实现 URL/provenance**

URL normalize 只做 scheme/host lowercase、IDNA validation、default port移除、path dot-segment规范化、fragment移除；保留 query原顺序和值，不静默删除 tracking 参数。dedupe 以 normalized URL为 key，保留最先出现 provenance并合并后续 source IDs。

`organize_provenance` 输入 claims + source records，输出 claim→source IDs、unbacked claims、duplicate URLs、source types/timestamps；不判断来源真假，不把 snippet 改写成事实。

- [ ] **Step 5: 注册 definitions、运行测试并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_analysis.py tests/procedures/test_provenance.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/procedures/bundled/analysis.py lunagentic_research_swarm/procedures/bundled/provenance.py lunagentic_research_swarm/procedures/bundled/provider.py tests/procedures/test_analysis.py tests/procedures/test_provenance.py
git commit -m "feat: add safe analysis and provenance procedures"
```

### Task 5: 实现可重建 LanceDB generation 索引

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `_manifest.json`
- Modify: `lunagentic_research_swarm/storage/migrations.py`
- Create: `lunagentic_research_swarm/storage/vectors.py`
- Modify: `lunagentic_research_swarm/services.py`
- Test: `tests/storage/test_vectors.py`
- Test: `tests/storage/test_vector_rebuild.py`

**Interfaces:**
- Consumes: `ctx.llm.embed`、SQLite summary layer/vector_jobs、`ctx.paths.data_dir / vectors/lancedb`。
- Produces: `VectorIndex.start/close/enqueue/rebuild/status/search`、generation/fingerprint/dimension metadata。

**Primary reference:** [LanceDB Python API](https://lancedb.github.io/lancedb/python/python/)；本计划锁定 [PyPI `lancedb` 0.34.x](https://pypi.org/project/lancedb/) 的 minor API。

- [ ] **Step 1: 写 mismatch、batch dimension 与 atomic switch 失败测试**

```python
@pytest.mark.asyncio
async def test_dimension_change_starts_new_generation_without_padding(vector_harness) -> None:
    await vector_harness.build_with_vectors([[1.0, 2.0, 3.0]])
    old = await vector_harness.status()
    vector_harness.embedder.return_vectors([[1.0, 2.0]])
    result = await vector_harness.index_new_source("summary-2")
    assert not result.success
    assert result.error.code == "embedding_generation_mismatch"
    assert (await vector_harness.status()).rebuilding
    assert old.dimension == 3


@pytest.mark.asyncio
async def test_batch_dimension_inconsistency_aborts_generation(vector_harness) -> None:
    vector_harness.embedder.return_vectors([[1.0, 2.0], [1.0, 2.0, 3.0]])
    with pytest.raises(EmbeddingGenerationMismatch):
        await vector_harness.rebuild(force=True)
    assert not (await vector_harness.status()).candidate_active
```

- [ ] **Step 2: 运行 vector tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_vectors.py tests/storage/test_vector_rebuild.py -v`

Expected: FAIL because LanceDB/vector module is missing。

- [ ] **Step 3: 加入 LanceDB 依赖与 migration 002**

三处同步加入 `lancedb>=0.34.0,<0.35.0`；该 minor 线提供 sync/async connect、create/open table 和 vector search。本计划采用 sync API包装 `asyncio.to_thread`，把 DB调用与 event loop隔离。

Migration 002 创建：

```sql
CREATE TABLE vector_generations (
  generation INTEGER PRIMARY KEY,
  selector TEXT NOT NULL,
  actual_model_name TEXT,
  model_fingerprint TEXT NOT NULL,
  dimension INTEGER,
  table_name TEXT NOT NULL UNIQUE,
  schema_version INTEGER NOT NULL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  activated_at REAL,
  retired_at REAL
);
CREATE TABLE vector_documents (
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  generation INTEGER NOT NULL REFERENCES vector_generations(generation),
  actual_model_name TEXT,
  model_fingerprint TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  indexed_at REAL NOT NULL,
  PRIMARY KEY(source_kind, source_id, generation)
);
CREATE UNIQUE INDEX one_active_vector_generation
ON vector_generations(status) WHERE status = 'active';
```

- [ ] **Step 4: 实现 embedding validation 与 generation fingerprint**

task selector通过 `ctx.llm.embed(texts=batch_texts, task_name=selector.task_name, max_concurrent=config.embedding.max_concurrent)`；model selector首发 embedding若公共 API仍不能物理 pin，明确 `physical_embedding_selector_unsupported`，不走文本 LLM pinning。每个 result必须含 nonempty finite vector与 model_name；batch actual model与 len一致。fingerprint=SHA256(selector + actual model + dimension + schema version)。

每个 source row存 kind/id/text metadata/feedback disposition 到 LanceDB；text只来自白名单。Lance table命名 `lrs_vectors_g{generation}`，首次 batch list-of-dict让 Lance推导 fixed-size vector schema，随后读取 schema dimension验证。

- [ ] **Step 5: 实现 mismatch rebuild 与 atomic activation**

selector/fingerprint/dimension/table schema mismatch：把 job标 `embedding_generation_mismatch`，创建 candidate generation；`past_cases` 此时返回 `vector_index_rebuilding`。从 SQLite 全量读取白名单 source并分批 embedding，每批严格验证；全部 add并行数/row count/schema后，在一个 SQLite transaction旧 active→retired、candidate→active。旧 table延迟到 configurable 24h 且至少保留一个 retired generation后清理。失败保留 old active + failed candidate/job，不删除权威数据。

manual `rebuild(force=False)` 无 mismatch时返回 already_current；force创建新 generation。空库时记录 idle/uninitialized，第一条 formalized task触发 generation。

- [ ] **Step 6: 运行 vector tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_vectors.py tests/storage/test_vector_rebuild.py tests/test_dependencies.py -v`

Expected: 全部 PASS，包括 selector/model/dimension/schema四 mismatch、failed candidate保留 old active、atomic switch、rebuild期间显式不可用。

```bash
git add pyproject.toml requirements.txt _manifest.json lunagentic_research_swarm/storage/migrations.py lunagentic_research_swarm/storage/vectors.py lunagentic_research_swarm/services.py tests/storage/test_vectors.py tests/storage/test_vector_rebuild.py tests/test_dependencies.py
git commit -m "feat: add rebuildable LanceDB case index"
```

### Task 6: 实现历史案例检索 Procedure

**Files:**
- Create: `lunagentic_research_swarm/procedures/bundled/past_cases.py`
- Modify: `lunagentic_research_swarm/procedures/bundled/provider.py`
- Test: `tests/procedures/test_past_cases.py`

**Interfaces:**
- Consumes: active `VectorIndex`、SQLite tasks/summaries/reports/feedback。
- Produces: `builtin.past_cases`，显式区分 accepted/mixed/rejected/unreviewed/outcome correction。

- [ ] **Step 1: 写 feedback-aware result 失败测试**

```python
@pytest.mark.asyncio
async def test_past_cases_labels_feedback_truth_status(past_case_harness) -> None:
    past_case_harness.seed("accepted", score=0.7)
    past_case_harness.seed("rejected", score=0.9)
    past_case_harness.seed(None, score=0.8)
    result = await past_case_harness.search("formalized task")
    assert [item["validation_status"] for item in result.data["cases"]] == [
        "accepted", "unreviewed", "rejected"
    ]
    assert result.data["cases"][-1]["use_as"] == "anti_pattern"
```

- [ ] **Step 2: 运行 past-case test 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_past_cases.py -v`

Expected: FAIL because procedure is missing。

- [ ] **Step 3: 实现 query、filter 与 transparent rerank**

query先 embedding，搜索 top `min(50, limit*5)`，排除当前 task/可选时间范围；把同一 task多个 source聚合。rerank公式公开返回：`similarity + accepted 0.15 + mixed 0.05 + outcome_confirmed 0.10 - rejected 0.20`，只改变排序不隐藏 rejected。每项返回 source IDs、formalized摘要、相关 summaries/report、feedback disposition、correction/outcome、similarity/rerank components、agent/model/procedure fingerprints（有则）。无 feedback标 unreviewed，不能视为成功。

vector rebuilding/unavailable/failed分别结构化返回，不伪装空 cases；没有匹配时 success=true + 空 cases。

- [ ] **Step 4: 运行 past-case test 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/procedures/test_past_cases.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/procedures/bundled/past_cases.py lunagentic_research_swarm/procedures/bundled/provider.py tests/procedures/test_past_cases.py
git commit -m "feat: add feedback-aware past case retrieval"
```

### Task 7: 实现 feedback、lesson 与 600 秒提醒

**Files:**
- Create: `lunagentic_research_swarm/feedback.py`
- Modify: `lunagentic_research_swarm/tools.py`
- Modify: `plugin.py`
- Modify: `lunagentic_research_swarm/runtime/controller.py`
- Modify: `lunagentic_research_swarm/services.py`
- Test: `tests/test_feedback.py`
- Test: `tests/integration/test_feedback_reminders.py`

**Interfaces:**
- Consumes: SQLite feedback/reminder/outbox、VectorIndex jobs、ResearchManager。
- Produces: `FeedbackService.submit/schedule/cancel_due_to_continue/process_due` 与第八个 Planner tool `submit_research_feedback`。

- [ ] **Step 1: 写 immutable event、状态与 reminder 失败测试**

```python
@pytest.mark.asyncio
async def test_feedback_is_immutable_and_supersession_is_explicit(feedback_harness) -> None:
    first = await feedback_harness.submit(task_id="lrs_1", disposition="mixed", corrections=["A应为B"])
    second = await feedback_harness.submit(
        task_id="lrs_1", disposition="superseded", supersedes_feedback_id=first.feedback_id,
        corrections=["最终应为C"],
    )
    rows = await feedback_harness.events()
    assert len(rows) == 2
    assert rows[0].feedback_id == first.feedback_id
    assert rows[1].supersedes_feedback_id == first.feedback_id


@pytest.mark.parametrize("status", ["COMPLETED", "COMPLETED_WITH_ERRORS", "STOPPED"])
@pytest.mark.asyncio
async def test_terminal_status_schedules_one_reminder(status, reminder_harness) -> None:
    await reminder_harness.finish(status)
    reminder_harness.clock.advance(600)
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1
    await reminder_harness.run_due()
    assert reminder_harness.maisaka.trigger_calls == 1
```

- [ ] **Step 2: 运行 feedback tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_feedback.py tests/integration/test_feedback_reminders.py -v`

Expected: FAIL because feedback service/tool is missing。

- [ ] **Step 3: 定义完整 feedback schema 与 lesson**

字段：`task_id`、可选 `round_number`、`disposition accepted|mixed|rejected|superseded`、`rating 1..5 optional`、`useful_findings[]`、`incorrect_findings[]`、`missing_information[]`、`decision`、`outcome`、`corrections[]`、`notes`、`supersedes_feedback_id`。superseded 必须引用同 Task已有 event；其他 disposition不得传 supersedes。所有 event append-only。

lesson 不调用第五种总结器角色；使用 deterministic renderer：Task ID/round + disposition + useful/corrections/missing/outcome 的非空短段，最大 8000 字，明确 `source_feedback_id`。lesson写 summary-layer source并排 vector job；不会修改 prompts/selectors/catalog。

- [ ] **Step 4: 注册正式 Planner tool**

`submit_research_feedback` 用上面字段 JSON schema，handler调用 service并返回 feedback ID、indexed lesson ID、当前 disposition。至此 `get_components()` 的 Tool names必须恰好为批准的八个。提交 feedback transaction取消该 round pending reminder。

- [ ] **Step 5: 实现 reminder/outbox 语义**

COMPLETED/COMPLETED_WITH_ERRORS/STOPPED transaction插 unique(round_id) reminder due=ended+feedback_wait_seconds。600s 到期且无该 round feedback/新 round时，outbox trigger Maisaka，intent要求检查 task/report并携带当前 task ID调用 `submit_research_feedback`；一次触发后 status=triggered，不重复。continue/new round把 pending标 cancelled；EXPIRED/INTERRUPTED/FAILED不插 reminder。

- [ ] **Step 6: 运行 feedback tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_feedback.py tests/integration/test_feedback_reminders.py tests/test_planner_tools.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/feedback.py lunagentic_research_swarm/tools.py lunagentic_research_swarm/runtime/controller.py lunagentic_research_swarm/services.py plugin.py tests/test_feedback.py tests/integration/test_feedback_reminders.py tests/test_planner_tools.py
git commit -m "feat: add research feedback and reminders"
```

### Task 8: 实现可选 raw debug 存储与确定性统计

**Files:**
- Create: `lunagentic_research_swarm/storage/debug.py`
- Create: `lunagentic_research_swarm/statistics.py`
- Modify: `lunagentic_research_swarm/runtime/turns.py`
- Modify: `lunagentic_research_swarm/procedures/executor.py`
- Test: `tests/storage/test_debug_storage.py`
- Test: `tests/test_statistics.py`

**Interfaces:**
- Consumes: 两个 storage toggles、LLM/procedure events、SQLite ledger/usage。
- Produces: `DebugStore` 独立 DB、`StatisticsService.task/plugin/cache`。

- [ ] **Step 1: 写四组合隐私矩阵失败测试**

```python
@pytest.mark.parametrize(
    ("transcripts", "payloads", "expect_transcript", "expect_payload"),
    [(False, False, False, False), (True, False, True, False),
     (False, True, False, True), (True, True, True, True)],
)
@pytest.mark.asyncio
async def test_raw_storage_toggles_are_independent(
    debug_harness, transcripts, payloads, expect_transcript, expect_payload
) -> None:
    await debug_harness.run_one_turn(transcripts=transcripts, payloads=payloads)
    assert debug_harness.has_transcript_rows() is expect_transcript
    assert debug_harness.has_payload_rows() is expect_payload
    assert not debug_harness.vector_text_contains("raw-agent-secret")
    assert not debug_harness.vector_text_contains("raw-procedure-secret")
```

- [ ] **Step 2: 运行 debug/stats tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_debug_storage.py tests/test_statistics.py -v`

Expected: FAIL because debug/statistics modules are missing。

- [ ] **Step 3: 实现独立 debug DB 与失败隔离**

两个开关都 false时不创建 `data_dir/debug/`。任一开启时创建 `debug/raw.sqlite3`；transcripts表存 task/round/branch/turn/messages/envelope，payloads表存 request/arguments/raw result。两者从不被 `load_summary_layer`/VectorIndex读取。branch summary transaction成功后 runtime messages立即清空，即使 debug保存；restart也不读 debug。

debug write失败不回滚权威 transition：记录 logger error并在 authority `extension_fingerprints` 或专用 minimal procedure error event写 `debug_storage_failed`（无 raw内容）。reasoning在写前已被 gateway丢弃，debug schema不设 reasoning column。

- [ ] **Step 4: 实现 SQL 聚合统计**

Task stats从 ledger/usage/procedure/branch/report重算：普通/总结器 call count、prompt/output/hit/miss、cache hit rate=`hit/(hit+miss)`（分母0为 null）、estimated/actual/unreconciled credits、总结器按同一价格目录计算但不扣研究余额的 `cost_equivalent_credits`、pool/debt、branches total/active/finalized/max depth、compact/checkpoint/protocol correction/continue counts、procedures success/error/external cost、duration/error counts。Plugin stats按物理 model/agent/procedure/task汇总，不读 raw。

每个 report保存的 stats与同 transaction snapshot一致；测试从账本重算并 equality compare。

- [ ] **Step 5: 运行 debug/stats tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/storage/test_debug_storage.py tests/test_statistics.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/storage/debug.py lunagentic_research_swarm/statistics.py lunagentic_research_swarm/runtime/turns.py lunagentic_research_swarm/procedures/executor.py tests/storage/test_debug_storage.py tests/test_statistics.py
git commit -m "feat: add private debug storage and statistics"
```

### Task 9: 实现 `/swarm` 用户命令与 health/status 输出

**Files:**
- Create: `lunagentic_research_swarm/commands.py`
- Modify: `plugin.py`
- Test: `tests/test_commands.py`

**Interfaces:**
- Consumes: ResearchManager/registries/statistics/vector/feedback/health、`ctx.send.text`。
- Produces: 已批准的命令集合与中文紧凑输出。

- [ ] **Step 1: 写命令组件和 vector rebuild 失败测试**

```python
def test_swarm_command_patterns_are_registered(plugin_module) -> None:
    names = {
        item["name"] for item in plugin_module.create_plugin().get_components()
        if item["type"] == "COMMAND"
    }
    assert names == {
        "swarm_status", "swarm_tasks", "swarm_stats", "swarm_agents",
        "swarm_procedures", "swarm_health", "swarm_vectors_status",
        "swarm_vectors_rebuild", "swarm_feedback",
    }


@pytest.mark.asyncio
async def test_force_vector_rebuild_command(command_harness) -> None:
    await command_harness.invoke("/swarm vectors rebuild --force", stream_id="s")
    assert command_harness.vector.rebuild_calls == [True]
    assert "已创建新的向量 generation" in command_harness.sent_text
```

- [ ] **Step 2: 运行 command tests 并确认失败**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_commands.py -v`

Expected: FAIL because commands are missing。

- [ ] **Step 3: 实现九个正则命令**

- `/swarm status [task_id]`：无 ID显示插件+running概要，有 ID显示 lifecycle/round/active/queue/deadline/credits/tokens/cache/reports。
- `/swarm tasks [status]`：最近任务。
- `/swarm stats [task_id]`：Task或plugin聚合。
- `/swarm agents` / `procedures`：live provider、enabled、selector/protocol、health，不显示 prompt/secrets。
- `/swarm health`：SQLite/vector/pinning/extension/recommended fetch/queue/outbox/reminder。
- `/swarm vectors status` 与 `/swarm vectors rebuild [--force]`。
- `/swarm feedback <task_id> <accepted|mixed|rejected> [notes]`：简化用户通路，调用同一 FeedbackService；复杂反馈用 Planner tool。

所有 handler从 kwargs解析 stream_id，不存在则错误；用 `ctx.send.text(text, stream_id)`。`commands.max_output_chars` 超限做结构化分页/计数摘要，不静默省略错误条目。

- [ ] **Step 4: 运行 command tests 并提交**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/test_commands.py -v`

Expected: 全部 PASS。

```bash
git add lunagentic_research_swarm/commands.py plugin.py tests/test_commands.py
git commit -m "feat: add swarm status and maintenance commands"
```

### Task 10: 完成中文文档、兼容性矩阵与 0.1.0 验收

**Files:**
- Create: `README.md`
- Create: `CHANGELOG.md`
- Create: `LICENSE`
- Create: `docs/extension-authoring.md`
- Create: `docs/credits-and-reporting.md`
- Create: `docs/privacy-and-recovery.md`
- Create: `tests/smoke_test.py`
- Create: `tests/integration/test_optional_providers.py`
- Create: `tests/integration/test_release_acceptance.py`
- Modify: `_manifest.json`
- Modify: `config.default.toml`

**Interfaces:**
- Consumes: 完整 0.1.0 plugin。
- Produces: 用户/扩展作者中文文档、推荐依赖说明、离线 smoke test、发布验收证据。

- [ ] **Step 1: 写 optional provider 与 release acceptance tests**

覆盖三种 fetch 状态：不存在→正常 load+health recommended_missing；存在有效 provider→catalog出现 fetch；卸载→新调用 edge procedure_unavailable，在途调用按 provider返回完成。文件 provider不存在不影响 core。第三方 agent/provider invalid batch在 health可见。

release flow 断言：start立即返回；formalized task入 SQLite/vector；默认 quick root可运行；intermediate先持久化后 Maisaka；final含 stats；feedback reminder；continue只用 summaries；raw默认不存在；all contexts release；八 tools/九 commands/九 agents/所有 bundled procedures完整。

- [ ] **Step 2: 写 README 与专题文档**

README 必须覆盖：

- 名称、Lunagentic 词源、agentic swarm长描述；GPT-5.6 Luna/DeepSeek-V4-Flash-0731/GPT-5.6 Sol/Claude Opus 5仅是部署示例。
- 安装、`config.default.toml`复制、task/model selector、物理 pinning fragility与 health。
- 1 price unit=100 credits、默认100、currency label忽略、Host price优先级与插件 override Note、免费缺省、500k/50k warning。
- 时间报告、60s grace、checkpoint/compact/terminate、pause/continue/stop行为。
- Planner tools、用户 commands、九 agents、内置 procedures。
- JSON default/native per-agent，native允许 toolcall无正文；格式错本地修复+一次同模型 correction。
- SQLite/LanceDB、embedding mismatch自动/手动 rebuild、默认隐私、crash→INTERRUPTED。
- feedback/reminder/学习边界。
- `fetch-url`推荐不必需；file depot独立未来 provider；LLMProvider不用于 pinning。

`extension-authoring.md` 给出完整 `describe_agents@1`、`describe_procedures@1`、`invoke_procedure@1` decorators/payload/result、refresh call、provider config ownership、remove in-flight语义。其他两份专题逐条写批准 math/report/privacy规则。

- [ ] **Step 3: 完善 manifest 描述和最终 config template**

manifest description 用中文表达：以 agentic research swarm architecture进行持续、多智能体深度调查；结合价格有竞争力的快速模型与大型知识模型；通过多代专职 agent、动态 credits、模型路由和上下文优化，提供近似可预测的时间/成本特征。不要宣称绑定某具体模型。dependencies与 pyproject/requirements同步，fetch-url不列依赖。

config.default 完整包含所有 section、九 agent可覆写示例（注释）、procedures、reporting/feedback/commands、四搜索引擎字段、pricing override醒目 Note；API keys空且不提交 live config。

- [ ] **Step 4: 写离线 smoke test**

`tests/smoke_test.py` 按工作区惯例注入 repo/SDK path，验证 import/factory/components/config schema、bundled definitions、dependency sync、无 reasoning schema、prompt文件可读；直接 `python tests/smoke_test.py` 最后打印明确 `ok`。

- [ ] **Step 5: 运行完整验收**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest -v`

Expected: 全部 PASS，0 failed/errored/skipped（只有明确平台条件的 Lance wheel测试可用 marker skip，并在 README 说明）。

Run: `PYTHONPATH=../maibot-plugin-sdk python tests/smoke_test.py`

Expected: exit 0，最后一行 `ok: Lunagentic Research Swarm 0.1.0 offline smoke test passed`。

Run: `python -m ruff check plugin.py lunagentic_research_swarm tests`

Expected: exit 0。不运行 workspace-wide format/build。

- [ ] **Step 6: 核对工作树与提交发布候选**

Run: `git diff --check`

Expected: 无输出。

Run: `git status --short`

Expected: 只列本 Task 新增/修改文件。

```bash
git add README.md CHANGELOG.md LICENSE docs _manifest.json config.default.toml tests/smoke_test.py tests/integration/test_optional_providers.py tests/integration/test_release_acceptance.py
git commit -m "docs: prepare Lunagentic Research Swarm 0.1.0"
```

## 本计划完成门槛

- 九个 agent 与所有非核心 Procedure 都由同一 extension validator/registry加载。
- 四搜索引擎只有配置有效时 advertised，结果有 query/engine/URL/provenance，secret不泄漏。
- calculator无 arbitrary eval，memory不拉二进制，URL normalize不改变 query语义。
- Lance generation保存 selector/model/fingerprint/dimension/schema，mismatch明确 rebuild且原子切换。
- past cases透明呈现 accepted/mixed/rejected/unreviewed，feedback不自动改 prompt/路由。
- 600s reminder对 completed/error/stopped一次，continue/feedback取消，expired/interrupted无提醒。
- raw toggles四组合、统计账本重算、九 commands、八 Planner tools通过。
- `fetch-url` 缺失/存在/卸载兼容，且不是 manifest hard dependency。
- 全套 pytest、offline smoke、窄 ruff、`git diff --check`通过，工作树在最终提交后干净。
