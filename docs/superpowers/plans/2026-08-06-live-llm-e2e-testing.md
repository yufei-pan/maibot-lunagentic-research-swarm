# Live LLM E2E Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add credentials-gated, marker-tiered live LLM E2E tests (A → B → stub-thorough → live-tools) that drive `RuntimeHarness` with a real OpenAI-compatible endpoint and LLM-as-judge quality checks.

**Architecture:** Extend `tests/live_llm.py` for credentials, `LiveLLMGateway`, and judges. Add `tests/live_harness.py` for effect-drain loop, stub/live procedures, and live summarizer. Thin wrappers on `RuntimeHarness` keep call sites short. No Host/SDK changes; no Mein deploy.

**Tech Stack:** Python 3.10+, pytest / pytest-asyncio, httpx, existing LRS `RuntimeEffectRunner` + `TurnWorker` + `ProcedureExecutor`, OpenAI-compatible Chat Completions.

**Spec:** `docs/superpowers/specs/2026-08-06-live-llm-e2e-testing-design.md`

## Global Constraints

- Do not modify MaiBot Host or SDK.
- Never commit real API keys, LAN hosts, or secrets. Docs/examples use placeholders only (`YOUR_API_KEY`, `127.0.0.1` or `*.example`).
- `.debug_api_call_credentials` remains gitignored; only `.debug_api_call_credentials.example` is tracked.
- Bare `pytest` may collect live tests; they **skip** without usable credentials. Select tiers with `-m live_llm…`.
- Prefer 简体中文 for user-facing / log strings in helpers; keep protocol/field names English.
- One judge JSON-parse retry only; no retry on task timeout or HTTP errors.
- Run offline tests with: `PYTHONPATH=.:../maibot-plugin-sdk pytest <path> -v`
- Run live tiers only when credentials are filled: `pytest -m live_llm_e2e -v` (etc.)

## File map

| File | Responsibility |
|---|---|
| `tests/live_llm.py` | Credentials load, `credentials_available` / `live_tools_available`, `chat_completion`, `LiveLLMGateway`, `light_judge` / `deep_judge` |
| `tests/live_harness.py` | Effect drain loop, stub procedure provider, live summarizer, artifact dump |
| `tests/fakes.py` | Thin `RuntimeHarness` wrappers (`use_live_llm`, `drive_live_until_terminal`, …) |
| `.debug_api_call_credentials.example` | Placeholder schema including timeouts + optional `[web_search]` |
| `pyproject.toml` | Register `live_llm_e2e`, `live_llm_thorough`, `live_llm_live_tools` markers |
| `tests/llm/test_live_credentials_unit.py` | Offline credential loader tests |
| `tests/llm/test_live_gateway_unit.py` | Offline mocked HTTP tests for gateway |
| `tests/llm/test_live_judge_unit.py` | Offline mocked judge tests |
| `tests/integration/test_live_drive_offline.py` | Offline drain-loop with `FakeLLMGateway` |
| `tests/integration/test_live_e2e_ab.py` | Markers A+B |
| `tests/integration/test_live_thorough_stub_tools.py` | Stub-tools thorough |
| `tests/integration/test_live_thorough_live_tools.py` | Real web_search thorough |
| `README.md` | Short “live LLM tests” how-to (no real endpoints) |

---

### Task 1: Credentials schema, markers, example file

**Files:**
- Modify: `tests/live_llm.py`
- Modify: `.debug_api_call_credentials.example`
- Modify: `pyproject.toml`
- Create: `tests/llm/test_live_credentials_unit.py`

**Interfaces:**
- Consumes: existing `LiveLLMCredentials`, `load_live_llm_credentials`, `credentials_available`
- Produces:
  - `LiveLLMCredentials` fields: `e2e_timeout_seconds: float`, `thorough_timeout_seconds: float`, `web_search_enabled: bool`, `web_search: dict[str, Any]`
  - `live_tools_available() -> bool`
  - pytest markers: `live_llm_e2e`, `live_llm_thorough`, `live_llm_live_tools`

- [ ] **Step 1: Write failing unit tests for new credential fields**

```python
# tests/llm/test_live_credentials_unit.py
from __future__ import annotations

from pathlib import Path

import pytest

from live_llm import live_tools_available, load_live_llm_credentials


def test_load_timeouts_and_web_search_defaults(tmp_path: Path) -> None:
    path = tmp_path / "creds.toml"
    path.write_text(
        'base_url = "http://127.0.0.1:9/v1"\n'
        'api_key = "sk-test"\n'
        'model = "m"\n',
        encoding="utf-8",
    )
    creds = load_live_llm_credentials(path)
    assert creds.e2e_timeout_seconds == 180.0
    assert creds.thorough_timeout_seconds == 900.0
    assert creds.web_search_enabled is False
    assert creds.web_search == {}


def test_live_tools_available_requires_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "creds.toml"
    path.write_text(
        'base_url = "http://127.0.0.1:9/v1"\n'
        'api_key = "sk-test"\n'
        'model = "m"\n'
        "web_search_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("live_llm.CREDENTIALS_PATH", path)
    assert live_tools_available() is True
```

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_live_credentials_unit.py -v`

Expected: FAIL (`LiveLLMCredentials` missing new fields / `live_tools_available` missing).

- [ ] **Step 3: Implement loader + markers + example**

Update `LiveLLMCredentials` and `load_live_llm_credentials` to read:

```python
e2e_timeout_seconds=float(raw.get("e2e_timeout_seconds", 180)),
thorough_timeout_seconds=float(raw.get("thorough_timeout_seconds", 900)),
web_search_enabled=bool(raw.get("web_search_enabled", False)),
web_search=dict(raw.get("web_search") or {}),
```

```python
def live_tools_available() -> bool:
    if not credentials_available():
        return False
    return bool(load_live_llm_credentials().web_search_enabled)
```

`pyproject.toml` markers:

```toml
markers = [
    "live_llm: 需要仓库根 .debug_api_call_credentials 的真实 LLM 调用",
    "live_llm_e2e: A/B 端到端（seed formalize + light judge）",
    "live_llm_thorough: stub tools 深度调研 + deep judge",
    "live_llm_live_tools: 真实 web_search + live LLM + deep judge",
]
```

Update `.debug_api_call_credentials.example` with placeholder timeouts and commented `[web_search]` — **no real hosts/keys**.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_live_credentials_unit.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/live_llm.py tests/llm/test_live_credentials_unit.py \
  .debug_api_call_credentials.example pyproject.toml
git commit -m "$(cat <<'EOF'
test: extend live LLM credentials and pytest markers

Add e2e/thorough timeouts, web_search_enabled gate, and marker registrations for live tiers.
EOF
)"
```

---

### Task 2: `LiveLLMGateway` (offline unit tests)

**Files:**
- Modify: `tests/live_llm.py`
- Create: `tests/llm/test_live_gateway_unit.py`

**Interfaces:**
- Consumes: `LiveLLMCredentials`, `chat_completion` patterns, `GenerationRequest` / `GenerationResult` / `TokenUsage` / `GenerationError` from `lunagentic_research_swarm.llm.gateway`
- Produces: `class LiveLLMGateway` with:
  - `calls: list[dict[str, Any]]`
  - `async def generate(self, request: GenerationRequest | None = None, *, selector: str | None = None, messages: Any = None, **kwargs: Any) -> GenerationResult`
  - Always uses credentials `model` / `temperature` (ignore selector model id)
  - Maps HTTP content → `GenerationResult.response`; maps `message.tool_calls` if present

- [ ] **Step 1: Write failing gateway unit test with mocked httpx**

```python
# tests/llm/test_live_gateway_unit.py
from __future__ import annotations

from typing import Any

import pytest

from live_llm import LiveLLMCredentials, LiveLLMGateway
from lunagentic_research_swarm.llm.gateway import GenerationRequest


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


@pytest.mark.asyncio
async def test_live_gateway_generate_maps_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = LiveLLMCredentials(
        base_url="http://127.0.0.1:9/v1",
        api_key="sk-test",
        model="local-model",
        temperature=1.0,
    )
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse(
                {
                    "model": "local-model",
                    "choices": [{"message": {"content": '{"report":"ok","procedures":[],"delegations":[]}'}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            )

    monkeypatch.setattr("live_llm.httpx.AsyncClient", _Client)
    gateway = LiveLLMGateway(creds)
    result = await gateway.generate(
        GenerationRequest(selector="task:mid_memory", messages=[{"role": "user", "content": "hi"}])
    )
    assert result.success is True
    assert "report" in result.response
    assert result.model_name == "local-model"
    assert captured["json"]["model"] == "local-model"
    assert captured["json"]["temperature"] == 1.0
    assert gateway.calls[0]["selector"] == "task:mid_memory"
```

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_live_gateway_unit.py -v`

Expected: FAIL (`LiveLLMGateway` missing).

- [ ] **Step 3: Implement `LiveLLMGateway`**

Implement in `tests/live_llm.py` mirroring `FakeLLMGateway.generate` request unpacking, posting to `{base_url}/chat/completions`, returning `GenerationResult` with `TokenUsage(source="actual")`. On HTTP/JSON failure raise (do not return soft success). Export in `__all__`.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_live_gateway_unit.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/live_llm.py tests/llm/test_live_gateway_unit.py
git commit -m "$(cat <<'EOF'
test: add LiveLLMGateway for OpenAI-compatible generate

Map chat completions into GenerationResult for TurnWorker live drives.
EOF
)"
```

---

### Task 3: Light / deep LLM judges (offline unit tests)

**Files:**
- Modify: `tests/live_llm.py`
- Create: `tests/llm/test_live_judge_unit.py`

**Interfaces:**
- Consumes: `chat_completion`, `LiveLLMCredentials`
- Produces:
  - `async def light_judge(credentials, *, objective: str, report: str) -> dict[str, Any]` with keys `pass: bool`, `reason: str`
  - `async def deep_judge(credentials, *, objective: str, report: str, evidence: str = "") -> dict[str, Any]` with keys `pass: bool`, `reason: str`, `scores: dict[str, int]` (`relevance`, `completeness`, `groundedness`)
  - Parse model JSON; on parse failure retry judge prompt once; then raise/`pass=False` with reason (never silent pass)
  - Deep pass rule: all three scores ≥ 3

- [ ] **Step 1: Write failing judge unit tests**

```python
# tests/llm/test_live_judge_unit.py
from __future__ import annotations

import json
from typing import Any

import pytest

from live_llm import LiveLLMCredentials, deep_judge, light_judge

CREDS = LiveLLMCredentials("http://127.0.0.1:9/v1", "sk", "m", temperature=1.0)


@pytest.mark.asyncio
async def test_light_judge_parses_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_chat(_creds: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"success": True, "response": json.dumps({"pass": True, "reason": "addresses objective"}, ensure_ascii=False)}

    monkeypatch.setattr("live_llm.chat_completion", _fake_chat)
    result = await light_judge(CREDS, objective="终结并报告", report="已终结：完成最小自测。")
    assert result["pass"] is True


@pytest.mark.asyncio
async def test_deep_judge_requires_scores_ge_3(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_chat(_creds: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "success": True,
            "response": json.dumps(
                {
                    "pass": True,
                    "reason": "ok",
                    "scores": {"relevance": 4, "completeness": 3, "groundedness": 2},
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr("live_llm.chat_completion", _fake_chat)
    result = await deep_judge(CREDS, objective="查 X", report="...", evidence="TOKEN")
    assert result["pass"] is False  # groundedness 2 < 3 overrides model pass
```

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_live_judge_unit.py -v`

Expected: FAIL (functions missing).

- [ ] **Step 3: Implement judges**

System/user prompts must demand a single JSON object. Prefer extracting the first `{...}` object from the reply (same spirit as envelope repairs). Enforce deep score gate in Python regardless of model `pass`.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/llm/test_live_judge_unit.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/live_llm.py tests/llm/test_live_judge_unit.py
git commit -m "$(cat <<'EOF'
test: add light and deep LLM judges for live E2E

JSON rubrics with one parse retry; deep pass requires all scores >= 3.
EOF
)"
```

---

### Task 4: Live effect-drain harness (offline with FakeLLM)

**Files:**
- Create: `tests/live_harness.py`
- Modify: `tests/fakes.py` (`RuntimeHarness` wrappers)
- Create: `tests/integration/test_live_drive_offline.py`

**Interfaces:**
- Consumes: `RuntimeHarness`, `RuntimeEffectRunner`, `TurnWorker`, `ProcedureExecutor`, core procedure catalog/handlers, `FakeLLMGateway`
- Produces:
  - `async def drive_until_terminal(harness, *, timeout_seconds: float, artifact_dir: Path | None = None) -> dict[str, Any]`
  - `def attach_effect_runner(harness, *, pricing: Any | None = None) -> RuntimeEffectRunner` — builds `TurnWorker(harness.llm, harness.procedures, procedure_factory=...)` and `bind_manager`
  - `RuntimeHarness.use_live_llm(creds)` → sets `self.llm = LiveLLMGateway(creds)`
  - `RuntimeHarness.drive_live_until_terminal(...)` → delegates
  - On timeout: write `final_status.json`, `scheduler_pending.txt`, last report text if any under `artifact_dir`, then raise `TimeoutError`

**Drain algorithm (must implement exactly):**

```python
async def drive_until_terminal(harness, *, timeout_seconds: float, artifact_dir=None):
    import time, asyncio
    from lunagentic_research_swarm.models import TaskStatus
    runner = attach_effect_runner(harness)
    deadline = time.monotonic() + float(timeout_seconds)
    terminal = {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
        "COMPLETED",
        "COMPLETED_WITH_ERRORS",
    }
    while time.monotonic() < deadline:
        status = await harness.manager.status(harness.task_id)
        raw = status.get("status")
        status_value = raw.value if hasattr(raw, "value") else str(raw)
        if status_value in terminal or raw in (TaskStatus.COMPLETED, TaskStatus.COMPLETED_WITH_ERRORS):
            return status
        if harness.scheduler.enqueued:
            effect = harness.scheduler.enqueued.pop(0)
            await runner.run(effect)
            continue
        await asyncio.sleep(0.05)
    # dump artifacts then raise TimeoutError
```

`procedure_factory` must use real `ProcedureExecutor` with catalog from effect payload / snapshot and `local_invokers` that include at least `core.terminate` (reuse bundled core registration pattern from `services.py` / `procedures.core`). For this offline task, keep `harness.procedures` as the API/local invoker backing terminate; do not require web_search yet.

- [ ] **Step 1: Write offline drain test**

```python
# tests/integration/test_live_drive_offline.py
from __future__ import annotations

import pytest
from fakes import FakeLLMResponse
from live_harness import drive_until_terminal

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import CORE_TERMINATE_ID


@pytest.mark.asyncio
async def test_drive_until_terminal_with_scripted_terminate(runtime_harness) -> None:
    harness = runtime_harness
    await harness.start("离线驱动自测", credits=40.0, time_budget=60)
    await harness.formalize(
        "正式任务：用一次 turn 调用 core.terminate 并给出简短 report。"
    )
    harness.llm.enqueue(
        FakeLLMResponse(
            payload={
                "report": "离线完成",
                "procedures": [{"procedure_id": CORE_TERMINATE_ID, "arguments": {}, "credits": 0}],
                "delegations": [],
            }
        )
    )
    status = await drive_until_terminal(harness, timeout_seconds=30)
    raw = status.get("status")
    value = raw.value if hasattr(raw, "value") else str(raw)
    assert value in {TaskStatus.COMPLETED.value, TaskStatus.COMPLETED_WITH_ERRORS.value}
```

If formalize already enqueued `PerformAgentCall`, the drain loop must consume it (do not manually build effects).

- [ ] **Step 2: Run to verify RED**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/integration/test_live_drive_offline.py -v`

Expected: FAIL (`live_harness` missing or terminate path incomplete).

- [ ] **Step 3: Implement `live_harness.py` + wrappers**

Wire core procedures so scripted terminate reaches a terminal task status. Skip/`pytest.fail` with a clear message if catalog wiring is wrong. Add `RuntimeHarness.use_live_llm` / `drive_live_until_terminal` thin wrappers.

- [ ] **Step 4: Run to verify GREEN**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/integration/test_live_drive_offline.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/live_harness.py tests/fakes.py tests/integration/test_live_drive_offline.py
git commit -m "$(cat <<'EOF'
test: add live effect-drain harness for RuntimeHarness

Drain FakeScheduler through RuntimeEffectRunner until terminal status.
EOF
)"
```

---

### Task 5: Live E2E A + B (`live_llm_e2e`)

**Files:**
- Create: `tests/integration/test_live_e2e_ab.py`

**Interfaces:**
- Consumes: `credentials_available`, `load_live_llm_credentials`, `RuntimeHarness.use_live_llm`, `drive_live_until_terminal` / `drive_until_terminal`, `light_judge`
- Produces: two async tests marked `live_llm_e2e`, skip if `not credentials_available()`

- [ ] **Step 1: Write A + B tests (skip without creds)**

```python
# tests/integration/test_live_e2e_ab.py
from __future__ import annotations

import pytest

from live_llm import credentials_available, light_judge, load_live_llm_credentials
from lunagentic_research_swarm.models import TaskStatus

pytestmark = [
    pytest.mark.live_llm_e2e,
    pytest.mark.skipif(not credentials_available(), reason="未配置可用的 .debug_api_call_credentials"),
]


def _status_value(status: dict) -> str:
    raw = status.get("status")
    return raw.value if hasattr(raw, "value") else str(raw)


@pytest.mark.asyncio
async def test_live_e2e_a_thin_terminate_and_light_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    formalized = (
        "# 正式任务\n用一次 turn 输出可解析 JSON envelope，"
        "调用 core.terminate，report 用一两句说明已完成最小自测。不要写长文。\n"
    )
    await harness.start("最小协议自测", credits=50.0, time_budget=120)
    await harness.formalize(formalized)  # seeded via FakeSummarizer
    harness.use_live_llm(creds)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.e2e_timeout_seconds,
        artifact_dir=tmp_path / "a",
    )
    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }
    # Resolve FINAL report text from harness.reports / store — follow test_research_flow.py accessors.
    final_text = str(getattr(harness.reports[-1], "body", None) or harness.reports[-1])
    assert final_text.strip(), "FINAL report must be non-empty"
    assert harness.llm.calls, "expected at least one live LLM call"
    verdict = await light_judge(creds, objective=formalized, report=final_text)
    assert verdict["pass"] is True, verdict


@pytest.mark.asyncio
async def test_live_e2e_b_root_delegates_child_then_finishes(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    formalized = (
        "# 正式任务\n根智能体必须委派至少一个子分支（同一 agent_id 即可），"
        "子分支完成简短 report 后 terminate；最终产出 FINAL。\n"
    )
    await harness.start("多分支自测", credits=80.0, time_budget=180)
    await harness.formalize(formalized)
    harness.use_live_llm(creds)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.e2e_timeout_seconds,
        artifact_dir=tmp_path / "b",
    )
    assert _status_value(status) in {
        TaskStatus.COMPLETED.value,
        TaskStatus.COMPLETED_WITH_ERRORS.value,
    }
    # Assert ≥1 child materialized (store parent_branch_id IS NOT NULL, or branch count / depth).
    child_rows = await harness.store_count_child_branches()  # implement helper if missing
    assert child_rows >= 1
    final_text = str(getattr(harness.reports[-1], "body", None) or harness.reports[-1])
    assert final_text.strip()
    verdict = await light_judge(creds, objective=formalized, report=final_text)
    assert verdict["pass"] is True, verdict
```

Tighten FINAL / child-branch accessors while implementing to match real `Report` / store APIs (follow `tests/integration/test_research_flow.py`). If `store_count_child_branches` does not exist, add a small helper on the harness or inline SQL against the integration SQLite.

- [ ] **Step 2: Run offline collection (skip)**

Run: `PYTHONPATH=.:../maibot-plugin-sdk pytest tests/integration/test_live_e2e_ab.py -v`

Expected: SKIP if placeholder credentials; or PASS/FAIL against real local endpoint when configured.

- [ ] **Step 3: Run live when credentials are real**

Run: `pytest tests/integration/test_live_e2e_ab.py -v -m live_llm_e2e`

Expected: PASS on local OpenAI-compatible endpoint with `temperature` from credentials.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_live_e2e_ab.py tests/fakes.py tests/live_harness.py
git commit -m "$(cat <<'EOF'
test: add live_llm_e2e A/B research slice tests

Seed formalize, live LLM drain, and light judge on FINAL reports.
EOF
)"
```

---

### Task 6: Stub procedures + thorough tier (`live_llm_thorough`)

**Files:**
- Modify: `tests/live_harness.py` (`use_stub_procedures`, `LiveSummarizer`)
- Modify: `tests/fakes.py` wrappers
- Create: `tests/integration/test_live_thorough_stub_tools.py`

**Interfaces:**
- Consumes: `builtin.web_search` procedure id, `deep_judge`, `thorough_timeout_seconds`
- Produces:
  - Stub search returning fixtures with distinctive token `LRS_STUB_FACT_CALIFORNIA_RAIL_42`
  - `LiveSummarizer` with `formalize_task` / `finalize_branch` / `finalize_task` calling live LLM (minimal prompts); used only on thorough tiers
  - Test marked `live_llm_thorough`

**Fixture token (exact):** `LRS_STUB_FACT_CALIFORNIA_RAIL_42`

- [ ] **Step 1: Write thorough stub-tools test**

```python
# tests/integration/test_live_thorough_stub_tools.py
from __future__ import annotations

import pytest

from live_llm import credentials_available, deep_judge, load_live_llm_credentials

STUB_FACT = "LRS_STUB_FACT_CALIFORNIA_RAIL_42"

pytestmark = [
    pytest.mark.live_llm_thorough,
    pytest.mark.skipif(not credentials_available(), reason="未配置可用的 .debug_api_call_credentials"),
]


@pytest.mark.asyncio
async def test_live_thorough_stub_web_search_and_deep_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    harness.use_live_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_stub_procedures(
        {
            "california": {
                "results": [
                    {
                        "title": "Stub Rail Doc",
                        "url": "https://example.invalid/rail",
                        "snippet": f"Budget note {STUB_FACT}",
                    }
                ]
            }
        }
    )
    await harness.start(
        "调查加州高铁预算时间线，必须先 web_search 再下结论，并在报告中引用检索到的关键事实。",
        credits=120.0,
        time_budget=600,
    )
    await harness.manager.wait_idle(harness.task_id)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=tmp_path / "thorough",
    )
    assert harness.stub_search_invokes >= 1
    final_text = str(getattr(harness.reports[-1], "body", None) or harness.reports[-1])
    verdict = await deep_judge(
        creds,
        objective="加州高铁预算时间线（须使用搜索）",
        report=final_text,
        evidence=STUB_FACT,
    )
    assert verdict["pass"] is True, verdict
```

- [ ] **Step 2: Implement stub provider + live summarizer**

`LiveSummarizer.formalize_task`: prompt the model to rewrite objective into a short formalized markdown task; return `SummaryResult(True, text, model, usage, None)`.

`use_stub_procedures`: register local invoker for `builtin.web_search` that matches query substrings to fixtures; always allow `core.terminate`.

Replace `harness.summarizer` **before** `start` so `_formalize` uses the live summarizer. Open any leftover FakeSummarizer gates if still present.

- [ ] **Step 3: Run live thorough**

Run: `pytest tests/integration/test_live_thorough_stub_tools.py -v -m live_llm_thorough`

Expected: PASS (may take minutes).

- [ ] **Step 4: Commit**

```bash
git add tests/live_harness.py tests/fakes.py tests/integration/test_live_thorough_stub_tools.py
git commit -m "$(cat <<'EOF'
test: add live_llm_thorough stub web_search E2E

Live formalize/summarize with deterministic search fixtures and deep judge.
EOF
)"
```

---

### Task 7: Live tools thorough (`live_llm_live_tools`)

**Files:**
- Modify: `tests/live_harness.py` (`use_live_procedures`)
- Modify: `tests/fakes.py`
- Create: `tests/integration/test_live_thorough_live_tools.py`
- Modify: `README.md` (short section)

**Interfaces:**
- Consumes: `live_tools_available()`, `WebSearchSection.model_validate` from credentials `web_search` dict + defaults, real `BundledProcedureProvider` / `WebSearchService`
- Produces: one test marked `live_llm_live_tools`, skip unless `live_tools_available()`

- [ ] **Step 1: Write live-tools test**

```python
# tests/integration/test_live_thorough_live_tools.py
from __future__ import annotations

import pytest

from live_llm import deep_judge, live_tools_available, load_live_llm_credentials

pytestmark = [
    pytest.mark.live_llm_live_tools,
    pytest.mark.skipif(not live_tools_available(), reason="未启用 web_search_enabled 或缺少 LLM 凭证"),
]


@pytest.mark.asyncio
async def test_live_thorough_real_web_search_and_deep_judge(runtime_harness, tmp_path) -> None:
    creds = load_live_llm_credentials()
    harness = runtime_harness
    harness.use_live_summarizer(creds)
    harness.use_live_llm(creds)
    harness.use_live_procedures(creds.web_search)
    await harness.start(
        "用真实网页搜索查证一个简单事实性问题，并给出有依据的简短结论。",
        credits=150.0,
        time_budget=900,
    )
    await harness.manager.wait_idle(harness.task_id)
    status = await harness.drive_live_until_terminal(
        timeout_seconds=creds.thorough_timeout_seconds,
        artifact_dir=tmp_path / "live_tools",
    )
    assert harness.live_search_invokes >= 1
    final_text = str(getattr(harness.reports[-1], "body", None) or harness.reports[-1])
    verdict = await deep_judge(
        creds,
        objective="真实网页搜索事实核查",
        report=final_text,
        evidence="",
    )
    assert verdict["pass"] is True, verdict
```

Do **not** assert exact SERP snippets.

- [ ] **Step 2: Implement `use_live_procedures`**

Build `WebSearchSection` from credentials (`enabled_engines` default `["duckduckgo"]` if omitted). Wire bundled provider into local invokers used by `attach_effect_runner`. Count invokes on the harness.

- [ ] **Step 3: README blurb**

Add a short “Live LLM tests” section listing markers and pointing at `.debug_api_call_credentials.example`. No real endpoints/keys.

- [ ] **Step 4: Run**

```bash
pytest tests/integration/test_live_thorough_live_tools.py -v -m live_llm_live_tools
```

Expected: SKIP unless `web_search_enabled = true`; PASS when search works.

- [ ] **Step 5: Commit**

```bash
git add tests/live_harness.py tests/fakes.py \
  tests/integration/test_live_thorough_live_tools.py README.md \
  .debug_api_call_credentials.example
git commit -m "$(cat <<'EOF'
test: add live_llm_live_tools real web_search E2E

Optional thorough lane with BundledProcedureProvider search and deep judge.
EOF
)"
```

---

### Task 8: Smoke regression + docs cross-check

**Files:**
- Modify: none unless failures found

- [ ] **Step 1: Run offline regression**

```bash
PYTHONPATH=.:../maibot-plugin-sdk pytest tests/integration/test_live_drive_offline.py \
  tests/llm/test_live_credentials_unit.py \
  tests/llm/test_live_gateway_unit.py \
  tests/llm/test_live_judge_unit.py \
  tests/llm/test_live_json_envelope.py \
  tests/runtime/test_effect_runner_correction.py \
  -v
```

Expected: unit/offline PASS; live envelope SKIP or PASS depending on credentials.

- [ ] **Step 2: Confirm markers**

```bash
pytest --markers | grep live_llm
```

Expected: four live markers listed.

- [ ] **Step 3: Commit only if fixes were needed**

```bash
git commit -m "$(cat <<'EOF'
test: fix live E2E follow-ups from regression pass
EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Markers `live_llm` / `e2e` / `thorough` / `live_tools` | 1 |
| Credentials timeouts + `web_search_enabled` | 1, 7 |
| `LiveLLMGateway` | 2 |
| Light judge on A/B | 3, 5 |
| Deep judge on thorough | 3, 6, 7 |
| Seed formalize A/B | 5 |
| Live formalize thorough | 6, 7 |
| Effect drain via `RuntimeEffectRunner` | 4 |
| Stub web_search fixtures | 6 |
| Real web_search optional lane | 7 |
| Artifact dump on timeout | 4 |
| No secrets in docs | 1, 7 |
| Offline default / skip without creds | 1, 5–7 |

## Self-review notes

- No TBD placeholders left in task steps. FINAL report accessors may need tightening against real `Report` fields during Task 5 — follow `test_research_flow.py`, do not invent Host APIs.
- Type names consistent: `LiveLLMGateway.generate` → `GenerationResult`; judges return `dict[str, Any]` with documented keys.
- Task 4 is the riskiest (procedure catalog + terminate wiring); keep offline scripted terminate green before enabling live A/B.
- Placeholder hosts in this plan are `127.0.0.1` / `example.invalid` only.
