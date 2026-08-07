"""Live effect-drain helpers for RuntimeHarness (FakeLLM offline + LiveLLM later)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import FormalizedTask, TaskStatus
from lunagentic_research_swarm.procedures.core import (
    CORE_CHECKPOINT_ID,
    CORE_COMPACT_ID,
    CORE_TERMINATE_ID,
)
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.reducer import PerformBranchSummary
from lunagentic_research_swarm.runtime.turns import TurnWorker

_CORE_IDS = (CORE_TERMINATE_ID, CORE_COMPACT_ID, CORE_CHECKPOINT_ID)
WEB_SEARCH_PROCEDURE_ID = "builtin.web_search"
_STUB_INVOKE_API = "builtin.invoke_procedure"


class _ZeroPricing:
    """Offline drain pricing: charge a tiny fixed amount so reconciliation stays valid."""

    def charge_actual(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(credits=0.0)


class LiveSummarizer:
    """Minimal live formalize / finalize for thorough tiers (no Host prompt files)."""

    def __init__(self, credentials: Any) -> None:
        self._credentials = credentials
        self.branch_requests: list[Any] = []
        self.task_requests: list[Any] = []

    async def _chat(self, messages: list[dict[str, Any]]) -> SummaryResult:
        from live_llm import chat_completion

        result = await chat_completion(self._credentials, messages)
        text = str((result or {}).get("response") or "").strip()
        model = str((result or {}).get("model_name") or (result or {}).get("model") or self._credentials.model)
        if not text:
            from lunagentic_research_swarm.llm.gateway import GenerationError

            return SummaryResult(False, "", model, None, GenerationError("summary_empty", "live summarizer 返回空文本"))
        return SummaryResult(True, text, model, None, None)

    async def formalize_task(self, request: Any) -> SummaryResult:
        raw = str(getattr(request, "raw_context", "") or "")
        chat = getattr(request, "chat_messages", ()) or ()
        chat_blob = chat if isinstance(chat, str) else json.dumps(chat, ensure_ascii=False, default=str)
        result = await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是任务形式化助手。把用户目标改写成简短正式任务（Markdown）。"
                        "保留目标中的硬性步骤与引用要求；若目标要求检索，写明须先检索再下结论。"
                        "只输出正式任务正文，不要解释。"
                    ),
                },
                {"role": "user", "content": f"raw_context:\n{raw}\n\nchat:\n{chat_blob}"},
            ]
        )
        if result.success:
            try:
                FormalizedTask.create(result.text)
            except ValueError as exc:
                from lunagentic_research_swarm.llm.gateway import GenerationError

                return SummaryResult(False, "", result.model_name, None, GenerationError("summary_empty", str(exc)))
        return result

    async def finalize_branch(self, request: Any) -> SummaryResult:
        self.branch_requests.append(request)
        history = list(getattr(request, "branch_history", ()) or ())
        formalized = getattr(getattr(request, "formalized_task", None), "text", "") or ""
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是分支总结器。根据分支历史写一段简洁中文摘要。"
                        "必须原样保留工具结果中的关键事实、数字与事实标记字符串；禁止改写或省略这些标记。"
                        "只输出摘要。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"formalized_task:\n{formalized}\n\n"
                        f"branch_history:\n{json.dumps(history, ensure_ascii=False, default=str)}"
                    ),
                },
            ]
        )

    async def finalize_task(self, request: Any) -> SummaryResult:
        self.task_requests.append(request)
        coverage = list(getattr(request, "coverage_summaries", ()) or ())
        formalized = getattr(getattr(request, "formalized_task", None), "text", "") or ""
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是任务最终报告撰写者。综合覆盖摘要写出简短最终报告（中文）。"
                        "报告必须包含检索到的关键事实，并原样粘贴工具结果中的事实标记字符串；"
                        "不要只写「已搜索」而不贴出标记本身。只输出报告正文。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"formalized_task:\n{formalized}\n\n"
                        f"coverage_summaries:\n{json.dumps(coverage, ensure_ascii=False, default=str)}"
                    ),
                },
            ]
        )

    async def compact_branch(self, request: Any) -> SummaryResult:
        history = list(getattr(request, "branch_history", ()) or ())
        formalized = getattr(getattr(request, "formalized_task", None), "text", "") or ""
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": "你是上下文压缩器。把分支历史压成短摘要，保留关键事实。只输出摘要。",
                },
                {
                    "role": "user",
                    "content": (
                        f"formalized_task:\n{formalized}\n\n"
                        f"branch_history:\n{json.dumps(history, ensure_ascii=False, default=str)}"
                    ),
                },
            ]
        )


def _web_search_catalog_entry() -> Any:
    definition = SimpleNamespace(
        procedure_id=WEB_SEARCH_PROCEDURE_ID,
        display_name="网页搜索",
        description=(
            "按 query 检索网页并返回 title/url/snippet。"
            "arguments: query (string, 必填)；engine / max_results 可选。"
            "调研类任务应优先调用本 Procedure 获取证据。"
        ),
        timeout_seconds=30.0,
        idempotent=True,
        arguments_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "engine": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
        enabled=True,
    )
    return SimpleNamespace(
        definition=definition,
        provider_plugin_id="builtin",
        api_name=_STUB_INVOKE_API,
        api_version="1",
        fingerprint=f"stub:{WEB_SEARCH_PROCEDURE_ID}",
    )


def _core_catalog_entry(procedure_id: str) -> Any:
    return SimpleNamespace(
        definition=SimpleNamespace(
            procedure_id=procedure_id,
            display_name=procedure_id,
            description=procedure_id,
            timeout_seconds=30.0,
            idempotent=True,
        ),
        provider_plugin_id="lrs.core",
        api_name=procedure_id,
        api_version="1",
        fingerprint=f"core:{procedure_id}",
    )


def _build_drain_procedure_catalog(*, include_web_search: bool) -> Any:
    entries: dict[str, Any] = {procedure_id: _core_catalog_entry(procedure_id) for procedure_id in _CORE_IDS}
    if include_web_search:
        entries[WEB_SEARCH_PROCEDURE_ID] = _web_search_catalog_entry()
    return _catalog_from_entries(entries)


def _catalog_from_entries(entries: dict[str, Any]) -> Any:
    catalog_ids = tuple(entries)

    def _prompt_entry(entry: Any) -> dict[str, Any]:
        definition = getattr(entry, "definition", entry)
        payload: dict[str, Any] = {
            "procedure_id": str(getattr(definition, "procedure_id", "")),
            "display_name": str(getattr(definition, "display_name", "")),
            "description": str(getattr(definition, "description", "")),
        }
        schema = getattr(definition, "arguments_schema", None)
        if isinstance(schema, Mapping):
            payload["arguments_schema"] = dict(schema)
        return payload

    class _DrainProcedureCatalog:
        fingerprint = "live-drain-procedures"
        ids = catalog_ids

        @property
        def entries(self) -> tuple[Any, ...]:
            return tuple(_prompt_entry(item) for item in entries.values())

        def get(self, procedure_id: str) -> Any | None:
            return entries.get(str(procedure_id))

    return _DrainProcedureCatalog()


def _match_stub_fixture(fixtures: Mapping[str, Any], query: str) -> Any:
    lowered = query.casefold()
    for key, payload in fixtures.items():
        if str(key).casefold() in lowered:
            return payload
    # Query substring miss: empty results (do not leak the first fixture).
    return {"results": []}


def use_stub_procedures(harness: Any, fixtures: Mapping[str, Any]) -> None:
    """Install canned ``builtin.web_search`` + catalog entries before ``start``/formalize."""

    fixture_map = {str(key): value for key, value in dict(fixtures or {}).items()}
    harness._stub_search_fixtures = fixture_map
    harness.stub_search_invokes = 0

    async def _stub_invoke(
        *,
        version: str = "1",
        procedure_id: str,
        request_id: str = "",
        arguments: Mapping[str, Any] | None = None,
        scoped_metadata: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del version, scoped_metadata, _kwargs
        pid = str(procedure_id or "")
        if pid != WEB_SEARCH_PROCEDURE_ID:
            return {
                "success": False,
                "data": None,
                "error": {"code": "procedure_unavailable", "message": f"stub 未实现 {pid}"},
                "metadata": {"request_id": request_id, "procedure_id": pid},
            }
        harness.stub_search_invokes = int(getattr(harness, "stub_search_invokes", 0) or 0) + 1
        args = dict(arguments or {})
        query = str(args.get("query") or "")
        payload = _match_stub_fixture(fixture_map, query)
        if isinstance(payload, Mapping) and "results" in payload:
            data = {
                "engine": str(args.get("engine") or "stub"),
                "query": query,
                "results": list(payload.get("results") or []),
            }
        elif isinstance(payload, Mapping):
            data = {
                "engine": str(args.get("engine") or "stub"),
                "query": query,
                "results": [dict(payload)],
            }
        else:
            data = {"engine": "stub", "query": query, "results": []}
        return {
            "success": True,
            "data": data,
            "error": None,
            "metadata": {"request_id": request_id, "procedure_id": pid, "stub": True},
        }

    harness._effect_local_invokers = {_STUB_INVOKE_API: _stub_invoke}
    _install_web_search_catalog(harness)


def use_live_procedures(harness: Any, web_search_config: Mapping[str, Any] | None = None) -> None:
    """Wire real ``BundledProcedureProvider`` / ``WebSearchService`` into drain local invokers."""

    from lunagentic_research_swarm.config import WebSearchSection
    from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
    from lunagentic_research_swarm.procedures.executor import bundled_procedure_invoker

    raw = dict(web_search_config or {})
    if "enabled_engines" not in raw or not raw.get("enabled_engines"):
        raw["enabled_engines"] = ["duckduckgo"]
    section = WebSearchSection.model_validate(raw)
    provider = BundledProcedureProvider(SimpleNamespace(), web_search_config=section)
    harness._bundled_procedure_provider = provider
    harness.live_search_invokes = 0

    base_invoke = bundled_procedure_invoker(provider)

    async def _live_invoke(
        *,
        version: str = "1",
        procedure_id: str,
        request_id: str = "",
        arguments: Mapping[str, Any] | None = None,
        scoped_metadata: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        pid = str(procedure_id or "")
        if pid == WEB_SEARCH_PROCEDURE_ID:
            harness.live_search_invokes = int(getattr(harness, "live_search_invokes", 0) or 0) + 1
        return await base_invoke(
            version=version,
            procedure_id=procedure_id,
            request_id=request_id,
            arguments=arguments,
            scoped_metadata=scoped_metadata,
            **_kwargs,
        )

    harness._effect_local_invokers = {_STUB_INVOKE_API: _live_invoke}
    # Prefer real web_search schema (engine required + enum) so the live LLM can call correctly.
    live_entry = _web_search_catalog_entry_from_provider(provider)
    catalog = _build_drain_procedure_catalog(include_web_search=True)
    if live_entry is not None:
        entries = {procedure_id: _core_catalog_entry(procedure_id) for procedure_id in _CORE_IDS}
        entries[WEB_SEARCH_PROCEDURE_ID] = live_entry
        catalog = _catalog_from_entries(entries)
    _install_procedure_catalog(harness, catalog)


def _web_search_catalog_entry_from_provider(provider: Any) -> Any | None:
    for payload in list(provider.describe() or []):
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("procedure_id") or "") != WEB_SEARCH_PROCEDURE_ID:
            continue
        definition = SimpleNamespace(
            procedure_id=WEB_SEARCH_PROCEDURE_ID,
            display_name=str(payload.get("display_name") or "网页搜索"),
            description=str(
                payload.get("description")
                or "按指定引擎执行网页搜索。arguments 必填 engine 与 query。"
            ),
            timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            idempotent=bool(payload.get("idempotent", True)),
            arguments_schema=dict(payload.get("arguments_schema") or {}),
            enabled=bool(payload.get("enabled", True)),
        )
        return SimpleNamespace(
            definition=definition,
            provider_plugin_id="builtin",
            api_name=_STUB_INVOKE_API,
            api_version="1",
            fingerprint=f"live:{WEB_SEARCH_PROCEDURE_ID}",
        )
    return None


def _install_web_search_catalog(harness: Any) -> None:
    _install_procedure_catalog(harness, _build_drain_procedure_catalog(include_web_search=True))


def _install_procedure_catalog(harness: Any, catalog: Any) -> None:
    snapshot = getattr(harness, "_shared_snapshot", None)
    if snapshot is None:
        return
    snapshot.procedure_catalog = catalog
    agent_catalog = getattr(snapshot, "agent_catalog", None)
    if agent_catalog is None:
        return

    def resolve_allowed_procedures(_agent_id: str, procedures: Any) -> tuple[str, ...]:
        ids = list(getattr(procedures, "ids", ()) or ())
        for core_id in _CORE_IDS:
            if core_id not in ids:
                ids.append(core_id)
        if WEB_SEARCH_PROCEDURE_ID not in ids:
            ids.append(WEB_SEARCH_PROCEDURE_ID)
        return tuple(ids)

    agent_catalog.resolve_allowed_procedures = resolve_allowed_procedures  # type: ignore[attr-defined]


def _ensure_drain_catalogs(harness: Any) -> None:
    """Patch frozen round snapshot so prepare_* + ProcedureExecutor can run offline."""

    manager = getattr(harness, "manager", None)
    task_id = str(getattr(harness, "task_id", "") or "")
    if manager is None or not task_id:
        raise RuntimeError("drive_until_terminal 需要已 formalize 的 RuntimeHarness（manager/task_id）")
    snapshot = manager._round_snapshots.get(task_id)
    if snapshot is None:
        raise RuntimeError("drive_until_terminal 缺少 frozen round snapshot")

    agent_catalog = getattr(snapshot, "agent_catalog", None)
    if agent_catalog is None:
        raise RuntimeError("round snapshot 缺少 agent_catalog")

    if not callable(getattr(agent_catalog, "resolve_allowed_procedures", None)):

        def resolve_allowed_procedures(_agent_id: str, procedures: Any) -> tuple[str, ...]:
            ids = list(getattr(procedures, "ids", ()) or ())
            for core_id in _CORE_IDS:
                if core_id not in ids:
                    ids.append(core_id)
            return tuple(ids)

        agent_catalog.resolve_allowed_procedures = resolve_allowed_procedures  # type: ignore[attr-defined]

    procedure_catalog = getattr(snapshot, "procedure_catalog", None)
    include_web_search = bool(getattr(harness, "_stub_search_fixtures", None)) or bool(
        getattr(harness, "_effect_local_invokers", None)
    )
    existing_ids = set(getattr(procedure_catalog, "ids", ()) or ()) if procedure_catalog is not None else set()
    needs_catalog = (
        procedure_catalog is None
        or not callable(getattr(procedure_catalog, "get", None))
        or CORE_TERMINATE_ID not in existing_ids
        or (include_web_search and WEB_SEARCH_PROCEDURE_ID not in existing_ids)
    )
    if needs_catalog:
        snapshot.procedure_catalog = _build_drain_procedure_catalog(include_web_search=include_web_search)

    # Integration harness price catalog lacks estimate_model_for_selector; free estimate.
    price_catalog = getattr(snapshot, "price_catalog", None)
    if price_catalog is not None and not callable(getattr(price_catalog, "estimate_model_for_selector", None)):
        snapshot.price_catalog = None


def attach_effect_runner(harness: Any, *, pricing: Any | None = None) -> RuntimeEffectRunner:
    """Build ``TurnWorker`` + ``RuntimeEffectRunner`` bound to ``harness.manager``."""

    if getattr(harness, "manager", None) is None:
        raise RuntimeError("attach_effect_runner 需要已 open/start 的 RuntimeHarness")
    _ensure_drain_catalogs(harness)

    # Production-aligned: core.terminate / compact / checkpoint are control flags from
    # ``split_procedure_requests`` inside ProcedureExecutor.invoke_many — they never call
    # ``local_invokers``. Only ordinary (non-core) procedures use local_invokers / host API
    # (see services._procedure_local_invokers → builtin.invoke_procedure). Offline drain
    # defaults to empty; stub/live thorough tiers attach via use_stub/use_live_procedures.
    local_invokers: dict[str, Any] = dict(getattr(harness, "_effect_local_invokers", None) or {})

    def procedure_factory(catalog: Any) -> ProcedureExecutor:
        return ProcedureExecutor(
            catalog,
            api=harness.procedures,
            summarizer=harness.summarizer,
            local_invokers=local_invokers,
        )

    worker = TurnWorker(
        harness.llm,
        harness.procedures,
        pricing=pricing if pricing is not None else _ZeroPricing(),
        procedure_factory=procedure_factory,
    )
    runner = RuntimeEffectRunner(worker)
    runner.bind_manager(harness.manager)
    return runner


async def _dump_timeout_artifacts(harness: Any, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {}
    try:
        if harness.manager is not None and harness.task_id:
            status = dict(await harness.manager.status(harness.task_id))
            raw = status.get("status")
            if hasattr(raw, "value"):
                status["status"] = raw.value
    except Exception as exc:
        status = {"error": f"{type(exc).__name__}: {exc}"}
    (artifact_dir / "final_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pending_lines: list[str] = []
    for item in list(getattr(getattr(harness, "scheduler", None), "enqueued", []) or []):
        kind = getattr(item, "kind", type(item).__name__)
        pending_lines.append(f"{kind!s} event_id={getattr(item, 'event_id', '')!s}")
    (artifact_dir / "scheduler_pending.txt").write_text(
        "\n".join(pending_lines) + ("\n" if pending_lines else ""),
        encoding="utf-8",
    )
    report_text = ""
    coordinator = getattr(harness, "coordinator", None)
    reports = list(getattr(coordinator, "reports", None) or [])
    if reports:
        last = reports[-1]
        report_text = str(getattr(last, "text", "") or getattr(last, "body", "") or last)
    if report_text:
        (artifact_dir / "last_report.txt").write_text(report_text, encoding="utf-8")
    stub_invokes = getattr(harness, "stub_search_invokes", None)
    if stub_invokes is not None:
        (artifact_dir / "stub_search_invokes.txt").write_text(f"{stub_invokes}\n", encoding="utf-8")
    live_invokes = getattr(harness, "live_search_invokes", None)
    if live_invokes is not None:
        (artifact_dir / "live_search_invokes.txt").write_text(f"{live_invokes}\n", encoding="utf-8")


def _record_branch_summary_reason(harness: Any, effect: Any) -> None:
    """Accumulate PerformBranchSummary.payload['reason'] for terminate-oracle tests."""

    if not isinstance(effect, PerformBranchSummary):
        return
    payload = getattr(effect, "payload", None)
    if not isinstance(payload, Mapping):
        return
    reason = payload.get("reason")
    if reason is None:
        return
    reasons = getattr(harness, "live_drain_branch_summary_reasons", None)
    if not isinstance(reasons, list):
        reasons = []
        harness.live_drain_branch_summary_reasons = reasons
    reasons.append(str(reason))


async def drive_until_terminal(
    harness: Any,
    *,
    timeout_seconds: float,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Drain ``FakeScheduler.enqueued`` through ``RuntimeEffectRunner`` until terminal status."""

    runner = attach_effect_runner(harness)
    deadline = time.monotonic() + float(timeout_seconds)
    harness.live_drain_branch_summary_reasons = []
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
            _record_branch_summary_reason(harness, effect)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(runner.run(effect), timeout=remaining)
            except TimeoutError:
                if artifact_dir is not None:
                    await _dump_timeout_artifacts(harness, Path(artifact_dir))
                raise TimeoutError(
                    f"drive_until_terminal 超时（{timeout_seconds}s）：task_id={harness.task_id!s} "
                    f"pending={len(harness.scheduler.enqueued)}（runner.run 未在剩余时间内完成）"
                ) from None
            continue
        await asyncio.sleep(0.05)
    if artifact_dir is not None:
        await _dump_timeout_artifacts(harness, Path(artifact_dir))
    raise TimeoutError(
        f"drive_until_terminal 超时（{timeout_seconds}s）：task_id={harness.task_id!s} "
        f"pending={len(harness.scheduler.enqueued)}"
    )


__all__ = [
    "LiveSummarizer",
    "WEB_SEARCH_PROCEDURE_ID",
    "attach_effect_runner",
    "drive_until_terminal",
    "use_live_procedures",
    "use_stub_procedures",
]
