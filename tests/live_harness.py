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
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.reducer import PerformBranchSummary
from lunagentic_research_swarm.runtime.turns import TurnWorker

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


def _match_stub_fixture(fixtures: Mapping[str, Any], query: str) -> Any:
    lowered = query.casefold()
    for key, payload in fixtures.items():
        if str(key).casefold() in lowered:
            return payload
    # Query substring miss: empty results (do not leak the first fixture).
    return {"results": []}


def use_stub_procedures(harness: Any, fixtures: Mapping[str, Any]) -> None:
    """Install canned ``builtin.web_search`` plus the real bundled research catalog."""

    from lunagentic_research_swarm.config import WebSearchSection
    from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
    from lunagentic_research_swarm.procedures.executor import bundled_procedure_invoker

    fixture_map = {str(key): value for key, value in dict(fixtures or {}).items()}
    harness._stub_search_fixtures = fixture_map
    harness.stub_search_invokes = 0

    ctx = _harness_procedure_ctx(harness)
    provider = BundledProcedureProvider(
        ctx,
        web_search_config=WebSearchSection(enabled_engines=["ddgs"]),
        store=getattr(harness, "store", None),
    )
    harness._bundled_procedure_provider = provider
    base_invoke = bundled_procedure_invoker(provider)

    async def _stub_invoke(
        *,
        version: str = "1",
        procedure_id: str,
        request_id: str = "",
        arguments: Mapping[str, Any] | None = None,
        scoped_metadata: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        pid = str(procedure_id or "")
        if pid != WEB_SEARCH_PROCEDURE_ID:
            return await base_invoke(
                version=version,
                procedure_id=procedure_id,
                request_id=request_id,
                arguments=arguments,
                scoped_metadata=scoped_metadata,
                **_kwargs,
            )
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
    catalog = _build_bundled_procedure_catalog(provider)
    harness._live_procedure_catalog = catalog
    _install_procedure_catalog(harness, catalog)


def use_live_procedures(harness: Any, web_search_config: Mapping[str, Any] | None = None) -> None:
    """Wire real ``BundledProcedureProvider`` and freeze its full research catalog."""

    from lunagentic_research_swarm.config import WebSearchSection
    from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider
    from lunagentic_research_swarm.procedures.executor import bundled_procedure_invoker

    raw = dict(web_search_config or {})
    if "enabled_engines" not in raw or not raw.get("enabled_engines"):
        raw["enabled_engines"] = ["ddgs"]
    section = WebSearchSection.model_validate(raw)
    ctx = _harness_procedure_ctx(harness)
    provider = BundledProcedureProvider(
        ctx,
        web_search_config=section,
        store=getattr(harness, "store", None),
    )
    harness._bundled_procedure_provider = provider
    harness.live_search_invokes = 0
    harness._stub_search_fixtures = None

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
    catalog = _build_bundled_procedure_catalog(provider)
    harness._live_procedure_catalog = catalog
    _install_procedure_catalog(harness, catalog)


def _harness_procedure_ctx(harness: Any) -> Any:
    manager = getattr(harness, "manager", None)
    ctx = getattr(manager, "ctx", None) if manager is not None else None
    return ctx if ctx is not None else SimpleNamespace()


def _build_bundled_procedure_catalog(provider: Any) -> Any:
    """Freeze the real bundled research catalog (``allowed_agents`` included).

    ``core.*`` stay outside this registry — system control section + runtime header.
    """

    from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition
    from lunagentic_research_swarm.procedures.registry import ProcedureRegistry

    payloads = [dict(item) for item in list(provider.describe() or []) if isinstance(item, Mapping)]
    by_id = {str(item.get("procedure_id") or ""): item for item in payloads}
    if WEB_SEARCH_PROCEDURE_ID not in by_id:
        fallback = _web_search_fallback_definition()
        payloads.append(fallback.model_dump(mode="json"))
    registry = ProcedureRegistry()
    registry.replace_provider("builtin", payloads)
    return registry.snapshot({})


def _web_search_fallback_definition() -> Any:
    from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition

    return ProcedureDefinition.model_validate(
        {
            "procedure_id": WEB_SEARCH_PROCEDURE_ID,
            "version": "1",
            "display_name": "网页搜索",
            "description": "按指定引擎执行网页搜索。arguments 必填 engine 与 query。",
            "arguments_schema": {
                "type": "object",
                "properties": {
                    "engine": {"type": "string", "enum": ["ddgs"]},
                    "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["engine", "query"],
                "additionalProperties": False,
            },
            "result_schema": {"type": "object"},
            "idempotent": True,
            "timeout_seconds": 30.0,
            "external_cost_kind": "provider_metered",
            "enabled": True,
            "allowed_agents": ["*"],
        }
    )


def _install_web_search_catalog(harness: Any) -> None:
    """Install full bundled catalog (name kept for older call sites)."""

    provider = getattr(harness, "_bundled_procedure_provider", None)
    if provider is None:
        from lunagentic_research_swarm.config import WebSearchSection
        from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider

        provider = BundledProcedureProvider(
            _harness_procedure_ctx(harness),
            web_search_config=WebSearchSection(enabled_engines=["ddgs"]),
            store=getattr(harness, "store", None),
        )
        harness._bundled_procedure_provider = provider
    catalog = _build_bundled_procedure_catalog(provider)
    harness._live_procedure_catalog = catalog
    _install_procedure_catalog(harness, catalog)


def _patch_resolve_allowed_procedures(agent_catalog: Any, *, include_web_search: bool) -> None:
    # Real AgentCatalogSnapshot already resolves via Procedure.allowed_agents (frozen).
    if callable(getattr(agent_catalog, "resolve_allowed_procedures", None)):
        return

    def resolve_allowed_procedures(agent_id: str, procedures: Any) -> tuple[str, ...]:
        resolve = getattr(procedures, "resolve_callable_procedures", None)
        if callable(resolve):
            return tuple(str(item) for item in resolve(agent_id))
        ids = list(getattr(procedures, "ids", ()) or ())
        if include_web_search and WEB_SEARCH_PROCEDURE_ID not in ids:
            ids.append(WEB_SEARCH_PROCEDURE_ID)
        return tuple(str(item) for item in ids)

    agent_catalog.resolve_allowed_procedures = resolve_allowed_procedures  # type: ignore[attr-defined]


def _install_procedure_catalog(harness: Any, catalog: Any, *, snapshot: Any | None = None) -> None:
    snap = snapshot if snapshot is not None else getattr(harness, "_shared_snapshot", None)
    if snap is None:
        return
    snap.procedure_catalog = catalog
    agent_catalog = getattr(snap, "agent_catalog", None)
    if agent_catalog is None:
        return
    _patch_resolve_allowed_procedures(agent_catalog, include_web_search=True)


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

    provider = getattr(harness, "_bundled_procedure_provider", None)
    store = getattr(harness, "store", None)
    if provider is not None and store is not None and hasattr(provider, "bind_case_index"):
        try:
            provider.bind_case_index(store=store)
        except Exception:
            pass

    live_catalog = getattr(harness, "_live_procedure_catalog", None)
    if live_catalog is not None:
        _install_procedure_catalog(harness, live_catalog, snapshot=snapshot)
    else:
        include_ws = bool(getattr(harness, "_stub_search_fixtures", None)) or bool(
            getattr(harness, "_effect_local_invokers", None)
        )
        _patch_resolve_allowed_procedures(agent_catalog, include_web_search=include_ws)
        procedure_catalog = getattr(snapshot, "procedure_catalog", None)
        existing_ids = set(getattr(procedure_catalog, "ids", ()) or ()) if procedure_catalog is not None else set()
        needs_catalog = (
            procedure_catalog is None
            or not callable(getattr(procedure_catalog, "get", None))
            or (include_ws and WEB_SEARCH_PROCEDURE_ID not in existing_ids)
            or not callable(getattr(procedure_catalog, "resolve_callable_procedures", None))
        )
        if needs_catalog and include_ws:
            _install_web_search_catalog(harness)
            live_catalog = getattr(harness, "_live_procedure_catalog", None)
            if live_catalog is not None:
                _install_procedure_catalog(harness, live_catalog, snapshot=snapshot)

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
    auto_advance_clock: bool = True,
) -> dict[str, Any]:
    """Drain ``FakeScheduler.enqueued`` through ``RuntimeEffectRunner`` until terminal status.

    When the queue is idle, optionally advance ``harness.clock`` so real deadline/grace
    timer tasks (``_sleep_until``) can fire — same machinery as production, not injected
    ``ReportDeadlineReached`` / ``GraceExpired`` events.
    """

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
        # Synthesis runs as create_task on the loop, not via FakeScheduler.  Idle
        # with pending=0 must still wait for it or we time out in FINALIZING/
        # REPORTING after the FINAL report is already persisted.
        coordinator = getattr(harness, "coordinator", None) or (
            harness.manager.report_coordinators.get(harness.task_id)
            if getattr(harness, "manager", None) is not None
            else None
        )
        harness.coordinator = coordinator
        synthesis_tasks = getattr(coordinator, "_synthesis_tasks", None) if coordinator is not None else None
        if synthesis_tasks:
            await coordinator.wait_for_synthesis()
            continue
        clock = getattr(harness, "clock", None)
        if auto_advance_clock and clock is not None and callable(getattr(clock, "advance", None)):
            clock.advance(1.0)
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
