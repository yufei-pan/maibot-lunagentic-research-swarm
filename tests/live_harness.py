"""Live effect-drain helpers for RuntimeHarness (FakeLLM offline + LiveLLM later)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lunagentic_research_swarm.models import TaskStatus
from lunagentic_research_swarm.procedures.core import (
    CORE_CHECKPOINT_ID,
    CORE_COMPACT_ID,
    CORE_TERMINATE_ID,
    execute_core_procedure,
)
from lunagentic_research_swarm.procedures.executor import ProcedureExecutor
from lunagentic_research_swarm.runtime.effect_runner import RuntimeEffectRunner
from lunagentic_research_swarm.runtime.turns import TurnWorker

_CORE_IDS = (CORE_TERMINATE_ID, CORE_COMPACT_ID, CORE_CHECKPOINT_ID)


class _ZeroPricing:
    """Offline drain pricing: charge a tiny fixed amount so reconciliation stays valid."""

    def charge_actual(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(credits=0.0)


async def _core_terminate_invoker(
    *,
    version: str = "1",
    procedure_id: str = CORE_TERMINATE_ID,
    request_id: str = "",
    arguments: dict[str, Any] | None = None,
    scoped_metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Local invoker for ``core.terminate`` (control path usually skips API; kept for wiring)."""

    _ = version, scoped_metadata
    result = await execute_core_procedure(str(procedure_id or CORE_TERMINATE_ID), dict(arguments or {}))
    payload = result.model_dump(mode="python")
    metadata = dict(payload.get("metadata") or {})
    if request_id:
        metadata["request_id"] = request_id
    payload["metadata"] = metadata
    return payload


def _core_local_invokers() -> dict[str, Any]:
    return {CORE_TERMINATE_ID: _core_terminate_invoker}


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
    needs_core_catalog = (
        procedure_catalog is None
        or not callable(getattr(procedure_catalog, "get", None))
        or CORE_TERMINATE_ID not in set(getattr(procedure_catalog, "ids", ()) or ())
    )
    if needs_core_catalog:
        core_entries = {
            procedure_id: SimpleNamespace(
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
            for procedure_id in _CORE_IDS
        }

        class _CoreProcedureCatalog:
            fingerprint = "live-drain-core-procedures"
            ids = _CORE_IDS
            entries = tuple(core_entries.values())

            def get(self, procedure_id: str) -> Any | None:
                return core_entries.get(str(procedure_id))

        snapshot.procedure_catalog = _CoreProcedureCatalog()

    # Integration harness price catalog lacks estimate_model_for_selector; free estimate.
    price_catalog = getattr(snapshot, "price_catalog", None)
    if price_catalog is not None and not callable(getattr(price_catalog, "estimate_model_for_selector", None)):
        snapshot.price_catalog = None


def attach_effect_runner(harness: Any, *, pricing: Any | None = None) -> RuntimeEffectRunner:
    """Build ``TurnWorker`` + ``RuntimeEffectRunner`` bound to ``harness.manager``."""

    if getattr(harness, "manager", None) is None:
        raise RuntimeError("attach_effect_runner 需要已 open/start 的 RuntimeHarness")
    _ensure_drain_catalogs(harness)

    local_invokers = _core_local_invokers()
    if CORE_TERMINATE_ID not in local_invokers:
        raise RuntimeError("local_invokers 必须包含 core.terminate")

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


async def drive_until_terminal(
    harness: Any,
    *,
    timeout_seconds: float,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Drain ``FakeScheduler.enqueued`` through ``RuntimeEffectRunner`` until terminal status."""

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
    if artifact_dir is not None:
        await _dump_timeout_artifacts(harness, Path(artifact_dir))
    raise TimeoutError(
        f"drive_until_terminal 超时（{timeout_seconds}s）：task_id={harness.task_id!s} "
        f"pending={len(harness.scheduler.enqueued)}"
    )


__all__ = [
    "attach_effect_runner",
    "drive_until_terminal",
]
