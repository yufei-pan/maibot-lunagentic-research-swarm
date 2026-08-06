"""Design §17.2 / config ``[reporting]`` — deliver_intermediate / deliver_final gates.

When a gate is False, the report remains durable (``reports`` row + text) but
Maisaka append/trigger outbox rows are not inserted for that kind.
"""

from __future__ import annotations

from typing import Any

import pytest

from lunagentic_research_swarm.models import ReportKind


async def _outbox_report_ids(harness: Any) -> list[str]:
    """All outbox report_ids for the harness task (any status)."""

    task_id = harness.task_id

    def _query(connection: Any) -> list[str]:
        rows = connection.execute(
            "SELECT report_id FROM maisaka_outbox WHERE task_id = ? ORDER BY created_at, outbox_id",
            (task_id,),
        ).fetchall()
        return [str(row["report_id"] if hasattr(row, "keys") else row[0]) for row in rows]

    return await harness.store.run_locked(_query)


async def _durable_reports(harness: Any) -> list[dict[str, Any]]:
    layer = await harness.store.load_summary_layer(harness.task_id)
    assert layer is not None
    return [dict(item) for item in layer.reports]


@pytest.mark.asyncio
async def test_spec_17_2_deliver_defaults_true_on_coordinator(runtime_harness) -> None:
    """§17.2 — default ``[reporting]`` deliver flags stay True on the coordinator."""

    harness = runtime_harness
    await harness.start("默认交付", credits=40.0, time_budget=30)
    await harness.formalize("正式默认交付")
    assert harness.coordinator is not None
    assert harness.coordinator.deliver_intermediate is True
    assert harness.coordinator.deliver_final is True


@pytest.mark.asyncio
async def test_spec_17_2_deliver_intermediate_false_skips_outbox_keeps_report(runtime_harness) -> None:
    """§17.2 — deliver_intermediate=False: INTERMEDIATE durable, no Maisaka outbox for it."""

    harness = runtime_harness
    harness.configure_runtime_limits(deliver_intermediate=False, deliver_final=True)
    await harness.start("中间不发", credits=100.0, time_budget=120)
    await harness.formalize("正式中间不发")
    assert harness.coordinator is not None
    assert harness.coordinator.deliver_intermediate is False
    assert harness.coordinator.deliver_final is True

    await harness.root_delegates({"A": 50.0, "B": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(120)
    await harness.run_until_idle()

    assert harness.reports[0].kind is ReportKind.INTERMEDIATE
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    reports = await _durable_reports(harness)
    assert reports[0]["kind"] == "INTERMEDIATE"
    assert str(reports[0]["text"]).strip()
    intermediate_id = str(reports[0]["report_id"])

    assert await harness.pending_outbox_count() == 0
    assert await _outbox_report_ids(harness) == []
    assert await harness.deliver_outbox() == 0
    assert harness.maisaka.append_calls == []

    await harness.finalize_all()
    assert harness.reports[-1].kind is ReportKind.FINAL
    kinds = await harness.persisted_report_kinds()
    assert kinds == ["INTERMEDIATE", "FINAL"]
    final_reports = await _durable_reports(harness)
    final = next(item for item in final_reports if item["kind"] == "FINAL")
    assert str(final["text"]).strip()
    final_id = str(final["report_id"])

    outbox_ids = await _outbox_report_ids(harness)
    assert intermediate_id not in outbox_ids
    assert final_id in outbox_ids
    assert await harness.pending_outbox_count() == 1

    delivered = await harness.deliver_outbox()
    assert delivered >= 2
    assert len(harness.maisaka.append_calls) == 1
    assert harness.maisaka.append_calls[0]["message_id"] == f"lrs-report:{final_id}"
    assert len(harness.maisaka.trigger_calls) == 1
    assert harness.maisaka.trigger_calls[0]["metadata"]["report_id"] == final_id


@pytest.mark.asyncio
async def test_spec_17_2_deliver_final_false_skips_outbox_keeps_final_report(runtime_harness) -> None:
    """§17.2 — deliver_final=False: FINAL durable; Maisaka not appended for final."""

    harness = runtime_harness
    harness.configure_runtime_limits(deliver_intermediate=True, deliver_final=False)
    await harness.start("最终不发", credits=100.0, time_budget=120)
    await harness.formalize("正式最终不发")
    assert harness.coordinator is not None
    assert harness.coordinator.deliver_intermediate is True
    assert harness.coordinator.deliver_final is False

    await harness.root_delegates({"A": 50.0, "B": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(120)
    await harness.run_until_idle()

    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    intermediate = (await _durable_reports(harness))[0]
    intermediate_id = str(intermediate["report_id"])
    assert await harness.pending_outbox_count() == 1
    assert await _outbox_report_ids(harness) == [intermediate_id]

    await harness.finalize_all()
    kinds = await harness.persisted_report_kinds()
    assert kinds == ["INTERMEDIATE", "FINAL"]
    final = next(item for item in await _durable_reports(harness) if item["kind"] == "FINAL")
    assert str(final["text"]).strip()
    final_id = str(final["report_id"])

    outbox_ids = await _outbox_report_ids(harness)
    assert outbox_ids == [intermediate_id]
    assert final_id not in outbox_ids
    assert await harness.pending_outbox_count() == 1

    delivered = await harness.deliver_outbox()
    assert delivered >= 2
    assert len(harness.maisaka.append_calls) == 1
    assert harness.maisaka.append_calls[0]["message_id"] == f"lrs-report:{intermediate_id}"
    assert all(
        call["metadata"]["report_id"] != final_id for call in harness.maisaka.trigger_calls
    )


@pytest.mark.asyncio
async def test_spec_17_2_deliver_both_false_no_maisaka_for_any_report(runtime_harness) -> None:
    """§17.2 — both gates False: both reports durable; deliver_once makes no appends."""

    harness = runtime_harness
    harness.configure_runtime_limits(deliver_intermediate=False, deliver_final=False)
    await harness.start("全不发", credits=100.0, time_budget=120)
    await harness.formalize("正式全不发")
    assert harness.coordinator is not None
    assert harness.coordinator.deliver_intermediate is False
    assert harness.coordinator.deliver_final is False

    await harness.root_delegates({"A": 50.0, "B": 50.0})
    await harness.branch_checkpoint("A")
    harness.clock.advance(120)
    await harness.run_until_idle()
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE"]
    assert str((await _durable_reports(harness))[0]["text"]).strip()
    assert await harness.pending_outbox_count() == 0

    await harness.finalize_all()
    assert await harness.persisted_report_kinds() == ["INTERMEDIATE", "FINAL"]
    for report in await _durable_reports(harness):
        assert str(report["text"]).strip()

    assert await _outbox_report_ids(harness) == []
    assert await harness.pending_outbox_count() == 0
    assert await harness.deliver_outbox() == 0
    assert harness.maisaka.append_calls == []
    assert harness.maisaka.trigger_calls == []
