"""Spec §21–22 / §18.2 / §25.4 — config·price reload, pause concurrency, raw privacy.

Complements ``test_lifecycle`` reload cases, ``test_scheduler`` pause caps, and
``test_debug_storage`` with discoverable ``test_spec_*`` names. Offline only.

§21.1 is split: same-manager formalized/User1 invariance vs public reload catalog/limits
on the receiving container — never assert formalized across disconnected stores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF, ON_MODEL_CONFIG_RELOAD
from storage.test_debug_storage import DebugHarness

from lunagentic_research_swarm.config import LRSConfig
from lunagentic_research_swarm.llm.summarizer import SummaryResult
from lunagentic_research_swarm.models import FormalizedTask
from lunagentic_research_swarm.runtime.context import RuntimeHeader, StablePromptBuilder
from lunagentic_research_swarm.runtime.scheduler import FairScheduler
from test_lifecycle import build_container, config_with

from .test_controller_start import FakePriceCatalog, harness  # noqa: F401
from .test_scheduler import FakeWorker, effect


FORMALIZED_TEXT = "不可变正式任务 α\r\n第二行，空格  不合并。"


@pytest.fixture
def fake_worker() -> FakeWorker:
    return FakeWorker()


def _prompt_user1_bytes(formalized: FormalizedTask, *, pricing: dict[str, Any]) -> bytes:
    builder = StablePromptBuilder(
        formalized_task=formalized,
        swarm_identity="麦麦深度调查组",
        bot_profile={"nickname": "麦麦", "personality": "认真", "behavior_style": "求证", "reply_style": "简洁"},
        agent_catalog={"agent.root": {"selector": "task:reasoning", "protocol": "json_envelope", "role": "调查"}},
        procedure_catalog={"builtin.echo": {"description": "回显"}},
        pricing=pricing,
    )
    messages = builder.messages_for_call(
        builder.root_context(coordinator="agent.root"),
        RuntimeHeader("root", 1, "root", 99, 8.0, 1, 0),
    )
    users = [message for message in messages if message["role"] == "user"]
    return str(users[0]["content"]).encode("utf-8")


@pytest.mark.asyncio
async def test_spec_21_1_next_round_snapshot_keeps_inflight_formalized_byte_identical(
    harness: tuple[Any, ...],
) -> None:
    """§21.1 — next-round snapshot mutation leaves in-flight formalized byte-identical.

    Pins the same manager/store/controller that holds the formalized task: User1 from
    ``StablePromptBuilder`` equals formalized bytes under pricing changes, and mutating
    the round snapshot does not rewrite store/controller formalized or issue a second
    ``update_task_formalization``.
    """

    manager, store, summarizer, _scheduler, _message, _config = harness
    snapshot = await manager._snapshot_provider()

    async def formalize_fixed(_request: Any) -> Any:
        summarizer.requests.append(_request)
        await summarizer.gate.wait()
        return SummaryResult(True, FORMALIZED_TEXT, "fake-model", None, None)

    summarizer.formalize_task = formalize_fixed  # type: ignore[method-assign]
    result = await manager.start(objective="调查 reload 边界", stream_id="s-reload", time_budget_seconds=120)
    task_id = result["task_id"]
    summarizer.gate.set()
    await manager.wait_idle(task_id)

    stored = await store.load_task(task_id)
    assert stored is not None and stored.formalized_task is not None
    formalized = stored.formalized_task
    before_bytes = formalized.text.encode("utf-8")
    before_sha = formalized.sha256
    assert before_bytes == FORMALIZED_TEXT.encode("utf-8")
    controller = manager._controllers[task_id]
    assert controller.state.formalized_task is not None
    assert controller.state.formalized_task.text.encode("utf-8") == before_bytes
    formalize_cmds = sum(1 for command in store.commands if getattr(command, "kind", None) == "update_task_formalization")
    assert formalize_cmds == 1

    pricing_before = {"agent.root": {"fingerprint": "price-1", "price_in": 1.0, "price_out": 2.0}}
    user1_before = _prompt_user1_bytes(formalized, pricing=pricing_before)

    # Same-manager next-round fields (what reload would swap for a *new* round).
    snapshot.price_catalog = FakePriceCatalog()
    snapshot.price_catalog.fingerprint = "price-reloaded"  # type: ignore[misc]
    snapshot.default_effort_credits = 999.0
    snapshot.root_default_selector = "model:reloaded"

    pricing_after = {"agent.root": {"fingerprint": "price-2", "price_in": 9.0, "price_out": 9.0}}
    user1_after = _prompt_user1_bytes(formalized, pricing=pricing_after)
    assert user1_before == before_bytes == user1_after

    reloaded = await store.load_task(task_id)
    assert reloaded is not None and reloaded.formalized_task is not None
    assert reloaded.formalized_task.text.encode("utf-8") == before_bytes
    assert reloaded.formalized_task.sha256 == before_sha
    assert controller.state.formalized_task.text.encode("utf-8") == before_bytes
    assert controller.state.formalized_task.sha256 == before_sha
    assert sum(1 for command in store.commands if getattr(command, "kind", None) == "update_task_formalization") == 1


@pytest.mark.asyncio
async def test_spec_21_1_public_reload_updates_catalog_and_live_limits(
    plugin_module: Any,
    tmp_path: Path,
) -> None:
    """§21.1 — public ``on_config_update`` updates price catalog / live limits on *that* container.

    Does not claim in-flight formalized invariance across a disconnected manager store:
    broadcast only mutates the receiving container's next-round catalogs and safety limits.
    """

    container, context, _fake_store, _factory, _events = build_container(
        plugin_module,
        tmp_path,
        snapshot_loader=lambda: {
            "models": [{"name": "m", "price_in": 1.0, "price_out": 2.0}],
            "model_task_config": {"utils": {"model_list": ["m"]}},
        },
    )
    await container.start()
    plugin = plugin_module.create_plugin()
    plugin._set_context(context)
    plugin.set_plugin_config(LRSConfig().model_dump(mode="python"))
    plugin._services = container

    await plugin.on_config_update(
        ON_MODEL_CONFIG_RELOAD,
        {
            "models": [{"name": "m", "price_in": 3.0, "price_out": 4.0}],
            "model_task_config": {"utils": {"model_list": ["m"]}},
        },
        "price-v2",
    )
    assert container.price_catalog.debug_snapshot()["models"]["m"]["price_in"] == 3.0
    assert container.price_catalog.debug_snapshot()["models"]["m"]["price_out"] == 4.0

    updated = config_with(
        plugin={"root_agent": "builtin.quick_thinker"},
        scheduler={"max_global_llm_concurrency": 4},
        budget={"default_effort_credits": 250.0},
    ).model_dump(mode="python")
    await plugin.on_config_update(CONFIG_RELOAD_SCOPE_SELF, updated, "self-v2")
    assert plugin._services.safety_limits["max_global_llm_concurrency"] == 4
    assert plugin._services.health()["config_reload"]["code"] == "live_limits_updated"
    assert "next_round" in plugin._services.health()["config_reload"]
    after = await container.snapshot_next_round()
    assert after.default_effort_credits == 250.0
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_spec_22_pause_honors_per_task_caps_and_does_not_starve_other_task(
    fake_worker: FakeWorker,
) -> None:
    """§22 — under pause, new agent/summarizer block; procedure still capped; other tasks keep turns."""

    fake_worker.block = True
    scheduler = FairScheduler(global_llm=2, per_task_llm=2, per_task_procedure=1, worker=fake_worker)
    await scheduler.start()

    await scheduler.enqueue(effect("A", "a-inflight", kind="agent"))
    await fake_worker.wait_started(1)
    scheduler.pause_task("A")

    await scheduler.enqueue(effect("A", "a-agent-blocked", kind="agent"))
    await scheduler.enqueue(effect("A", "a-summary-blocked", kind="summarizer"))
    await scheduler.enqueue(effect("A", "a-proc-0", kind="procedure"))
    await scheduler.enqueue(effect("A", "a-proc-1", kind="procedure"))
    await scheduler.enqueue(effect("B", "b0", kind="agent"))

    await fake_worker.wait_started(3)
    assert "a-inflight" in fake_worker.started
    assert "a-proc-0" in fake_worker.started
    assert "b0" in fake_worker.started
    assert "a-agent-blocked" not in fake_worker.started
    assert "a-summary-blocked" not in fake_worker.started
    assert "a-proc-1" not in fake_worker.started  # per-task procedure cap = 1

    snapshot = scheduler.stats()
    assert snapshot["tasks"]["A"]["procedure_active"] == 1
    assert snapshot["tasks"]["B"]["llm_active"] == 1

    fake_worker.release.set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_spec_18_2_default_config_leaves_no_raw_agent_transcript_in_summary_layer(
    tmp_path: Path,
) -> None:
    """§18.2 / §25.4 — default storage toggles off; durable summary layer has no raw agent transcript."""

    config = LRSConfig()
    assert config.storage.store_agent_transcripts is False
    assert config.storage.store_raw_procedure_payloads is False

    debug_harness = DebugHarness(tmp_path)
    await debug_harness.open()
    try:
        await debug_harness.run_one_turn(
            transcripts=config.storage.store_agent_transcripts,
            payloads=config.storage.store_raw_procedure_payloads,
        )
        assert not debug_harness.has_transcript_rows()
        assert not debug_harness.has_payload_rows()
        assert not debug_harness.debug_dir().exists()

        layer = await debug_harness.store.load_summary_layer("lrs_debug")
        assert layer is not None
        encoded = json.dumps(
            {
                "formalized": layer.formalized_task.text if layer.formalized_task else "",
                "summaries": [dict(item) for item in layer.summaries],
                "reports": [dict(item) for item in layer.reports],
            },
            ensure_ascii=False,
            default=str,
        )
        assert "raw-agent-secret" not in encoded
        assert "raw-procedure-secret" not in encoded
        assert not debug_harness.vector_text_contains("raw-agent-secret")
    finally:
        await debug_harness.close()
