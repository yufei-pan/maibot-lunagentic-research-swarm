"""`/swarm` 用户命令：中文紧凑状态、统计与维护输出。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from maibot_sdk import Command

from lunagentic_research_swarm.storage.sqlite import BUSY_STATUSES

_FETCH_PROCEDURE_ID = "fetch_url.fetch"
_FETCH_PROVIDER_ID = "com.0-hz.fetch-url"
_BUSY_STATUS_VALUES = frozenset(status.value for status in BUSY_STATUSES)


def _groups(kwargs: Mapping[str, Any]) -> dict[str, str]:
    raw = kwargs.get("matched_groups")
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items() if v is not None and str(v) != ""}
    return {}


def _require_stream_id(kwargs: Mapping[str, Any]) -> tuple[str | None, str | None]:
    stream_id = kwargs.get("stream_id")
    if not isinstance(stream_id, str) or not stream_id.strip():
        return None, "缺少 stream_id，无法回复命令结果"
    return stream_id.strip(), None


def clip_command_output(text: str, max_chars: int, *, error_lines: Sequence[str] = ()) -> str:
    """超限时做结构化截断，并保留错误条目摘要（不静默省略）。

    始终保证 ``len(result) <= max(1000, max_chars)``。错误摘要带
    ``共 N / 显示 K`` 计数；装不下的错误条目被省略时也会写明总数与显示数。
    """

    limit = max(1000, int(max_chars))
    body = str(text or "")
    errors = [str(line) for line in error_lines if str(line).strip()]
    total = len(errors)

    def _error_footer(shown: int) -> str:
        header = f"—— 错误摘要（共 {total} / 显示 {shown}）——"
        if shown <= 0:
            return "\n" + header
        lines = "\n".join(f"- {errors[i]}" for i in range(shown))
        return "\n" + header + "\n" + lines

    def _truncate_body(room: int) -> str:
        """在 ``room`` 字符预算内放入正文（必要时加截断头）。"""
        if room <= 0:
            return ""
        if len(body) <= room:
            return body
        header = f"……（输出已截断，原文约 {len(body)} 字，上限 {limit}）\n"
        if len(header) + 1 > room:
            # 预算极紧：尽量保留原文前缀
            return body[:room]
        keep = max(0, room - len(header) - 1)
        return header + body[:keep] + "…"

    if total == 0:
        if len(body) <= limit:
            return body
        return _truncate_body(limit)

    # 优先为错误摘要预留空间：取最大可显示条数 K，再把剩余预算给正文。
    footer = _error_footer(0)
    for candidate in range(total, -1, -1):
        trial = _error_footer(candidate)
        if len(trial) <= limit:
            footer = trial
            break
    else:
        # 理论上 K=0 的 footer 很短；若仍超限则硬截断 footer 本身。
        return footer[:limit]

    room = limit - len(footer)
    prefix = _truncate_body(room) if room > 0 else ""
    if not prefix:
        # 无正文空间时去掉 footer 前导换行，避免浪费首字符。
        return (footer[1:] if footer.startswith("\n") else footer)[:limit]
    out = prefix + footer
    return out if len(out) <= limit else out[:limit]


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.{digits}f}"


def format_plugin_overview(*, root_agent: str, running: Sequence[Mapping[str, Any]], health: Mapping[str, Any]) -> str:
    lines = [
        "麦麦深度调查组 · 概要",
        f"根智能体：{root_agent or '-'}",
        f"健康：sqlite={_status_of(health.get('sqlite'))} vector={_status_of(health.get('vector_index'))} "
        f"pinning={_status_of(health.get('physical_pinning'))}",
        f"运行中任务：{len(running)}",
    ]
    for item in running[:20]:
        lines.append(
            f"- {item.get('task_id')}  {item.get('status')}  round={item.get('round_number')}  "
            f"leaves={len(item.get('active_leaves') or [])}"
        )
    if len(running) > 20:
        lines.append(f"……另有 {len(running) - 20} 个运行中任务")
    return "\n".join(lines)


def format_task_status(
    status: Mapping[str, Any],
    *,
    stats: Mapping[str, Any] | None = None,
    queue: Mapping[str, Any] | None = None,
    deadline: Any = None,
    reports: int | None = None,
) -> str:
    leaves = status.get("active_leaves") or []
    credits = sum(float(leaf.get("credits") or 0) for leaf in leaves if isinstance(leaf, Mapping))
    lines = [
        f"任务 {status.get('task_id')}",
        f"生命周期：{status.get('status')}",
        f"round：{status.get('round_number')}（{status.get('round_id') or '-'}） generation={status.get('generation')}",
        f"活动叶子：{len(leaves)}  credits合计={_fmt_num(credits)}",
    ]
    if queue:
        lines.append(
            f"调度：active={queue.get('active', 0)} queued={queue.get('queued', 0)} "
            f"paused={queue.get('paused', False)}"
        )
    if deadline is not None:
        lines.append(f"deadline：{deadline}")
    if stats:
        lines.extend(
            [
                f"credits：pool={_fmt_num(stats.get('credit_pool'))} actual={_fmt_num(stats.get('actual_credits'))} "
                f"est={_fmt_num(stats.get('estimated_credits'))} unrec={_fmt_num(stats.get('unreconciled_credits'))}",
                f"tokens：prompt={stats.get('prompt_tokens', 0)} completion={stats.get('completion_tokens', 0)} "
                f"cache_hit={stats.get('cache_hit_tokens', 0)} miss={stats.get('cache_miss_tokens', 0)} "
                f"hit_rate={_fmt_rate(stats.get('cache_hit_rate'))}",
                f"分支：total={stats.get('branches_total', 0)} active={stats.get('branches_active', 0)} "
                f"finalized={stats.get('branches_finalized', 0)}",
                f"procedure成功/失败：{stats.get('procedures_success', 0)}/{stats.get('procedures_error', 0)} "
                f"错误计数={stats.get('error_count', 0)}",
            ]
        )
    if reports is not None:
        lines.append(f"报告数：{reports}")
    return "\n".join(lines)


def format_task_list(tasks: Sequence[Mapping[str, Any]], *, status_filter: str | None = None) -> str:
    rows = list(tasks)
    if status_filter:
        rows = [item for item in rows if str(item.get("status") or "") == status_filter]
    lines = [f"任务列表（{len(rows)}）" + (f" filter={status_filter}" if status_filter else "")]
    if not rows:
        lines.append("（无匹配任务）")
        return "\n".join(lines)
    for item in rows[:50]:
        lines.append(
            f"- {item.get('task_id')}  {item.get('status')}  round={item.get('round_number')}  "
            f"created={item.get('created_at') or '-'}"
        )
    if len(rows) > 50:
        lines.append(f"……另有 {len(rows) - 50} 条（已截断列表）")
    return "\n".join(lines)


def format_task_stats(stats: Mapping[str, Any]) -> str:
    lines = [
        f"统计 · 任务 {stats.get('task_id')}",
        f"agent/summarizer 调用：{stats.get('agent_calls', 0)}/{stats.get('summarizer_calls', 0)}",
        f"prompt_tokens={stats.get('prompt_tokens', 0)} completion_tokens={stats.get('completion_tokens', 0)}",
        f"cache_hit={stats.get('cache_hit_tokens', 0)} cache_miss={stats.get('cache_miss_tokens', 0)} "
        f"hit_rate={_fmt_rate(stats.get('cache_hit_rate'))}",
        f"credits actual={_fmt_num(stats.get('actual_credits'))} estimated={_fmt_num(stats.get('estimated_credits'))} "
        f"unreconciled={_fmt_num(stats.get('unreconciled_credits'))} "
        f"cost_equivalent={_fmt_num(stats.get('cost_equivalent_credits'))}",
        f"pool={_fmt_num(stats.get('credit_pool'))} debt={_fmt_num(stats.get('credit_debt'))}",
        f"branches total/active/finalized={stats.get('branches_total', 0)}/"
        f"{stats.get('branches_active', 0)}/{stats.get('branches_finalized', 0)} "
        f"max_depth={stats.get('max_branch_depth', 0)}",
        f"compact/checkpoint/continue={stats.get('compact_count', 0)}/"
        f"{stats.get('checkpoint_count', 0)}/{stats.get('continue_count', 0)}",
        f"procedure 成功/失败={stats.get('procedures_success', 0)}/{stats.get('procedures_error', 0)} "
        f"错误={stats.get('error_count', 0)} duration_ms={stats.get('duration_ms_total', 0)}",
    ]
    return "\n".join(lines)


def format_plugin_stats(stats: Mapping[str, Any]) -> str:
    models = stats.get("models") or {}
    agents = stats.get("agents") or {}
    procedures = stats.get("procedures") or {}
    tasks = stats.get("tasks") or {}
    lines = [
        "统计 · 插件聚合",
        f"模型数={len(models)} 智能体数={len(agents)} Procedure数={len(procedures)} 任务数={len(tasks)}",
    ]
    for name, bucket in sorted(models.items(), key=lambda item: item[0])[:20]:
        if not isinstance(bucket, Mapping):
            continue
        lines.append(
            f"- model {name}: calls={bucket.get('calls', 0)} "
            f"prompt={bucket.get('prompt_tokens', 0)} completion={bucket.get('completion_tokens', 0)} "
            f"credits={_fmt_num(bucket.get('actual_credits'))}"
        )
    for name, bucket in sorted(agents.items(), key=lambda item: item[0])[:20]:
        if not isinstance(bucket, Mapping):
            continue
        lines.append(f"- agent {name}: branches={bucket.get('branches', 0)} finalized={bucket.get('finalized', 0)}")
    for name, bucket in sorted(procedures.items(), key=lambda item: item[0])[:20]:
        if not isinstance(bucket, Mapping):
            continue
        lines.append(
            f"- procedure {name}: calls={bucket.get('calls', 0)} "
            f"成功={bucket.get('success', 0)} 失败={bucket.get('error', 0)}"
        )
    omitted = max(0, len(models) - 20) + max(0, len(agents) - 20) + max(0, len(procedures) - 20)
    if omitted:
        lines.append(f"……另有 {omitted} 条分布未展开")
    return "\n".join(lines)


def format_agents(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"智能体目录（{len(rows)}）"]
    for row in rows:
        lines.append(
            f"- {row.get('agent_id')}  provider={row.get('provider')}  enabled={row.get('enabled')}  "
            f"selector={row.get('selector')}  protocol={row.get('protocol')}  health={row.get('health')}"
        )
    return "\n".join(lines)


def format_procedures(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"Procedure 目录（{len(rows)}）"]
    for row in rows:
        lines.append(
            f"- {row.get('procedure_id')}  provider={row.get('provider')}  enabled={row.get('enabled')}  "
            f"health={row.get('health')}"
        )
    return "\n".join(lines)


_HEALTH_BAD_STATUSES = frozenset({"degraded", "failed", "invalid", "recommended_missing", "unavailable", "critical"})


def collect_health_errors(payload: Mapping[str, Any]) -> list[str]:
    """收集顶层与 extension_providers 嵌套中的不健康项，供 clip 错误脚注。"""

    errors: list[str] = []
    for key, value in payload.items():
        if key == "extension_providers" and isinstance(value, Mapping):
            for kind, kind_map in value.items():
                if not isinstance(kind_map, Mapping):
                    continue
                for provider_id, status in sorted(kind_map.items(), key=lambda item: str(item[0])):
                    if not isinstance(status, Mapping):
                        continue
                    if str(status.get("status") or "") in _HEALTH_BAD_STATUSES:
                        code = status.get("code") or status.get("status")
                        errors.append(f"extension_providers.{kind}.{provider_id}: {code}")
            continue
        if isinstance(value, Mapping) and str(value.get("status") or "") in _HEALTH_BAD_STATUSES:
            code = value.get("code") or value.get("status")
            errors.append(f"{key}: {code}")
    return errors


def format_health(payload: Mapping[str, Any]) -> str:
    lines = ["健康检查"]
    for key in (
        "sqlite",
        "vector_index",
        "physical_pinning",
        "extension_discovery",
        "maisaka_outbox",
        "root_agent",
        "root_selector",
        "summarizer_selector",
        "config_reload",
    ):
        if key in payload:
            lines.append(f"- {key}: {_compact_status(payload.get(key))}")
    providers = payload.get("extension_providers") or {}
    if isinstance(providers, Mapping) and providers:
        lines.append("- extension_providers:")
        for kind in ("agents", "procedures"):
            kind_map = providers.get(kind) or {}
            if not isinstance(kind_map, Mapping) or not kind_map:
                continue
            for provider_id, status in sorted(kind_map.items(), key=lambda item: str(item[0])):
                lines.append(f"  - {kind}/{provider_id}: {_compact_status(status)}")
    fetch = payload.get("recommended_fetch") or {}
    lines.append(f"- recommended_fetch: {_compact_status(fetch)}")
    queue = payload.get("queue") or {}
    if queue:
        lines.append(
            f"- queue: active={queue.get('active', 0)} queued={queue.get('queued', 0)}"
        )
    reminder = payload.get("reminder") or {}
    if reminder:
        lines.append(f"- reminder: {_compact_status(reminder)}")
    errors = payload.get("errors") or []
    if errors:
        lines.append("错误：")
        for item in errors:
            lines.append(f"  ! {item}")
    return "\n".join(lines)


def format_vector_status(status: Any, jobs: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "向量索引状态",
        f"idle={getattr(status, 'idle', None)} rebuilding={getattr(status, 'rebuilding', None)}",
        f"active_generation={getattr(status, 'active_generation', None)} dimension={getattr(status, 'dimension', None)}",
        f"selector={getattr(status, 'selector', None)} fingerprint={getattr(status, 'model_fingerprint', None)}",
        f"failed_candidate={getattr(status, 'failed_candidate', None)} "
        f"retired={getattr(status, 'retired_generations', ())}",
    ]
    err_code = getattr(status, "last_error_code", None)
    if err_code:
        lines.append(f"last_error={err_code}: {getattr(status, 'last_error_message', '')}")
    lines.append(f"jobs={len(jobs)}")
    for job in list(jobs)[:10]:
        lines.append(
            f"- {job.get('job_id')} {job.get('status')} {job.get('source_kind')}:{job.get('source_id')} "
            f"err={job.get('error_code') or '-'}"
        )
    return "\n".join(lines)


def format_vector_rebuild_result(result: Any) -> tuple[str, list[str]]:
    success = bool(getattr(result, "success", False))
    code = getattr(result, "code", None)
    data = getattr(result, "data", None) or {}
    error = getattr(result, "error", None)
    errors: list[str] = []
    if success:
        if code == "already_current":
            return "向量索引已是当前 generation，无需重建。", errors
        if code == "empty":
            return "无可索引语料，未创建向量 generation。", errors
        generation = data.get("generation") if isinstance(data, Mapping) else None
        if generation is not None:
            return (
                f"已创建新的向量 generation {generation}（dimension={data.get('dimension')} count={data.get('count')}）",
                errors,
            )
        return f"向量重建完成（code={code}）。", errors
    message = getattr(error, "message", None) or str(error or "向量重建失败")
    err_code = getattr(error, "code", None) or code or "vector_rebuild_failed"
    errors.append(f"{err_code}: {message}")
    return f"向量重建失败：{message}", errors


def _status_of(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("status") or value.get("code") or "unknown")
    return str(value or "unknown")


def _compact_status(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    parts = [f"status={value.get('status', 'unknown')}"]
    for key in ("code", "agent_id", "idle", "active_generation", "reminders_enabled", "detail"):
        if key in value and value[key] is not None:
            parts.append(f"{key}={value[key]}")
    return " ".join(parts)


def _list_agent_rows(services: Any) -> list[dict[str, Any]]:
    registry = getattr(services, "agent_registry", None)
    if registry is None:
        return []
    health_map = dict(getattr(registry, "health", {}) or {})
    rows: list[dict[str, Any]] = []
    providers = getattr(registry, "_providers", {}) or {}
    for provider_id, batch in providers.items():
        health = health_map.get(provider_id)
        health_status = getattr(health, "status", None) or (
            health.get("status") if isinstance(health, Mapping) else "unknown"
        )
        for definition in getattr(batch, "definitions", ()) or ():
            rows.append(
                {
                    "agent_id": getattr(definition, "agent_id", None),
                    "provider": provider_id,
                    "enabled": bool(getattr(definition, "enabled", True)),
                    "selector": getattr(definition, "model_selector", ""),
                    "protocol": getattr(definition, "protocol", ""),
                    "health": health_status,
                }
            )
    rows.sort(key=lambda item: str(item.get("agent_id") or ""))
    return rows


def _list_procedure_rows(services: Any) -> list[dict[str, Any]]:
    registry = getattr(services, "procedure_registry", None)
    if registry is None:
        return []
    health_map = dict(getattr(registry, "health", {}) or {})
    rows: list[dict[str, Any]] = []
    providers = getattr(registry, "_providers", {}) or {}
    for provider_id, batch in providers.items():
        health = health_map.get(provider_id)
        health_status = getattr(health, "status", None) or (
            health.get("status") if isinstance(health, Mapping) else "unknown"
        )
        for definition in getattr(batch, "definitions", ()) or ():
            rows.append(
                {
                    "procedure_id": getattr(definition, "procedure_id", None),
                    "provider": provider_id,
                    "enabled": bool(getattr(definition, "enabled", True)),
                    "health": health_status,
                }
            )
    rows.sort(key=lambda item: str(item.get("procedure_id") or ""))
    return rows


def _recommended_fetch_status(services: Any) -> dict[str, Any]:
    registry = getattr(services, "procedure_registry", None)
    if registry is None:
        return {"status": "recommended_missing", "code": "procedure_registry_unavailable", "detail": _FETCH_PROCEDURE_ID}
    if hasattr(registry, "is_live") and callable(registry.is_live):
        if registry.is_live(_FETCH_PROCEDURE_ID):
            return {"status": "healthy", "detail": _FETCH_PROCEDURE_ID}
    # fallback: scan providers
    for row in _list_procedure_rows(services):
        if row.get("procedure_id") == _FETCH_PROCEDURE_ID and row.get("enabled"):
            return {"status": "healthy", "detail": _FETCH_PROCEDURE_ID, "provider": row.get("provider")}
    health_map = dict(getattr(registry, "health", {}) or {})
    provider_health = health_map.get(_FETCH_PROVIDER_ID)
    if provider_health is not None:
        status = getattr(provider_health, "status", None) or (
            provider_health.get("status") if isinstance(provider_health, Mapping) else "unknown"
        )
        if status == "invalid":
            return {"status": "degraded", "code": "fetch_provider_invalid", "detail": _FETCH_PROCEDURE_ID}
    return {"status": "recommended_missing", "code": "fetch_url_missing", "detail": _FETCH_PROCEDURE_ID}


def _queue_snapshot(services: Any) -> dict[str, Any]:
    scheduler = getattr(services, "scheduler", None)
    stats_fn = getattr(scheduler, "stats", None) if scheduler is not None else None
    if not callable(stats_fn):
        return {}
    try:
        payload = stats_fn()
    except Exception:
        return {"status": "degraded", "code": "scheduler_stats_failed"}
    global_stats = {}
    if isinstance(payload, Mapping):
        global_stats = dict(payload.get("global") or {})
        if not global_stats:
            # aggregate from tasks
            active = queued = 0
            for item in (payload.get("tasks") or {}).values():
                if isinstance(item, Mapping):
                    active += int(item.get("active") or 0)
                    queued += int(item.get("queued") or 0)
            global_stats = {"active": active, "queued": queued}
    return global_stats


def _reminder_snapshot(services: Any) -> dict[str, Any]:
    feedback = getattr(services, "feedback", None)
    status = getattr(services, "_status", {}) or {}
    feedback_status = dict(status.get("feedback") or {}) if isinstance(status, Mapping) else {}
    enabled = bool(getattr(feedback, "reminders_enabled", feedback_status.get("reminders_enabled", False)))
    return {
        "status": feedback_status.get("status") or ("healthy" if feedback is not None else "unknown"),
        "reminders_enabled": enabled,
    }


def _commands_config(plugin: Any) -> Any:
    services = getattr(plugin, "_services", None)
    if services is not None:
        config = getattr(services, "_config", None)
        if config is not None and getattr(config, "commands", None) is not None:
            return config.commands
    try:
        config = plugin.config
        return getattr(config, "commands", None)
    except Exception:
        return None


def _max_output_chars(plugin: Any) -> int:
    section = _commands_config(plugin)
    value = getattr(section, "max_output_chars", 12000) if section is not None else 12000
    try:
        return max(1000, int(value))
    except (TypeError, ValueError):
        return 12000


def _commands_enabled(plugin: Any) -> bool:
    section = _commands_config(plugin)
    if section is None:
        return True
    return bool(getattr(section, "enabled", True))


def _allow_vector_rebuild(plugin: Any) -> bool:
    section = _commands_config(plugin)
    if section is None:
        return False
    return bool(getattr(section, "allow_vector_rebuild", False))


def _maintenance_allowlist(plugin: Any) -> list[str]:
    """维护白名单：优先 `maintenance_allowed_user_ids`（Host user_id），兼容旧字段名。"""

    section = _commands_config(plugin)
    if section is None:
        return []
    raw = getattr(section, "maintenance_allowed_user_ids", None)
    if raw is None:
        raw = getattr(section, "maintenance_allowed_person_ids", None)
    return [str(item).strip() for item in list(raw or []) if str(item).strip()]


def _maintenance_allowed(plugin: Any, kwargs: Mapping[str, Any]) -> bool:
    """空白名单 = 不限制；非空时与 Host 命令 RPC 的 `user_id` 对齐。"""

    allowed = _maintenance_allowlist(plugin)
    if not allowed:
        return True
    message = kwargs.get("message") if isinstance(kwargs.get("message"), Mapping) else {}
    candidates = [
        kwargs.get("user_id"),
        message.get("user_id") if isinstance(message, Mapping) else None,
        kwargs.get("person_id"),
    ]
    allowed_set = set(allowed)
    for value in candidates:
        if isinstance(value, str) and value.strip() and value.strip() in allowed_set:
            return True
    return False


class SwarmCommandsMixin:
    """九个 `/swarm` 正则命令；由插件类混入。"""

    async def _swarm_send(self, text: str, stream_id: str, *, errors: Sequence[str] = ()) -> tuple[bool, str, bool]:
        clipped = clip_command_output(text, _max_output_chars(self), error_lines=errors)
        await self.ctx.send.text(clipped, stream_id)
        return True, clipped, True

    async def _swarm_fail(self, message: str, stream_id: str | None) -> tuple[bool, str, bool]:
        if stream_id:
            await self.ctx.send.text(message, stream_id)
        return False, message, True

    def _swarm_services(self) -> Any | None:
        return getattr(self, "_services", None)

    @Command(
        "swarm_status",
        description="查看深度调查任务状态或插件运行概要",
        pattern=r"^/swarm\s+status(?:\s+(?P<task_id>\S+))?$",
    )
    async def swarm_status(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        groups = _groups(kwargs)
        task_id = groups.get("task_id")
        services = self._swarm_services()
        manager = getattr(self, "_manager", None) or getattr(services, "manager", None)
        if manager is None:
            return await self._swarm_fail("研究运行时尚未初始化", stream_id)
        try:
            if not task_id:
                tasks = await manager.list_tasks(stream_id=stream_id)
                running = [
                    item
                    for item in tasks
                    if isinstance(item, Mapping)
                    and str(item.get("status") or "") in _BUSY_STATUS_VALUES
                ]
                health = services.health() if services is not None and hasattr(services, "health") else {}
                root = ""
                if services is not None:
                    config = getattr(services, "_config", None)
                    root = str(getattr(getattr(config, "plugin", None), "root_agent", "") or "")
                text = format_plugin_overview(root_agent=root, running=running, health=health)
                return await self._swarm_send(text, stream_id)
            status = await manager.status(task_id, stream_id=stream_id)
            stats = None
            if services is not None and getattr(services, "statistics", None) is not None:
                stats = await services.statistics.task(task_id)
            queue = None
            scheduler = getattr(services, "scheduler", None) if services is not None else None
            if scheduler is not None and callable(getattr(scheduler, "stats", None)):
                queue = dict((scheduler.stats().get("tasks") or {}).get(task_id) or {})
            deadline = None
            reports: int | None = None
            coordinator = (
                getattr(manager, "report_coordinators", {}).get(task_id)
                if hasattr(manager, "report_coordinators")
                else None
            )
            if coordinator is not None:
                deadline = getattr(coordinator, "deadline_at", None)
                coord_reports = getattr(coordinator, "reports", None)
                if coord_reports is not None:
                    reports = len(coord_reports)
            if reports is None:
                store = getattr(manager, "store", None)
                if store is None and services is not None:
                    store = getattr(services, "store", None)
                if store is not None and callable(getattr(store, "load_summary_layer", None)):
                    layer = await store.load_summary_layer(task_id)
                    if layer is not None:
                        reports = len(getattr(layer, "reports", ()) or [])
                    else:
                        reports = 0
                else:
                    reports = 0
            text = format_task_status(
                status, stats=stats, queue=queue, deadline=deadline, reports=reports
            )
            return await self._swarm_send(text, stream_id)
        except PermissionError as exc:
            return await self._swarm_fail(str(exc), stream_id)
        except LookupError as exc:
            return await self._swarm_fail(str(exc), stream_id)
        except Exception as exc:
            return await self._swarm_fail(f"查询状态失败：{exc}", stream_id)

    @Command(
        "swarm_tasks",
        description="列出最近深度调查任务",
        pattern=r"^/swarm\s+tasks(?:\s+(?P<status>\S+))?$",
    )
    async def swarm_tasks(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        services = self._swarm_services()
        manager = getattr(self, "_manager", None) or getattr(services, "manager", None)
        if manager is None:
            return await self._swarm_fail("研究运行时尚未初始化", stream_id)
        status_filter = _groups(kwargs).get("status")
        try:
            tasks = await manager.list_tasks(stream_id=stream_id)
            text = format_task_list(tasks, status_filter=status_filter)
            return await self._swarm_send(text, stream_id)
        except Exception as exc:
            return await self._swarm_fail(f"列出任务失败：{exc}", stream_id)

    @Command(
        "swarm_stats",
        description="查看任务或插件聚合统计",
        pattern=r"^/swarm\s+stats(?:\s+(?P<task_id>\S+))?$",
    )
    async def swarm_stats(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        services = self._swarm_services()
        statistics = getattr(services, "statistics", None) if services is not None else None
        if statistics is None:
            return await self._swarm_fail("统计服务尚未初始化", stream_id)
        task_id = _groups(kwargs).get("task_id")
        try:
            if task_id:
                text = format_task_stats(await statistics.task(task_id))
            else:
                text = format_plugin_stats(await statistics.plugin())
            return await self._swarm_send(text, stream_id)
        except Exception as exc:
            return await self._swarm_fail(f"查询统计失败：{exc}", stream_id)

    @Command(
        "swarm_agents",
        description="列出 live 智能体目录（不含 prompt/密钥）",
        pattern=r"^/swarm\s+agents$",
    )
    async def swarm_agents(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        services = self._swarm_services()
        if services is None:
            return await self._swarm_fail("基础服务尚未初始化", stream_id)
        rows = _list_agent_rows(services)
        return await self._swarm_send(format_agents(rows), stream_id)

    @Command(
        "swarm_procedures",
        description="列出 live Procedure 目录",
        pattern=r"^/swarm\s+procedures$",
    )
    async def swarm_procedures(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        services = self._swarm_services()
        if services is None:
            return await self._swarm_fail("基础服务尚未初始化", stream_id)
        rows = _list_procedure_rows(services)
        return await self._swarm_send(format_procedures(rows), stream_id)

    @Command(
        "swarm_health",
        description="查看 SQLite/向量/pinning/扩展/推荐 fetch/队列/outbox/提醒健康度",
        pattern=r"^/swarm\s+health$",
    )
    async def swarm_health(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        services = self._swarm_services()
        if services is None or not hasattr(services, "health"):
            return await self._swarm_fail("基础服务尚未初始化", stream_id)
        try:
            refresh = getattr(services, "refresh_vector_index_health", None)
            if callable(refresh):
                await refresh()
            payload = dict(services.health())
        except Exception as exc:
            return await self._swarm_fail(f"读取健康状态失败：{exc}", stream_id)
        payload["recommended_fetch"] = _recommended_fetch_status(services)
        payload["queue"] = _queue_snapshot(services)
        payload["reminder"] = _reminder_snapshot(services)
        errors = collect_health_errors(payload)
        text = format_health(payload)
        return await self._swarm_send(text, stream_id, errors=errors)

    @Command(
        "swarm_vectors_status",
        description="查看向量索引 generation/selector/job 状态",
        pattern=r"^/swarm\s+vectors\s+status$",
    )
    async def swarm_vectors_status(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        services = self._swarm_services()
        vector = getattr(services, "vector_index", None) if services is not None else None
        if vector is None:
            return await self._swarm_fail("向量索引不可用", stream_id)
        try:
            status = await vector.status()
            jobs = await vector.list_jobs() if hasattr(vector, "list_jobs") else []
            text = format_vector_status(status, jobs)
            errors = []
            if getattr(status, "last_error_code", None):
                errors.append(f"{status.last_error_code}: {getattr(status, 'last_error_message', '')}")
            return await self._swarm_send(text, stream_id, errors=errors)
        except Exception as exc:
            return await self._swarm_fail(f"查询向量状态失败：{exc}", stream_id)

    @Command(
        "swarm_vectors_rebuild",
        description="手动重建向量索引；--force 即使 fingerprint 未变也创建新 generation",
        pattern=r"^/swarm\s+vectors\s+rebuild(?:\s+(?P<force>--force))?$",
    )
    async def swarm_vectors_rebuild(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        if not _allow_vector_rebuild(self):
            return await self._swarm_fail("已禁止向量重建（commands.allow_vector_rebuild=false）", stream_id)
        if not _maintenance_allowed(self, {**kwargs, "stream_id": stream_id}):
            return await self._swarm_fail("当前用户不在维护人员白名单中", stream_id)
        services = self._swarm_services()
        vector = getattr(services, "vector_index", None) if services is not None else None
        if vector is None:
            return await self._swarm_fail("向量索引不可用", stream_id)
        force = _groups(kwargs).get("force") == "--force"
        try:
            result = await vector.rebuild(force=force)
            refresh = getattr(services, "refresh_vector_index_health", None) if services is not None else None
            if callable(refresh):
                with suppress(Exception):
                    await refresh()
            text, errors = format_vector_rebuild_result(result)
            ok = bool(getattr(result, "success", False))
            clipped = clip_command_output(text, _max_output_chars(self), error_lines=errors)
            await self.ctx.send.text(clipped, stream_id)
            return ok, clipped, True
        except Exception as exc:
            return await self._swarm_fail(f"向量重建失败：{exc}", stream_id)

    @Command(
        "swarm_feedback",
        description="提交简化反馈：accepted/mixed/rejected；复杂反馈请用 Planner 工具",
        pattern=r"^/swarm\s+feedback\s+(?P<task_id>\S+)\s+(?P<disposition>accepted|mixed|rejected)(?:\s+(?P<notes>.+))?$",
    )
    async def swarm_feedback(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id, error = _require_stream_id({**kwargs, "stream_id": stream_id})
        if error:
            return False, error, True
        if not _commands_enabled(self):
            return await self._swarm_fail("命令已禁用（commands.enabled=false）", stream_id)
        assert stream_id is not None
        groups = _groups(kwargs)
        task_id = groups.get("task_id", "")
        disposition = groups.get("disposition", "")
        notes = groups.get("notes")
        services = self._swarm_services()
        feedback = getattr(services, "feedback", None) if services is not None else None
        if feedback is None:
            return await self._swarm_fail("反馈服务尚未初始化", stream_id)
        try:
            result = await feedback.submit(task_id=task_id, disposition=disposition, notes=notes)
            text = (
                f"反馈已提交：feedback_id={result.feedback_id} lesson_id={getattr(result, 'lesson_id', None)} "
                f"disposition={result.disposition} indexing={getattr(result, 'lesson_indexing', 'skipped')}"
            )
            errors = []
            index_error = getattr(result, "lesson_index_error", None)
            if index_error:
                errors.append(str(index_error))
            return await self._swarm_send(text, stream_id, errors=errors)
        except LookupError as exc:
            return await self._swarm_fail(str(exc), stream_id)
        except ValueError as exc:
            return await self._swarm_fail(str(exc), stream_id)
        except Exception as exc:
            return await self._swarm_fail(f"反馈提交失败：{exc}", stream_id)


__all__ = [
    "SwarmCommandsMixin",
    "clip_command_output",
    "collect_health_errors",
    "format_agents",
    "format_health",
    "format_plugin_overview",
    "format_plugin_stats",
    "format_procedures",
    "format_task_list",
    "format_task_stats",
    "format_task_status",
    "format_vector_rebuild_result",
    "format_vector_status",
]
