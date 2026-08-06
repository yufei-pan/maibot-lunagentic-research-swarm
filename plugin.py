from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from maibot_sdk import API, CONFIG_RELOAD_SCOPE_SELF, MaiBotPlugin, ON_MODEL_CONFIG_RELOAD, Tool

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from lunagentic_research_swarm.commands import SwarmCommandsMixin  # noqa: E402
from lunagentic_research_swarm.config import LRSConfig, normalize_config  # noqa: E402
from lunagentic_research_swarm.feedback import validate_feedback_arguments  # noqa: E402
from lunagentic_research_swarm.services import LRSServiceContainer  # noqa: E402
from lunagentic_research_swarm.tools import (  # noqa: E402
    CONTEXT_SCHEMA,
    CONTINUE_SCHEMA,
    FEEDBACK_SCHEMA,
    LIST_SCHEMA,
    START_SCHEMA,
    STOP_SCHEMA,
    TASK_SCHEMA,
    failure_result,
    invoke_manager,
    manager_error,
    mutation_failure_result,
    parse_iso_timestamp,
    public_task_dto,
    success_result,
    validate_adjustment,
    validate_effort,
    validate_iso_timestamp,
    validate_nonblank,
    validate_status,
    validate_time_budget,
)


class LunagenticResearchSwarmPlugin(SwarmCommandsMixin, MaiBotPlugin):
    plugin_id = "com.0-hz.lunagentic-research-swarm"
    config_reload_subscriptions = {ON_MODEL_CONFIG_RELOAD}
    config_model = LRSConfig

    def __init__(self) -> None:
        super().__init__()
        self._services: LRSServiceContainer | None = None
        self._manager: Any | None = None

    def normalize_plugin_config(self, config_data: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
        merged, changed, notes = normalize_config(config_data, LRSConfig().model_dump(mode="python"))
        if notes:
            self.ctx.logger.info("LRS 配置迁移：%s", "；".join(notes))
        return merged, changed

    async def on_load(self) -> None:
        config = self.config
        assert isinstance(config, LRSConfig)
        if not config.plugin.enabled:
            # 与其他一方插件一致：`[plugin] enabled = false` 只加载插件本体，
            # 不启动研究运行时，也不注册任何后台任务。
            self._services = None
            self._manager = None
            self.ctx.logger.info("麦麦深度调查组已按配置禁用（plugin.enabled=false）")
            return
        services = LRSServiceContainer(self.ctx, config)
        self._services = services
        try:
            await services.start()
            self._manager = services.manager
        except BaseException:
            try:
                await services.close()
            except Exception:
                self.ctx.logger.exception("LRS 加载失败后的基础服务清理也发生错误")
            raise
        self.ctx.logger.info("麦麦深度调查组基础组件已加载")

    async def on_unload(self) -> None:
        if self._services is None:
            self.ctx.logger.info("麦麦深度调查组未启动，无需卸载")
            return
        await self._services.close()
        self._manager = None
        self.ctx.logger.info("麦麦深度调查组基础组件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        services = self._require_services()
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            detached = LRSConfig.model_validate(config_data)
            await services.update_self_config(detached, version=version)
            self.ctx.logger.info(
                "LRS 插件配置已更新：version=%s；安全限制与刷新间隔已生效，目录、selector 与 prompt 从下一 round 生效",
                version,
            )
        elif scope == ON_MODEL_CONFIG_RELOAD:
            if services.update_model_snapshot(config_data):
                self.ctx.logger.info("LRS 模型价格与 task 列表快照已更新：version=%s", version)
            else:
                self.ctx.logger.warning("LRS 模型价格与 task 列表快照未应用：version=%s", version)

    def _require_services(self) -> LRSServiceContainer:
        if self._services is None:
            raise RuntimeError("LRS 基础服务尚未初始化")
        return self._services

    @API(
        "refresh_extensions",
        description="请求麦麦深度调查组重新扫描智能体与 Procedure provider",
        version="1",
        public=True,
    )
    async def refresh_extensions(self) -> dict[str, Any]:
        return await self._require_services().refresh_extensions(reason="provider_request")

    def _require_manager(self) -> Any | None:
        if self._manager is not None:
            return self._manager
        if self._services is not None:
            return getattr(self._services, "manager", None)
        return None

    def _manager_unavailable(self, *, task_id: str | None = None, mutating: bool = False) -> dict[str, Any]:
        if mutating:
            return mutation_failure_result("manager_unavailable", "研究运行时尚未初始化", task_id=task_id)
        return failure_result("manager_unavailable", "研究运行时尚未初始化", task_id=task_id)

    @Tool(
        "start_deep_research",
        description="异步启动一次深度调查；启动后使用 status、pause、continue、stop、add_context 工具交互。",
        parameters=START_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def start_deep_research(
        self,
        objective: str,
        time_budget_seconds: int | None = None,
        effort_level: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if (message := validate_nonblank(objective, "objective")) is not None:
            return mutation_failure_result("invalid_argument", message)
        if (message := validate_time_budget(time_budget_seconds)) is not None:
            return mutation_failure_result("invalid_argument", message)
        if (message := validate_effort(effort_level)) is not None:
            return mutation_failure_result("invalid_argument", message)
        planner_context = kwargs.get("planner_context")
        if planner_context is not None and not isinstance(planner_context, str):
            return mutation_failure_result("invalid_argument", "planner_context 必须为字符串")
        stream_id = kwargs.get("stream_id")
        if (message := validate_nonblank(stream_id, "stream_id")) is not None:
            return mutation_failure_result("stream_id_required", message)
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable(mutating=True)
        if time_budget_seconds is None:
            time_budget_seconds = self._default_time_budget_seconds()
        try:
            result = await invoke_manager(
                manager,
                "start",
                objective=objective,
                stream_id=stream_id,
                time_budget_seconds=time_budget_seconds,
                effort_level=float(effort_level),
                planner_context=planner_context,
            )
        except Exception as exc:
            return manager_error(exc, mutating=True)
        if not isinstance(result, dict):
            return mutation_failure_result("manager_error", "研究运行时返回了无效结果")
        return success_result(
            result,
            effective_time_budget_seconds=time_budget_seconds,
            adjustment=float(result.get("initial_credits", 0.0)),
            task_id=str(result.get("task_id") or "") or None,
        )

    @Tool(
        "pause_deep_research",
        description="暂停深度调查；等待当前允许的在途工作安全结算，不会启动新的研究调用。",
        parameters=TASK_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def pause_deep_research(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        stream_id, error = self._stream_from_host(kwargs, task_id=task_id, mutating=True)
        if error is not None:
            return error
        return await self._call_mutating("pause", task_id, stream_id=stream_id)

    @Tool(
        "continue_deep_research",
        description="继续或重启深度调查，并可增加/减少 signed credits 与重置报告时间预算。",
        parameters=CONTINUE_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def continue_deep_research(
        self,
        task_id: str,
        time_budget_seconds: int | None = None,
        credit_adjustment: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if (message := validate_nonblank(task_id, "task_id")) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        if (message := validate_time_budget(time_budget_seconds)) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        if (message := validate_adjustment(credit_adjustment)) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        stream_id, error = self._stream_from_host(kwargs, task_id=task_id, mutating=True)
        if error is not None:
            return error
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable(task_id=task_id, mutating=True)
        try:
            result = await invoke_manager(
                manager,
                "continue_task",
                task_id,
                time_budget_seconds=time_budget_seconds,
                credit_adjustment=float(credit_adjustment),
                stream_id=stream_id,
            )
        except Exception as exc:
            return manager_error(exc, task_id=task_id, mutating=True, effective_time_budget_seconds=time_budget_seconds, adjustment=float(credit_adjustment))
        if not isinstance(result, dict):
            return mutation_failure_result("manager_error", "研究运行时返回了无效结果", task_id=task_id, effective_time_budget_seconds=time_budget_seconds, adjustment=float(credit_adjustment))
        return success_result(
            result,
            effective_time_budget_seconds=time_budget_seconds,
            adjustment=float(credit_adjustment),
            task_id=task_id,
        )

    @Tool(
        "stop_deep_research",
        description="停止深度调查并丢弃迟到结果；不会调用总结器。",
        parameters=STOP_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def stop_deep_research(self, task_id: str, reason: str = "", **kwargs: Any) -> dict[str, Any]:
        if (message := validate_nonblank(task_id, "task_id")) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        if not isinstance(reason, str):
            return mutation_failure_result("invalid_argument", "reason 必须为字符串", task_id=task_id)
        stream_id, error = self._stream_from_host(kwargs, task_id=task_id, mutating=True)
        if error is not None:
            return error
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable(task_id=task_id, mutating=True)
        try:
            result = await invoke_manager(manager, "stop", task_id, reason=reason, stream_id=stream_id)
        except Exception as exc:
            return manager_error(exc, task_id=task_id, mutating=True, adjustment=0.0)
        if not isinstance(result, dict):
            return mutation_failure_result("manager_error", "研究运行时返回了无效结果", task_id=task_id, adjustment=0.0)
        return success_result(result, adjustment=0.0, task_id=task_id)

    @Tool(
        "add_research_context",
        description="向仍在运行或可继续的深度调查广播补充信息。",
        parameters=CONTEXT_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def add_research_context(self, task_id: str, information: str, **kwargs: Any) -> dict[str, Any]:
        if (message := validate_nonblank(task_id, "task_id")) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        if (message := validate_nonblank(information, "information")) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        stream_id, error = self._stream_from_host(kwargs, task_id=task_id, mutating=True)
        if error is not None:
            return error
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable(task_id=task_id, mutating=True)
        try:
            result = await invoke_manager(manager, "add_context", task_id, information, stream_id=stream_id)
        except Exception as exc:
            return manager_error(exc, task_id=task_id, mutating=True, adjustment=0.0)
        if not isinstance(result, dict):
            return mutation_failure_result("manager_error", "研究运行时返回了无效结果", task_id=task_id, adjustment=0.0)
        return success_result(result, adjustment=0.0, task_id=task_id)

    @Tool(
        "get_research_status",
        description="查询指定深度调查任务的运行状态、round、分支与已释放上下文状态。",
        parameters=TASK_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def get_research_status(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        if (message := validate_nonblank(task_id, "task_id")) is not None:
            return failure_result("invalid_argument", message)
        stream_id, error = self._stream_from_host(kwargs, task_id=task_id)
        if error is not None:
            return error
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable(task_id=task_id)
        try:
            result = await invoke_manager(manager, "status", task_id, stream_id=stream_id)
        except Exception as exc:
            return manager_error(exc, task_id=task_id)
        if not isinstance(result, dict):
            return failure_result("manager_error", "研究运行时返回了无效结果", task_id=task_id)
        return {"success": True, **public_task_dto(result, task_id=task_id)}

    @Tool(
        "list_research_tasks",
        description="列出深度调查任务；可按状态和 ISO 时间范围过滤，最多返回 100 条。",
        parameters=LIST_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def list_research_tasks(
        self,
        status: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if (message := validate_status(status)) is not None:
            return failure_result("invalid_argument", message)
        for value, field in ((created_after, "created_after"), (created_before, "created_before")):
            if (message := validate_iso_timestamp(value, field)) is not None:
                return failure_result("invalid_argument", message)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            return failure_result("invalid_argument", "limit 必须在 1 到 100 之间")
        if created_after is not None and created_before is not None:
            if parse_iso_timestamp(created_after) > parse_iso_timestamp(created_before):
                return failure_result("invalid_argument", "created_after 不能晚于 created_before")
        stream_id, error = self._stream_from_host(kwargs)
        if error is not None:
            return error
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable()
        try:
            tasks = await invoke_manager(manager, "list_tasks", stream_id=stream_id)
        except Exception as exc:
            return manager_error(exc)
        if not isinstance(tasks, list):
            return failure_result("manager_error", "研究运行时返回了无效任务列表")
        if status is not None:
            tasks = [item for item in tasks if isinstance(item, Mapping) and item.get("status") == status]
        if created_after is not None:
            boundary = parse_iso_timestamp(created_after)
            tasks = [item for item in tasks if isinstance(item, Mapping) and _after_boundary(item.get("created_at"), boundary)]
        if created_before is not None:
            boundary = parse_iso_timestamp(created_before)
            tasks = [item for item in tasks if isinstance(item, Mapping) and _before_boundary(item.get("created_at"), boundary)]
        return {
            "success": True,
            "tasks": [public_task_dto(item) for item in tasks[:limit] if isinstance(item, Mapping)],
            "limit": limit,
        }

    @Tool(
        "submit_research_feedback",
        description="提交对深度调查结果的质量反馈、纠正与实际 outcome；事件追加写入并生成可检索 lesson。",
        parameters=FEEDBACK_SCHEMA,
        core_tool=True,
        visibility="visible",
    )
    async def submit_research_feedback(
        self,
        task_id: str,
        disposition: str,
        round_number: int | None = None,
        rating: int | None = None,
        useful_findings: list[str] | None = None,
        incorrect_findings: list[str] | None = None,
        missing_information: list[str] | None = None,
        decision: str | None = None,
        outcome: str | None = None,
        corrections: list[str] | None = None,
        notes: str | None = None,
        supersedes_feedback_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "task_id": task_id,
            "disposition": disposition,
            "round_number": round_number,
            "rating": rating,
            "useful_findings": useful_findings,
            "incorrect_findings": incorrect_findings,
            "missing_information": missing_information,
            "decision": decision,
            "outcome": outcome,
            "corrections": corrections,
            "notes": notes,
            "supersedes_feedback_id": supersedes_feedback_id,
        }
        if (message := validate_feedback_arguments(payload)) is not None:
            return failure_result("invalid_argument", message, task_id=task_id if isinstance(task_id, str) else None)
        services = self._services
        feedback = getattr(services, "feedback", None) if services is not None else None
        if feedback is None:
            return failure_result("feedback_unavailable", "反馈服务尚未初始化", task_id=task_id)
        try:
            result = await feedback.submit(
                task_id=task_id,
                disposition=disposition,
                round_number=round_number,
                rating=rating,
                useful_findings=useful_findings,
                incorrect_findings=incorrect_findings,
                missing_information=missing_information,
                decision=decision,
                outcome=outcome,
                corrections=corrections,
                notes=notes,
                supersedes_feedback_id=supersedes_feedback_id,
            )
        except LookupError as exc:
            return failure_result("task_not_found", str(exc)[:256], task_id=task_id)
        except ValueError as exc:
            return failure_result("invalid_argument", str(exc)[:256], task_id=task_id)
        except Exception:
            return failure_result("feedback_error", "反馈提交失败", task_id=task_id)
        payload: dict[str, Any] = {
            "success": True,
            "feedback_id": result.feedback_id,
            "lesson_id": result.lesson_id,
            "disposition": result.disposition,
            "task_id": task_id,
            "round_id": result.round_id,
            "lesson_indexing": getattr(result, "lesson_indexing", "skipped"),
        }
        index_error = getattr(result, "lesson_index_error", None)
        if index_error:
            payload["lesson_index_error"] = index_error
        return payload

    def _default_time_budget_seconds(self) -> int:
        if self._services is not None:
            return int(self._services.safety_limits.get("default_time_budget_seconds", 120))
        return 120

    async def _call_mutating(self, method_name: str, task_id: str, *, stream_id: str) -> dict[str, Any]:
        if (message := validate_nonblank(task_id, "task_id")) is not None:
            return mutation_failure_result("invalid_argument", message, task_id=task_id)
        manager = self._require_manager()
        if manager is None:
            return self._manager_unavailable(task_id=task_id, mutating=True)
        try:
            result = await invoke_manager(manager, method_name, task_id, stream_id=stream_id)
        except Exception as exc:
            return manager_error(exc, task_id=task_id, mutating=True, adjustment=0.0)
        if not isinstance(result, dict):
            return mutation_failure_result("manager_error", "研究运行时返回了无效结果", task_id=task_id, adjustment=0.0)
        return success_result(result, adjustment=0.0, task_id=task_id)

    def _stream_from_host(
        self, kwargs: Mapping[str, Any], *, task_id: str | None = None, mutating: bool = False
    ) -> tuple[str | None, dict[str, Any] | None]:
        stream_id = kwargs.get("stream_id")
        if (message := validate_nonblank(stream_id, "stream_id")) is not None:
            if mutating:
                return None, mutation_failure_result("stream_id_required", message, task_id=task_id)
            return None, failure_result("stream_id_required", message, task_id=task_id)
        return str(stream_id), None


def _after_boundary(value: Any, boundary: datetime) -> bool:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return timestamp >= boundary


def _before_boundary(value: Any, boundary: datetime) -> bool:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return timestamp <= boundary


def create_plugin() -> LunagenticResearchSwarmPlugin:
    return LunagenticResearchSwarmPlugin()
