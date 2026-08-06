"""统一的异步 Procedure 执行器。

普通 Procedure 只经由冻结 round catalog 指向 provider 的
``plugin_id.invoke_procedure@1``；core 控制由 :mod:`.core` 单独处理。执行器是
worker 层值对象，不直接改变 branch/credits，也不把原始 provider payload 写入
持久层。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import replace as dataclasses_replace
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ValidationError

from lunagentic_research_swarm.extensions.contracts import ProcedureInvocation, ProcedureResult
from lunagentic_research_swarm.procedures.core import (
    CORE_COMPACT_ID,
    CoreProcedureContext,
    execute_core_procedure,
    split_procedure_requests,
)
from lunagentic_research_swarm.procedures.registry import ProcedureCatalogSnapshot
from lunagentic_research_swarm.runtime.events import ProcedureBatchCompleted

LocalProcedureInvoker = Callable[..., Awaitable[Any]]


class CompositeProcedureAPI:
    """统一 provider API：先查本地 invoker（按 api_name），再回退 Host ``ctx.api``。

    本地 builtin 与第三方 Host API 共用同一 ``call(api_name, version=..., ...)`` 形状，
    避免在 scheduler / executor 里按 procedure_id 特判。
    """

    def __init__(
        self,
        host_api: Any | None = None,
        local_invokers: Mapping[str, LocalProcedureInvoker] | None = None,
    ) -> None:
        self._host = host_api
        self._local = {str(name): invoker for name, invoker in dict(local_invokers or {}).items()}

    @property
    def local_api_names(self) -> frozenset[str]:
        return frozenset(self._local)

    async def call(self, name: str, *, version: str = "1", **kwargs: Any) -> Any:
        invoker = self._local.get(str(name))
        if invoker is not None:
            return await invoker(version=version, **kwargs)
        if self._host is None:
            raise RuntimeError("Procedure executor 缺少 ctx.api，且无匹配的本地 invoker")
        call = getattr(self._host, "call", None)
        if not callable(call):
            if callable(self._host):
                call = self._host
            else:
                raise RuntimeError("ctx.api 必须提供 call()")
        return await call(name, version=version, **kwargs)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Provider contract 只允许结构化 data/error/metadata；这些字段若原样跨过
# event/storage 边界就会把 raw payload、原始 provenance 或 reasoning 带入默认库。
# 其余业务字段（例如 title、answer、items）不做猜测式裁剪。
_SENSITIVE_PAYLOAD_KEYS = frozenset(
    {
        "reasoning",
        "raw_payload",
        "raw_provenance",
        "raw_payloads",
        "raw_result",
        "raw_arguments",
        "provenance",
        "payload",
        "transcript",
        "messages",
    }
)


def _sanitize_payload(value: Any) -> Any:
    """递归移除 provider raw/reasoning 字段，同时保留普通业务 JSON。"""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_payload(item)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_PAYLOAD_KEYS
        }
    if isinstance(value, list | tuple):
        return [_sanitize_payload(item) for item in value]
    return value


def _request_value(request: Any, key: str, default: Any = None) -> Any:
    if isinstance(request, Mapping):
        return request.get(key, default)
    return getattr(request, key, default)


@dataclass(frozen=True, slots=True)
class ProcedureExecutionResult:
    """单个请求的可持久化摘要；没有 raw provider payload 字段。"""

    procedure_id: str
    request_id: str
    result: ProcedureResult
    provider_plugin_id: str = ""
    api_name: str = ""
    api_version: str = "1"
    attempts: int = 1
    duration_ms: int = 0
    call_id: str = ""

    @property
    def procedure_result(self) -> ProcedureResult:
        """兼容调用方使用的别名。"""

        return self.result

    @property
    def success(self) -> bool:
        return self.result.success

    @property
    def data(self) -> Mapping[str, Any] | None:
        return self.result.data

    @property
    def error(self) -> Mapping[str, Any] | None:
        return self.result.error

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.result.metadata

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "request_id": self.request_id,
            "result": self.result.model_dump(mode="json"),
            "provider_plugin_id": self.provider_plugin_id,
            "api_name": self.api_name,
            "api_version": self.api_version,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "call_id": self.call_id,
        }


# ``ProcedureResultItem`` is the name used by a few integrations and is intentionally
# the same value type rather than a subclass (which would complicate event decoding).
ProcedureResultItem = ProcedureExecutionResult


def _structured_error(
    code: str,
    message: str,
    *,
    provider_plugin_id: str = "",
    procedure_id: str = "",
    api_name: str = "",
    api_version: str = "1",
    request_id: str = "",
    duration_ms: int = 0,
    agent_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ProcedureResult:
    result_metadata: dict[str, Any] = {
        "provider_plugin_id": provider_plugin_id,
        "procedure_id": procedure_id,
        "api_name": api_name,
        "api_version": api_version,
        "request_id": request_id,
        "duration_ms": duration_ms,
    }
    if agent_id:
        result_metadata["agent_id"] = agent_id
    if metadata:
        result_metadata.update(_sanitize_payload(dict(metadata)))
        if agent_id:
            result_metadata["agent_id"] = agent_id
    if "attempts" in result_metadata:
        result_metadata.setdefault("attempt", result_metadata["attempts"])
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata=result_metadata,
    )


def procedure_result_summary(item: ProcedureExecutionResult) -> dict[str, Any]:
    """供 branch 上下文/审计使用的结构化摘要；已剥离敏感 raw 字段。"""

    result = item.result
    summary: dict[str, Any] = {
        "procedure_id": item.procedure_id,
        "request_id": item.request_id,
        # 智能体自带的关联 ID（若有），便于把结果对回它自己的请求。
        **({"call_id": item.call_id} if item.call_id else {}),
        "success": bool(getattr(result, "success", False)),
        "provider_plugin_id": item.provider_plugin_id,
        "duration_ms": int(item.duration_ms),
    }
    data = getattr(result, "data", None)
    error = getattr(result, "error", None)
    if data is not None:
        summary["data"] = _sanitize_payload(data)
    if error is not None:
        summary["error"] = _sanitize_payload(error)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("agent_id"):
        summary["agent_id"] = str(metadata["agent_id"])
    return summary


def fold_procedure_results_into_messages(
    messages: Sequence[Mapping[str, Any]],
    results: Sequence[ProcedureExecutionResult],
) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    """把普通 Procedure 结果按调用顺序写入可变 history，供子分支/总结继承。"""

    folded = [dict(item) for item in messages if isinstance(item, Mapping)]
    summaries: list[dict[str, Any]] = []
    for item in results:
        if str(item.procedure_id).startswith("core."):
            continue
        summary = procedure_result_summary(item)
        summaries.append(summary)
        folded.append(
            {
                "role": "user",
                "content": "procedure_result:\n"
                + json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str),
            }
        )
    return tuple(folded), summaries


class ProcedureExecutor:
    """调用冻结 Procedure catalog 的异步执行器。

    ``catalog`` 可以是 ``ProcedureCatalogSnapshot``，也可以是提供 ``get`` 的
    catalog-like 对象；``api`` 可以是 ``ctx.api`` 或直接实现 ``call`` 的 fake。
    构造器额外接受 ``ctx`` 以便 runtime 直接传入插件上下文；``local_invokers``
    按 catalog ``api_name`` 路由到进程内 provider（如 builtin），再回退 Host API。
    """

    def __init__(
        self,
        catalog: Any | None = None,
        api: Any | None = None,
        *,
        ctx: Any | None = None,
        summarizer: Any | None = None,
        agent_id: str = "lrs.executor",
        local_invokers: Mapping[str, LocalProcedureInvoker] | None = None,
        debug_store: Any | None = None,
    ) -> None:
        # 允许历史调用方使用 ProcedureExecutor(ctx.api, catalog)。
        if catalog is not None and hasattr(catalog, "call") and api is not None and hasattr(api, "get"):
            catalog, api = api, catalog
        if api is None and ctx is not None:
            api = getattr(ctx, "api", None)
        if local_invokers:
            api = CompositeProcedureAPI(host_api=api, local_invokers=local_invokers)
        self.catalog = catalog or ProcedureCatalogSnapshot(())
        self.api = api
        self.summarizer = summarizer
        self.agent_id = str(agent_id or "lrs.executor")
        self.debug_store = debug_store

    def _entry(self, procedure_id: str) -> Any | None:
        getter = getattr(self.catalog, "get", None)
        if callable(getter):
            return getter(procedure_id)
        if isinstance(self.catalog, Mapping):
            return self.catalog.get(procedure_id)
        return None

    @staticmethod
    def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
        if isinstance(entry, Mapping):
            return entry.get(key, default)
        return getattr(entry, key, default)

    @staticmethod
    def _definition(entry: Any) -> Any:
        value = ProcedureExecutor._entry_value(entry, "definition")
        return value if value is not None else entry

    @staticmethod
    def _effect_payload(effect: Any) -> Mapping[str, Any]:
        if isinstance(effect, Mapping):
            return effect
        payload = getattr(effect, "payload", None)
        if isinstance(payload, Mapping):
            return payload
        return MappingProxyType({})

    @classmethod
    def _context(cls, effect: Any) -> dict[str, Any]:
        payload = dict(cls._effect_payload(effect))

        def value(key: str, default: Any = None) -> Any:
            item = payload.get(key)
            if (key not in payload or item is None) and not isinstance(effect, Mapping):
                item = getattr(effect, key, default)
            if item is None:
                item = default
            return item

        task_id = value("task_id", "task-unknown")
        round_id = value("round_id", "round-unknown")
        branch_id = value("branch_id", "branch-unknown")
        turn_id = value("turn_id", value("call_id", "turn-unknown"))
        agent_id = value("agent_id", "lrs.executor")
        allowed_raw = value("allowed_procedures")
        allowed: frozenset[str] | None
        if allowed_raw is None:
            # 未由 prepare_procedure_effect 注入时保持兼容：不额外限制 catalog。
            allowed = None
        elif allowed_raw == "*" or allowed_raw == ["*"] or allowed_raw == ("*",):
            allowed = None
        elif isinstance(allowed_raw, (str, bytes, bytearray)):
            allowed = frozenset({str(allowed_raw)})
        elif isinstance(allowed_raw, Sequence):
            allowed = frozenset(str(item) for item in allowed_raw)
        else:
            allowed = frozenset()
        return {
            "task_id": str(task_id or "task-unknown"),
            "round_id": str(round_id or "round-unknown"),
            "branch_id": str(branch_id or "branch-unknown"),
            "turn_id": str(turn_id or "turn-unknown"),
            "agent_id": str(agent_id or "lrs.executor"),
            "call_id": str(value("call_id", "") or ""),
            "event_id": str(value("event_id", "") or ""),
            "generation": int(value("generation", getattr(effect, "generation", 0)) or 0),
            "allowed_procedures": allowed,
        }

    @classmethod
    def _requests(cls, effect: Any, requests: Sequence[Any] | None) -> list[Any]:
        if requests is not None:
            return list(requests)
        payload = cls._effect_payload(effect)
        for key in ("requests", "procedure_requests", "procedures"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return list(value)
        value = getattr(effect, "requests", None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        if isinstance(effect, Sequence) and not isinstance(effect, (str, bytes, bytearray)):
            return list(effect)
        return []

    @staticmethod
    def _stable_request_id(context: Mapping[str, Any], request: Any, index: int) -> str:
        explicit = _request_value(request, "request_id")
        if isinstance(explicit, str) and explicit.strip():
            return explicit[:128]
        identity = {
            "task_id": context["task_id"],
            "round_id": context["round_id"],
            "branch_id": context["branch_id"],
            "turn_id": context["turn_id"],
            "procedure_id": str(_request_value(request, "procedure_id", "")),
            "index": index,
            "arguments": _jsonable(_request_value(request, "arguments", {})),
        }
        return "proc_" + _fingerprint(identity)[:56]

    @staticmethod
    def _metadata(context: Mapping[str, Any]) -> dict[str, Any]:
        # 这是唯一发送给 provider 的上下文；不放 messages、其它 branch 或 raw payload。
        return {
            "task_id": context["task_id"],
            "round_id": context["round_id"],
            "branch_id": context["branch_id"],
            "turn_id": context["turn_id"],
            "agent_id": context["agent_id"],
        }

    async def _api_call(
        self,
        entry: Any,
        procedure_id: str,
        invocation: ProcedureInvocation,
    ) -> Any:
        if self.api is None:
            raise RuntimeError("Procedure executor 缺少 ctx.api")
        api_name = str(self._entry_value(entry, "api_name", ""))
        api_version = str(self._entry_value(entry, "api_version", "1"))
        if not api_name:
            raise RuntimeError("Procedure catalog 缺少 provider API identity")
        call = getattr(self.api, "call", None)
        if not callable(call):
            if callable(self.api):
                call = self.api
            else:
                raise RuntimeError("ctx.api 必须提供 call()")
        # API contract 的固定 envelope：procedure_id、request_id、arguments、scoped_metadata。
        return await call(
            api_name,
            version=api_version,
            procedure_id=procedure_id,
            request_id=invocation.request_id,
            arguments=dict(invocation.arguments),
            scoped_metadata=dict(invocation.scoped_metadata),
        )

    @staticmethod
    def _normalize_result(
        raw: Any,
        *,
        entry: Any,
        procedure_id: str,
        request_id: str,
        duration_ms: int,
        attempts: int,
        agent_id: str = "",
    ) -> ProcedureResult:
        try:
            result = ProcedureResult.model_validate(raw, strict=True)
        except (TypeError, ValueError, ValidationError):
            return _structured_error(
                "provider_contract_invalid",
                "提供方返回值不符合 ProcedureResult 契约",
                provider_plugin_id=str(ProcedureExecutor._entry_value(entry, "provider_plugin_id", "")),
                procedure_id=procedure_id,
                api_name=str(ProcedureExecutor._entry_value(entry, "api_name", "")),
                api_version=str(ProcedureExecutor._entry_value(entry, "api_version", "1")),
                request_id=request_id,
                duration_ms=duration_ms,
                agent_id=agent_id,
                metadata={"attempts": attempts},
            )
        metadata = dict(_sanitize_payload(result.metadata))
        # These values are authority data from the frozen catalog/request. Provider
        # metadata is untrusted telemetry and must never override them.
        metadata["provider_plugin_id"] = str(ProcedureExecutor._entry_value(entry, "provider_plugin_id", ""))
        metadata["api_name"] = str(ProcedureExecutor._entry_value(entry, "api_name", ""))
        metadata["api_version"] = str(ProcedureExecutor._entry_value(entry, "api_version", "1"))
        metadata["procedure_id"] = procedure_id
        metadata["request_id"] = request_id
        if agent_id:
            metadata["agent_id"] = agent_id
        sanitized_data = _sanitize_payload(result.data)
        sanitized_error = _sanitize_payload(result.error)
        metadata["duration_ms"] = duration_ms
        metadata["attempts"] = attempts
        metadata["attempt"] = attempts
        # 重新走 contract validator，确保 executor 生成的 metadata 与 provider
        # 原结果一样是递归冻结的 JSON，而不是 model_copy(update=...) 产生的可变 dict。
        return ProcedureResult.model_validate(
            {
                "success": result.success,
                "data": sanitized_data,
                "error": sanitized_error,
                "metadata": metadata,
            },
            strict=True,
        )

    async def _invoke_one(self, request: Any, context: Mapping[str, Any], index: int) -> ProcedureExecutionResult:
        procedure_id = str(_request_value(request, "procedure_id", "") or "")
        request_id = self._stable_request_id(context, request, index)
        agent_id = str(context.get("agent_id") or "")
        allowed = context.get("allowed_procedures")
        if isinstance(allowed, frozenset) and procedure_id not in allowed:
            result = _structured_error(
                "procedure_not_allowed",
                "当前 agent 的 allowed_procedures 不允许调用该 Procedure",
                procedure_id=procedure_id,
                request_id=request_id,
                agent_id=agent_id,
            )
            return ProcedureExecutionResult(procedure_id, request_id, result)
        entry = self._entry(procedure_id)
        if entry is None:
            result = _structured_error(
                "procedure_unavailable",
                "Procedure 不在当前 round catalog 中",
                procedure_id=procedure_id,
                request_id=request_id,
                agent_id=agent_id,
            )
            return ProcedureExecutionResult(procedure_id, request_id, result)

        definition = self._definition(entry)
        provider_id = str(self._entry_value(entry, "provider_plugin_id", ""))
        api_name = str(self._entry_value(entry, "api_name", ""))
        api_version = str(self._entry_value(entry, "api_version", "1"))
        try:
            invocation = ProcedureInvocation.model_validate(
                {
                    "request_id": request_id,
                    "task_id": context["task_id"],
                    "round_id": context["round_id"],
                    "branch_id": context["branch_id"],
                    "turn_id": context["turn_id"],
                    "agent_id": context["agent_id"],
                    "arguments": dict(_request_value(request, "arguments", {}) or {}),
                    "scoped_metadata": self._metadata(context),
                },
                strict=True,
            )
        except (TypeError, ValueError, ValidationError):
            result = _structured_error(
                "invalid_arguments",
                "Procedure arguments 或调用上下文无效",
                provider_plugin_id=provider_id,
                procedure_id=procedure_id,
                api_name=api_name,
                api_version=api_version,
                request_id=request_id,
                agent_id=agent_id,
            )
            return ProcedureExecutionResult(procedure_id, request_id, result, provider_id, api_name, api_version)

        started = time.perf_counter()
        timeout_seconds = float(getattr(definition, "timeout_seconds", 30.0))
        idempotent = bool(getattr(definition, "idempotent", False))
        attempts = 0
        while True:
            attempts += 1
            try:
                call = self._api_call(entry, procedure_id, invocation)
                if timeout_seconds > 0:
                    raw = await asyncio.wait_for(call, timeout=timeout_seconds)
                else:
                    raw = await call
            except asyncio.TimeoutError:
                duration_ms = int((time.perf_counter() - started) * 1000)
                result = _structured_error(
                    "procedure_timeout",
                    "Procedure 调用超时；结果状态不确定，未自动重试",
                    provider_plugin_id=provider_id,
                    procedure_id=procedure_id,
                    api_name=api_name,
                    api_version=api_version,
                    request_id=request_id,
                    duration_ms=duration_ms,
                    agent_id=agent_id,
                    metadata={"attempts": attempts},
                )
                return ProcedureExecutionResult(
                    procedure_id, request_id, result, provider_id, api_name, api_version, attempts, duration_ms
                )
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                retryable = getattr(exc, "retryable", None) is True
                if idempotent and retryable and attempts == 1:
                    continue
                result = _structured_error(
                    "provider_call_failed",
                    "Procedure provider 调用失败",
                    provider_plugin_id=provider_id,
                    procedure_id=procedure_id,
                    api_name=api_name,
                    api_version=api_version,
                    request_id=request_id,
                    duration_ms=duration_ms,
                    agent_id=agent_id,
                    metadata={"attempts": attempts, "retryable": retryable},
                )
                return ProcedureExecutionResult(
                    procedure_id, request_id, result, provider_id, api_name, api_version, attempts, duration_ms
                )

            duration_ms = int((time.perf_counter() - started) * 1000)
            result = self._normalize_result(
                raw,
                entry=entry,
                procedure_id=procedure_id,
                request_id=request_id,
                duration_ms=duration_ms,
                attempts=attempts,
                agent_id=agent_id,
            )
            retryable = bool(result.error and result.error.get("retryable") is True)
            has_business_result = result.data is not None
            if not result.success and idempotent and retryable and not has_business_result and attempts == 1:
                continue
            await self._maybe_save_payload(
                context=context,
                request=request,
                request_id=request_id,
                procedure_id=procedure_id,
                raw_result=raw,
            )
            return ProcedureExecutionResult(
                procedure_id, request_id, result, provider_id, api_name, api_version, attempts, duration_ms
            )

    async def _maybe_save_payload(
        self,
        *,
        context: Mapping[str, Any],
        request: Any,
        request_id: str,
        procedure_id: str,
        raw_result: Any,
    ) -> None:
        store = self.debug_store
        if store is None or not getattr(store, "store_raw_procedure_payloads", False):
            return
        save = getattr(store, "save_payload", None)
        if not callable(save):
            return
        arguments = dict(_request_value(request, "arguments", {}) or {})
        request_payload = {
            "procedure_id": procedure_id,
            "request_id": request_id,
            "arguments": arguments,
        }
        try:
            await save(
                task_id=context["task_id"],
                round_id=context["round_id"],
                branch_id=context["branch_id"],
                turn_id=context["turn_id"],
                request_id=request_id,
                procedure_id=procedure_id,
                request=request_payload,
                arguments=arguments,
                raw_result=_jsonable(raw_result),
            )
        except Exception:
            record = getattr(store, "_record_failure", None)
            if callable(record):
                try:
                    await record(
                        kind="payload",
                        error=RuntimeError("debug payload 写入失败"),
                        task_id=context["task_id"],
                        round_id=context["round_id"],
                    )
                except Exception:
                    pass

    async def invoke_many(
        self,
        effect: Any,
        requests: Sequence[Any] | None = None,
    ) -> ProcedureBatchCompleted:
        """并发调用普通 Procedure，执行 core.compact，并按输入顺序生成完成事件。"""

        context = self._context(effect)
        payload = dict(self._effect_payload(effect))
        all_requests = self._requests(effect, requests)
        ordinary, controls = split_procedure_requests(all_requests)
        results = list(
            await asyncio.gather(*(self._invoke_one(request, context, index) for index, request in enumerate(ordinary)))
        )
        # 把智能体自带的可选 call_id 透传回结果，供它把结果对回自己的请求。
        results = [
            dataclasses_replace(item, call_id=str(_request_value(request, "call_id", "") or ""))
            for item, request in zip(results, ordinary)
        ]
        parent_messages = tuple(dict(item) for item in payload.get("messages", ()) if isinstance(item, Mapping))
        # spec §8.2/§9.1：本 turn 的 envelope `report` 是该智能体的工作输出（明确
        # 不含 provider reasoning），必须进入可变 history，子分支才能继承父节点的
        # 输出，分支总结/compact 也才有内容可总结。
        report = str(payload.get("report", "") or "").strip()
        if report:
            parent_messages = (*parent_messages, {"role": "assistant", "content": report})
        # 普通结果先进入可变 history，再 compact，使子分支继承 procedure 输出。
        parent_messages, _summaries = fold_procedure_results_into_messages(parent_messages, results)
        if controls.compact:
            compact_result = await self._execute_compact(context=context, payload=payload, messages=parent_messages)
            results.append(compact_result)
            data = getattr(compact_result.result, "data", None)
            if (
                bool(getattr(compact_result.result, "success", False))
                and isinstance(data, Mapping)
                and data.get("compacted")
            ):
                summary = str(data.get("summary") or "")
                parent_messages = self._rewrite_compacted_messages(parent_messages, summary)
        result_id = "proc_result_" + _fingerprint([item.as_dict() for item in results])[:56]
        event_id = (
            context["event_id"]
            or "evt_proc_"
            + _fingerprint(
                {
                    "task_id": context["task_id"],
                    "round_id": context["round_id"],
                    "call_id": context["call_id"],
                    "result_id": result_id,
                }
            )[:56]
        )
        return ProcedureBatchCompleted(
            event_id=event_id,
            task_id=context["task_id"],
            round_id=context["round_id"],
            generation=context["generation"],
            branch_id=context["branch_id"],
            call_id=context["call_id"],
            result_id=result_id,
            results=tuple(results),
            controls=controls,
            parent_messages=parent_messages,
        )

    async def _execute_compact(
        self,
        *,
        context: Mapping[str, Any],
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> ProcedureExecutionResult:
        request_id = "core_compact_" + _fingerprint(
            {
                "task_id": context["task_id"],
                "round_id": context["round_id"],
                "call_id": context["call_id"],
                "branch_id": context["branch_id"],
            }
        )[:48]
        started = time.perf_counter()
        formalized = payload.get("formalized_task")
        history = tuple(dict(item) for item in messages)
        result = await execute_core_procedure(
            CORE_COMPACT_ID,
            {},
            summarizer=self.summarizer,
            context=CoreProcedureContext(formalized_task=formalized, branch_history=history),
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        agent_id = str(context.get("agent_id") or "")
        metadata = dict(_sanitize_payload(getattr(result, "metadata", {}) or {}))
        metadata["agent_id"] = agent_id or "lrs.executor"
        metadata["procedure_id"] = CORE_COMPACT_ID
        metadata["request_id"] = request_id
        metadata["duration_ms"] = duration_ms
        metadata["attempts"] = 1
        metadata["attempt"] = 1
        normalized = ProcedureResult.model_validate(
            {
                "success": bool(getattr(result, "success", False)),
                "data": _sanitize_payload(getattr(result, "data", None)),
                "error": _sanitize_payload(getattr(result, "error", None)),
                "metadata": metadata,
            },
            strict=True,
        )
        return ProcedureExecutionResult(
            CORE_COMPACT_ID,
            request_id,
            normalized,
            "core",
            "",
            "1",
            1,
            duration_ms,
        )

    @staticmethod
    def _rewrite_compacted_messages(
        messages: Sequence[Mapping[str, Any]],
        summary: str,
    ) -> tuple[dict[str, Any], ...]:
        """Keep stable system/formalized prefix; replace mutable history with compact summary."""

        prefix: list[dict[str, Any]] = []
        index = 0
        if index < len(messages) and str(messages[index].get("role", "")) == "system":
            prefix.append(dict(messages[index]))
            index += 1
        if index < len(messages) and str(messages[index].get("role", "")) == "user":
            prefix.append(dict(messages[index]))
            index += 1
        prefix.append({"role": "assistant", "content": f"分支压缩摘要：{summary}"})
        return tuple(prefix)


def bundled_procedure_invoker(provider: Any) -> LocalProcedureInvoker:
    """把 ``BundledProcedureProvider.invoke`` 适配成 ``api.call`` 同形的本地 invoker。"""

    async def _invoke(
        *,
        version: str = "1",
        procedure_id: str,
        request_id: str = "",
        arguments: Mapping[str, Any] | None = None,
        scoped_metadata: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        del version, request_id, _kwargs
        result = await provider.invoke(
            procedure_id,
            arguments or {},
            scoped_metadata=scoped_metadata,
        )
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json")
        return result

    return _invoke


__all__ = [
    "CompositeProcedureAPI",
    "LocalProcedureInvoker",
    "ProcedureExecutionResult",
    "ProcedureExecutor",
    "ProcedureResultItem",
    "bundled_procedure_invoker",
    "fold_procedure_results_into_messages",
    "procedure_result_summary",
]
