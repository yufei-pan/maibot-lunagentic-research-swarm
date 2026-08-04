"""Task-aware bounded fair scheduler for LRS runtime effects.

The scheduler owns only local queue/worker bookkeeping.  Workers receive an
effect and its generation token, but never mutate reducer state.  In
particular, a generation cancellation does not attempt to cancel an upstream
Host request; the effect's generation remains available to the reducer so a
late completion can be ignored there.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from lunagentic_research_swarm.runtime.reducer import Effect


class GenerationToken(int):
    """An integer-compatible token passed to a worker for one effect.

    ``int(token)`` and equality with an effect's generation work for simple
    workers, while ``token.cancelled`` lets adapters observe local stop state.
    Cancellation is deliberately advisory: it never interrupts a Host call.
    """

    def __new__(cls, scheduler: FairScheduler, task_id: str, generation: int) -> GenerationToken:
        value = int.__new__(cls, generation)
        value._scheduler = scheduler  # type: ignore[attr-defined]
        value.task_id = task_id  # type: ignore[attr-defined]
        value.generation = generation  # type: ignore[attr-defined]
        return value

    @property
    def cancelled(self) -> bool:
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is None:
            return False
        return scheduler._is_generation_cancelled(self.task_id, int(self))

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled

    def is_current(self) -> bool:
        return not self.cancelled


class _ImmediateInt(int):
    """Return value that works for both sync and ``await`` cancellation callers."""

    def __new__(cls, value: int) -> _ImmediateInt:
        return int.__new__(cls, value)

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _value() -> int:
            return int(self)

        return _value().__await__()


@dataclass(slots=True)
class _QueuedEffect:
    effect: Effect
    queued_at: float
    future: asyncio.Future[Any]


@dataclass(slots=True)
class _ActiveEffect:
    entry: _QueuedEffect
    task: asyncio.Task[Any]
    started_at: float
    token: GenerationToken


class FairScheduler:
    """Bounded scheduler with priority queues and per-priority task fairness.

    ``worker`` may be an async/sync callable accepting ``(effect, token)`` or
    just ``(effect)``.  For object workers, ``run``, ``execute``, ``handle`` or
    ``process`` are accepted as equivalent entry points.  A worker is invoked
    exactly once; retry policy belongs to the Host and is intentionally absent
    here.
    """

    _PRIORITY_RANK = {
        "barrier": 400,
        "critical": 350,
        "high": 300,
        "normal": 200,
        "low": 100,
    }

    def __init__(
        self,
        *,
        global_llm: int = 16,
        per_task_llm: int = 8,
        per_task_procedure: int = 16,
        worker: Any,
        on_result: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = {
            "global_llm": self._validate_limit(global_llm, "global_llm"),
            "per_task_llm": self._validate_limit(per_task_llm, "per_task_llm"),
            "per_task_procedure": self._validate_limit(per_task_procedure, "per_task_procedure"),
        }
        self._worker = worker
        self._on_result = on_result
        self._clock = clock

        # priority -> task -> FIFO effects.  The separate cursor maps preserve
        # RR state even when a new task joins while one task is already wide.
        self._queues: OrderedDict[str, OrderedDict[str, deque[_QueuedEffect]]] = OrderedDict()
        self._cursor: dict[str, str | None] = {}
        self._dispatched: dict[str, bool] = {}
        self._paused: set[str] = set()
        self._cancelled_through: dict[str, int] = {}

        self._active: dict[int, _ActiveEffect] = {}
        self._next_active_id = 0
        self._global_llm_active = 0
        self._task_llm_active: Counter[str] = Counter()
        self._task_procedure_active: Counter[str] = Counter()
        self._errors: list[dict[str, str]] = []

        self._started = False
        self._closing = False
        self._closed = False
        self._wake: asyncio.Event | None = None
        self._dispatcher: asyncio.Task[Any] | None = None
        self._close_task: asyncio.Task[None] | None = None

    @staticmethod
    def _validate_limit(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} 必须是非负整数")
        return value

    @property
    def global_llm(self) -> int:
        return self._limits["global_llm"]

    @property
    def per_task_llm(self) -> int:
        return self._limits["per_task_llm"]

    @property
    def per_task_procedure(self) -> int:
        return self._limits["per_task_procedure"]

    async def start(self) -> FairScheduler:
        """Start the dispatcher; calling this repeatedly is idempotent."""

        if self._closed or self._closing:
            raise RuntimeError("scheduler 已关闭")
        if self._started:
            return self
        self._started = True
        self._wake = asyncio.Event()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="lrs-fair-scheduler")
        return self

    async def enqueue(self, effect: Effect) -> bool:
        """Queue an effect and return whether it was accepted locally."""

        if self._closed or self._closing:
            return False
        if not self._started:
            await self.start()
        task_id = str(effect.task_id)
        generation = int(effect.generation)
        if self._is_generation_cancelled(task_id, generation):
            return False
        loop = asyncio.get_running_loop()
        entry = _QueuedEffect(effect=effect, queued_at=self._clock(), future=loop.create_future())
        priority = self._priority_key(effect.priority)
        task_queues = self._queues.setdefault(priority, OrderedDict())
        if task_id not in task_queues:
            task_queues[task_id] = deque()
            if self._cursor.get(priority) is None:
                self._cursor[priority] = task_id
            elif self._dispatched.get(priority, False):
                # A task arriving after dispatch goes immediately before the
                # next cursor, preventing a wide task from taking another turn.
                cursor = self._cursor.get(priority)
                if cursor in task_queues and cursor != task_id:
                    task_queues.move_to_end(task_id, last=False)
                    self._cursor[priority] = task_id
        task_queues[task_id].append(entry)
        self._signal()
        return True

    def pause_task(self, task_id: str) -> None:
        """Stop new agent/summarizer launches while preserving in-flight work."""

        self._paused.add(str(task_id))
        self._signal()

    def resume_task(self, task_id: str) -> None:
        self._paused.discard(str(task_id))
        self._signal()

    # Continue is a useful controller-facing spelling for resume.
    continue_task = resume_task
    unpause_task = resume_task

    def cancel_generation(self, task_id: str, generation: int | None = None) -> _ImmediateInt:
        """Drop queued effects from a generation and mark active tokens cancelled.

        Active wrappers are never cancelled.  If ``generation`` is omitted, the
        highest generation currently known for this task is used, allowing the
        common ``stop_generation(task_id)`` form to invalidate the current work.
        """

        task_id = str(task_id)
        if generation is None:
            generations = [
                int(active.entry.effect.generation)
                for active in self._active.values()
                if str(active.entry.effect.task_id) == task_id
            ]
            generations.extend(
                int(entry.effect.generation)
                for task_queues in self._queues.values()
                for queued in task_queues.get(task_id, ())
                for entry in (queued,)
            )
            generation = max(generations, default=self._cancelled_through.get(task_id, -1))
        generation = int(generation)
        self._cancelled_through[task_id] = max(self._cancelled_through.get(task_id, -1), generation)

        removed = 0
        for priority, task_queues in list(self._queues.items()):
            queue = task_queues.get(task_id)
            if queue is None:
                continue
            kept: deque[_QueuedEffect] = deque()
            while queue:
                entry = queue.popleft()
                if int(entry.effect.generation) <= generation:
                    removed += 1
                    self._cancel_future(entry.future)
                else:
                    kept.append(entry)
            if kept:
                task_queues[task_id] = kept
            else:
                task_queues.pop(task_id, None)
            self._cleanup_priority(priority, task_queues)
        self._signal()
        return _ImmediateInt(removed)

    # Historical wording in the runtime brief.
    stop_generation = cancel_generation

    def update_limits(
        self,
        *,
        global_llm: int | None = None,
        per_task_llm: int | None = None,
        per_task_procedure: int | None = None,
        **aliases: Any,
    ) -> None:
        """Apply live limits; lowering only affects future launches."""

        values = {
            "global_llm": global_llm if global_llm is not None else aliases.get("max_global_llm_concurrency"),
            "per_task_llm": per_task_llm if per_task_llm is not None else aliases.get("max_task_llm_concurrency"),
            "per_task_procedure": (
                per_task_procedure
                if per_task_procedure is not None
                else aliases.get("max_task_procedure_concurrency")
            ),
        }
        for name, value in values.items():
            if value is not None:
                self._limits[name] = self._validate_limit(value, name)
        self._signal()

    set_limits = update_limits

    def update_config(self, config: Any) -> None:
        """Read scheduler values from a config section or mapping."""

        if isinstance(config, Mapping):
            self.update_limits(**dict(config))
            return
        self.update_limits(
            global_llm=getattr(config, "max_global_llm_concurrency", None),
            per_task_llm=getattr(config, "max_task_llm_concurrency", None),
            per_task_procedure=getattr(config, "max_task_procedure_concurrency", None),
        )

    async def close(self) -> None:
        """Stop intake, cancel queued local futures, and await started wrappers."""

        caller = asyncio.current_task()
        if self._close_task is not None:
            if caller is not None and any(active.task is caller for active in self._active.values()):
                return
            await asyncio.shield(self._close_task)
            return

        self._closing = True
        for task_queues in self._queues.values():
            for queue in task_queues.values():
                while queue:
                    self._cancel_future(queue.popleft().future)
        self._queues.clear()
        self._cursor.clear()
        self._dispatched.clear()
        self._signal()
        self._close_task = asyncio.create_task(self._finish_close(caller), name="lrs-fair-scheduler-close")
        await asyncio.shield(self._close_task)

    async def _finish_close(self, excluded_task: asyncio.Task[Any] | None) -> None:
        if self._dispatcher is not None and self._dispatcher is not excluded_task:
            await self._dispatcher
        active_tasks = [active.task for active in self._active.values() if active.task is not excluded_task]
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        self._closed = True

    def stats(self) -> dict[str, Any]:
        """Return queue/capacity telemetry without effect payloads or prompts."""

        now = self._clock()
        queued_entries = [entry for task_queues in self._queues.values() for queue in task_queues.values() for entry in queue]
        task_ids = {str(entry.effect.task_id) for entry in queued_entries}
        task_ids.update(str(active.entry.effect.task_id) for active in self._active.values())
        task_ids.update(self._paused)

        tasks: dict[str, Any] = {}
        for task_id in sorted(task_ids):
            queued = [entry for entry in queued_entries if str(entry.effect.task_id) == task_id]
            active = [item for item in self._active.values() if str(item.entry.effect.task_id) == task_id]
            kind_stats: dict[str, dict[str, Any]] = {}
            for entry in queued:
                kind = str(entry.effect.kind)
                data = kind_stats.setdefault(kind, {"active": 0, "queued": 0, "wait_latency_seconds": 0.0})
                data["queued"] += 1
                data["wait_latency_seconds"] += max(0.0, now - entry.queued_at)
            for item in active:
                kind = str(item.entry.effect.kind)
                data = kind_stats.setdefault(kind, {"active": 0, "queued": 0, "wait_latency_seconds": 0.0})
                data["active"] += 1
                data["wait_latency_seconds"] += max(0.0, item.started_at - item.entry.queued_at)
            waits = [max(0.0, now - entry.queued_at) for entry in queued]
            waits.extend(max(0.0, item.started_at - item.entry.queued_at) for item in active)
            average_wait = sum(waits) / len(waits) if waits else 0.0
            tasks[task_id] = {
                "active": len(active),
                "queued": len(queued),
                "llm_active": self._task_llm_active.get(task_id, 0),
                "llm_limit": self.per_task_llm,
                "procedure_active": self._task_procedure_active.get(task_id, 0),
                "procedure_limit": self.per_task_procedure,
                "paused": task_id in self._paused,
                "kind": kind_stats,
                "active_by_kind": {key: value["active"] for key, value in kind_stats.items()},
                "queued_by_kind": {key: value["queued"] for key, value in kind_stats.items()},
                "wait_latency_seconds": average_wait,
                "wait_latency": {
                    "samples": len(waits),
                    "average_seconds": average_wait,
                    "max_seconds": max(waits, default=0.0),
                },
            }

        global_kind_stats: dict[str, dict[str, Any]] = {}
        for entry in queued_entries:
            kind = str(entry.effect.kind)
            data = global_kind_stats.setdefault(kind, {"active": 0, "queued": 0, "wait_latency_seconds": 0.0})
            data["queued"] += 1
            data["wait_latency_seconds"] += max(0.0, now - entry.queued_at)
        for item in self._active.values():
            kind = str(item.entry.effect.kind)
            data = global_kind_stats.setdefault(kind, {"active": 0, "queued": 0, "wait_latency_seconds": 0.0})
            data["active"] += 1
            data["wait_latency_seconds"] += max(0.0, item.started_at - item.entry.queued_at)
        global_waits = [max(0.0, now - entry.queued_at) for entry in queued_entries]
        global_waits.extend(
            max(0.0, item.started_at - item.entry.queued_at) for item in self._active.values()
        )
        global_average_wait = sum(global_waits) / len(global_waits) if global_waits else 0.0
        global_stats = {
            "active": len(self._active),
            "queued": len(queued_entries),
            "llm_active": self._global_llm_active,
            "llm_limit": self.global_llm,
            "limits": dict(self._limits),
            "kind": global_kind_stats,
            "active_by_kind": {key: value["active"] for key, value in global_kind_stats.items()},
            "queued_by_kind": {key: value["queued"] for key, value in global_kind_stats.items()},
            "wait_latency_seconds": global_average_wait,
            "wait_latency": {
                "samples": len(global_waits),
                "average_seconds": global_average_wait,
                "max_seconds": max(global_waits, default=0.0),
            },
        }
        return {
            "started": self._started and not self._closed,
            "closing": self._closing,
            "closed": self._closed,
            "active": len(self._active),
            "queued": len(queued_entries),
            "global": global_stats,
            "tasks": tasks,
            "errors": tuple(dict(error) for error in self._errors),
        }

    async def wait_for_idle(self, timeout: float | None = None) -> None:
        """Wait until no local work remains; useful for controller shutdown/tests."""

        async def _wait() -> None:
            while self._active or self._queued_count():
                if self._wake is None:
                    await asyncio.sleep(0)
                    continue
                self._wake.clear()
                if self._active or self._queued_count():
                    await self._wake.wait()

        if timeout is None:
            await _wait()
        else:
            await asyncio.wait_for(_wait(), timeout=timeout)

    async def _dispatch_loop(self) -> None:
        assert self._wake is not None
        try:
            while not self._closing:
                await self._wake.wait()
                self._wake.clear()
                while not self._closing:
                    entry = self._take_runnable()
                    if entry is None:
                        break
                    self._launch(entry)
        except asyncio.CancelledError:
            if not self._closing:
                raise

    def _launch(self, entry: _QueuedEffect) -> None:
        effect = entry.effect
        resource = self._resource_kind(effect.kind)
        self._reserve(resource, str(effect.task_id))
        token = GenerationToken(self, str(effect.task_id), int(effect.generation))
        task = asyncio.create_task(self._run_entry(entry, token), name=f"lrs-effect-{effect.event_id or effect.kind}")
        active_id = self._next_active_id
        self._next_active_id += 1
        self._active[active_id] = _ActiveEffect(entry, task, self._clock(), token)

    async def _run_entry(self, entry: _QueuedEffect, token: GenerationToken) -> None:
        active_item = next((item for item in self._active.values() if item.entry is entry), None)
        resource = self._resource_kind(entry.effect.kind)
        task_id = str(entry.effect.task_id)
        try:
            result = await self._invoke_worker(entry.effect, token)
            if self._on_result is not None:
                await self._invoke_callback(self._on_result, entry.effect, result, token)
            self._set_future_result(entry.future, result)
        except asyncio.CancelledError:
            self._cancel_future(entry.future)
            raise
        except Exception as exc:
            self._errors.append({"kind": type(exc).__name__})
            if not entry.future.done():
                entry.future.set_exception(exc)
                entry.future.add_done_callback(self._consume_future_exception)
        finally:
            for active_id, item in list(self._active.items()):
                if item.entry is entry:
                    self._active.pop(active_id, None)
                    break
            self._release(resource, task_id)
            self._signal()

    async def _invoke_worker(self, effect: Effect, token: GenerationToken) -> Any:
        target = self._worker
        if not callable(target):
            for name in ("run", "execute", "handle", "process", "start"):
                candidate = getattr(target, name, None)
                if candidate is not None:
                    target = candidate
                    break
            else:
                raise TypeError("worker 必须是可调用对象或提供 run/execute/handle/process/start")
        value = self._call_adapted(target, effect, token)
        if inspect.isawaitable(value):
            return await value
        return value

    async def _invoke_callback(self, callback: Callable[..., Any], effect: Effect, result: Any, token: GenerationToken) -> None:
        value = self._call_adapted_callback(callback, effect, result, token)
        if inspect.isawaitable(value):
            await value

    @staticmethod
    def _call_adapted(target: Callable[..., Any], effect: Effect, token: GenerationToken) -> Any:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return target(effect, token)
        for args, kwargs in (
            ((effect,), {"generation": token}),
            ((effect,), {"token": token}),
            ((effect, token), {}),
            ((effect,), {}),
        ):
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            return target(*args, **kwargs)
        raise TypeError("worker 签名必须接受 effect 或 effect+generation token")

    @staticmethod
    def _call_adapted_callback(callback: Callable[..., Any], effect: Effect, result: Any, token: GenerationToken) -> Any:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(effect, result, token)
        for args in ((effect, result, token), (effect, result), (result,)):
            try:
                signature.bind(*args)
            except TypeError:
                continue
            return callback(*args)
        raise TypeError("on_result 签名必须接受 effect/result")

    def _take_runnable(self) -> _QueuedEffect | None:
        self._discard_cancelled_queued()
        for priority in sorted(self._queues, key=self._priority_rank, reverse=True):
            task_queues = self._queues.get(priority)
            if task_queues is None:
                continue
            selected = self._take_from_priority(priority, task_queues)
            if selected is not None:
                return selected
            if priority == "barrier" and task_queues:
                return None
        return None

    def _take_from_priority(
        self, priority: str, task_queues: OrderedDict[str, deque[_QueuedEffect]]
    ) -> _QueuedEffect | None:
        if not task_queues:
            self._cleanup_priority(priority, task_queues)
            return None
        keys = list(task_queues)
        cursor = self._cursor.get(priority)
        start = keys.index(cursor) if cursor in keys else 0
        for offset in range(len(keys)):
            task_id = keys[(start + offset) % len(keys)]
            queue = task_queues[task_id]
            selected_index: int | None = None
            for index, entry in enumerate(queue):
                if self._can_start(entry.effect):
                    selected_index = index
                    break
            if selected_index is None:
                continue
            queue.rotate(-selected_index)
            entry = queue.popleft()
            queue.rotate(selected_index)
            selected_index_before_cleanup = keys.index(task_id)
            if not queue:
                task_queues.pop(task_id, None)
            remaining = list(task_queues)
            if remaining:
                if task_id in task_queues:
                    self._cursor[priority] = remaining[(remaining.index(task_id) + 1) % len(remaining)]
                else:
                    self._cursor[priority] = remaining[selected_index_before_cleanup % len(remaining)]
            else:
                self._cleanup_priority(priority, task_queues)
            self._dispatched[priority] = True
            return entry
        return None

    def _discard_cancelled_queued(self) -> None:
        for priority, task_queues in list(self._queues.items()):
            for task_id, queue in list(task_queues.items()):
                kept: deque[_QueuedEffect] = deque()
                while queue:
                    entry = queue.popleft()
                    if self._is_generation_cancelled(str(entry.effect.task_id), int(entry.effect.generation)):
                        self._cancel_future(entry.future)
                    else:
                        kept.append(entry)
                if kept:
                    task_queues[task_id] = kept
                else:
                    task_queues.pop(task_id, None)
            self._cleanup_priority(priority, task_queues)

    def _cleanup_priority(self, priority: str, task_queues: OrderedDict[str, deque[_QueuedEffect]]) -> None:
        if task_queues:
            if self._cursor.get(priority) not in task_queues:
                self._cursor[priority] = next(iter(task_queues))
            return
        self._queues.pop(priority, None)
        self._cursor.pop(priority, None)
        self._dispatched.pop(priority, None)

    def _can_start(self, effect: Effect) -> bool:
        task_id = str(effect.task_id)
        if self._is_generation_cancelled(task_id, int(effect.generation)):
            return False
        resource = self._resource_kind(effect.kind)
        if task_id in self._paused and resource in {"agent", "summarizer"}:
            return False
        if resource == "agent":
            return self._global_llm_active < self.global_llm and self._task_llm_active[task_id] < self.per_task_llm
        if resource == "summarizer":
            return self._global_llm_active < self.global_llm
        if resource == "procedure":
            return self._task_procedure_active[task_id] < self.per_task_procedure
        return True

    def _reserve(self, resource: str, task_id: str) -> None:
        if resource in {"agent", "summarizer"}:
            self._global_llm_active += 1
        if resource == "agent":
            self._task_llm_active[task_id] += 1
        elif resource == "procedure":
            self._task_procedure_active[task_id] += 1

    def _release(self, resource: str, task_id: str) -> None:
        if resource in {"agent", "summarizer"}:
            self._global_llm_active = max(0, self._global_llm_active - 1)
        if resource == "agent":
            self._task_llm_active[task_id] = max(0, self._task_llm_active[task_id] - 1)
        elif resource == "procedure":
            self._task_procedure_active[task_id] = max(0, self._task_procedure_active[task_id] - 1)

    @classmethod
    def _priority_key(cls, priority: Any) -> str:
        value = str(getattr(priority, "value", priority) or "normal").lower()
        return value

    @classmethod
    def _priority_rank(cls, priority: str) -> int:
        return cls._PRIORITY_RANK.get(priority, cls._PRIORITY_RANK["normal"])

    @staticmethod
    def _resource_kind(kind: Any) -> str:
        value = str(getattr(kind, "value", kind) or "").lower()
        if "procedure" in value:
            return "procedure"
        if value in {"agent", "llm", "agent_call", "performagentcall"} or (
            "agent" in value and "summary" not in value and "summar" not in value
        ):
            return "agent"
        if value in {"summarizer", "summary", "formalization", "formalize", "compact", "report"} or (
            "summary" in value or "summar" in value or value.startswith("formal")
        ):
            return "summarizer"
        return "other"

    def _is_generation_cancelled(self, task_id: str, generation: int) -> bool:
        return int(generation) <= self._cancelled_through.get(str(task_id), -1)

    def _queued_count(self) -> int:
        return sum(len(queue) for task_queues in self._queues.values() for queue in task_queues.values())

    def _signal(self) -> None:
        if self._wake is not None:
            self._wake.set()

    @staticmethod
    def _cancel_future(future: asyncio.Future[Any]) -> None:
        if not future.done():
            future.cancel()

    @staticmethod
    def _set_future_result(future: asyncio.Future[Any], result: Any) -> None:
        if not future.done():
            future.set_result(result)

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()


__all__ = ["FairScheduler", "GenerationToken"]
