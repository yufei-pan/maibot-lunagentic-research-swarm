"""研究 credits 数学、不可变账本条目与 LLM 预扣/核销命令。

这个模块只构造值对象和 :class:`~lunagentic_research_swarm.storage.sqlite.StoreCommand`，
不访问 store，也不启动 effect。调用方可以把返回的 commands 与同一状态转移中的
其他命令一起提交。
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from lunagentic_research_swarm.llm.pricing import ChargedUsage, PriceCatalog, TokenUsage, normalize_actual_usage
from lunagentic_research_swarm.models import CreditLedgerEntry, new_ledger_id
from lunagentic_research_swarm.storage.sqlite import StoreCommand


def _finite(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} 必须为有限数")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须为有限数")
    return normalized


def _nonnegative_finite(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} 必须为非负有限数")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} 必须为非负有限数")
    return normalized


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须为非负整数")
    return value


def _timestamp(value: float | datetime) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return _finite(value, "created_at")


def _immutable_balances(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(values))


def _new_usage_id() -> str:
    return f"usage_{uuid.uuid4().hex}"


def _contains_reasoning(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(key == "reasoning" or _contains_reasoning(item) for key, item in value.items())
    if isinstance(value, list | tuple):
        return any(_contains_reasoning(item) for item in value)
    return False


def _metadata_json(metadata: Mapping[str, Any] | None) -> str:
    safe = dict(metadata or {})
    if _contains_reasoning(safe):
        raise ValueError("credits metadata 不得包含 reasoning")
    try:
        return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("credits metadata 必须可编码为 JSON") from exc


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """子分支分配结果；mapping 与结果对象都不可变。"""

    allocations: Mapping[str, float]
    returned_to_pool: float
    launch_allowed: bool
    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "allocations", _immutable_balances(self.allocations))
        object.__setattr__(self, "returned_to_pool", _finite(self.returned_to_pool, "returned_to_pool"))
        object.__setattr__(self, "scale", _finite(self.scale, "scale"))


@dataclass(frozen=True, slots=True)
class RedistributionResult:
    """continue 屏障完成后的活动叶子与 dormant pool。"""

    balances: Mapping[str, float]
    pool_after: float
    restart_balance: float | None = None
    can_start_root: bool = False
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "balances", _immutable_balances(self.balances))
        object.__setattr__(self, "pool_after", _finite(self.pool_after, "pool_after"))
        if self.restart_balance is not None:
            object.__setattr__(self, "restart_balance", _finite(self.restart_balance, "restart_balance"))


@dataclass(frozen=True, slots=True)
class ReservationResult:
    """一次 input reservation 的不可变 command bundle。"""

    estimated_charge: float
    commands: tuple[StoreCommand, ...]
    usage_command: StoreCommand
    ledger_command: StoreCommand | None
    ledger_entry: CreditLedgerEntry | None
    usage_id: str
    task_id: str
    round_id: str
    branch_id: str | None
    call_id: str
    role: str
    selector: str
    estimated_model_name: str | None
    price_source: str
    price_fingerprint: str
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    duration_ms: int
    created_at: float

    @property
    def amount(self) -> float:
        return self.estimated_charge

    @property
    def reservation_command(self) -> StoreCommand:
        return self.ledger_command or self.usage_command


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """一次调用完成后的 usage telemetry 与 signed ledger adjustment。"""

    status: Literal["actual", "estimated", "estimated_unreconciled"]
    estimated_charge: float
    actual_charge: float | None
    adjustment: float
    commands: tuple[StoreCommand, ...]
    usage_command: StoreCommand
    ledger_command: StoreCommand | None
    charged: ChargedUsage | None = None

    @property
    def actual_credits(self) -> float:
        """兼容 pricing 层命名；无实际 usage 时返回保守预估。"""

        return self.estimated_charge if self.actual_charge is None else self.actual_charge

    @property
    def adjustment_credits(self) -> float:
        return self.adjustment


# 公开别名让后续 reducer 可以按语义命名 command bundle。
InputReservation = ReservationResult
UsageReconciliation = ReconciliationResult


def _scaled_with_last_item_remainder(
    items: Sequence[tuple[str, float]], scale: float, target: float
) -> dict[str, float]:
    if not items:
        return {}
    allocations = {key: value * scale for key, value in items[:-1]}
    remainder = target - math.fsum(allocations.values())
    # 浮点乘法可能让非末项之和比 target 多一个 ulp；把这个余差从
    # 最后一个已分配的非末项吸收，避免本应为零的末项变成负 credits。
    if remainder < 0 and scale >= 0:
        deficit = -remainder
        for key, _ in reversed(items[:-1]):
            reducible = min(allocations[key], deficit)
            allocations[key] -= reducible
            deficit -= reducible
            if deficit <= 0:
                break
        remainder = target - math.fsum(allocations.values())
    last_key, _ = items[-1]
    allocations[last_key] = max(0.0, remainder) if scale >= 0 else remainder
    return allocations


def allocate_children(balance: float, requests: Sequence[tuple[str, float]]) -> AllocationResult:
    """按父分支余额比例缩放委派请求，零余额仍允许启动。"""

    normalized_balance = _finite(balance, "balance")
    normalized_requests = [
        (str(key), _nonnegative_finite(value, f"requests[{key}]")) for key, value in requests
    ]
    if normalized_balance < 0:
        return AllocationResult({}, 0.0, launch_allowed=False, scale=0.0)
    requested_total = math.fsum(value for _, value in normalized_requests)
    if not math.isfinite(requested_total):
        raise ValueError("requests 总和必须为有限数")
    if requested_total == 0:
        return AllocationResult(
            {key: 0.0 for key, _ in normalized_requests}, normalized_balance, launch_allowed=True, scale=1.0
        )
    scale = min(1.0, normalized_balance / requested_total)
    allocations = _scaled_with_last_item_remainder(
        normalized_requests, scale, min(normalized_balance, requested_total)
    )
    returned = normalized_balance - math.fsum(allocations.values())
    if abs(returned) <= 1e-12 * max(1.0, abs(normalized_balance)):
        returned = 0.0
    return AllocationResult(allocations, returned, launch_allowed=True, scale=scale)


def settle_branch(balance: float, pool: float) -> float:
    """把终结分支的有符号余额原样结算到 dormant pool。"""

    return math.fsum((_finite(pool, "pool"), _finite(balance, "balance")))


def redistribute_pool(
    pool: float,
    adjustment: float,
    leaves: Mapping[str, float],
) -> RedistributionResult:
    """仅在 continue 时把 pool 与有符号 adjustment 分配给活动叶子。"""

    available = math.fsum((_finite(pool, "pool"), _finite(adjustment, "adjustment")))
    normalized_leaves = {str(key): _finite(value, f"leaves[{key}]") for key, value in leaves.items()}
    if not normalized_leaves:
        can_start_root = available >= 0
        return RedistributionResult(
            {},
            pool_after=available,
            restart_balance=available,
            can_start_root=can_start_root,
            finish_reason=None if can_start_root else "task_finished_insufficient_funds",
        )

    positive = [(key, balance) for key, balance in normalized_leaves.items() if balance > 0]
    if positive:
        positive_total = math.fsum(balance for _, balance in positive)
        changes = _scaled_with_last_item_remainder(positive, available / positive_total, available)
    else:
        even = available / len(normalized_leaves)
        changes = _scaled_with_last_item_remainder(
            [(key, 1.0) for key in normalized_leaves], even, available
        )
    balances = {key: balance + changes.get(key, 0.0) for key, balance in normalized_leaves.items()}
    return RedistributionResult(balances, pool_after=0.0)


def assert_task_credit_equation(
    *,
    initial: float,
    signed_adjustments: Sequence[float],
    charged_agent_costs: Sequence[float],
    active_balances: Mapping[str, float],
    dormant_pool: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-7,
) -> None:
    """断言研究 credits 守恒，不把总结器/Procedure 费用混入等式。"""

    initial_value = _finite(initial, "initial")
    adjustments = [_finite(value, "signed_adjustments") for value in signed_adjustments]
    charges = [_nonnegative_finite(value, "charged_agent_costs") for value in charged_agent_costs]
    active = {str(key): _finite(value, f"active_balances[{key}]") for key, value in active_balances.items()}
    pool = _finite(dormant_pool, "dormant_pool")
    expected = math.fsum((initial_value, math.fsum(adjustments), -math.fsum(charges)))
    observed = math.fsum((*active.values(), pool))
    if not math.isclose(expected, observed, rel_tol=rel_tol, abs_tol=abs_tol):
        raise AssertionError(
            "task credits 不守恒："
            f"expected={expected!r}, observed={observed!r}, difference={observed - expected!r}"
        )


def _token_fields(
    usage: ChargedUsage | TokenUsage | Mapping[str, Any] | None,
    *,
    prompt_tokens: Any = 0,
    completion_tokens: Any = 0,
    cache_hit_tokens: Any = 0,
    cache_miss_tokens: Any = 0,
) -> tuple[int, int, int, int]:
    if isinstance(usage, ChargedUsage):
        usage = usage.usage
    if isinstance(usage, TokenUsage):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cache_hit_tokens = usage.cache_hit_tokens
        cache_miss_tokens = usage.cache_miss_tokens
    elif isinstance(usage, Mapping):
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        cache_hit_tokens = usage.get("cache_hit_tokens", cache_hit_tokens)
        cache_miss_tokens = usage.get("cache_miss_tokens", cache_miss_tokens)
    values = (
        _nonnegative_int(prompt_tokens, "prompt_tokens"),
        _nonnegative_int(completion_tokens, "completion_tokens"),
        _nonnegative_int(cache_hit_tokens, "cache_hit_tokens"),
        _nonnegative_int(cache_miss_tokens, "cache_miss_tokens"),
    )
    return values


def _role_is_research(role: str) -> bool:
    return role.casefold() not in {
        "summarizer",
        "summary",
        "formalizer",
        "formalization",
        "checkpoint_summarizer",
        "branch_summarizer",
        "task_summarizer",
    }


def reserve_input(
    estimated_charge: float | ChargedUsage | None = None,
    *,
    task_id: str = "",
    round_id: str = "",
    branch_id: str | None = None,
    call_id: str = "",
    usage_id: str | None = None,
    ledger_id: str | None = None,
    role: str = "agent",
    selector: str = "",
    estimated_model_name: str | None = None,
    price_source: str = "host_config",
    price_fingerprint: str = "",
    estimated_usage: ChargedUsage | TokenUsage | Mapping[str, Any] | None = None,
    catalog: PriceCatalog | None = None,
    prompt_tokens: Any = 0,
    completion_tokens: Any = 0,
    cache_hit_tokens: Any = 0,
    cache_miss_tokens: Any = 0,
    balance_after: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    duration_ms: int = 0,
    created_at: float | datetime = 0.0,
) -> ReservationResult:
    """构造 input reservation 的 usage 与 research ledger commands。

    ``estimated_charge`` 可直接给出，也可传入 ``ChargedUsage``。若 role 是总结器，
    只保留 usage telemetry，不产生研究 credits ledger。
    """

    charged_estimate: ChargedUsage | None = None
    if isinstance(estimated_charge, ChargedUsage):
        charged_estimate = estimated_charge
        estimated_charge = charged_estimate.credits
    if isinstance(estimated_usage, ChargedUsage):
        charged_estimate = estimated_usage
        if estimated_charge is None:
            estimated_charge = charged_estimate.credits
    if estimated_charge is None:
        raise ValueError("必须提供 estimated_charge 或 ChargedUsage")
    estimate = _nonnegative_finite(estimated_charge, "estimated_charge")
    prompt, completion, hit, miss = _token_fields(
        estimated_usage,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
    )
    if charged_estimate is not None:
        if estimated_model_name is None:
            estimated_model_name = charged_estimate.price.model_name
        if not price_source or price_source == "host_config":
            price_source = charged_estimate.price.source
    if catalog is not None and not price_fingerprint:
        price_fingerprint = catalog.fingerprint
    if not isinstance(role, str) or not role:
        raise ValueError("role 必须为非空字符串")
    if not isinstance(selector, str):
        raise ValueError("selector 必须为字符串")
    duration = _nonnegative_int(duration_ms, "duration_ms")
    timestamp = _timestamp(created_at)
    usage_key = usage_id or _new_usage_id()
    call_key = call_id or usage_key
    metadata_values = {
        "estimated_model_name": estimated_model_name,
        "actual_model_name": None,
        "price_source": price_source,
        "price_fingerprint": price_fingerprint,
        "reconciliation_status": "reserved",
        **dict(metadata or {}),
    }
    metadata_text = _metadata_json(metadata_values)
    usage_values: dict[str, Any] = {
        "usage_id": usage_key,
        "task_id": task_id,
        "round_id": round_id,
        "branch_id": branch_id,
        "call_id": call_key,
        "role": role,
        "selector": selector,
        "estimated_model_name": estimated_model_name,
        "actual_model_name": None,
        "price_source": price_source,
        "price_fingerprint": price_fingerprint,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "estimated_charge": estimate,
        "actual_charge": None,
        "adjustment": None,
        "reconciliation_status": "reserved",
        "duration_ms": duration,
        "created_at": timestamp,
    }
    usage_command = StoreCommand("insert_llm_usage", usage_values)

    ledger_entry: CreditLedgerEntry | None = None
    ledger_command: StoreCommand | None = None
    if _role_is_research(role):
        ledger_key = ledger_id or new_ledger_id()
        ledger_entry = CreditLedgerEntry(
            ledger_id=ledger_key,
            task_id=task_id,
            round_id=round_id,
            branch_id=branch_id,
            call_id=call_key,
            entry_kind="input_reservation",
            amount=-estimate,
            balance_after=balance_after,
            metadata_json=metadata_text,
            created_at=timestamp,
        )
        ledger_command = StoreCommand(
            "insert_credit_ledger",
            {
                "ledger_id": ledger_entry.ledger_id,
                "task_id": ledger_entry.task_id,
                "round_id": ledger_entry.round_id,
                "branch_id": ledger_entry.branch_id,
                "call_id": ledger_entry.call_id,
                "entry_kind": ledger_entry.entry_kind,
                "amount": ledger_entry.amount,
                "balance_after": ledger_entry.balance_after,
                "metadata_json": ledger_entry.metadata_json,
                "created_at": ledger_entry.created_at,
            },
        )

    commands = (usage_command,) if ledger_command is None else (usage_command, ledger_command)
    return ReservationResult(
        estimated_charge=estimate,
        commands=commands,
        usage_command=usage_command,
        ledger_command=ledger_command,
        ledger_entry=ledger_entry,
        usage_id=usage_key,
        task_id=task_id,
        round_id=round_id,
        branch_id=branch_id,
        call_id=call_key,
        role=role,
        selector=selector,
        estimated_model_name=estimated_model_name,
        price_source=price_source,
        price_fingerprint=price_fingerprint,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        duration_ms=duration,
        created_at=timestamp,
    )


def reconcile_usage(
    reservation: ReservationResult | None = None,
    *,
    catalog: PriceCatalog | None = None,
    actual_model_name: str | None = None,
    usage: TokenUsage | Mapping[str, Any] | None = None,
    success: bool = True,
    estimated_charge: float | None = None,
    actual_charge: float | None = None,
    task_id: str = "",
    round_id: str = "",
    branch_id: str | None = None,
    call_id: str = "",
    role: str = "agent",
    selector: str = "",
    estimated_model_name: str | None = None,
    price_source: str = "host_config",
    price_fingerprint: str = "",
    usage_id: str | None = None,
    balance_after: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    duration_ms: int | None = None,
    created_at: float | datetime = 0.0,
) -> ReconciliationResult:
    """使用 whole-call 实际模型/usage 核销 reservation。

    没有 Host usage 的失败不会释放 reservation，状态固定为
    ``estimated_unreconciled``，且不会产生补偿 ledger entry。
    """

    if reservation is not None:
        if not isinstance(reservation, ReservationResult):
            raise TypeError("reservation 必须为 ReservationResult")
        estimated = reservation.estimated_charge
        task_id = reservation.task_id
        round_id = reservation.round_id
        branch_id = reservation.branch_id
        call_id = reservation.call_id
        role = reservation.role
        selector = reservation.selector
        estimated_model_name = reservation.estimated_model_name
        price_source = reservation.price_source
        price_fingerprint = reservation.price_fingerprint
        if duration_ms is None:
            duration_ms = reservation.duration_ms
        if created_at == 0.0:
            created_at = reservation.created_at
        reserved_tokens = (
            reservation.prompt_tokens,
            reservation.completion_tokens,
            reservation.cache_hit_tokens,
            reservation.cache_miss_tokens,
        )
        if usage_id is None:
            usage_id = reservation.usage_id
    else:
        if estimated_charge is None:
            raise ValueError("无 reservation 时必须提供 estimated_charge")
        estimated = _nonnegative_finite(estimated_charge, "estimated_charge")
        reserved_tokens = (0, 0, 0, 0)
    if reservation is None:
        estimated = _nonnegative_finite(estimated, "estimated_charge")
    duration = _nonnegative_int(0 if duration_ms is None else duration_ms, "duration_ms")
    timestamp = _timestamp(created_at)
    usage_key = usage_id or _new_usage_id()

    normalized_usage: TokenUsage | None = None
    charged: ChargedUsage | None = None
    status: Literal["actual", "estimated", "estimated_unreconciled"]
    actual: float | None
    adjustment: float
    if usage is None:
        status = "estimated" if success else "estimated_unreconciled"
        actual = None
        adjustment = 0.0
        prompt, completion, hit, miss = reserved_tokens
    else:
        prompt, completion, hit, miss = _token_fields(usage)
        normalized_usage = normalize_actual_usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
        )
        model_name = actual_model_name or estimated_model_name or ""
        if actual_charge is not None:
            actual = _nonnegative_finite(actual_charge, "actual_charge")
        elif catalog is not None:
            charged = catalog.charge_actual(
                actual_model_name=model_name,
                prompt_tokens=normalized_usage.prompt_tokens,
                completion_tokens=normalized_usage.completion_tokens,
                cache_hit_tokens=normalized_usage.cache_hit_tokens,
                cache_miss_tokens=normalized_usage.cache_miss_tokens,
            )
            actual = charged.credits
        else:
            raise ValueError("有 usage 时必须提供 catalog 或 actual_charge")
        status = "actual" if actual_model_name else "estimated"
        adjustment = estimated - actual

        # 核销审计必须描述实际计费快照，而不是沿用 reservation 时的
        # 估算模型/价格来源。没有实际模型名时，使用明确的 estimated
        # fallback 模型解析当前目录；该状态仍由上面的 status 标记。
        if catalog is not None:
            provenance_model_name = actual_model_name or estimated_model_name or ""
            resolved = catalog.resolve_model(provenance_model_name, actual=bool(actual_model_name))
            price_source = charged.price.source if charged is not None else resolved.source
            price_fingerprint = catalog.fingerprint

    metadata_values = {
        "estimated_model_name": estimated_model_name,
        "actual_model_name": actual_model_name,
        "price_source": price_source,
        "price_fingerprint": price_fingerprint,
        "reconciliation_status": status,
        "estimated_charge": estimated,
        "actual_charge": actual,
        "adjustment": adjustment,
        **dict(metadata or {}),
    }
    metadata_text = _metadata_json(metadata_values)
    usage_values: dict[str, Any] = {
        "usage_id": f"{usage_key}:reconciled",
        "task_id": task_id,
        "round_id": round_id,
        "branch_id": branch_id,
        "call_id": call_id or usage_key,
        "role": role,
        "selector": selector,
        "estimated_model_name": estimated_model_name,
        "actual_model_name": actual_model_name,
        "price_source": price_source,
        "price_fingerprint": price_fingerprint,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "estimated_charge": estimated,
        "actual_charge": actual,
        "adjustment": adjustment,
        "reconciliation_status": status,
        "duration_ms": duration,
        "created_at": timestamp,
    }
    usage_command = StoreCommand("insert_llm_usage", usage_values)

    ledger_command: StoreCommand | None = None
    if usage is not None and _role_is_research(role):
        entry = CreditLedgerEntry(
            ledger_id=new_ledger_id(),
            task_id=task_id,
            round_id=round_id,
            branch_id=branch_id,
            call_id=call_id or usage_key,
            entry_kind="input_reconciliation",
            amount=adjustment,
            balance_after=balance_after,
            metadata_json=metadata_text,
            created_at=timestamp,
        )
        ledger_command = StoreCommand(
            "insert_credit_ledger",
            {
                "ledger_id": entry.ledger_id,
                "task_id": entry.task_id,
                "round_id": entry.round_id,
                "branch_id": entry.branch_id,
                "call_id": entry.call_id,
                "entry_kind": entry.entry_kind,
                "amount": entry.amount,
                "balance_after": entry.balance_after,
                "metadata_json": entry.metadata_json,
                "created_at": entry.created_at,
            },
        )
    commands = (usage_command,) if ledger_command is None else (usage_command, ledger_command)
    return ReconciliationResult(
        status=status,
        estimated_charge=estimated,
        actual_charge=actual,
        adjustment=adjustment,
        commands=commands,
        usage_command=usage_command,
        ledger_command=ledger_command,
        charged=charged,
    )
