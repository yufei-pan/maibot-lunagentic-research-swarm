"""扩展契约共用的严格验证与稳定指纹。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

EXTENSION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,127}$")


class FrozenList(list[Any]):
    """仍按 JSON array 序列化、但拒绝原地改写的列表。"""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("扩展契约已经冻结")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __copy__(self) -> "FrozenList":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenList":
        return self


class FrozenDict(dict[str, Any]):
    """仍按 JSON object 序列化、但拒绝原地改写的字典。"""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("扩展契约已经冻结")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenDict":
        return self


def freeze_json(value: Any) -> Any:
    """递归复制并冻结 JSON 容器，同时保持 Pydantic 的 JSON 序列化能力。"""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object 的 key 必须为字符串")
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_json(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("扩展契约值必须是有限且可序列化的 JSON 值")


def validate_extension_id(value: str, *, field_name: str) -> str:
    """验证 agent/Procedure 的公共 ID 语法。"""

    if not isinstance(value, str) or EXTENSION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} 必须匹配 ^[a-z0-9][a-z0-9_.-]{{2,127}}$")
    return value


def authorized_provider_namespace(provider_id: str) -> str:
    """由 Host plugin ID 纯函数决定唯一可声明的扩展命名空间。"""

    if not isinstance(provider_id, str) or not provider_id or provider_id != provider_id.strip():
        raise ValueError("provider plugin ID 不能为空或包含首尾空白")
    namespace = "builtin" if provider_id == "builtin" else provider_id.rsplit(".", 1)[-1].replace("-", "_")
    if re.fullmatch(r"[a-z0-9][a-z0-9_]{0,127}", namespace) is None:
        raise ValueError(f"provider plugin ID {provider_id} 无法映射为合法命名空间")
    return namespace


def validate_model_selector(value: str) -> str:
    """只接受带非空后缀的 task:/model: selector，不做静默 fallback。"""

    if not isinstance(value, str) or value != value.strip():
        raise ValueError("模型选择器必须使用 task:<名称> 或 model:<名称>")
    prefix, separator, suffix = value.partition(":")
    if separator != ":" or prefix not in {"task", "model"} or not suffix or suffix != suffix.strip():
        raise ValueError("模型选择器必须使用 task:<名称> 或 model:<名称>")
    return value


def validate_agent_batch(provider_id: str, definitions: Sequence[Any]) -> list[Any]:
    """对一个 provider 的 agent 批次做与 registry 相同的严格校验；整批通过或整批失败。"""

    # 延迟导入以避免与 contracts 的循环依赖。
    from lunagentic_research_swarm.extensions.contracts import AgentDefinition

    authorized_namespace = authorized_provider_namespace(provider_id)
    raw_definitions: list[Mapping[str, Any]] = []
    for item in definitions:
        if isinstance(item, AgentDefinition):
            raw_definitions.append(item.model_dump(mode="python"))
        elif isinstance(item, Mapping):
            raw_definitions.append(item)
        else:
            raise TypeError("agent provider 批次只能包含 AgentDefinition 或 Mapping")
    checked = [AgentDefinition.model_validate(item) for item in raw_definitions]
    ids = [item.agent_id for item in checked]
    if len(set(ids)) != len(ids):
        raise ValueError("agent provider 批次包含重复 ID")
    for agent_id in ids:
        if agent_id.partition(".")[0] != authorized_namespace:
            raise ValueError(
                f"agent_id {agent_id} 不属于 provider {provider_id} 获授权的命名空间 {authorized_namespace}"
            )
    return checked


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _fingerprint_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical fingerprint 的 JSON object key 必须为字符串")
        return {key: _fingerprint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fingerprint_value(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("canonical fingerprint 输入必须是有限且可序列化的 JSON 值")


def canonical_fingerprint(values: Any, *, id_field: str | None = None) -> str:
    """对 JSON 值生成 keys 排序、UTF-8、无空白的 SHA-256 指纹。"""

    normalized = _fingerprint_value(values)
    if id_field is not None:
        if not isinstance(normalized, list):
            raise TypeError("带 id_field 的指纹输入必须为 JSON array")
        normalized = sorted(normalized, key=lambda item: str(item[id_field]))
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
