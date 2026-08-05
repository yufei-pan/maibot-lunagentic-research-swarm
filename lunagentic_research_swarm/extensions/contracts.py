"""agent 与 Procedure provider 共享的严格公共契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .validation import freeze_json, validate_extension_id, validate_model_selector


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    provider_plugin_id: str
    status: Literal["healthy", "invalid", "removed"]
    errors: tuple[str, ...] = ()
    fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class CatalogDelta:
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtensionRefreshEvent:
    """一次 refresh 对一个 provider 或 discovery 边界的不可变审计事件。"""

    provider_plugin_id: str
    extension_kind: Literal["agents", "procedures", "discovery"]
    availability: Literal["available", "invalid", "removed"]
    fingerprint: str
    errors: tuple[str, ...]
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(self.errors))


@dataclass(frozen=True, slots=True)
class ExtensionRefreshDelta:
    """一次 refresh 完成后交给 service 的完整持久化增量。"""

    events: tuple[ExtensionRefreshEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class AgentDefinition(_StrictContract):
    agent_id: str
    version: str = Field(min_length=1, max_length=32)
    display_name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=2000)
    character_prompt: str = Field(min_length=1, max_length=20000)
    model_selector: str
    protocol: Literal["json_envelope", "native_tools"] = "json_envelope"
    allowed_procedures: list[str] = Field(default_factory=lambda: ["*"], min_length=1)
    can_be_root: StrictBool = False
    auto_compact_tokens: int | None = Field(default=None, ge=1024)
    enabled: StrictBool = True

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, value: str) -> str:
        value = validate_extension_id(value, field_name="agent_id")
        if value == "summarizer" or value.startswith("core.") or value.endswith(".summarizer"):
            raise ValueError("agent_id 不得冒充 core 或 summarizer")
        return value

    @field_validator("model_selector", mode="before")
    @classmethod
    def _validate_selector(cls, value: Any) -> str:
        return validate_model_selector(value)

    @field_validator("allowed_procedures")
    @classmethod
    def _validate_allowed_procedures(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("allowed_procedures 不得重复")
        if "*" in values:
            if values != ["*"]:
                raise ValueError("通配符 * 不能与具体 Procedure ID 混用")
            return freeze_json(values)
        for value in values:
            validate_extension_id(value, field_name="procedure_id")
        return freeze_json(values)


class ProcedureDefinition(_StrictContract):
    procedure_id: str
    version: str = Field(min_length=1, max_length=32)
    display_name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=2000)
    arguments_schema: dict[str, Any]
    result_schema: dict[str, Any]
    idempotent: StrictBool = False
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    external_cost_kind: Literal["none", "provider_metered"] = "none"
    enabled: StrictBool = True

    @field_validator("procedure_id")
    @classmethod
    def _validate_procedure_id(cls, value: str) -> str:
        value = validate_extension_id(value, field_name="procedure_id")
        if value.startswith("core."):
            raise ValueError("procedure_id 不得使用保留的 core 命名空间")
        return value

    @field_validator("arguments_schema", "result_schema", mode="before")
    @classmethod
    def _validate_object_schema(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("type") != "object":
            raise ValueError("JSON schema 必须声明 type=object")
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("JSON schema 必须可安全序列化") from exc
        return value

    @field_validator("arguments_schema", "result_schema")
    @classmethod
    def _freeze_object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        return freeze_json(value)


class ProcedureInvocation(_StrictContract):
    request_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    round_id: str = Field(min_length=1, max_length=128)
    branch_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=128)
    agent_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    scoped_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, value: str) -> str:
        return validate_extension_id(value, field_name="agent_id")

    @field_validator("arguments", "scoped_metadata")
    @classmethod
    def _freeze_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return freeze_json(value)


class ProcedureResult(_StrictContract):
    success: StrictBool
    data: dict[str, Any] | None
    error: dict[str, Any] | None
    metadata: dict[str, Any]

    @field_validator("data", "error", "metadata")
    @classmethod
    def _freeze_payload(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return freeze_json(value)

    @model_validator(mode="after")
    def _validate_outcome(self) -> "ProcedureResult":
        if self.success and self.error is not None:
            raise ValueError("成功结果的 error 必须为 null")
        if not self.success and self.error is None:
            raise ValueError("失败结果必须包含 error")
        return self
