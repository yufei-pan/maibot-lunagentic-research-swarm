"""记忆类 builtin Procedures：安全参数、限额与 Host capability 失败语义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider


@dataclass
class FakeMessageCapability:
    recent_calls: list[tuple[str, int]] = field(default_factory=list)
    by_id_calls: list[dict[str, Any]] = field(default_factory=list)
    time_range_calls: list[dict[str, Any]] = field(default_factory=list)
    fail: Exception | None = None
    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"message_id": "m1", "timestamp": "2026-01-01T00:00:00Z", "text": "hello"},
            {"message_id": "m2", "timestamp": "2026-01-01T00:01:00Z", "text": "world"},
        ]
    )

    async def get_recent(self, chat_id: str, limit: int = 10) -> Any:
        self.recent_calls.append((chat_id, limit))
        if self.fail is not None:
            raise self.fail
        return list(self.messages)[:limit]

    async def get_by_id(
        self,
        message_id: str,
        *,
        chat_id: str = "",
        stream_id: str = "",
        include_binary_data: bool = False,
    ) -> Any:
        self.by_id_calls.append(
            {
                "message_id": message_id,
                "chat_id": chat_id,
                "stream_id": stream_id,
                "include_binary_data": include_binary_data,
            }
        )
        if self.fail is not None:
            raise self.fail
        if include_binary_data:
            return {"message_id": message_id, "image_base64": "SECRET"}
        return {"message_id": message_id, "timestamp": "2026-01-01T00:00:00Z", "text": "body"}

    async def build_readable(self, messages: Any, **kwargs: Any) -> Any:
        if self.fail is not None:
            raise self.fail
        texts = []
        for item in messages or []:
            if isinstance(item, dict):
                texts.append(str(item.get("text") or item.get("message_id") or ""))
            else:
                texts.append(str(item))
        return "\n".join(texts)

    async def get_by_time_in_chat(self, chat_id: str, start_time: str, end_time: str, **kwargs: Any) -> Any:
        self.time_range_calls.append(
            {"chat_id": chat_id, "start_time": start_time, "end_time": end_time, **kwargs}
        )
        if self.fail is not None:
            raise self.fail
        return list(self.messages)


@dataclass
class FakeChatCapability:
    calls: list[tuple[str, str]] = field(default_factory=list)
    fail: Exception | None = None

    async def get_all_streams(self, platform: str = "qq") -> Any:
        self.calls.append(("all", platform))
        if self.fail is not None:
            raise self.fail
        return [{"stream_id": "s-all", "platform": platform}]

    async def get_group_streams(self, platform: str = "qq") -> Any:
        self.calls.append(("group", platform))
        if self.fail is not None:
            raise self.fail
        return [{"stream_id": "s-group", "platform": platform}]

    async def get_private_streams(self, platform: str = "qq") -> Any:
        self.calls.append(("private", platform))
        if self.fail is not None:
            raise self.fail
        return [{"stream_id": "s-private", "platform": platform}]


@dataclass
class FakePersonCapability:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail: Exception | None = None

    async def get_id(self, platform: str, user_id: str) -> Any:
        self.calls.append(("id", {"platform": platform, "user_id": user_id}))
        if self.fail is not None:
            raise self.fail
        return "person-1"

    async def get_id_by_name(self, person_name: str) -> Any:
        self.calls.append(("name", {"person_name": person_name}))
        if self.fail is not None:
            raise self.fail
        return "person-2"

    async def get_value(self, person_id: str, field_name: str) -> Any:
        self.calls.append(("field", {"person_id": person_id, "field_name": field_name}))
        if self.fail is not None:
            raise self.fail
        return "Alice"


@dataclass
class FakeKnowledgeCapability:
    calls: list[tuple[str, int]] = field(default_factory=list)
    fail: Exception | None = None

    async def search(self, query: str, limit: int = 5) -> Any:
        self.calls.append((query, limit))
        if self.fail is not None:
            raise self.fail
        return [{"title": "hit", "snippet": "snippet"}]


@dataclass
class FakeCtx:
    chat: FakeChatCapability = field(default_factory=FakeChatCapability)
    message: FakeMessageCapability = field(default_factory=FakeMessageCapability)
    person: FakePersonCapability = field(default_factory=FakePersonCapability)
    knowledge: FakeKnowledgeCapability = field(default_factory=FakeKnowledgeCapability)


@pytest.fixture
def memory_provider() -> BundledProcedureProvider:
    return BundledProcedureProvider(FakeCtx())


@pytest.mark.asyncio
async def test_recent_messages_never_requests_binary_data(memory_provider) -> None:
    result = await memory_provider.invoke("builtin.message_recent", {"stream_id": "s", "limit": 8})
    assert result.success
    assert memory_provider.ctx.message.recent_calls == [("s", 8)]
    assert "image_base64" not in repr(result.data)


@pytest.mark.asyncio
async def test_memory_limits_are_explicit(memory_provider) -> None:
    result = await memory_provider.invoke("builtin.message_recent", {"stream_id": "s", "limit": 500})
    assert not result.success
    assert result.error.code == "invalid_arguments"


@pytest.mark.asyncio
async def test_message_by_id_never_requests_binary_data(memory_provider) -> None:
    result = await memory_provider.invoke(
        "builtin.message_by_id",
        {"message_id": "m1", "stream_id": "s"},
    )
    assert result.success
    assert memory_provider.ctx.message.by_id_calls == [
        {
            "message_id": "m1",
            "chat_id": "",
            "stream_id": "s",
            "include_binary_data": False,
        }
    ]
    assert "image_base64" not in repr(result.data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("procedure_id", "arguments"),
    [
        ("builtin.message_recent", {"stream_id": "s", "limit": 0}),
        ("builtin.message_recent", {"stream_id": "s", "limit": 51}),
        ("builtin.message_time_range", {"stream_id": "s", "start_time": "a", "end_time": "b", "limit": 0}),
        ("builtin.message_time_range", {"stream_id": "s", "start_time": "a", "end_time": "b", "limit": 101}),
        ("builtin.knowledge_search", {"query": "q", "limit": 0}),
        ("builtin.knowledge_search", {"query": "q", "limit": 21}),
        ("builtin.knowledge_search", {"query": "", "limit": 5}),
        ("builtin.knowledge_search", {"query": "x" * 2001, "limit": 5}),
    ],
)
async def test_explicit_limits_reject_out_of_range(
    memory_provider: BundledProcedureProvider,
    procedure_id: str,
    arguments: dict[str, Any],
) -> None:
    result = await memory_provider.invoke(procedure_id, arguments)
    assert not result.success
    assert result.error.code == "invalid_arguments"


@pytest.mark.asyncio
async def test_sdk_failure_returns_host_capability_failed_not_empty_success(
    memory_provider: BundledProcedureProvider,
) -> None:
    memory_provider.ctx.message.fail = RuntimeError("host down")
    result = await memory_provider.invoke("builtin.message_recent", {"stream_id": "s", "limit": 8})
    assert not result.success
    assert result.error.code == "host_capability_failed"
    assert result.data is None


@pytest.mark.asyncio
async def test_chat_streams_dispatch_by_kind(memory_provider: BundledProcedureProvider) -> None:
    for kind, method in (("all", "all"), ("group", "group"), ("private", "private")):
        result = await memory_provider.invoke(
            "builtin.chat_streams",
            {"kind": kind, "platform": "qq"},
        )
        assert result.success
        assert result.data["items"]
        assert memory_provider.ctx.chat.calls[-1] == (method, "qq")


@pytest.mark.asyncio
async def test_message_recent_keeps_minimal_ids_and_readable(
    memory_provider: BundledProcedureProvider,
) -> None:
    result = await memory_provider.invoke("builtin.message_recent", {"stream_id": "s", "limit": 2})
    assert result.success
    assert result.data["stream_id"] == "s"
    assert result.data["truncated"] == 2
    assert "readable" in result.data
    assert result.data["items"] == [
        {"message_id": "m1", "timestamp": "2026-01-01T00:00:00Z"},
        {"message_id": "m2", "timestamp": "2026-01-01T00:01:00Z"},
    ]


@pytest.mark.asyncio
async def test_message_time_range_truncates_to_limit(memory_provider: BundledProcedureProvider) -> None:
    memory_provider.ctx.message.messages = [
        {"message_id": f"m{i}", "timestamp": f"t{i}", "text": f"body-{i}"} for i in range(5)
    ]
    result = await memory_provider.invoke(
        "builtin.message_time_range",
        {"stream_id": "s", "start_time": "a", "end_time": "b", "limit": 2},
    )
    assert result.success
    assert result.data["truncated"] == 2
    assert len(result.data["items"]) == 2
    assert result.data["stream_id"] == "s"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "arguments", "expected_call"),
    [
        ("id", {"mode": "id", "platform": "qq", "user_id": "u1"}, ("id", {"platform": "qq", "user_id": "u1"})),
        ("name", {"mode": "name", "person_name": "Alice"}, ("name", {"person_name": "Alice"})),
        (
            "field",
            {"mode": "field", "person_id": "p1", "field_name": "name"},
            ("field", {"person_id": "p1", "field_name": "name"}),
        ),
    ],
)
async def test_person_lookup_calls_matching_capability(
    memory_provider: BundledProcedureProvider,
    mode: str,
    arguments: dict[str, Any],
    expected_call: tuple[str, dict[str, Any]],
) -> None:
    result = await memory_provider.invoke("builtin.person_lookup", arguments)
    assert result.success
    assert memory_provider.ctx.person.calls == [expected_call]
    assert result.data["items"]


@pytest.mark.asyncio
async def test_knowledge_search_returns_query_and_items(memory_provider: BundledProcedureProvider) -> None:
    result = await memory_provider.invoke(
        "builtin.knowledge_search",
        {"query": "麦麦", "limit": 3},
    )
    assert result.success
    assert result.data["query"] == "麦麦"
    assert result.data["truncated"] == 1
    assert memory_provider.ctx.knowledge.calls == [("麦麦", 3)]
