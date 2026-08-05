"""URL 标准化、去重与 provenance 整理：保真 query，不改写事实。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.procedures.bundled.provenance import (
    normalize_url,
    normalize_urls,
    organize_provenance,
)
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider


def test_url_normalizer_does_not_reorder_semantic_query() -> None:
    value = normalize_url("HTTPS://Example.COM:443/a?b=2&b=1#fragment")
    assert value == "https://example.com/a?b=2&b=1"


def test_url_normalizer_removes_dot_segments_and_keeps_tracking() -> None:
    value = normalize_url("https://EXAMPLE.com/a/./b/../c?utm_source=x&id=1")
    assert value == "https://example.com/a/c?utm_source=x&id=1"


def test_url_normalizer_rejects_invalid_idna() -> None:
    with pytest.raises(ValueError):
        # 空标签 + 零宽字符：idna 编码失败
        normalize_url("https://.\u200b.com/path")


def test_url_normalizer_preserves_ipv6_brackets() -> None:
    assert normalize_url("https://[::1]:443/a") == "https://[::1]/a"
    assert normalize_url("http://[2001:db8::1]:8080/x") == "http://[2001:db8::1]:8080/x"
    assert normalize_url("https://[::1]/path") == "https://[::1]/path"


def test_normalize_urls_dedupes_and_merges_source_ids() -> None:
    result = normalize_urls(
        [
            {
                "url": "HTTPS://Example.COM:443/a?b=2&b=1#x",
                "source_id": "s1",
                "title": "first",
                "snippet": "keep-me",
            },
            {
                "url": "https://example.com/a?b=2&b=1",
                "source_id": "s2",
                "title": "second",
                "snippet": "ignore-me",
            },
            {
                "url": "https://other.example/z",
                "source_id": "s3",
                "title": "other",
                "snippet": "solo",
            },
        ]
    )
    assert result.success
    items = result.data["items"]
    assert len(items) == 2
    first = items[0]
    assert first["url"] == "https://example.com/a?b=2&b=1"
    assert first["source_ids"] == ["s1", "s2"]
    assert first["title"] == "first"
    assert first["snippet"] == "keep-me"
    assert result.data["duplicate_urls"] == ["https://example.com/a?b=2&b=1"]


def test_organize_provenance_maps_claims_without_rewriting_snippets() -> None:
    result = organize_provenance(
        claims=[
            {"claim_id": "c1", "text": "地球是圆的", "source_ids": ["s1", "s2"]},
            {"claim_id": "c2", "text": "无依据断言", "source_ids": []},
            {"claim_id": "c3", "text": "缺失来源", "source_ids": ["missing"]},
        ],
        sources=[
            {
                "source_id": "s1",
                "url": "HTTPS://Example.COM:443/a?b=2&b=1#frag",
                "source_type": "web",
                "timestamp": "2026-01-01T00:00:00Z",
                "snippet": "原始摘要，勿改写",
            },
            {
                "source_id": "s2",
                "url": "https://example.com/a?b=2&b=1",
                "source_type": "web",
                "timestamp": "2026-01-02T00:00:00Z",
                "snippet": "重复 URL",
            },
            {
                "source_id": "s3",
                "url": "https://other.example/doc",
                "source_type": "memory",
                "timestamp": "2026-01-03T00:00:00Z",
                "snippet": "未引用",
            },
        ],
    )
    assert result.success
    data = result.data
    assert data["claim_sources"]["c1"] == ["s1", "s2"]
    assert data["unbacked_claims"] == ["c2", "c3"]
    assert "https://example.com/a?b=2&b=1" in data["duplicate_urls"]
    by_id = {row["source_id"]: row for row in data["sources"]}
    assert by_id["s1"]["source_type"] == "web"
    assert by_id["s1"]["timestamp"] == "2026-01-01T00:00:00Z"
    assert by_id["s1"]["snippet"] == "原始摘要，勿改写"
    assert by_id["s1"]["url"] == "https://example.com/a?b=2&b=1"


def test_organize_provenance_rejects_duplicate_claim_ids() -> None:
    result = organize_provenance(
        claims=[
            {"claim_id": "c1", "text": "无依据", "source_ids": []},
            {"claim_id": "c1", "text": "有依据", "source_ids": ["s1"]},
        ],
        sources=[
            {
                "source_id": "s1",
                "url": "https://example.com/",
                "source_type": "web",
                "timestamp": "2026-01-01T00:00:00Z",
                "snippet": "s",
            }
        ],
    )
    assert not result.success
    assert result.error.code == "invalid_arguments"
    assert "claim_id" in result.error.message


def test_organize_provenance_rejects_duplicate_source_ids() -> None:
    result = organize_provenance(
        claims=[{"claim_id": "c1", "text": "x", "source_ids": ["s1"]}],
        sources=[
            {
                "source_id": "s1",
                "url": "https://example.com/a",
                "source_type": "web",
                "timestamp": "2026-01-01T00:00:00Z",
                "snippet": "a",
            },
            {
                "source_id": "s1",
                "url": "https://example.com/b",
                "source_type": "web",
                "timestamp": "2026-01-02T00:00:00Z",
                "snippet": "b",
            },
        ],
    )
    assert not result.success
    assert result.error.code == "invalid_arguments"
    assert "source_id" in result.error.message


@pytest.mark.asyncio
async def test_provider_exposes_provenance_procedures() -> None:
    provider = BundledProcedureProvider(SimpleNamespace())
    ids = {item["procedure_id"] for item in provider.describe()}
    assert {"builtin.normalize_urls", "builtin.organize_provenance"} <= ids

    normalized = await provider.invoke(
        "builtin.normalize_urls",
        {
            "items": [
                {"url": "HTTPS://Example.COM/a#x", "source_id": "a"},
                {"url": "https://example.com/a", "source_id": "b"},
            ]
        },
    )
    assert normalized.success
    assert len(normalized.data["items"]) == 1
    assert normalized.data["items"][0]["source_ids"] == ["a", "b"]

    organized = await provider.invoke(
        "builtin.organize_provenance",
        {
            "claims": [{"claim_id": "c1", "text": "x", "source_ids": ["s1"]}],
            "sources": [
                {
                    "source_id": "s1",
                    "url": "https://example.com/",
                    "source_type": "web",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "snippet": "s",
                }
            ],
        },
    )
    assert organized.success
    assert organized.data["claim_sources"]["c1"] == ["s1"]
    assert organized.data["unbacked_claims"] == []
