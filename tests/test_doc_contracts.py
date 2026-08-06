"""锁定用户文档与主设计规格中的 procedure credits / contractor 契约措辞。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_main_design_spec_documents_procedure_credits_and_contractor() -> None:
    text = _read("docs/superpowers/specs/2026-08-03-lunagentic-research-swarm-design.md")
    assert "Procedure 不扣研究 credits" not in text
    assert "research_credits_charged" in text
    assert "credit_budget" in text
    assert "builtin.contractor" in text
    assert "2026-08-06-contractor-procedure-design.md" in text
    assert '"credits": 0.0' in text or '"credits": 0' in text
    assert "timeout_seconds" in text and "允许为 `0`" in text
    assert "**自动** compact 不扣研究 credits" in text
    assert "**智能体请求的** `core.compact`" in text


def test_readme_lists_contractor_and_procedure_budget_hint() -> None:
    text = _read("README.md")
    assert "builtin.contractor" in text
    assert "research_credits_charged" in text
    assert "credits" in text and "预算提示" in text


def test_extension_authoring_documents_charge_and_budget() -> None:
    text = _read("docs/extension-authoring.md")
    assert "research_credits_charged" in text
    assert "credit_budget" in text
    assert "timeout_seconds=0" in text
    assert "external_cost" in text


def test_credits_and_reporting_documents_contractor_billing() -> None:
    text = _read("docs/credits-and-reporting.md")
    assert "research_credits_charged" in text
    assert "credit_budget" in text
    assert "builtin.contractor" in text
    assert "预算提示" in text
    assert "自动 compact" in text
