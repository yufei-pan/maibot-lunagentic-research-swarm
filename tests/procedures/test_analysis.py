"""受限计算、统计与单位换算：拒绝任意代码，明确错误边界。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunagentic_research_swarm.procedures.bundled.analysis import (
    calculate,
    convert_units,
    statistics,
)
from lunagentic_research_swarm.procedures.bundled.provider import BundledProcedureProvider


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "(1).__class__", "[x for x in range(3)]", "2 ** 100000"],
)
def test_calculator_rejects_unsafe_expression(expression) -> None:
    result = calculate(expression)
    assert not result.success
    assert result.error.code == "unsafe_expression"


def test_calculator_evaluates_safe_expression() -> None:
    result = calculate("(1 + 2) * 3 - 4 / 2")
    assert result.success
    assert result.data["value"] == 7.0


def test_calculator_rejects_oversized_ast() -> None:
    # 65 个加法 → 超过 128 节点（Expression + 常量/算子树）
    expression = "+".join(["1"] * 70)
    result = calculate(expression)
    assert not result.success
    assert result.error.code == "unsafe_expression"


def test_calculator_rejects_expression_longer_than_2000() -> None:
    # 在 ast.parse 之前按 schema maxLength 拒绝，避免同步解析阻塞
    expression = "1+" * 1001  # 2002 chars
    assert len(expression) > 2000
    result = calculate(expression)
    assert not result.success
    assert result.error.code == "unsafe_expression"
    assert "长度" in result.error.message


def test_calculator_rejects_huge_literal() -> None:
    result = calculate("1e101")
    assert not result.success
    assert result.error.code == "unsafe_expression"


def test_calculator_rejects_arithmetic_overflow() -> None:
    # 指数在允许范围，但运算溢出 float → 结构化 unsafe_expression，不抛出
    result = calculate("1e100 ** 100")
    assert not result.success
    assert result.error.code == "unsafe_expression"


def test_calculator_rejects_oversized_integer_literal() -> None:
    # 超大整数字面量在 float() 转换时 OverflowError → unsafe_expression
    result = calculate("1" + "0" * 400)
    assert not result.success
    assert result.error.code == "unsafe_expression"


def test_calculator_division_by_zero() -> None:
    result = calculate("1 / 0")
    assert not result.success
    assert result.error.code == "division_by_zero"


@pytest.mark.parametrize(
    ("operation", "values", "expected"),
    [
        ("mean", [1.0, 2.0, 3.0], 2.0),
        ("median", [1.0, 3.0, 2.0], 2.0),
        ("min", [3.0, 1.0, 2.0], 1.0),
        ("max", [3.0, 1.0, 2.0], 3.0),
        ("stdev", [1.0, 2.0, 3.0], pytest.approx(1.0)),
        ("pstdev", [1.0, 2.0, 3.0], pytest.approx(0.816496580927726)),
    ],
)
def test_statistics_basic_operations(operation, values, expected) -> None:
    result = statistics(operation, values)
    assert result.success
    assert result.data["operation"] == operation
    assert result.data["value"] == expected


def test_statistics_quantiles() -> None:
    result = statistics("quantiles", [1.0, 2.0, 3.0, 4.0], n=4)
    assert result.success
    assert result.data["operation"] == "quantiles"
    assert len(result.data["value"]) == 3


def test_statistics_rejects_non_finite_and_oversized() -> None:
    assert not statistics("mean", [1.0, float("nan")]).success
    assert statistics("mean", [1.0, float("nan")]).error.code == "invalid_arguments"
    too_many = [1.0] * 10001
    oversized = statistics("mean", too_many)
    assert not oversized.success
    assert oversized.error.code == "invalid_arguments"


def test_statistics_quantiles_n_bounds() -> None:
    assert statistics("quantiles", [1.0, 2.0], n=1).error.code == "invalid_arguments"
    assert statistics("quantiles", [1.0, 2.0], n=101).error.code == "invalid_arguments"


def test_statistics_rejects_non_finite_computed_quantiles() -> None:
    # 有限输入仍可能外推为 inf；须结构化 invalid_arguments，而非 ValidationError
    result = statistics("quantiles", [-1e308, 1e308], n=100)
    assert not result.success
    assert result.error.code == "invalid_arguments"
    assert "有限" in result.error.message


def test_convert_units_length_and_temperature() -> None:
    meters = convert_units(1.0, "km", "m")
    assert meters.success
    assert meters.data["value"] == 1000.0

    kelvin = convert_units(0.0, "C", "K")
    assert kelvin.success
    assert kelvin.data["value"] == pytest.approx(273.15)

    fahrenheit = convert_units(100.0, "C", "F")
    assert fahrenheit.success
    assert fahrenheit.data["value"] == pytest.approx(212.0)


def test_convert_units_rejects_unknown_and_cross_dimension() -> None:
    unknown = convert_units(1.0, "parsec", "m")
    assert not unknown.success
    assert unknown.error.code == "unknown_unit"

    cross = convert_units(1.0, "m", "kg")
    assert not cross.success
    assert cross.error.code == "incompatible_units"


def test_convert_units_rejects_non_finite_result() -> None:
    # 有限输入换算溢出 → 结构化失败，不抛 ValidationError
    overflow = convert_units(1e308, "TB", "B")
    assert not overflow.success
    assert overflow.error.code == "invalid_arguments"
    assert "有限" in overflow.error.message

    temp = convert_units(1e308, "C", "F")
    assert not temp.success
    assert temp.error.code == "invalid_arguments"


@pytest.mark.asyncio
async def test_provider_exposes_analysis_procedures() -> None:
    provider = BundledProcedureProvider(SimpleNamespace())
    ids = {item["procedure_id"] for item in provider.describe()}
    assert {"builtin.calculate", "builtin.statistics", "builtin.convert_units"} <= ids

    calc = await provider.invoke("builtin.calculate", {"expression": "2 + 2"})
    assert calc.success
    assert calc.data["value"] == 4.0

    stats = await provider.invoke(
        "builtin.statistics",
        {"operation": "mean", "values": [1, 2, 3]},
    )
    assert stats.success
    assert stats.data["value"] == 2.0

    units = await provider.invoke(
        "builtin.convert_units",
        {"value": 1, "from_unit": "kg", "to_unit": "g"},
    )
    assert units.success
    assert units.data["value"] == 1000.0
