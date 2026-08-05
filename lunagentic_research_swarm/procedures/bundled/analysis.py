"""受限计算、基础统计与显式单位换算 Procedures。"""

from __future__ import annotations

import ast
import math
import operator
import statistics as stats_lib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from lunagentic_research_swarm.extensions.contracts import ProcedureDefinition, ProcedureResult

Handler = Callable[[Any, Mapping[str, Any]], Awaitable[ProcedureResult]]

_MAX_AST_NODES = 128
_MAX_ABS_NUMBER = 1e100
_MAX_ABS_EXPONENT = 100
_MAX_STATS_VALUES = 10000
_STATS_OPS = frozenset({"mean", "median", "stdev", "pstdev", "min", "max", "quantiles"})

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 显式因子表：同维度单位 → 基准单位系数（温度单独处理）
_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "yd": 0.9144,
    "ft": 0.3048,
    "in": 0.0254,
}
_MASS_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "g": 0.001,
    "mg": 1e-6,
    "lb": 0.45359237,
    "oz": 0.028349523125,
}
_TIME_TO_S: dict[str, float] = {
    "s": 1.0,
    "ms": 0.001,
    "min": 60.0,
    "h": 3600.0,
    "day": 86400.0,
}
_DATA_TO_B: dict[str, float] = {
    "B": 1.0,
    "KB": 1000.0,
    "MB": 1_000_000.0,
    "GB": 1_000_000_000.0,
    "TB": 1_000_000_000_000.0,
    "KiB": 1024.0,
    "MiB": 1024.0**2,
    "GiB": 1024.0**3,
    "TiB": 1024.0**4,
}
_TEMPERATURE_UNITS = frozenset({"C", "K", "F"})
_DIMENSION_TABLES: tuple[tuple[str, dict[str, float]], ...] = (
    ("length", _LENGTH_TO_M),
    ("mass", _MASS_TO_KG),
    ("time", _TIME_TO_S),
    ("data", _DATA_TO_B),
)


def _failure(code: str, message: str) -> ProcedureResult:
    return ProcedureResult(
        success=False,
        data=None,
        error={"code": code, "message": message},
        metadata={},
    )


def _success(data: Mapping[str, Any]) -> ProcedureResult:
    return ProcedureResult(
        success=True,
        data=dict(data),
        error=None,
        metadata={},
    )


class _UnsafeExpression(Exception):
    pass


class _DivisionByZero(Exception):
    pass


def _check_number(value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _UnsafeExpression("仅允许数值常量")
    number = float(value)
    if not math.isfinite(number):
        raise _UnsafeExpression("数值必须有限")
    if abs(number) > _MAX_ABS_NUMBER:
        raise _UnsafeExpression("数值绝对值超过上限")
    return number


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return _check_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_ABS_EXPONENT:
                raise _UnsafeExpression("指数绝对值超过上限")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0.0:
            raise _DivisionByZero("除数不能为 0")
        try:
            result = _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise _DivisionByZero("除数不能为 0") from exc
        if not isinstance(result, int | float) or isinstance(result, bool):
            raise _UnsafeExpression("运算结果必须为数值")
        if not math.isfinite(float(result)):
            raise _UnsafeExpression("运算结果必须有限")
        if abs(float(result)) > _MAX_ABS_NUMBER:
            raise _UnsafeExpression("运算结果绝对值超过上限")
        return float(result)
    raise _UnsafeExpression("表达式包含不允许的语法")


def calculate(expression: str) -> ProcedureResult:
    """受限 AST 求值：仅数值常量与四则/整除/取模/幂/一元正负。"""

    if not isinstance(expression, str) or not expression.strip():
        return _failure("invalid_arguments", "expression 必须为非空字符串")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return _failure("unsafe_expression", "表达式无法解析")
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        return _failure("unsafe_expression", "表达式节点数超过上限")
    try:
        value = _eval_node(tree)
    except _DivisionByZero as exc:
        return _failure("division_by_zero", str(exc))
    except _UnsafeExpression as exc:
        return _failure("unsafe_expression", str(exc))
    return _success({"expression": expression, "value": value})


def statistics(operation: str, values: Sequence[Any], n: int | None = None) -> ProcedureResult:
    """固定操作集的基础统计；输入最多 10000 个有限数值。"""

    if operation not in _STATS_OPS:
        return _failure("invalid_arguments", "operation 必须为 mean|median|stdev|pstdev|min|max|quantiles")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return _failure("invalid_arguments", "values 必须为数组")
    if len(values) > _MAX_STATS_VALUES:
        return _failure("invalid_arguments", f"values 最多 {_MAX_STATS_VALUES} 个")
    if len(values) == 0:
        return _failure("invalid_arguments", "values 不能为空")
    numbers: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return _failure("invalid_arguments", "values 必须全部为有限数值")
        number = float(item)
        if not math.isfinite(number):
            return _failure("invalid_arguments", "values 必须全部为有限数值")
        numbers.append(number)

    try:
        if operation == "mean":
            value: Any = stats_lib.mean(numbers)
        elif operation == "median":
            value = stats_lib.median(numbers)
        elif operation == "stdev":
            if len(numbers) < 2:
                return _failure("invalid_arguments", "stdev 至少需要 2 个数值")
            value = stats_lib.stdev(numbers)
        elif operation == "pstdev":
            value = stats_lib.pstdev(numbers)
        elif operation == "min":
            value = min(numbers)
        elif operation == "max":
            value = max(numbers)
        else:
            if n is None:
                return _failure("invalid_arguments", "quantiles 需要参数 n")
            if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n > 100:
                return _failure("invalid_arguments", "quantiles 的 n 必须在 2..100")
            value = stats_lib.quantiles(numbers, n=n)
    except stats_lib.StatisticsError as exc:
        return _failure("invalid_arguments", str(exc))

    return _success({"operation": operation, "value": value})


def _temperature_convert(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return value
    # 先转到摄氏
    if from_unit == "C":
        celsius = value
    elif from_unit == "K":
        celsius = value - 273.15
    else:  # F
        celsius = (value - 32.0) * 5.0 / 9.0
    if to_unit == "C":
        return celsius
    if to_unit == "K":
        return celsius + 273.15
    return celsius * 9.0 / 5.0 + 32.0


def _lookup_dimension(unit: str) -> tuple[str, dict[str, float]] | None:
    for name, table in _DIMENSION_TABLES:
        if unit in table:
            return name, table
    return None


def convert_units(value: float | int, from_unit: str, to_unit: str) -> ProcedureResult:
    """显式因子/偏移表换算；未知单位与跨维度拒绝。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return _failure("invalid_arguments", "value 必须为有限数值")
    number = float(value)
    if not math.isfinite(number):
        return _failure("invalid_arguments", "value 必须为有限数值")
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        return _failure("invalid_arguments", "from_unit/to_unit 必须为字符串")
    if not from_unit or not to_unit:
        return _failure("invalid_arguments", "from_unit/to_unit 不能为空")

    if from_unit in _TEMPERATURE_UNITS or to_unit in _TEMPERATURE_UNITS:
        if from_unit not in _TEMPERATURE_UNITS or to_unit not in _TEMPERATURE_UNITS:
            return _failure("incompatible_units", "温度单位不能与其他维度混用")
        result = _temperature_convert(number, from_unit, to_unit)
        return _success(
            {
                "value": result,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "input": number,
            }
        )

    from_dim = _lookup_dimension(from_unit)
    to_dim = _lookup_dimension(to_unit)
    if from_dim is None:
        return _failure("unknown_unit", f"未知单位：{from_unit}")
    if to_dim is None:
        return _failure("unknown_unit", f"未知单位：{to_unit}")
    if from_dim[0] != to_dim[0]:
        return _failure("incompatible_units", f"不能跨维度换算：{from_dim[0]} → {to_dim[0]}")
    table = from_dim[1]
    result = number * table[from_unit] / table[to_unit]
    return _success(
        {
            "value": result,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "input": number,
        }
    )


_ANALYSIS_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "procedure_id": "builtin.calculate",
        "display_name": "受限计算",
        "description": "对受限算术表达式求值；拒绝任意代码执行。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "value": {"type": "number"},
            },
        },
    },
    {
        "procedure_id": "builtin.statistics",
        "display_name": "基础统计",
        "description": "对有限数值列表执行 mean/median/stdev/pstdev/min/max/quantiles。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["mean", "median", "stdev", "pstdev", "min", "max", "quantiles"],
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": _MAX_STATS_VALUES,
                },
                "n": {"type": "integer", "minimum": 2, "maximum": 100},
            },
            "required": ["operation", "values"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "value": {},
            },
        },
    },
    {
        "procedure_id": "builtin.convert_units",
        "display_name": "单位换算",
        "description": "在 length/mass/time/temperature/data size 维度内做显式单位换算。",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "minLength": 1},
                "to_unit": {"type": "string", "minLength": 1},
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": False,
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
                "input": {"type": "number"},
            },
        },
    },
)


def analysis_procedure_definitions() -> list[ProcedureDefinition]:
    """构造分析类 Procedure 定义。"""

    definitions: list[ProcedureDefinition] = []
    for item in _ANALYSIS_DEFINITIONS:
        payload = {
            **item,
            "version": "1",
            "idempotent": True,
            "timeout_seconds": 30.0,
            "external_cost_kind": "none",
            "enabled": True,
        }
        definitions.append(ProcedureDefinition.model_validate(payload))
    return definitions


async def _calculate(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        return _failure("invalid_arguments", "expression 必须为字符串")
    return calculate(expression)


async def _statistics(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    operation = arguments.get("operation")
    values = arguments.get("values")
    n = arguments.get("n")
    if not isinstance(operation, str):
        return _failure("invalid_arguments", "operation 必须为字符串")
    if not isinstance(values, list):
        return _failure("invalid_arguments", "values 必须为数组")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int)):
        return _failure("invalid_arguments", "n 必须为整数")
    return statistics(operation, values, n=n)


async def _convert_units(_ctx: Any, arguments: Mapping[str, Any]) -> ProcedureResult:
    value = arguments.get("value")
    from_unit = arguments.get("from_unit")
    to_unit = arguments.get("to_unit")
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        return _failure("invalid_arguments", "from_unit/to_unit 必须为字符串")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _failure("invalid_arguments", "value 必须为数值")
    return convert_units(value, from_unit, to_unit)


ANALYSIS_HANDLERS: dict[str, Handler] = {
    "builtin.calculate": _calculate,
    "builtin.statistics": _statistics,
    "builtin.convert_units": _convert_units,
}
