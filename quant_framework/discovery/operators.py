"""Whitelisted DSL operators — no arbitrary code execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from discovery.types import NUMERIC_TYPES, ValueType


class OperatorId(str, Enum):
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY_BOUNDED = "MULTIPLY_BOUNDED"
    DIVIDE_PROTECTED = "DIVIDE_PROTECTED"
    ABS = "ABS"
    NEGATE = "NEGATE"
    MIN = "MIN"
    MAX = "MAX"
    CLIP = "CLIP"
    LAG = "LAG"
    DIFFERENCE = "DIFFERENCE"
    PERCENT_CHANGE = "PERCENT_CHANGE"
    ROLLING_MEAN = "ROLLING_MEAN"
    ROLLING_MEDIAN = "ROLLING_MEDIAN"
    ROLLING_STD = "ROLLING_STD"
    ROLLING_MIN = "ROLLING_MIN"
    ROLLING_MAX = "ROLLING_MAX"
    ROLLING_RANK = "ROLLING_RANK"
    ROLLING_ZSCORE = "ROLLING_ZSCORE"
    EMA = "EMA"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    GREATER_EQUAL = "GREATER_EQUAL"
    LESS_EQUAL = "LESS_EQUAL"
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"
    BETWEEN = "BETWEEN"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_SIGNAL = "EXIT_SIGNAL"
    TIME_EXIT = "TIME_EXIT"
    SESSION_EXIT = "SESSION_EXIT"
    ATR_STOP = "ATR_STOP"
    ATR_TARGET = "ATR_TARGET"
    TRAILING_STOP = "TRAILING_STOP"
    REGIME_GATE = "REGIME_GATE"
    LIQUIDITY_GATE = "LIQUIDITY_GATE"
    VOLATILITY_GATE = "VOLATILITY_GATE"


FORBIDDEN_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "subprocess",
        "os",
        "sys",
        "pickle",
        "ctypes",
    }
)


@dataclass(frozen=True)
class OperatorSpec:
    op_id: OperatorId
    input_types: tuple[frozenset[ValueType], ...]
    output_type: ValueType
    arity: int
    complexity_cost: float
    causal: bool = True
    required_feature_capabilities: tuple[str, ...] = ()
    lookback_child: int | None = None
    max_lookback: int | None = None


def _num() -> frozenset[ValueType]:
    return frozenset(NUMERIC_TYPES)


def build_operator_registry() -> dict[OperatorId, OperatorSpec]:
    n = _num()
    b = frozenset({ValueType.BOOLEAN})
    s = frozenset({ValueType.SCALAR})
    specs = [
        OperatorSpec(OperatorId.ADD, (n, n), ValueType.SCALAR, 2, 1.0),
        OperatorSpec(OperatorId.SUBTRACT, (n, n), ValueType.SCALAR, 2, 1.0),
        OperatorSpec(OperatorId.MULTIPLY_BOUNDED, (n, n), ValueType.SCALAR, 2, 1.2),
        OperatorSpec(OperatorId.DIVIDE_PROTECTED, (n, n), ValueType.RATIO, 2, 1.5),
        OperatorSpec(OperatorId.ABS, (n,), ValueType.SCALAR, 1, 0.5),
        OperatorSpec(OperatorId.NEGATE, (n,), ValueType.SCALAR, 1, 0.5),
        OperatorSpec(OperatorId.MIN, (n, n), ValueType.SCALAR, 2, 0.8),
        OperatorSpec(OperatorId.MAX, (n, n), ValueType.SCALAR, 2, 0.8),
        OperatorSpec(OperatorId.CLIP, (n, n, n), ValueType.SCALAR, 3, 1.2),
        OperatorSpec(OperatorId.LAG, (n, s), ValueType.SCALAR, 2, 1.0, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.DIFFERENCE, (n,), ValueType.SCALAR, 1, 1.0),
        OperatorSpec(OperatorId.PERCENT_CHANGE, (n,), ValueType.RETURN, 1, 1.0),
        OperatorSpec(OperatorId.ROLLING_MEAN, (n, s), ValueType.SCALAR, 2, 1.5, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.ROLLING_MEDIAN, (n, s), ValueType.SCALAR, 2, 1.8, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.ROLLING_STD, (n, s), ValueType.VOLATILITY, 2, 1.8, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.ROLLING_MIN, (n, s), ValueType.SCALAR, 2, 1.5, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.ROLLING_MAX, (n, s), ValueType.SCALAR, 2, 1.5, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.ROLLING_RANK, (n, s), ValueType.RANK, 2, 2.0, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.ROLLING_ZSCORE, (n, s), ValueType.ZSCORE, 2, 2.0, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.EMA, (n, s), ValueType.SCALAR, 2, 1.5, lookback_child=1, max_lookback=100),
        OperatorSpec(OperatorId.GREATER_THAN, (n, n), ValueType.BOOLEAN, 2, 1.0),
        OperatorSpec(OperatorId.LESS_THAN, (n, n), ValueType.BOOLEAN, 2, 1.0),
        OperatorSpec(OperatorId.GREATER_EQUAL, (n, n), ValueType.BOOLEAN, 2, 1.0),
        OperatorSpec(OperatorId.LESS_EQUAL, (n, n), ValueType.BOOLEAN, 2, 1.0),
        OperatorSpec(OperatorId.CROSS_ABOVE, (n, n), ValueType.BOOLEAN, 2, 1.5),
        OperatorSpec(OperatorId.CROSS_BELOW, (n, n), ValueType.BOOLEAN, 2, 1.5),
        OperatorSpec(OperatorId.BETWEEN, (n, n, n), ValueType.BOOLEAN, 3, 1.5),
        OperatorSpec(OperatorId.AND, (b, b), ValueType.BOOLEAN, 2, 1.0),
        OperatorSpec(OperatorId.OR, (b, b), ValueType.BOOLEAN, 2, 1.0),
        OperatorSpec(OperatorId.NOT, (b,), ValueType.BOOLEAN, 1, 0.5),
        OperatorSpec(OperatorId.ENTRY_LONG, (b,), ValueType.ORDER_INTENT, 1, 1.0),
        OperatorSpec(OperatorId.ENTRY_SHORT, (b,), ValueType.ORDER_INTENT, 1, 1.0),
        OperatorSpec(OperatorId.EXIT_SIGNAL, (b,), ValueType.ORDER_INTENT, 1, 1.0),
        OperatorSpec(OperatorId.TIME_EXIT, (s,), ValueType.ORDER_INTENT, 1, 0.8),
        OperatorSpec(OperatorId.SESSION_EXIT, (b,), ValueType.ORDER_INTENT, 1, 0.8),
        OperatorSpec(
            OperatorId.ATR_STOP,
            (frozenset({ValueType.VOLATILITY, ValueType.SCALAR}), s),
            ValueType.ORDER_INTENT,
            2,
            1.5,
        ),
        OperatorSpec(
            OperatorId.ATR_TARGET,
            (frozenset({ValueType.VOLATILITY, ValueType.SCALAR}), s),
            ValueType.ORDER_INTENT,
            2,
            1.5,
        ),
        OperatorSpec(
            OperatorId.TRAILING_STOP,
            (frozenset({ValueType.VOLATILITY, ValueType.SCALAR}), s),
            ValueType.ORDER_INTENT,
            2,
            2.0,
        ),
        OperatorSpec(
            OperatorId.REGIME_GATE,
            (b, frozenset({ValueType.REGIME, ValueType.BOOLEAN})),
            ValueType.BOOLEAN,
            2,
            1.5,
        ),
        OperatorSpec(
            OperatorId.LIQUIDITY_GATE,
            (b, frozenset({ValueType.VOLUME, ValueType.RATIO, ValueType.SCALAR})),
            ValueType.BOOLEAN,
            2,
            1.5,
        ),
        OperatorSpec(
            OperatorId.VOLATILITY_GATE,
            (b, frozenset({ValueType.VOLATILITY, ValueType.RATIO, ValueType.SCALAR})),
            ValueType.BOOLEAN,
            2,
            1.5,
        ),
    ]
    return {s.op_id: s for s in specs}


OPERATOR_REGISTRY = build_operator_registry()
