"""DSL value types for the typed Strategy DSL (Phase 6C)."""

from __future__ import annotations

from enum import Enum


class ValueType(str, Enum):
    PRICE = "PRICE"
    RETURN = "RETURN"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    SPREAD = "SPREAD"
    RATIO = "RATIO"
    ZSCORE = "ZSCORE"
    RANK = "RANK"
    BOOLEAN = "BOOLEAN"
    TIME = "TIME"
    REGIME = "REGIME"
    POSITION_STATE = "POSITION_STATE"
    ORDER_INTENT = "ORDER_INTENT"
    SCALAR = "SCALAR"  # numeric constant / free parameter


NUMERIC_TYPES = frozenset(
    {
        ValueType.PRICE,
        ValueType.RETURN,
        ValueType.VOLATILITY,
        ValueType.VOLUME,
        ValueType.SPREAD,
        ValueType.RATIO,
        ValueType.ZSCORE,
        ValueType.RANK,
        ValueType.SCALAR,
        ValueType.TIME,
        # Numeric regime codes in [-1, 1] (trend / vol state) — comparable & ABS-able.
        ValueType.REGIME,
    }
)


class CreationMethod(str, Enum):
    RANDOM = "RANDOM"
    MUTATION = "MUTATION"
    CROSSOVER = "CROSSOVER"
    SEED_TEMPLATE = "SEED_TEMPLATE"
    LEGACY_IMPORT = "LEGACY_IMPORT"


class NodeKind(str, Enum):
    FEATURE = "FEATURE"
    CONSTANT = "CONSTANT"
    PARAMETER = "PARAMETER"
    OPERATOR = "OPERATOR"
