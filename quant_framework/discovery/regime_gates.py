"""Compile FamilySpec regime_constraints into executable DSL regime_gates."""

from __future__ import annotations

from discovery.expression_tree import ExprNode, constant_node, feature_node, op_node
from discovery.operators import OperatorId
from discovery.types import ValueType

UNSUPPORTED_FAMILY_CONSTRAINT = "UNSUPPORTED_FAMILY_CONSTRAINT"

# Supported constraint → executable gate builders.
# Regime feature numeric contracts (causal, no future data):
#   regime.trend_state ∈ [-1, 1]
#     +1 ≈ strong uptrend, -1 ≈ strong downtrend, 0 ≈ range
#     |x| >= 0.25 → trending; |x| < 0.25 → range
#   regime.volatility_state ∈ [-1, 1]
#     +1 ≈ expansion, -1 ≈ compression, 0 ≈ normal
#     x >= 0.25 → expansion; x <= -0.25 → compression


def _trend() -> ExprNode:
    return feature_node("regime.trend_state", ValueType.REGIME)


def _vol() -> ExprNode:
    return feature_node("regime.volatility_state", ValueType.REGIME)


def _minutes() -> ExprNode:
    return feature_node("temp.minutes_since_open", ValueType.TIME)


def _volume_pct() -> ExprNode:
    return feature_node("liq.volume_pct_20", ValueType.RANK)


def _gate(cond: ExprNode, regime_child: ExprNode) -> ExprNode:
    return op_node(OperatorId.REGIME_GATE, cond, regime_child)


def compile_regime_constraint(constraint: str) -> ExprNode:
    """Compile one constraint string into a REGIME_GATE tree."""
    key = str(constraint).strip()
    if key == "require_trend_regime":
        # |trend_state| >= 0.25
        abs_trend = op_node(OperatorId.ABS, _trend())
        cond = op_node(OperatorId.GREATER_EQUAL, abs_trend, constant_node(0.25))
        return _gate(cond, _trend())
    if key == "prefer_range_regime":
        abs_trend = op_node(OperatorId.ABS, _trend())
        cond = op_node(OperatorId.LESS_THAN, abs_trend, constant_node(0.25))
        return _gate(cond, _trend())
    if key == "prefer_vol_expansion":
        cond = op_node(OperatorId.GREATER_EQUAL, _vol(), constant_node(0.25))
        return _gate(cond, _vol())
    if key == "intraday_session_only":
        # Session minutes in [0, 390]
        ge = op_node(OperatorId.GREATER_EQUAL, _minutes(), constant_node(0.0))
        le = op_node(OperatorId.LESS_EQUAL, _minutes(), constant_node(390.0))
        cond = op_node(OperatorId.AND, ge, le)
        # Second child must be REGIME or BOOLEAN — use boolean session flag via vol proxy
        # Prefer boolean second child (allowed by REGIME_GATE).
        return op_node(OperatorId.REGIME_GATE, cond, cond)
    if key == "prefer_liquid_session":
        cond = op_node(OperatorId.GREATER_THAN, _volume_pct(), constant_node(0.3))
        return op_node(OperatorId.REGIME_GATE, cond, cond)
    raise ValueError(f"{UNSUPPORTED_FAMILY_CONSTRAINT}: {key!r}")


def compile_regime_constraints(constraints: tuple[str, ...] | list[str]) -> tuple[ExprNode, ...]:
    """Compile all constraints; never silently drop unsupported ones."""
    out: list[ExprNode] = []
    for c in constraints:
        out.append(compile_regime_constraint(c))
    return tuple(out)
