"""Dimension-aware threshold domains for DSL generation and prechecks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from discovery.grammar import FeatureLeaf, Grammar
from discovery.types import ValueType


INVALID_FEATURE_THRESHOLD_DOMAIN = "INVALID_FEATURE_THRESHOLD_DOMAIN"


@dataclass(frozen=True)
class FeatureDomain:
    feature_id: str
    units: str
    value_type: ValueType
    valid_range: tuple[float, float]
    typical_range: tuple[float, float]
    lookback_range: tuple[int, int] = (1, 100)

    def sample_threshold(self, rng: np.random.Generator) -> float:
        lo, hi = self.typical_range
        if self.units in {"category", "regime_enum"}:
            # Discrete valid regime codes.
            choices = [lo, 0.0, hi] if lo < 0 else [lo, hi]
            return float(rng.choice(choices))
        return float(rng.uniform(lo, hi))

    def accepts(self, value: float) -> bool:
        lo, hi = self.valid_range
        if not np.isfinite(value):
            return False
        return lo - 1e-12 <= float(value) <= hi + 1e-12


# Units → default domain when feature-specific metadata is absent.
_UNIT_DEFAULTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "return": ((-0.25, 0.25), (-0.02, 0.02)),
    "log_return": ((-0.25, 0.25), (-0.02, 0.02)),
    "zscore": ((-6.0, 6.0), (-3.0, 3.0)),
    "rank": ((0.0, 1.0), (0.05, 0.95)),
    "ratio": ((-2.0, 2.0), (-0.1, 0.1)),
    "minutes": ((0.0, 24 * 60.0), (0.0, 390.0)),
    "volatility": ((0.0, 1e6), (0.0, 50.0)),
    "price": ((-1e9, 1e9), (-100.0, 100.0)),
    "correlation": ((-1.0, 1.0), (-0.8, 0.8)),
    "category": ((-1.0, 1.0), (-1.0, 1.0)),
    "regime_code": ((-1.0, 1.0), (-1.0, 1.0)),
}


_FEATURE_OVERRIDES: dict[str, FeatureDomain] = {
    "price.simple_return_1": FeatureDomain(
        "price.simple_return_1", "return", ValueType.RETURN, (-0.25, 0.25), (-0.015, 0.015)
    ),
    "price.log_return_1": FeatureDomain(
        "price.log_return_1", "log_return", ValueType.RETURN, (-0.25, 0.25), (-0.015, 0.015)
    ),
    "price.return_5": FeatureDomain(
        "price.return_5", "return", ValueType.RETURN, (-0.5, 0.5), (-0.04, 0.04)
    ),
    "price.close_to_open": FeatureDomain(
        "price.close_to_open", "return", ValueType.RETURN, (-0.25, 0.25), (-0.03, 0.03)
    ),
    "price.gap_size": FeatureDomain(
        # Raw price gap — prefer normalized close_to_open for thresholds.
        "price.gap_size", "price", ValueType.PRICE, (-1e6, 1e6), (-50.0, 50.0)
    ),
    "price.rolling_z_20": FeatureDomain(
        "price.rolling_z_20", "zscore", ValueType.ZSCORE, (-6.0, 6.0), (-3.0, 3.0)
    ),
    "price.dist_rolling_mean_20": FeatureDomain(
        "price.dist_rolling_mean_20", "ratio", ValueType.RATIO, (-1.0, 1.0), (-0.05, 0.05)
    ),
    "price.breakout_distance_20": FeatureDomain(
        "price.breakout_distance_20", "ratio", ValueType.RATIO, (-1.0, 1.0), (-0.02, 0.05)
    ),
    "price.rolling_rank_20": FeatureDomain(
        "price.rolling_rank_20", "rank", ValueType.RANK, (0.0, 1.0), (0.05, 0.95)
    ),
    "liq.dist_session_vwap": FeatureDomain(
        "liq.dist_session_vwap", "ratio", ValueType.RATIO, (-1.0, 1.0), (-0.02, 0.02)
    ),
    "liq.volume_pct_20": FeatureDomain(
        "liq.volume_pct_20", "rank", ValueType.RANK, (0.0, 1.0), (0.1, 0.9)
    ),
    "vol.atr_14": FeatureDomain(
        "vol.atr_14", "volatility", ValueType.VOLATILITY, (0.0, 1e6), (0.1, 100.0)
    ),
    "vol.norm_atr_14": FeatureDomain(
        "vol.norm_atr_14", "ratio", ValueType.RATIO, (0.0, 1.0), (0.001, 0.05)
    ),
    "vol.realized_20": FeatureDomain(
        "vol.realized_20", "volatility", ValueType.VOLATILITY, (0.0, 1.0), (0.0, 0.05)
    ),
    "vol.range_compression_20": FeatureDomain(
        "vol.range_compression_20", "ratio", ValueType.RATIO, (0.0, 2.0), (0.2, 1.2)
    ),
    "temp.minutes_since_open": FeatureDomain(
        "temp.minutes_since_open", "minutes", ValueType.TIME, (0.0, 24 * 60.0), (0.0, 390.0)
    ),
    "regime.trend_state": FeatureDomain(
        "regime.trend_state",
        "regime_code",
        ValueType.REGIME,
        (-1.0, 1.0),
        (-1.0, 1.0),
    ),
    "regime.volatility_state": FeatureDomain(
        "regime.volatility_state",
        "regime_code",
        ValueType.REGIME,
        (-1.0, 1.0),
        (-1.0, 1.0),
    ),
}


def domain_for_feature(feature_id: str, value_type: ValueType | None = None) -> FeatureDomain:
    if feature_id in _FEATURE_OVERRIDES:
        return _FEATURE_OVERRIDES[feature_id]
    vt = value_type or ValueType.SCALAR
    units_map = {
        ValueType.RETURN: "return",
        ValueType.ZSCORE: "zscore",
        ValueType.RANK: "rank",
        ValueType.RATIO: "ratio",
        ValueType.TIME: "minutes",
        ValueType.VOLATILITY: "volatility",
        ValueType.PRICE: "price",
        ValueType.REGIME: "regime_code",
    }
    units = units_map.get(vt, "ratio")
    valid, typical = _UNIT_DEFAULTS.get(units, ((-10.0, 10.0), (-1.0, 1.0)))
    return FeatureDomain(feature_id, units, vt, valid, typical)


def sample_threshold_for_leaf(leaf: FeatureLeaf, rng: np.random.Generator) -> float:
    return domain_for_feature(leaf.feature_id, leaf.value_type).sample_threshold(rng)


def validate_threshold_against_feature(
    feature_id: str,
    value: float,
    *,
    value_type: ValueType | None = None,
) -> str | None:
    """Return INVALID_FEATURE_THRESHOLD_DOMAIN reason or None if ok."""
    dom = domain_for_feature(feature_id, value_type)
    if not dom.accepts(float(value)):
        return (
            f"{INVALID_FEATURE_THRESHOLD_DOMAIN}: {feature_id} threshold {value} "
            f"outside valid_range={dom.valid_range} (units={dom.units})"
        )
    return None


def validate_tree_threshold_domains(node: Any, grammar: Grammar) -> str | None:
    """Walk comparison nodes; reject constants outside the left feature domain."""
    from discovery.expression_tree import ExprNode, NodeKind
    from discovery.operators import OperatorId

    if node is None:
        return None
    assert isinstance(node, ExprNode)
    cmp_ops = {
        OperatorId.LESS_THAN,
        OperatorId.GREATER_THAN,
        OperatorId.LESS_EQUAL,
        OperatorId.GREATER_EQUAL,
    }
    if node.kind is NodeKind.OPERATOR and node.name in {o.value for o in cmp_ops}:
        left, right = node.children[0], node.children[1]
        if left.kind is NodeKind.FEATURE and right.kind in {NodeKind.CONSTANT, NodeKind.PARAMETER}:
            try:
                leaf = grammar.leaf_for(left.name)
                vt = leaf.value_type
            except KeyError:
                vt = left.value_type
            val = float(right.meta.get("default", right.meta.get("value", 0.0)))
            if right.kind is NodeKind.CONSTANT:
                val = float(right.meta.get("value", 0.0))
            reason = validate_threshold_against_feature(left.name, val, value_type=vt)
            if reason:
                return reason
    for child in node.children:
        reason = validate_tree_threshold_domains(child, grammar)
        if reason:
            return reason
    return None
