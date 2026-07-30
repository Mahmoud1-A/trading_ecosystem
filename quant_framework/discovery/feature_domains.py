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


# ATR stop/target multipliers are not tied to a specific feature_id (they scale
# whichever ATR-family feature feeds ATR_STOP/ATR_TARGET/TRAILING_STOP), so they
# get their own fixed domains rather than a per-feature lookup.
ATR_STOP_MULT_DOMAIN = FeatureDomain(
    "atr_stop_mult", "atr_multiplier", ValueType.SCALAR, (0.05, 20.0), (0.5, 5.0)
)
ATR_TARGET_MULT_DOMAIN = FeatureDomain(
    "atr_target_mult", "atr_multiplier", ValueType.SCALAR, (0.05, 30.0), (0.5, 8.0)
)


@dataclass(frozen=True)
class ThresholdViolation:
    """One semantic-domain violation found while validating an executable tree."""

    tree_path: str
    node_path: str
    feature_id: str | None
    parameter_or_constant: str
    tested_value: float
    valid_range: tuple[float, float]
    units: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tree_path": self.tree_path,
            "node_path": self.node_path,
            "feature_id": self.feature_id,
            "parameter_or_constant": self.parameter_or_constant,
            "tested_value": self.tested_value,
            "valid_range": list(self.valid_range),
            "units": self.units,
            "reason": self.reason,
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


def _unwrap_abs(node: Any) -> tuple[Any, bool]:
    from discovery.operators import OperatorId
    from discovery.types import NodeKind

    if node.kind is NodeKind.OPERATOR and node.name == OperatorId.ABS.value and node.children:
        return node.children[0], True
    return node, False


def _widen_for_abs(dom: FeatureDomain) -> FeatureDomain:
    lo, hi = dom.valid_range
    bound = max(abs(lo), abs(hi))
    tlo, thi = dom.typical_range
    tbound = max(abs(tlo), abs(thi)) or bound
    return FeatureDomain(dom.feature_id, dom.units, dom.value_type, (0.0, bound), (0.0, tbound))


def domain_for_side(node: Any, grammar: Grammar) -> FeatureDomain | None:
    """Resolve the feature domain that a comparison/BETWEEN operand belongs to.

    Handles a bare ``FEATURE`` node or ``ABS(FEATURE)`` (in which case the
    valid range is widened to ``[0, max(|lo|, |hi|)]``). Returns ``None`` when
    ``node`` is not feature-derived (e.g. an operator chain over a feature),
    since we cannot safely infer a domain for arbitrary compositions.
    """
    from discovery.types import NodeKind

    base, wrapped = _unwrap_abs(node)
    if base.kind is not NodeKind.FEATURE:
        return None
    try:
        vt = grammar.leaf_for(base.name).value_type
    except KeyError:
        vt = base.value_type
    dom = domain_for_feature(base.name, vt)
    return _widen_for_abs(dom) if wrapped else dom


def domain_for_context(
    parent_op: Any,
    idx_in_parent: int | None,
    sibling: Any,
    grammar: Grammar,
) -> FeatureDomain | None:
    """Resolve the applicable sampling/validation domain for a node's position.

    Used by mutation (to sample in-domain replacement constants) and by the
    evaluator-time validator (to know what to check). ``parent_op`` is the
    :class:`~discovery.operators.OperatorId` of the immediate parent operator
    node (or ``None`` at the root); ``idx_in_parent`` is this node's child
    index under that parent; ``sibling`` is the other operand to compare
    against (or the tested value for a ``BETWEEN`` bound).
    """
    from discovery.operators import OperatorId

    if parent_op is None or idx_in_parent is None:
        return None
    if parent_op is OperatorId.ATR_TARGET and idx_in_parent == 1:
        return ATR_TARGET_MULT_DOMAIN
    if parent_op in (OperatorId.ATR_STOP, OperatorId.TRAILING_STOP) and idx_in_parent == 1:
        return ATR_STOP_MULT_DOMAIN
    cmp_ops = {
        OperatorId.LESS_THAN,
        OperatorId.GREATER_THAN,
        OperatorId.LESS_EQUAL,
        OperatorId.GREATER_EQUAL,
    }
    if (parent_op in cmp_ops or parent_op is OperatorId.BETWEEN) and sibling is not None:
        return domain_for_side(sibling, grammar)
    return None


def _numeric_leaf_value(node: Any) -> float | None:
    from discovery.types import NodeKind

    if node.kind is NodeKind.CONSTANT:
        return float(node.meta.get("value", 0.0))
    if node.kind is NodeKind.PARAMETER:
        return float(node.meta.get("default", 0.0))
    return None


def _leaf_label(node: Any) -> str:
    from discovery.types import NodeKind

    if node.kind is NodeKind.CONSTANT:
        return "constant"
    if node.kind is NodeKind.PARAMETER:
        return f"parameter:{node.name}"
    return node.kind.value


def collect_threshold_violations(
    node: Any,
    grammar: Grammar,
    *,
    tree_path: str,
    node_path: str = "root",
) -> list[ThresholdViolation]:
    """Walk one executable tree and collect semantic-domain violations.

    Runs at evaluator time (unlike :func:`validate_tree_threshold_domains`,
    which only guards generation) so it also catches mutation, crossover,
    imported, manually constructed, and robustness-perturbed candidates.
    Handles, in either operand order: ``FEATURE op CONSTANT/PARAMETER``,
    ``ABS(FEATURE) op CONSTANT/PARAMETER``, ``BETWEEN(FEATURE|ABS(FEATURE),
    lower, upper)``, and ``ATR_STOP``/``ATR_TARGET``/``TRAILING_STOP``
    multipliers.
    """
    from discovery.operators import OperatorId
    from discovery.types import NodeKind

    violations: list[ThresholdViolation] = []
    if node is None:
        return violations

    if node.kind is NodeKind.OPERATOR:
        op_name = node.name
        children = node.children
        cmp_ops = {
            OperatorId.LESS_THAN.value,
            OperatorId.GREATER_THAN.value,
            OperatorId.LESS_EQUAL.value,
            OperatorId.GREATER_EQUAL.value,
        }

        if op_name in cmp_ops and len(children) == 2:
            left, right = children
            for label, feature_side, other_side in (
                ("left", left, right),
                ("right", right, left),
            ):
                base, _ = _unwrap_abs(feature_side)
                if base.kind is not NodeKind.FEATURE or other_side.kind not in (
                    NodeKind.CONSTANT,
                    NodeKind.PARAMETER,
                ):
                    continue
                dom = domain_for_side(feature_side, grammar)
                val = _numeric_leaf_value(other_side)
                if dom is not None and val is not None and not dom.accepts(val):
                    violations.append(
                        ThresholdViolation(
                            tree_path=tree_path,
                            node_path=f"{node_path}.{label}_threshold",
                            feature_id=base.name,
                            parameter_or_constant=_leaf_label(other_side),
                            tested_value=val,
                            valid_range=dom.valid_range,
                            units=dom.units,
                            reason=(
                                f"{INVALID_FEATURE_THRESHOLD_DOMAIN}: {base.name} threshold "
                                f"{val} outside valid_range={dom.valid_range} (units={dom.units})"
                            ),
                        )
                    )

        elif op_name == OperatorId.BETWEEN.value and len(children) == 3:
            value_side, lower, upper = children
            base, _ = _unwrap_abs(value_side)
            if base.kind is NodeKind.FEATURE:
                dom = domain_for_side(value_side, grammar)
                if dom is not None:
                    for label, bound_node in (("lower", lower), ("upper", upper)):
                        if bound_node.kind not in (NodeKind.CONSTANT, NodeKind.PARAMETER):
                            continue
                        val = _numeric_leaf_value(bound_node)
                        if val is not None and not dom.accepts(val):
                            violations.append(
                                ThresholdViolation(
                                    tree_path=tree_path,
                                    node_path=f"{node_path}.between_{label}",
                                    feature_id=base.name,
                                    parameter_or_constant=_leaf_label(bound_node),
                                    tested_value=val,
                                    valid_range=dom.valid_range,
                                    units=dom.units,
                                    reason=(
                                        f"{INVALID_FEATURE_THRESHOLD_DOMAIN}: BETWEEN {label} "
                                        f"bound {val} for {base.name} outside "
                                        f"valid_range={dom.valid_range} (units={dom.units})"
                                    ),
                                )
                            )

        elif (
            op_name in {OperatorId.ATR_STOP.value, OperatorId.ATR_TARGET.value, OperatorId.TRAILING_STOP.value}
            and len(children) == 2
        ):
            mult_node = children[1]
            if mult_node.kind in (NodeKind.CONSTANT, NodeKind.PARAMETER):
                dom = (
                    ATR_TARGET_MULT_DOMAIN
                    if op_name == OperatorId.ATR_TARGET.value
                    else ATR_STOP_MULT_DOMAIN
                )
                val = _numeric_leaf_value(mult_node)
                if val is not None and not dom.accepts(val):
                    violations.append(
                        ThresholdViolation(
                            tree_path=tree_path,
                            node_path=f"{node_path}.multiplier",
                            feature_id=None,
                            parameter_or_constant=_leaf_label(mult_node),
                            tested_value=val,
                            valid_range=dom.valid_range,
                            units=dom.units,
                            reason=(
                                f"{INVALID_FEATURE_THRESHOLD_DOMAIN}: {op_name} multiplier "
                                f"{val} outside valid_range={dom.valid_range} (units={dom.units})"
                            ),
                        )
                    )

    for i, child in enumerate(node.children):
        violations.extend(
            collect_threshold_violations(child, grammar, tree_path=tree_path, node_path=f"{node_path}.{i}")
        )
    return violations


def collect_candidate_threshold_violations(candidate: Any, grammar: Grammar) -> list[ThresholdViolation]:
    """Validate every executable tree on a candidate: entry, exit, stop, target,
    sizing, and each regime gate — regardless of how the candidate was created
    (generation, mutation, crossover, import, manual construction, or
    robustness perturbation)."""
    violations: list[ThresholdViolation] = []
    violations.extend(collect_threshold_violations(candidate.entry_tree, grammar, tree_path="entry_tree"))
    violations.extend(collect_threshold_violations(candidate.exit_tree, grammar, tree_path="exit_tree"))
    violations.extend(collect_threshold_violations(candidate.stop, grammar, tree_path="stop"))
    violations.extend(collect_threshold_violations(candidate.target, grammar, tree_path="target"))
    violations.extend(collect_threshold_violations(candidate.sizing, grammar, tree_path="sizing"))
    for i, gate in enumerate(candidate.regime_gates):
        violations.extend(
            collect_threshold_violations(gate, grammar, tree_path=f"regime_gate[{i}]")
        )
    return violations
