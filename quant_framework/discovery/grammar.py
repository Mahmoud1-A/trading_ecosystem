"""Grammar limits and feature leaf declarations for the Strategy DSL."""

from __future__ import annotations

from dataclasses import dataclass, field

from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.types import ValueType


GRAMMAR_VERSION = "strategy_dsl_v1"


@dataclass(frozen=True)
class GrammarLimits:
    max_tree_depth: int = 5
    max_nodes: int = 25
    max_distinct_features: int = 5
    max_free_parameters: int = 8
    max_entry_conditions: int = 4
    max_exit_conditions: int = 4
    max_cross_asset_inputs: int = 2
    max_regime_gates: int = 2
    max_position_sizing_modifiers: int = 2
    max_rolling_lookback: int = 100


@dataclass(frozen=True)
class FeatureLeaf:
    feature_id: str
    value_type: ValueType
    is_cross_asset: bool = False


# Default approved feature leaves the DSL may reference (IDs match Phase 6B catalog)
DEFAULT_FEATURE_LEAVES: tuple[FeatureLeaf, ...] = (
    FeatureLeaf("price.simple_return_1", ValueType.RETURN),
    FeatureLeaf("price.log_return_1", ValueType.RETURN),
    FeatureLeaf("price.return_5", ValueType.RETURN),
    FeatureLeaf("price.close_to_open", ValueType.RETURN),
    FeatureLeaf("price.gap_size", ValueType.PRICE),
    FeatureLeaf("price.rolling_z_20", ValueType.ZSCORE),
    FeatureLeaf("price.dist_rolling_mean_20", ValueType.RATIO),
    FeatureLeaf("price.breakout_distance_20", ValueType.RATIO),
    FeatureLeaf("price.rolling_rank_20", ValueType.RANK),
    FeatureLeaf("liq.dist_session_vwap", ValueType.RATIO),
    FeatureLeaf("liq.volume_pct_20", ValueType.RANK),
    FeatureLeaf("vol.atr_14", ValueType.VOLATILITY),
    FeatureLeaf("vol.norm_atr_14", ValueType.RATIO),
    FeatureLeaf("vol.realized_20", ValueType.VOLATILITY),
    FeatureLeaf("vol.range_compression_20", ValueType.RATIO),
    FeatureLeaf("temp.minutes_since_open", ValueType.TIME),
    FeatureLeaf("temp.dow", ValueType.SCALAR),
    # Regime leaves — required so REGIME_GATE child[1] never falls back to SCALAR
    FeatureLeaf("regime.trend_state", ValueType.REGIME),
    FeatureLeaf("regime.volatility_state", ValueType.REGIME),
    FeatureLeaf("xasset.lagged_return", ValueType.RETURN, is_cross_asset=True),
    FeatureLeaf("xasset.rolling_corr_20", ValueType.RATIO, is_cross_asset=True),
)


@dataclass
class Grammar:
    limits: GrammarLimits = field(default_factory=GrammarLimits)
    feature_leaves: tuple[FeatureLeaf, ...] = DEFAULT_FEATURE_LEAVES
    version: str = GRAMMAR_VERSION
    allowed_operators: frozenset[OperatorId] | None = None
    family_id: str | None = None

    def operators_returning(self, value_type: ValueType) -> list[OperatorId]:
        ops = [oid for oid, spec in OPERATOR_REGISTRY.items() if spec.output_type is value_type]
        if self.allowed_operators is not None:
            ops = [oid for oid in ops if oid in self.allowed_operators]
        return ops

    def allows_operator(self, oid: OperatorId) -> bool:
        if self.allowed_operators is None:
            return True
        return oid in self.allowed_operators

    def feature_ids(self) -> tuple[str, ...]:
        return tuple(f.feature_id for f in self.feature_leaves)

    def leaf_for(self, feature_id: str) -> FeatureLeaf:
        for leaf in self.feature_leaves:
            if leaf.feature_id == feature_id:
                return leaf
        raise KeyError(f"Unknown feature leaf {feature_id!r}")
