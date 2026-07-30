"""Causal Feature Store — Phase 6B."""

from features.catalog import FeatureCatalog
from features.contracts import (
    FeatureApprovalStatus,
    FeatureCategory,
    FeatureContract,
    FeatureFrame,
    MissingValuePolicy,
)
from features.definitions import build_default_catalog
from features.generator import FEATURE_SET_VERSION, FeatureGenerator, FeatureGeneratorConfig
from features.liquidity import causal_session_vwap
from features.normalization import TrainOnlyQuantileThresholds, TrainOnlyScaler
from features.validation import (
    FutureInvarianceResult,
    assert_no_vault_columns,
    assert_tick_volume_not_mislabeled,
    assert_warmup_enforced,
    future_invariance_test,
    miner_feature_ids,
)

__all__ = [
    "FEATURE_SET_VERSION",
    "FeatureApprovalStatus",
    "FeatureCatalog",
    "FeatureCategory",
    "FeatureContract",
    "FeatureFrame",
    "FeatureGenerator",
    "FeatureGeneratorConfig",
    "FutureInvarianceResult",
    "MissingValuePolicy",
    "TrainOnlyQuantileThresholds",
    "TrainOnlyScaler",
    "assert_no_vault_columns",
    "assert_tick_volume_not_mislabeled",
    "assert_warmup_enforced",
    "build_default_catalog",
    "causal_session_vwap",
    "future_invariance_test",
    "miner_feature_ids",
]
