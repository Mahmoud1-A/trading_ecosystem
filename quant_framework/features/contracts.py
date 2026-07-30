"""Feature contracts — approval, causality, and capability requirements (Phase 6B)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from data.events.enums import DataCapability, EventSchemaType


class FeatureApprovalStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    APPROVED = "APPROVED"
    DISABLED = "DISABLED"
    DEPRECATED = "DEPRECATED"


class FeatureCategory(str, Enum):
    PRICE = "PRICE"
    LIQUIDITY = "LIQUIDITY"
    VOLATILITY = "VOLATILITY"
    MICROSTRUCTURE = "MICROSTRUCTURE"
    TEMPORAL = "TEMPORAL"
    CROSS_ASSET = "CROSS_ASSET"
    REGIME = "REGIME"


class MissingValuePolicy(str, Enum):
    PROPAGATE_NAN = "PROPAGATE_NAN"
    DROP_UNTIL_WARMUP = "DROP_UNTIL_WARMUP"
    ZERO_FILL_IN_SESSION = "ZERO_FILL_IN_SESSION"
    # Never silently forward-fill across sessions/closures


@dataclass(frozen=True)
class FeatureContract:
    """
    Complete declaration for one feature.

    The Alpha Miner may consume only APPROVED causal features whose required
    provider capabilities are available.
    """

    feature_id: str
    feature_version: str
    feature_name: str
    category: FeatureCategory
    description: str
    economic_rationale: str
    input_event_types: tuple[EventSchemaType, ...]
    source_symbols: tuple[str, ...]
    permitted_asset_classes: tuple[str, ...]
    lookback_bars: int
    warm_up_bars: int
    update_frequency: str
    source_timestamp_rule: str
    availability_timestamp_rule: str
    maximum_staleness: str
    missing_value_policy: MissingValuePolicy
    output_units: str
    causal: bool
    requires_training_fit: bool
    complexity_cost: float
    dependencies: tuple[str, ...]
    implementation_hash: str
    approval_status: FeatureApprovalStatus
    required_capabilities: tuple[DataCapability, ...] = ()
    notes: str = ""

    def miner_eligible(self, available_capabilities: set[DataCapability] | None = None) -> bool:
        if self.approval_status is not FeatureApprovalStatus.APPROVED:
            return False
        if not self.causal:
            return False
        caps = available_capabilities or set()
        return all(c in caps for c in self.required_capabilities)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["missing_value_policy"] = self.missing_value_policy.value
        d["approval_status"] = self.approval_status.value
        d["input_event_types"] = [e.value for e in self.input_event_types]
        d["required_capabilities"] = [c.value for c in self.required_capabilities]
        return d


@dataclass
class FeatureFrame:
    """Computed feature matrix with explicit availability timestamps."""

    values: Any  # pd.DataFrame
    availability_timestamps: Any  # pd.Series aligned to values.index
    source_timestamps: Any  # pd.Series
    feature_ids: tuple[str, ...]
    feature_set_version: str
    feature_set_hash: str
    code_hash: str
    disabled_features: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def as_persistable(self) -> Any:
        import pandas as pd

        out = self.values.copy()
        out["availability_timestamp"] = self.availability_timestamps
        out["source_timestamp"] = self.source_timestamps
        return out
