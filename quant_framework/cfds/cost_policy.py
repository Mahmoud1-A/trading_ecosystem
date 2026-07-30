"""
CFD cost-model policy and research eligibility (Phase 12.2).

A real CFD dataset must never reach the Alpha Miner without a valid cost model.
Zero spread is never an acceptable default: when the source carries bid/ask we
derive the observed spread causally, and when it does not we require an
explicitly configured spread/commission/slippage model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from features.microstructure import quoted_spread


class CostModelEligibility(str, Enum):
    ELIGIBLE_WITH_OBSERVED_SPREAD = "ELIGIBLE_WITH_OBSERVED_SPREAD"
    ELIGIBLE_WITH_MODELED_SPREAD = "ELIGIBLE_WITH_MODELED_SPREAD"
    NOT_ELIGIBLE_MISSING_COST_MODEL = "NOT_ELIGIBLE_MISSING_COST_MODEL"


@dataclass(frozen=True)
class ObservedSpreadStats:
    """Causally derived quoted-spread distribution in price points."""

    sample_count: int
    mean: float
    median: float
    p25: float
    p75: float
    p95: float
    p99: float
    maximum: float
    minimum: float
    zero_or_negative_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModeledCostSpec:
    """Explicit cost assumptions used when the source has no bid/ask."""

    base_spread_points: float
    stressed_spread_points: float
    commission_per_lot: float
    slippage_points: float
    financing_configured: bool
    time_of_day_spread_curve: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.base_spread_points <= 0:
            errors.append("base_spread_points must be > 0 (zero spread is never a default)")
        if self.stressed_spread_points <= 0:
            errors.append("stressed_spread_points must be > 0")
        if self.stressed_spread_points < self.base_spread_points:
            errors.append("stressed_spread_points must be >= base_spread_points")
        if self.commission_per_lot < 0:
            errors.append("commission_per_lot cannot be negative")
        if self.slippage_points < 0:
            errors.append("slippage_points cannot be negative")
        if not self.financing_configured:
            errors.append("financing assumptions must be configured for CFD research")
        return errors


@dataclass(frozen=True)
class CostModelAssessment:
    eligibility: CostModelEligibility
    reasons: list[str] = field(default_factory=list)
    observed_spread: ObservedSpreadStats | None = None
    modeled_cost: ModeledCostSpec | None = None

    @property
    def is_eligible(self) -> bool:
        return self.eligibility is not CostModelEligibility.NOT_ELIGIBLE_MISSING_COST_MODEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligibility": self.eligibility.value,
            "is_eligible": self.is_eligible,
            "reasons": list(self.reasons),
            "observed_spread": self.observed_spread.as_dict() if self.observed_spread else None,
            "modeled_cost": self.modeled_cost.as_dict() if self.modeled_cost else None,
        }


def derive_observed_spread(quotes: pd.DataFrame) -> ObservedSpreadStats:
    """
    Compute the quoted-spread distribution from bid/ask rows.

    Uses only information contained in each quote, so the result is causal and
    carries no forward-looking bias.
    """
    spread = quoted_spread(quotes).astype(float)
    clean = spread.dropna()
    if clean.empty:
        raise ValueError("no usable bid/ask rows to derive spread")
    return ObservedSpreadStats(
        sample_count=int(clean.size),
        mean=float(clean.mean()),
        median=float(clean.median()),
        p25=float(clean.quantile(0.25)),
        p75=float(clean.quantile(0.75)),
        p95=float(clean.quantile(0.95)),
        p99=float(clean.quantile(0.99)),
        maximum=float(clean.max()),
        minimum=float(clean.min()),
        zero_or_negative_count=int((clean <= 0).sum()),
    )


def assess_cost_model(
    *,
    spread_available: bool,
    quotes: pd.DataFrame | None = None,
    modeled_cost: ModeledCostSpec | None = None,
) -> CostModelAssessment:
    """Decide whether a CFD dataset carries a usable cost model."""
    if spread_available and quotes is not None and not quotes.empty:
        try:
            stats = derive_observed_spread(quotes)
        except (ValueError, KeyError) as exc:
            return CostModelAssessment(
                eligibility=CostModelEligibility.NOT_ELIGIBLE_MISSING_COST_MODEL,
                reasons=[f"observed spread could not be derived: {exc}"],
            )
        reasons: list[str] = []
        if stats.zero_or_negative_count:
            reasons.append(
                f"{stats.zero_or_negative_count} quote(s) had non-positive spread"
            )
        if stats.median <= 0:
            return CostModelAssessment(
                eligibility=CostModelEligibility.NOT_ELIGIBLE_MISSING_COST_MODEL,
                reasons=reasons + ["median observed spread is non-positive"],
                observed_spread=stats,
            )
        return CostModelAssessment(
            eligibility=CostModelEligibility.ELIGIBLE_WITH_OBSERVED_SPREAD,
            reasons=reasons
            or ["observed quoted spread derived causally from source bid/ask"],
            observed_spread=stats,
            modeled_cost=modeled_cost,
        )

    if modeled_cost is None:
        return CostModelAssessment(
            eligibility=CostModelEligibility.NOT_ELIGIBLE_MISSING_COST_MODEL,
            reasons=[
                "source has no bid/ask spread and no modeled cost spec was supplied"
            ],
        )
    errors = modeled_cost.validation_errors()
    if errors:
        return CostModelAssessment(
            eligibility=CostModelEligibility.NOT_ELIGIBLE_MISSING_COST_MODEL,
            reasons=errors,
            modeled_cost=modeled_cost,
        )
    return CostModelAssessment(
        eligibility=CostModelEligibility.ELIGIBLE_WITH_MODELED_SPREAD,
        reasons=["explicit spread/commission/slippage/financing model configured"],
        modeled_cost=modeled_cost,
    )
