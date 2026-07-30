"""
Generic, configurable CFD prop-firm research profile (Phase 12.2).

No named prop firm's rules are hardcoded. Named-company profiles must later be
created as versioned configurations derived from the exact purchased account.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class PropProfileError(ValueError):
    """Raised when prop account constraints are internally inconsistent."""


@dataclass(frozen=True)
class GenericCFDPropProfile:
    """Account-level constraints a research strategy must respect."""

    profile_id: str
    profile_version: str
    initial_balance: float
    daily_loss_limit: float
    maximum_total_drawdown: float
    trailing_drawdown: bool
    profit_target: float
    minimum_trading_days: int
    maximum_position_exposure: float
    leverage: float
    overnight_permitted: bool
    weekend_permitted: bool
    commission_per_lot: float
    base_spread_points: float
    stressed_spread_points: float
    slippage_points: float
    news_restrictions: dict[str, Any] | None = None
    consistency_rules: dict[str, Any] | None = None
    notes: str = ""
    derived_from_named_firm: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise PropProfileError("initial_balance must be positive")
        if self.daily_loss_limit <= 0 or self.maximum_total_drawdown <= 0:
            raise PropProfileError("loss limits must be positive")
        if self.daily_loss_limit > self.maximum_total_drawdown:
            raise PropProfileError(
                "daily_loss_limit cannot exceed maximum_total_drawdown"
            )
        if self.profit_target <= 0:
            raise PropProfileError("profit_target must be positive")
        if self.minimum_trading_days < 0:
            raise PropProfileError("minimum_trading_days cannot be negative")
        if self.maximum_position_exposure <= 0:
            raise PropProfileError("maximum_position_exposure must be positive")
        if self.leverage <= 0:
            raise PropProfileError("leverage must be positive")
        if self.base_spread_points < 0 or self.stressed_spread_points < 0:
            raise PropProfileError("spread assumptions cannot be negative")
        if self.stressed_spread_points < self.base_spread_points:
            raise PropProfileError(
                "stressed_spread_points must be >= base_spread_points"
            )

    @property
    def daily_loss_limit_pct(self) -> float:
        return self.daily_loss_limit / self.initial_balance

    @property
    def max_drawdown_pct(self) -> float:
        return self.maximum_total_drawdown / self.initial_balance

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["daily_loss_limit_pct"] = self.daily_loss_limit_pct
        payload["max_drawdown_pct"] = self.max_drawdown_pct
        payload["warnings"] = list(self.warnings)
        return payload


def default_generic_prop_profile(
    *,
    initial_balance: float = 100_000.0,
    profile_version: str = "generic_cfd_prop_v1",
) -> GenericCFDPropProfile:
    """
    Neutral evaluation-style constraints for research only.

    The percentages here are common industry shapes, not any specific firm's
    published rulebook. Replace with the real account terms before relying on
    pass/fail conclusions.
    """
    return GenericCFDPropProfile(
        profile_id="generic_cfd_prop",
        profile_version=profile_version,
        initial_balance=initial_balance,
        daily_loss_limit=0.05 * initial_balance,
        maximum_total_drawdown=0.10 * initial_balance,
        trailing_drawdown=False,
        profit_target=0.10 * initial_balance,
        minimum_trading_days=5,
        maximum_position_exposure=2.0 * initial_balance,
        leverage=20.0,
        overnight_permitted=True,
        weekend_permitted=False,
        commission_per_lot=0.0,
        base_spread_points=0.8,
        stressed_spread_points=4.0,
        slippage_points=0.2,
        news_restrictions=None,
        consistency_rules=None,
        derived_from_named_firm=False,
        notes=(
            "Generic research placeholder. Not derived from any named prop firm's "
            "rulebook."
        ),
        warnings=(
            "Replace with a versioned profile built from the exact purchased "
            "account before drawing pass/fail conclusions.",
        ),
    )
