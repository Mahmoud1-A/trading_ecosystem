"""Prop-firm risk profile configuration (Pydantic v2)."""

from __future__ import annotations

from datetime import time
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from config.models import DrawdownType, NonNegativeFloat, PositiveFloat, UnitInterval


class PropProfile(BaseModel):
    """
    Configurable prop-firm constraint profile.

    Hard limits mirror challenge/funded rules. Soft/internal limits are used by the
    risk FSM for CAUTION / REDUCE_ONLY transitions (Phase 3).
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(default="default_prop", min_length=1)
    prop_hard_daily_loss_limit: float = Field(
        ...,
        lt=0,
        description="Hard daily loss as negative fraction of equity, e.g. -0.015 for -1.5%.",
    )
    prop_total_drawdown_limit: float = Field(
        ...,
        lt=0,
        description="Hard total drawdown as negative fraction of equity, e.g. -0.05.",
    )
    drawdown_type: DrawdownType = DrawdownType.STATIC
    internal_soft_daily_limit: float = Field(
        ...,
        lt=0,
        description="Soft daily loss fraction triggering CAUTION (must be > hard limit).",
    )
    internal_risk_budget: UnitInterval = Field(
        0.5,
        description="Fraction of per-trade risk allowed under CAUTION.",
    )
    daily_reset_timezone: str = Field(
        "America/New_York",
        description="IANA timezone for daily PnL reset.",
    )
    daily_reset_time: time = Field(
        default_factory=lambda: time(17, 0),
        description="Local clock time at which the daily loss window resets.",
    )
    include_unrealized_pnl: bool = True
    close_positions_on_hard_breach: bool = True
    cancel_pending_orders_on_hard_breach: bool = True
    allow_overnight_positions: bool = False
    max_positions: int = Field(3, ge=1)
    max_symbol_exposure: PositiveFloat = Field(
        1.0,
        description="Max absolute lots (or contracts) per symbol.",
    )
    max_portfolio_exposure: PositiveFloat = Field(
        5.0,
        description="Max absolute aggregate lots across the book.",
    )
    soft_to_reduce_only_ratio: UnitInterval = Field(
        0.85,
        description=(
            "When soft daily loss reaches this fraction of the distance to the hard "
            "limit (from 0), transition CAUTION -> REDUCE_ONLY."
        ),
    )

    @field_validator("daily_reset_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def _validate_limit_ordering(self) -> PropProfile:
        if self.internal_soft_daily_limit <= self.prop_hard_daily_loss_limit:
            raise ValueError(
                "internal_soft_daily_limit must be strictly greater than "
                "prop_hard_daily_loss_limit (e.g. -0.010 > -0.015)."
            )
        if self.prop_total_drawdown_limit >= 0:
            raise ValueError("prop_total_drawdown_limit must be negative")
        return self

    def zoneinfo(self) -> ZoneInfo:
        return ZoneInfo(self.daily_reset_timezone)


def default_prop_profile(
    style: Literal["challenge", "funded", "conservative"] = "challenge",
) -> PropProfile:
    """Factory for common prop-firm templates."""
    templates: dict[str, dict[str, object]] = {
        "challenge": {
            "name": "challenge_1p5_daily_5_total",
            "prop_hard_daily_loss_limit": -0.015,
            "prop_total_drawdown_limit": -0.05,
            "internal_soft_daily_limit": -0.010,
            "drawdown_type": DrawdownType.STATIC,
        },
        "funded": {
            "name": "funded_1p5_daily_trailing_5",
            "prop_hard_daily_loss_limit": -0.015,
            "prop_total_drawdown_limit": -0.05,
            "internal_soft_daily_limit": -0.010,
            "drawdown_type": DrawdownType.TRAILING,
        },
        "conservative": {
            "name": "conservative_1_daily_4_total",
            "prop_hard_daily_loss_limit": -0.01,
            "prop_total_drawdown_limit": -0.04,
            "internal_soft_daily_limit": -0.006,
            "drawdown_type": DrawdownType.STATIC,
            "internal_risk_budget": 0.35,
            "max_positions": 2,
        },
    }
    return PropProfile(**templates[style])  # type: ignore[arg-type]
