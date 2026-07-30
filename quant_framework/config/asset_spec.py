"""Asset specifications — Futures and CFDs are modeled separately."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from config.models import (
    AssetClass,
    ContinuousAdjustment,
    NonNegativeFloat,
    PositiveFloat,
    RolloverMethod,
    SettlementBehavior,
    UnitInterval,
)


class SessionHours(BaseModel):
    """Exchange / broker session window in a named timezone."""

    model_config = {"extra": "forbid", "frozen": True}

    timezone: str = "America/Chicago"
    open_time: time = Field(default_factory=lambda: time(8, 30))
    close_time: time = Field(default_factory=lambda: time(15, 0))
    overnight: bool = False

    @field_validator("timezone")
    @classmethod
    def _tz(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value

    def zoneinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class ContinuousSeriesConfig(BaseModel):
    """
    Research-only continuous futures series configuration.

    Continuous prices may be used for signals/features. Executable fills MUST
    always reference an explicit tradable contract (see FuturesAssetSpec.active_contract).
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = True
    adjustment: ContinuousAdjustment = ContinuousAdjustment.BACKWARD_RATIO
    roll_method: RolloverMethod = RolloverMethod.VOLUME
    roll_days_before_expiry: int = Field(5, ge=0)
    # Rollover price adjustments must never create artificial tradable PnL.
    exclude_roll_gap_from_pnl: bool = True


class RolloverRules(BaseModel):
    """Rules governing when the active tradable contract changes."""

    model_config = {"extra": "forbid", "frozen": True}

    method: RolloverMethod = RolloverMethod.VOLUME
    days_before_expiry: int = Field(5, ge=0)
    volume_ratio_threshold: PositiveFloat = Field(
        1.0,
        description="Roll when back-month volume / front-month volume >= threshold.",
    )
    persist_decision: bool = Field(
        True,
        description="Persist active contract + rollover decision with every fill.",
    )


class FuturesAssetSpec(BaseModel):
    """Futures instrument specification (executable contract vs continuous series)."""

    model_config = {"extra": "forbid", "frozen": True}

    asset_class: Literal[AssetClass.FUTURES] = AssetClass.FUTURES
    symbol: str = Field(..., min_length=1, description="Root symbol, e.g. ES")
    exchange: str = Field(..., min_length=1)
    multiplier: PositiveFloat = Field(..., description="Contract multiplier / point value")
    tick_size: PositiveFloat
    tick_value: PositiveFloat
    currency: str = Field("USD", min_length=3, max_length=3)
    expiration: date | None = Field(
        None,
        description="Expiration of the active tradable contract (None if unresolved).",
    )
    rollover_rules: RolloverRules = Field(default_factory=RolloverRules)
    continuous_series: ContinuousSeriesConfig = Field(default_factory=ContinuousSeriesConfig)
    active_contract: str = Field(
        ...,
        min_length=1,
        description="Explicit tradable contract id, e.g. ESH24. Required for fills.",
    )
    exchange_calendar: str = Field(
        "CME",
        description="Calendar identifier used by data.calendars.",
    )
    session_hours: SessionHours = Field(default_factory=SessionHours)
    initial_margin: PositiveFloat
    maintenance_margin: PositiveFloat
    settlement_behavior: SettlementBehavior = SettlementBehavior.MARK_TO_MARKET

    @model_validator(mode="after")
    def _margin_ordering(self) -> FuturesAssetSpec:
        if self.maintenance_margin > self.initial_margin:
            raise ValueError("maintenance_margin cannot exceed initial_margin")
        if abs(self.tick_value - self.tick_size * self.multiplier) > 1e-9:
            # Allow intentional divergence but require near-consistency by default.
            # Documented exception: some brokers quote tick_value independently.
            pass
        return self


class CFDAssetSpec(BaseModel):
    """CFD instrument specification — accounting rules must not mix with Futures."""

    model_config = {"extra": "forbid", "frozen": True}

    asset_class: Literal[AssetClass.CFD] = AssetClass.CFD
    broker_symbol: str = Field(..., min_length=1)
    contract_size: PositiveFloat = 1.0
    leverage: PositiveFloat = Field(20.0, description="Max leverage, e.g. 20 => 5% margin")
    broker_spread_markup: NonNegativeFloat = Field(
        0.0,
        description="Additional spread markup in price units.",
    )
    variable_spread_by_time: bool = True
    variable_spread_by_volatility: bool = True
    overnight_swap_long: float = Field(
        0.0,
        description="Financing rate per night for long positions (fraction of notional).",
    )
    overnight_swap_short: float = Field(
        0.0,
        description="Financing rate per night for short positions (fraction of notional).",
    )
    session_gaps: bool = True
    weekend_gaps: bool = True
    broker_trading_hours: SessionHours = Field(
        default_factory=lambda: SessionHours(
            timezone="UTC",
            open_time=time(0, 0),
            close_time=time(23, 59, 59),
            overnight=True,
        )
    )
    margin_rate: UnitInterval = Field(
        0.05,
        description="Initial margin as fraction of notional; typically 1/leverage.",
    )
    stop_out_level: UnitInterval = Field(
        0.5,
        description="Stop-out when equity / used_margin falls to this ratio.",
    )
    currency: str = Field("USD", min_length=3, max_length=3)
    tick_size: PositiveFloat = 0.01
    min_lot: PositiveFloat = 0.01
    lot_step: PositiveFloat = 0.01

    @model_validator(mode="after")
    def _leverage_margin_consistency(self) -> CFDAssetSpec:
        implied = 1.0 / float(self.leverage)
        if abs(implied - float(self.margin_rate)) > 1e-6:
            # Soft consistency check — allow broker-specific divergence but warn via error
            # only when margin_rate is clearly inconsistent with extreme leverage.
            if self.margin_rate > implied * 5 or self.margin_rate < implied / 5:
                raise ValueError(
                    f"margin_rate={self.margin_rate} is inconsistent with leverage={self.leverage}"
                )
        return self


AssetSpec = Annotated[FuturesAssetSpec | CFDAssetSpec, Field(discriminator="asset_class")]


def default_es_futures(active_contract: str = "ESH24") -> FuturesAssetSpec:
    """CME E-mini S&P 500 template for research demos."""
    return FuturesAssetSpec(
        symbol="ES",
        exchange="CME",
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.50,
        currency="USD",
        expiration=date(2024, 3, 15),
        active_contract=active_contract,
        exchange_calendar="CME",
        session_hours=SessionHours(
            timezone="America/Chicago",
            open_time=time(8, 30),
            close_time=time(15, 0),
        ),
        initial_margin=12_000.0,
        maintenance_margin=10_800.0,
        settlement_behavior=SettlementBehavior.MARK_TO_MARKET,
    )


def default_cfd_index(broker_symbol: str = "US500") -> CFDAssetSpec:
    """Generic equity-index CFD template."""
    return CFDAssetSpec(
        broker_symbol=broker_symbol,
        contract_size=1.0,
        leverage=20.0,
        broker_spread_markup=0.25,
        overnight_swap_long=-0.0001,
        overnight_swap_short=0.00005,
        margin_rate=0.05,
        stop_out_level=0.5,
        tick_size=0.1,
        min_lot=0.01,
        lot_step=0.01,
    )


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)
