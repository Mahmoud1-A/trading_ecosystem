"""
CFD research instrument specification (Phase 12.2).

Deliberately does not expose futures concepts (contract multiplier as a CME
multiplier, tick value, initial/maintenance margin per contract, expiration,
rollover rules). A CFD is sized by ``value_per_point`` and ``lot`` and is
financed overnight rather than rolled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cfds.financing_profile import CFDFinancingProfile, dukascopy_us500_financing_profile
from cfds.symbol_mapping import US500_CFD, assert_not_futures_identifier
from config.asset_spec import CFDAssetSpec, SessionHours

# Fields that only make sense for exchange-listed futures. Presence of any of
# these on a CFD spec indicates the two instrument models have been conflated.
FORBIDDEN_FUTURES_FIELDS = frozenset(
    {
        "multiplier",
        "tick_value",
        "expiration",
        "active_contract",
        "rollover_rules",
        "continuous_series",
        "initial_margin",
        "maintenance_margin",
        "exchange",
    }
)


class CFDInstrumentError(ValueError):
    """Raised when a CFD instrument spec is invalid or futures-contaminated."""


@dataclass(frozen=True)
class TradingSession:
    """A single broker trading window in the broker's own timezone."""

    name: str
    open_time: time
    close_time: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "weekdays": list(self.weekdays),
        }


@dataclass(frozen=True)
class MaintenanceWindow:
    """Recurring daily broker downtime, e.g. the platform rollover break."""

    name: str
    start_time: time
    end_time: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "weekdays": list(self.weekdays),
        }


@dataclass(frozen=True)
class CFDCommissionModel:
    """Commission terms. CFD index products are frequently spread-only."""

    per_lot: float = 0.0
    percent_of_notional: float = 0.0
    minimum: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CFDSlippageModel:
    """Slippage assumptions in price points."""

    base_points: float = 0.0
    stressed_points: float = 0.0
    volatility_multiplier: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CFDResearchInstrument:
    """Configuration for one tradable CFD used in research."""

    internal_symbol: str
    provider_symbol: str
    asset_class: str
    quote_currency: str
    value_per_point: float
    minimum_lot: float
    lot_step: float
    minimum_price_increment: float
    typical_spread_points: float
    stressed_spread_points: float
    commission_model: CFDCommissionModel
    slippage_model: CFDSlippageModel
    margin_requirement: float
    leverage: float
    financing: CFDFinancingProfile
    broker_timezone: str
    trading_sessions: tuple[TradingSession, ...]
    maintenance_windows: tuple[MaintenanceWindow, ...] = field(default_factory=tuple)
    target_broker_symbol: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.asset_class.upper() != "CFD":
            raise CFDInstrumentError(
                f"CFDResearchInstrument requires asset_class=CFD, got {self.asset_class!r}"
            )
        assert_not_futures_identifier(self.provider_symbol, field_name="provider_symbol")
        assert_not_futures_identifier(self.internal_symbol, field_name="internal_symbol")
        if self.target_broker_symbol:
            assert_not_futures_identifier(
                self.target_broker_symbol, field_name="target_broker_symbol"
            )
        try:
            ZoneInfo(self.broker_timezone)
        except ZoneInfoNotFoundError as exc:
            raise CFDInstrumentError(f"Unknown IANA timezone: {self.broker_timezone}") from exc
        if self.value_per_point <= 0:
            raise CFDInstrumentError("value_per_point must be positive")
        if self.minimum_lot <= 0 or self.lot_step <= 0:
            raise CFDInstrumentError("minimum_lot and lot_step must be positive")
        if self.minimum_price_increment <= 0:
            raise CFDInstrumentError("minimum_price_increment must be positive")
        if self.typical_spread_points < 0 or self.stressed_spread_points < 0:
            raise CFDInstrumentError("spread assumptions cannot be negative")
        if self.stressed_spread_points < self.typical_spread_points:
            raise CFDInstrumentError(
                "stressed_spread_points must be >= typical_spread_points"
            )
        if not 0.0 < self.margin_requirement <= 1.0:
            raise CFDInstrumentError("margin_requirement must be a fraction in (0, 1]")
        if self.leverage <= 0:
            raise CFDInstrumentError("leverage must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "internal_symbol": self.internal_symbol,
            "provider_symbol": self.provider_symbol,
            "target_broker_symbol": self.target_broker_symbol,
            "asset_class": self.asset_class,
            "quote_currency": self.quote_currency,
            "value_per_point": self.value_per_point,
            "minimum_lot": self.minimum_lot,
            "lot_step": self.lot_step,
            "minimum_price_increment": self.minimum_price_increment,
            "typical_spread_points": self.typical_spread_points,
            "stressed_spread_points": self.stressed_spread_points,
            "commission_model": self.commission_model.as_dict(),
            "slippage_model": self.slippage_model.as_dict(),
            "margin_requirement": self.margin_requirement,
            "leverage": self.leverage,
            "financing": self.financing.as_dict(),
            "broker_timezone": self.broker_timezone,
            "trading_sessions": [s.as_dict() for s in self.trading_sessions],
            "maintenance_windows": [w.as_dict() for w in self.maintenance_windows],
            "notes": self.notes,
            "futures_multiplier": None,
            "futures_tick_value": None,
            "futures_expiration": None,
            "futures_rollover": "NOT_APPLICABLE",
        }

    def to_asset_spec(self) -> CFDAssetSpec:
        """Build the engine-level CFD spec used by execution and financing."""
        return CFDAssetSpec(
            broker_symbol=self.target_broker_symbol or self.provider_symbol,
            contract_size=self.value_per_point,
            leverage=self.leverage,
            broker_spread_markup=0.0,
            overnight_swap_long=self.financing.overnight_financing_long,
            overnight_swap_short=self.financing.overnight_financing_short,
            margin_rate=self.margin_requirement,
            stop_out_level=0.5,
            tick_size=self.minimum_price_increment,
            min_lot=self.minimum_lot,
            lot_step=self.lot_step,
            broker_trading_hours=SessionHours(
                timezone=self.broker_timezone,
                open_time=self.trading_sessions[0].open_time,
                close_time=self.trading_sessions[0].close_time,
                overnight=True,
            ),
        )


def default_us500_cfd_instrument(
    *,
    provider_symbol: str = "USA500IDXUSD",
    financing: CFDFinancingProfile | None = None,
) -> CFDResearchInstrument:
    """
    Research default for the S&P 500 index CFD.

    Sizing follows the common "1 unit = 1 index point" CFD convention rather
    than the CME E-mini multiplier of 50. These values describe the research
    source and must be re-specified for any real execution target.
    """
    return CFDResearchInstrument(
        internal_symbol=US500_CFD,
        provider_symbol=provider_symbol,
        asset_class="CFD",
        quote_currency="USD",
        value_per_point=1.0,
        minimum_lot=0.01,
        lot_step=0.01,
        minimum_price_increment=0.001,
        typical_spread_points=0.5,
        stressed_spread_points=3.0,
        commission_model=CFDCommissionModel(per_lot=0.0, percent_of_notional=0.0),
        slippage_model=CFDSlippageModel(
            base_points=0.1, stressed_points=0.75, volatility_multiplier=1.5
        ),
        margin_requirement=0.05,
        leverage=20.0,
        financing=financing or dukascopy_us500_financing_profile(),
        broker_timezone="UTC",
        trading_sessions=(
            TradingSession(
                name="index_cfd_continuous",
                open_time=time(0, 0),
                close_time=time(23, 59, 59),
            ),
        ),
        maintenance_windows=(
            MaintenanceWindow(
                name="daily_rollover_break",
                start_time=time(21, 55),
                end_time=time(22, 5),
            ),
        ),
        notes=(
            "Generic index-CFD research sizing. Not CME ES: no 50x multiplier, "
            "no $12.50 tick value, no expiration, no rollover."
        ),
    )
