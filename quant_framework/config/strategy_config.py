"""Strategy, regime, and sizing configuration (no hardcoded params in alpha/)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from config.models import Timeframe, UnitInterval


class StrategyParams(BaseModel):
    """Intraday mean-reversion parameters — consumed by alpha/ modules only."""

    model_config = {"extra": "forbid", "frozen": True}

    family: str = "mean_reversion_vwap_bb"
    lookback: int = Field(20, ge=5)
    z_entry: float = Field(2.0, gt=0)
    z_exit: float = Field(0.25, ge=0)
    bb_std: float = Field(2.0, gt=0)
    use_vwap_zscore: bool = True
    max_holding_bars: int = Field(60, ge=1)
    allow_short: bool = True
    stop_atr_mult: float = Field(1.5, gt=0)
    target_atr_mult: float = Field(2.0, gt=0)
    vol_filter_min_atr: float | None = Field(None, ge=0)
    vol_filter_max_atr: float | None = Field(None, ge=0)
    use_regime_filter: bool = True
    session_flatten: bool = True
    atr_period: int = Field(14, ge=2)

    @model_validator(mode="after")
    def _z_order(self) -> StrategyParams:
        if self.z_exit >= self.z_entry:
            raise ValueError("z_exit must be strictly less than z_entry")
        return self


class RegimeParams(BaseModel):
    """Higher-timeframe regime detection knobs."""

    model_config = {"extra": "forbid", "frozen": True}

    htf: Timeframe = Timeframe.H1
    atr_period: int = Field(14, ge=2)
    adx_period: int = Field(14, ge=2)
    vol_lookback: int = Field(20, ge=5)
    adx_trend_threshold: float = Field(25.0, gt=0)
    atr_vol_percentile: float = Field(70.0, ge=0, le=100)
    ma_slope_lookback: int = Field(20, ge=3)
    range_compression_threshold: float = Field(0.85, gt=0)
    trade_only_range: bool = True


class SizingParams(BaseModel):
    """Volatility-targeted position sizing parameters."""

    model_config = {"extra": "forbid", "frozen": True}

    risk_per_trade_pct: UnitInterval = Field(0.005, description="Fraction of equity risked per trade")
    target_vol_annual: float = Field(0.15, gt=0)
    vol_lookback: int = Field(20, ge=5)
    atr_stop_mult: float = Field(1.5, gt=0)
    min_lot: float = Field(0.01, gt=0)
    lot_step: float = Field(0.01, gt=0)
    max_lots: float = Field(10.0, gt=0)
    bars_per_year: int = Field(252 * 78, ge=1)
