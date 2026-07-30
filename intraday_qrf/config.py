"""Centralized configuration for the Intraday Quantitative Research Framework."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Timeframe(str, Enum):
    M1 = "1min"
    M5 = "5min"
    H1 = "1h"
    D1 = "1D"


class AssetClass(str, Enum):
    FUTURES = "futures"
    CFD = "cfd"


class AssetSpec(BaseModel):
    """Contract / instrument specification."""

    symbol: str = Field(..., min_length=1)
    asset_class: AssetClass = AssetClass.FUTURES
    tick_size: float = Field(0.25, gt=0, description="Minimum price increment")
    tick_value: float = Field(12.50, gt=0, description="Currency value per tick")
    point_value: float = Field(50.0, gt=0, description="Currency value per full point")
    lot_size: float = Field(1.0, gt=0, description="Contract multiplier / lot size")
    currency: str = "USD"

    @property
    def value_per_point(self) -> float:
        return self.point_value * self.lot_size


class FrictionConfig(BaseModel):
    """Realistic execution friction parameters."""

    spread_ticks: float = Field(1.0, ge=0, description="Base bid-ask spread in ticks")
    spread_vol_multiplier: float = Field(
        0.15, ge=0, description="Extra spread = multiplier * ATR / tick_size"
    )
    slippage_ticks_mean: float = Field(0.5, ge=0)
    slippage_ticks_std: float = Field(0.35, ge=0)
    slippage_seed: int | None = Field(42, description="RNG seed for reproducible slippage")
    commission_per_lot: float = Field(2.50, ge=0, description="Round-trip commission per lot")
    commission_per_side: bool = Field(
        False, description="If True, commission_per_lot is charged per fill side"
    )


class StrategyParams(BaseModel):
    """Intraday statistical mean-reversion parameters (no hardcoded values in alpha/)."""

    lookback: int = Field(20, ge=5, description="Rolling window for VWAP / BB / Z-score")
    z_entry: float = Field(2.0, gt=0, description="|Z| threshold to enter")
    z_exit: float = Field(0.25, ge=0, description="|Z| threshold to flatten")
    bb_std: float = Field(2.0, gt=0, description="Bollinger Band standard deviations")
    use_vwap_zscore: bool = Field(True, description="Z-score vs session VWAP vs BB mid")
    max_holding_bars: int = Field(60, ge=1, description="Force flat after N bars")
    allow_short: bool = True
    session_start: str = Field("09:30", pattern=r"^\d{2}:\d{2}$")
    session_end: str = Field("15:55", pattern=r"^\d{2}:\d{2}$")
    flatten_before_close: bool = True

    @model_validator(mode="after")
    def _validate_z_levels(self) -> StrategyParams:
        if self.z_exit >= self.z_entry:
            raise ValueError("z_exit must be strictly less than z_entry")
        return self


class RegimeParams(BaseModel):
    """Higher-timeframe regime detection knobs."""

    htf: Timeframe = Timeframe.H1
    atr_period: int = Field(14, ge=2)
    adx_period: int = Field(14, ge=2)
    vol_lookback: int = Field(20, ge=5)
    adx_trend_threshold: float = Field(25.0, gt=0)
    atr_vol_percentile: float = Field(
        70.0, ge=0, le=100, description="ATR percentile above which vol is 'elevated'"
    )
    trade_only_range: bool = Field(
        True, description="Mean-reversion only trades when regime == RANGE"
    )


class RiskLimits(BaseModel):
    """Prop-firm style hard risk constraints."""

    starting_equity: float = Field(100_000.0, gt=0)
    risk_per_trade_pct: float = Field(0.5, gt=0, le=5.0, description="% equity risked per trade")
    atr_stop_mult: float = Field(1.5, gt=0, description="Stop distance = ATR * mult")
    atr_period: int = Field(14, ge=2)
    target_vol_annual: float = Field(
        0.15, gt=0, description="Volatility targeting annualized vol"
    )
    vol_lookback: int = Field(20, ge=5)
    max_position_lots: float = Field(10.0, gt=0)
    min_position_lots: float = Field(0.01, gt=0)
    daily_kill_switch_pct: float = Field(
        -1.5, lt=0, description="Halt trading if daily PnL % <= this (prop max DD)"
    )
    max_drawdown_pct: float = Field(
        -5.0, lt=0, description="Halt if peak-to-trough equity DD % <= this"
    )
    soft_drawdown_pct: float = Field(
        -3.0, lt=0, description="Reduce size when DD breaches soft guardrail"
    )
    soft_size_multiplier: float = Field(0.5, gt=0, le=1.0)

    @model_validator(mode="after")
    def _dd_ordering(self) -> RiskLimits:
        if self.soft_drawdown_pct <= self.max_drawdown_pct:
            raise ValueError("soft_drawdown_pct must be greater than max_drawdown_pct (e.g. -3 > -5)")
        if self.daily_kill_switch_pct >= 0:
            raise ValueError("daily_kill_switch_pct must be negative")
        return self


class WalkForwardConfig(BaseModel):
    """Walk-forward optimization split configuration."""

    is_ratio: float = Field(0.70, gt=0, lt=1.0, description="In-sample fraction")
    n_folds: int = Field(3, ge=1)
    purge_bars: int = Field(0, ge=0, description="Embargo/purge between IS and OOS")
    optimize_metric: Literal["sharpe", "sortino", "profit_factor", "expectancy"] = "sharpe"
    param_grid: dict[str, list[float | int]] = Field(
        default_factory=lambda: {
            "lookback": [15, 20, 30],
            "z_entry": [1.5, 2.0, 2.5],
            "z_exit": [0.15, 0.25, 0.5],
        }
    )


class BacktestConfig(BaseModel):
    """Top-level backtest run configuration."""

    timeframe: Timeframe = Timeframe.M5
    bars_per_day: int = Field(78, ge=1, description="5m RTH ~78 bars; 1m ~390")
    annualization_factor: int | None = Field(
        None, description="Override bars/year for Sharpe; else inferred"
    )
    data_path: Path | None = None
    output_dir: Path = Path("intraday_qrf/output")
    save_equity_chart: bool = True
    equity_chart_name: str = "equity_curve.png"

    @field_validator("data_path", "output_dir", mode="before")
    @classmethod
    def _to_path(cls, v: object) -> Path | None:
        if v is None:
            return None
        return Path(v)


class FrameworkConfig(BaseModel):
    """Root configuration aggregating all subsystems."""

    asset: AssetSpec = Field(
        default_factory=lambda: AssetSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURES,
            tick_size=0.25,
            tick_value=12.50,
            point_value=50.0,
            lot_size=1.0,
        )
    )
    strategy: StrategyParams = Field(default_factory=StrategyParams)
    regime: RegimeParams = Field(default_factory=RegimeParams)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    friction: FrictionConfig = Field(default_factory=FrictionConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    def bars_per_year(self) -> int:
        if self.backtest.annualization_factor is not None:
            return self.backtest.annualization_factor
        # ~252 RTH sessions
        return 252 * self.backtest.bars_per_day


def default_config() -> FrameworkConfig:
    """Factory for a production-ready default ES 5-minute prop-firm setup."""
    return FrameworkConfig()
