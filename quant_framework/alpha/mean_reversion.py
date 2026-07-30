"""Intraday VWAP / Bollinger / Z-Score mean reversion strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha.protocol import StrategyProtocol, StrategyState
from config.models import RegimeLabel, Timeframe
from config.strategy_config import RegimeParams, StrategyParams
from engine.events import InformationTiming, SignalEvent
from engine.fills import Fill
from engine.portfolio import Portfolio
from engine.vectorized_features import compute_feature_frame, timing_for_row
from regime.detector import RegimeDetector


class MeanReversionStrategy:
    """
    Baseline intraday statistical mean-reversion model.

    All parameters come from ``StrategyParams`` / ``RegimeParams`` — nothing hardcoded.
    """

    def __init__(
        self,
        params: StrategyParams,
        *,
        regime_params: RegimeParams | None = None,
        symbol: str = "ES",
    ) -> None:
        self._params = params
        self._regime_params = regime_params or RegimeParams()
        self._symbol = symbol
        self._regime = RegimeDetector(self._regime_params)

    @property
    def family(self) -> str:
        return self._params.family

    @property
    def params(self) -> StrategyParams:
        return self._params

    def required_timeframes(self) -> list[Timeframe]:
        tfs = [Timeframe.M5]
        if self._params.use_regime_filter:
            tfs.append(self._regime_params.htf)
        return tfs

    def required_features(self) -> list[str]:
        base = ["vwap", "z_vwap", "bb_z", "bb_mid", "atr", "availability_timestamp"]
        if self._params.use_regime_filter:
            base.append("regime")
        return base

    def compute_features(
        self,
        bars: pd.DataFrame,
        *,
        regime_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if "timestamp" in bars.columns:
            work = bars.set_index("timestamp")
        else:
            work = bars.copy()
        if work.index.tz is None:
            raise ValueError("bars must be timezone-aware")

        if "bar_end" not in work.columns:
            delta = work.index.to_series().diff().median()
            work = work.copy()
            work["bar_end"] = work.index + delta

        features = compute_feature_frame(
            work,
            lookback=self._params.lookback,
            bb_std=self._params.bb_std,
            atr_period=self._params.atr_period,
            bar_end_col="bar_end",
        )
        if self._params.use_regime_filter:
            regime = regime_frame if regime_frame is not None else self._regime.detect(work.reset_index())
            if "regime" in regime.columns:
                features = features.join(regime[["regime", "regime_availability_timestamp"]], how="left")
            else:
                features["regime"] = RegimeLabel.UNKNOWN.value
        features["combined_z"] = self._combined_z(features)
        return features

    def _combined_z(self, df: pd.DataFrame) -> pd.Series:
        parts: list[pd.Series] = []
        if self._params.use_vwap_zscore and "z_vwap" in df.columns:
            parts.append(df["z_vwap"])
        if "bb_z" in df.columns:
            parts.append(df["bb_z"])
        if not parts:
            return pd.Series(np.nan, index=df.index, name="combined_z")
        stacked = pd.concat(parts, axis=1)
        return stacked.mean(axis=1).rename("combined_z")

    def _vol_ok(self, row: pd.Series) -> bool:
        atr = float(row.get("atr", np.nan))
        if not np.isfinite(atr):
            return False
        if self._params.vol_filter_min_atr is not None and atr < self._params.vol_filter_min_atr:
            return False
        if self._params.vol_filter_max_atr is not None and atr > self._params.vol_filter_max_atr:
            return False
        return True

    def _regime_ok(self, row: pd.Series) -> bool:
        if not self._params.use_regime_filter:
            return True
        regime = row.get("regime", RegimeLabel.UNKNOWN.value)
        return self._regime.is_tradeable(str(regime))

    def generate_signals(
        self,
        row: pd.Series,
        *,
        bar_index: int,
        portfolio: Portfolio,
        state: StrategyState,
    ) -> list[SignalEvent]:
        timing = timing_for_row(row)
        pos = portfolio.get_position(self._symbol)
        z = float(row.get("combined_z", np.nan))
        atr = float(row.get("atr", np.nan))
        if not np.isfinite(z):
            return []

        signals: list[SignalEvent] = []

        # Exit logic for open position
        if not pos.is_flat:
            state.bars_in_position += 1
            exit_signal = False
            if abs(z) <= self._params.z_exit:
                exit_signal = True
            if state.bars_in_position >= self._params.max_holding_bars:
                exit_signal = True
            if exit_signal:
                signals.append(
                    SignalEvent(
                        timing=timing,
                        symbol=self._symbol,
                        side="FLAT",
                        quantity=abs(pos.quantity),
                        reason="mean_reversion_exit",
                    )
                )
                return signals
            return signals

        # Entry logic — flat book
        if not self._vol_ok(row) or not self._regime_ok(row):
            return []

        if z <= -self._params.z_entry:
            side = "BUY"
        elif z >= self._params.z_entry and self._params.allow_short:
            side = "SELL"
        else:
            return []

        stop = target = None
        if np.isfinite(atr):
            if side == "BUY":
                stop = float(row["close"]) - self._params.stop_atr_mult * atr
                target = float(row["close"]) + self._params.target_atr_mult * atr
            else:
                stop = float(row["close"]) + self._params.stop_atr_mult * atr
                target = float(row["close"]) - self._params.target_atr_mult * atr

        signals.append(
            SignalEvent(
                timing=timing,
                symbol=self._symbol,
                side=side,
                quantity=1.0,  # sized downstream by PositionSizer
                stop_price=stop,
                target_price=target,
                reason="mean_reversion_entry",
                meta={"combined_z": z, "atr": atr},
            )
        )
        return signals

    def on_fill(self, fill: Fill, state: StrategyState) -> None:
        if fill.side.value == "BUY":
            state.entry_side = "LONG"
        else:
            state.entry_side = "SHORT"
        state.entry_price = fill.price
        state.bars_in_position = 0

    def on_session_end(
        self,
        ts: pd.Timestamp,
        portfolio: Portfolio,
        state: StrategyState,
    ) -> list[SignalEvent]:
        if not self._params.session_flatten:
            return []
        pos = portfolio.get_position(self._symbol)
        if pos.is_flat:
            return []
        timing = InformationTiming(
            source_timestamp=ts,
            availability_timestamp=ts,
            decision_timestamp=ts,
        )
        return [
            SignalEvent(
                timing=timing,
                symbol=self._symbol,
                side="FLAT",
                quantity=abs(pos.quantity),
                reason="session_close",
            )
        ]
