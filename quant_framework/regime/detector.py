"""Higher-timeframe regime detection with explicit availability timestamps."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.models import RegimeLabel, Timeframe
from config.strategy_config import RegimeParams
from data.resampler import resample_ohlcv
from engine.events import require_aware
from engine.vectorized_features import atr as compute_atr

logger = logging.getLogger(__name__)

TF_RULE = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.H1: "1h",
    Timeframe.D1: "1D",
}


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = compute_atr(df, 1).replace(0, np.nan)
    atr_w = compute_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_w
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period, adjust=False).mean().rename(f"adx_{period}")


class RegimeDetector:
    """
    Vectorized HTF regime labels mapped back to LTF bars.

    Regime known at HTF bar close is only available to LTF decisions at/after
    that timestamp — enforced via ``availability_timestamp`` column.
    """

    def __init__(self, params: RegimeParams) -> None:
        self.params = params

    def compute_htf_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Build HTF indicator frame indexed by HTF bar open."""
        rule = TF_RULE[self.params.htf]
        if "timestamp" in bars.columns:
            work = bars.set_index("timestamp")
        else:
            work = bars.copy()
        if work.index.tz is None:
            raise ValueError("RegimeDetector requires timezone-aware bars")

        htf = resample_ohlcv(work.reset_index(), rule, validate=False).set_index("timestamp")
        delta = pd.tseries.frequencies.to_offset(rule)
        htf["bar_end"] = htf.index + delta
        htf["atr"] = compute_atr(htf, self.params.atr_period)
        htf["adx"] = _adx(htf, self.params.adx_period)
        htf["log_ret"] = np.log(htf["close"] / htf["close"].shift(1))
        htf["realized_vol"] = htf["log_ret"].rolling(self.params.vol_lookback, min_periods=self.params.vol_lookback).std(
            ddof=0
        )
        htf["atr_pct"] = htf["atr"].rolling(self.params.vol_lookback, min_periods=self.params.vol_lookback).apply(
            lambda s: float(pd.Series(s).rank(pct=True).iloc[-1]) * 100 if len(s) else np.nan,
            raw=False,
        )
        ma = htf["close"].rolling(self.params.ma_slope_lookback, min_periods=self.params.ma_slope_lookback).mean()
        htf["ma_slope"] = (ma - ma.shift(1)) / htf["close"].replace(0, np.nan)
        tr = htf["high"] - htf["low"]
        htf["range_compression"] = tr / tr.rolling(self.params.vol_lookback, min_periods=self.params.vol_lookback).mean()
        htf["availability_timestamp"] = htf["bar_end"]
        htf["source_timestamp"] = htf.index
        return htf

    def classify_row(self, row: pd.Series) -> RegimeLabel:
        adx = float(row.get("adx", np.nan))
        atr_pct = float(row.get("atr_pct", np.nan))
        slope = float(row.get("ma_slope", np.nan))
        compression = float(row.get("range_compression", np.nan))

        if not np.isfinite(adx):
            return RegimeLabel.UNKNOWN

        if np.isfinite(atr_pct):
            if atr_pct >= self.params.atr_vol_percentile:
                return RegimeLabel.HIGH_VOLATILITY
            if atr_pct <= (100 - self.params.atr_vol_percentile):
                return RegimeLabel.LOW_VOLATILITY

        if adx >= self.params.adx_trend_threshold:
            if slope > 0:
                return RegimeLabel.TREND_UP
            if slope < 0:
                return RegimeLabel.TREND_DOWN

        if np.isfinite(compression) and compression <= self.params.range_compression_threshold:
            return RegimeLabel.RANGE

        if adx < self.params.adx_trend_threshold * 0.75:
            return RegimeLabel.RANGE

        return RegimeLabel.UNKNOWN

    def map_to_ltf(self, ltf: pd.DataFrame, htf_features: pd.DataFrame) -> pd.DataFrame:
        """
        Map HTF regime to each LTF bar using merge_asof on availability_timestamp.

        An LTF bar may only see HTF regime whose availability_timestamp <= LTF decision time.
        """
        if "timestamp" in ltf.columns:
            ltf_work = ltf.copy()
        else:
            ltf_work = ltf.reset_index()
        if "timestamp" not in ltf_work.columns:
            ltf_work = ltf_work.rename(columns={ltf_work.columns[0]: "timestamp"})

        ltf_work = ltf_work.sort_values("timestamp")
        if "bar_end" not in ltf_work.columns:
            delta = ltf_work["timestamp"].diff().median()
            ltf_work["bar_end"] = ltf_work["timestamp"] + delta
        ltf_work["decision_timestamp"] = ltf_work["bar_end"]

        htf_reg = htf_features.copy().reset_index()
        if "timestamp" not in htf_reg.columns:
            htf_reg = htf_reg.rename(columns={htf_reg.columns[0]: "timestamp"})
        htf_reg["regime"] = htf_reg.apply(self.classify_row, axis=1)
        htf_reg["regime"] = htf_reg["regime"].apply(lambda r: r.value if hasattr(r, "value") else str(r))

        merged = pd.merge_asof(
            ltf_work.sort_values("decision_timestamp"),
            htf_reg[["availability_timestamp", "regime", "adx", "atr_pct", "ma_slope"]].rename(
                columns={"availability_timestamp": "regime_available_at"}
            ).sort_values("regime_available_at"),
            left_on="decision_timestamp",
            right_on="regime_available_at",
            direction="backward",
        )
        merged["regime_availability_timestamp"] = merged["regime_available_at"]
        return merged.set_index("timestamp")

    def detect(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: HTF features + LTF-aligned regime column."""
        htf = self.compute_htf_features(bars)
        return self.map_to_ltf(bars, htf)

    def is_tradeable(self, regime: str | RegimeLabel) -> bool:
        label = RegimeLabel(regime) if isinstance(regime, str) else regime
        if not self.params.trade_only_range:
            return label not in {RegimeLabel.UNKNOWN, RegimeLabel.HIGH_VOLATILITY}
        return label == RegimeLabel.RANGE
