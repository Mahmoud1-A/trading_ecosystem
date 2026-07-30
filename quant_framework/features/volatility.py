"""Causal volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev = bars["close"].shift(1)
    return pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev).abs(),
            (bars["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(bars).rolling(period, min_periods=period).mean()


def realized_volatility(close: pd.Series, lookback: int) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(lookback, min_periods=lookback).std(ddof=0)


def parkinson_volatility(bars: pd.DataFrame, lookback: int) -> pd.Series:
    # Parkinson: sqrt(1/(4 ln2) * mean(ln(H/L)^2))
    hl = np.log(bars["high"] / bars["low"].replace(0, np.nan)) ** 2
    const = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(const * hl.rolling(lookback, min_periods=lookback).mean())


def range_compression(bars: pd.DataFrame, lookback: int) -> pd.Series:
    rng = bars["high"] - bars["low"]
    mean_rng = rng.rolling(lookback, min_periods=lookback).mean().replace(0, np.nan)
    return rng / mean_rng


def signed_vol(close: pd.Series, lookback: int, *, side: str) -> pd.Series:
    rets = close.pct_change()
    if side == "down":
        masked = rets.where(rets < 0)
    else:
        masked = rets.where(rets > 0)
    return masked.rolling(lookback, min_periods=max(2, lookback // 2)).std(ddof=0)


def compute_volatility_features(bars: pd.DataFrame) -> pd.DataFrame:
    a = atr(bars, 14)
    return pd.DataFrame(
        {
            "vol.atr_14": a,
            "vol.norm_atr_14": a / bars["close"].replace(0, np.nan),
            "vol.atr_pct_50": a.rolling(50, min_periods=50).apply(
                lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
            ),
            "vol.realized_20": realized_volatility(bars["close"], 20),
            "vol.parkinson_20": parkinson_volatility(bars, 20),
            "vol.range_compression_20": range_compression(bars, 20),
            "vol.downside_20": signed_vol(bars["close"], 20, side="down"),
            "vol.upside_20": signed_vol(bars["close"], 20, side="up"),
            "regime.vol_of_vol_20": a.diff().rolling(20, min_periods=20).std(ddof=0),
        },
        index=bars.index,
    )
