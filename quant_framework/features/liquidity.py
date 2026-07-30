"""Causal VWAP and liquidity features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_session_vwap(bars: pd.DataFrame, *, session_key: pd.Series | None = None) -> pd.Series:
    """
    Cumulative session VWAP through the current bar only.

    Never uses future bars in the same session. Full-session VWAP is therefore
    unavailable before the session completes.
    """
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    if session_key is None:
        # Exchange-local calendar day of the bar index (must be tz-aware)
        session_key = bars.index.tz_convert(bars.index.tz).normalize()
    pv = typical * bars["volume"]
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = bars["volume"].groupby(session_key).cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).rename("causal_session_vwap")


def rolling_vwap(bars: pd.DataFrame, lookback: int) -> pd.Series:
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    pv = (typical * bars["volume"]).rolling(lookback, min_periods=lookback).sum()
    vol = bars["volume"].rolling(lookback, min_periods=lookback).sum().replace(0, np.nan)
    return (pv / vol).rename(f"rolling_vwap_{lookback}")


def volume_percentile(volume: pd.Series, lookback: int) -> pd.Series:
    return volume.rolling(lookback, min_periods=lookback).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
    )


def causal_relative_tod_volume(bars: pd.DataFrame) -> pd.Series:
    """
    volume / expanding mean of same clock-time volume observed so far.

    Uses only past occurrences of the same time-of-day — no full-history fit.
    """
    tod = bars.index.strftime("%H:%M")
    expanding_mean = bars["volume"].groupby(tod).transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return (bars["volume"] / expanding_mean.replace(0, np.nan)).rename("rel_tod_volume")


def compute_liquidity_features(bars: pd.DataFrame) -> pd.DataFrame:
    vwap = causal_session_vwap(bars)
    return pd.DataFrame(
        {
            "liq.session_vwap": vwap,
            "liq.dist_session_vwap": bars["close"] / vwap - 1.0,
            "liq.rolling_vwap_20": rolling_vwap(bars, 20),
            "liq.volume_pct_20": volume_percentile(bars["volume"], 20),
            "liq.rel_tod_volume": causal_relative_tod_volume(bars),
        },
        index=bars.index,
    )
