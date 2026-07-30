"""Causal regime features for DSL regime gates.

Numeric contracts (documented for FeatureContract notes + generator):

regime.trend_state ∈ [-1, 1]
    Causal EMA slope normalized by ATR.
    +1 ≈ strong uptrend, -1 ≈ strong downtrend, ~0 ≈ range.
    Uses only past bars (EMA of prior closes via standard causal ewm).

regime.volatility_state ∈ [-1, 1]
    Causal realized-vol vs its rolling median.
    +1 ≈ expansion, -1 ≈ compression, ~0 ≈ normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars["close"].astype(float)
    h = bars["high"].astype(float)
    l = bars["low"].astype(float)
    # Causal ATR proxy (Wilder-like) without lookahead.
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()

    ema_fast = c.ewm(span=10, adjust=False, min_periods=10).mean()
    ema_slow = c.ewm(span=30, adjust=False, min_periods=30).mean()
    # Slope of fast EMA over 5 bars, normalized by ATR → clip to [-1, 1]
    slope = ema_fast - ema_fast.shift(5)
    trend = (slope / (atr + 1e-9)).clip(-3.0, 3.0) / 3.0
    # Blend level separation for stronger trend signal.
    sep = ((ema_fast - ema_slow) / (atr + 1e-9)).clip(-3.0, 3.0) / 3.0
    trend_state = (0.5 * trend + 0.5 * sep).clip(-1.0, 1.0)

    # Realized vol (std of 1-bar returns) vs its 20-bar rolling median.
    rets = c.pct_change()
    realized = rets.rolling(20, min_periods=20).std(ddof=0)
    med = realized.rolling(40, min_periods=20).median()
    vol_raw = ((realized - med) / (med + 1e-9)).clip(-3.0, 3.0) / 3.0
    volatility_state = vol_raw.clip(-1.0, 1.0)

    return pd.DataFrame(
        {
            "regime.trend_state": trend_state.astype(float),
            "regime.volatility_state": volatility_state.astype(float),
        },
        index=bars.index,
    )


__all__ = ["compute_regime_features"]
