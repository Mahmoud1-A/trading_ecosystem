"""Causal price and return features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_return(close: pd.Series, lag: int = 1) -> pd.Series:
    return close.pct_change(lag)


def log_return(close: pd.Series, lag: int = 1) -> pd.Series:
    return np.log(close / close.shift(lag))


def close_to_open(open_: pd.Series, close: pd.Series) -> pd.Series:
    return open_ / close.shift(1) - 1.0


def open_to_close(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close / open_ - 1.0


def hl_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low


def normalized_body(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    return (close - open_) / rng


def upper_wick_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    return (high - np.maximum(open_, close)) / rng


def lower_wick_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    return (np.minimum(open_, close) - low) / rng


def distance_to_rolling_mean(close: pd.Series, lookback: int) -> pd.Series:
    mu = close.rolling(lookback, min_periods=lookback).mean()
    return close / mu - 1.0


def rolling_rank(close: pd.Series, lookback: int) -> pd.Series:
    return close.rolling(lookback, min_periods=lookback).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
    )


def rolling_zscore(close: pd.Series, lookback: int) -> pd.Series:
    mu = close.rolling(lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std(ddof=0).replace(0, np.nan)
    return (close - mu) / sd


def breakout_distance(close: pd.Series, lookback: int) -> pd.Series:
    # Causal: rolling max of prior bars only (exclude current via shift)
    prior_max = close.shift(1).rolling(lookback, min_periods=lookback).max()
    return close / prior_max - 1.0


def breakdown_distance(close: pd.Series, lookback: int) -> pd.Series:
    # Causal downside symmetric: rolling min of prior bars only.
    prior_min = close.shift(1).rolling(lookback, min_periods=lookback).min()
    return close / prior_min - 1.0


def gap_size(open_: pd.Series, close: pd.Series) -> pd.Series:
    return open_ - close.shift(1)


def compute_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    return pd.DataFrame(
        {
            "price.simple_return_1": simple_return(c, 1),
            "price.log_return_1": log_return(c, 1),
            "price.return_5": simple_return(c, 5),
            "price.close_to_open": close_to_open(o, c),
            "price.open_to_close": open_to_close(o, c),
            "price.hl_range": hl_range(h, l),
            "price.norm_body": normalized_body(o, h, l, c),
            "price.upper_wick": upper_wick_ratio(o, h, l, c),
            "price.lower_wick": lower_wick_ratio(o, h, l, c),
            "price.dist_rolling_mean_20": distance_to_rolling_mean(c, 20),
            "price.rolling_rank_20": rolling_rank(c, 20),
            "price.rolling_z_20": rolling_zscore(c, 20),
            "price.breakout_distance_20": breakout_distance(c, 20),
            "price.breakdown_distance_20": breakdown_distance(c, 20),
            "price.gap_size": gap_size(o, c),
        },
        index=bars.index,
    )
