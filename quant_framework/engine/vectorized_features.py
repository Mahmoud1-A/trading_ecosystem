"""
Vectorized feature computation with explicit information availability timestamps.

Does NOT blindly shift(1) everywhere. Each feature row declares when the value
became available (bar close) vs when it may be used for a decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.events import InformationTiming, require_aware


@dataclass(frozen=True)
class FeatureSeries:
    """A named feature column plus its availability rule."""

    name: str
    values: pd.Series
    # Features computed from bar t OHLC become available at bar_end(t).
    available_at: pd.Series  # timestamps aligned to values.index


def session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = df.index.normalize()
    cum_pv = (typical * df["volume"]).groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).rename("vwap")


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    mu = series.rolling(lookback, min_periods=lookback).mean()
    sd = series.rolling(lookback, min_periods=lookback).std(ddof=0)
    return ((series - mu) / sd.replace(0, np.nan)).rename(f"z_{lookback}")


def bollinger_z(close: pd.Series, lookback: int, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + n_std * sd,
            "bb_lower": mid - n_std * sd,
            "bb_z": (close - mid) / sd.replace(0, np.nan),
        },
        index=close.index,
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean().rename("atr")


def compute_feature_frame(
    bars: pd.DataFrame,
    *,
    lookback: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    bar_end_col: str | None = "bar_end",
) -> pd.DataFrame:
    """
    Vectorized features indexed by bar open timestamp.

    Adds:
    - feature columns (vwap, z_vwap, bb_*, atr, ...)
    - availability_timestamp: when the feature value is known (bar close)
    - decision_eligible: True on this row for decisions made at availability_timestamp
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by timezone-aware bar open timestamps")
    if bars.index.tz is None:
        raise ValueError("bars index must be timezone-aware")

    out = bars.copy()
    if bar_end_col and bar_end_col in out.columns:
        availability = pd.to_datetime(out[bar_end_col], utc=False)
    else:
        # Infer bar end from median delta
        if len(out.index) >= 2:
            delta = out.index.to_series().diff().median()
            availability = out.index + delta
        else:
            availability = out.index

    availability = pd.DatetimeIndex(availability)
    if availability.tz is None:
        raise ValueError("availability timestamps must be timezone-aware")

    out["vwap"] = session_vwap(out)
    out["z_vwap"] = rolling_zscore(out["close"] - out["vwap"], lookback)
    bb = bollinger_z(out["close"], lookback, bb_std)
    out = out.join(bb)
    out["atr"] = atr(out, atr_period)
    out["availability_timestamp"] = availability
    out["source_timestamp"] = out.index
    # Decision at close of bar t uses features with availability == bar_end(t)
    out["decision_timestamp"] = out["availability_timestamp"]
    return out


def timing_for_row(row: pd.Series, *, latency_submission: pd.Timedelta | None = None) -> InformationTiming:
    """Build InformationTiming for a feature/signal row (pre-submission)."""
    source = require_aware(row["source_timestamp"], name="source_timestamp")
    avail = require_aware(row["availability_timestamp"], name="availability_timestamp")
    decision = require_aware(row["decision_timestamp"], name="decision_timestamp")
    timing = InformationTiming(
        source_timestamp=source,
        availability_timestamp=avail,
        decision_timestamp=decision,
    )
    if latency_submission is not None:
        timing = timing.with_submission(decision, latency=latency_submission.to_pytimedelta())
    return timing


def assert_no_lookahead(feature_value_time: pd.Timestamp, decision_time: pd.Timestamp) -> None:
    """Hard check: feature source/availability must not post-date the decision."""
    fv = require_aware(feature_value_time, name="feature_value_time")
    dt = require_aware(decision_time, name="decision_time")
    if fv > dt:
        raise ValueError(
            f"Look-ahead violation: feature available at {fv} used for decision at {dt}"
        )
