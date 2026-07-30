"""Cross-asset hypothesis features — causal as-of alignment only."""

from __future__ import annotations

import pandas as pd

from data.synchronization.asof_join import asof_join


def lagged_return(secondary_close: pd.Series, lag: int = 1) -> pd.Series:
    return secondary_close.pct_change(lag)


def causal_rolling_correlation(
    primary: pd.Series,
    secondary: pd.Series,
    lookback: int,
) -> pd.Series:
    return primary.pct_change().rolling(lookback, min_periods=lookback).corr(secondary.pct_change())


def align_secondary_asof(
    primary_bars: pd.DataFrame,
    secondary_bars: pd.DataFrame,
    *,
    decision_col: str = "decision_timestamp",
    availability_col: str = "availability_timestamp",
) -> pd.DataFrame:
    """
    Align secondary bars to primary decision times with backward as-of only.

    Future secondary events are never joined.
    """
    left = primary_bars.copy()
    right = secondary_bars.copy()
    if decision_col not in left.columns:
        left = left.reset_index().rename(columns={left.columns[0] if False else "index": decision_col})
        # Prefer explicit timestamps
    if "availability_timestamp" not in right.columns and right.index.name is None:
        right = right.copy()
        right["availability_timestamp"] = right.index
        right = right.reset_index(drop=True)
    if decision_col not in left.columns:
        left = left.copy()
        left[decision_col] = left.index
        left = left.reset_index(drop=True)
    if availability_col not in right.columns:
        raise ValueError("secondary bars require availability_timestamp")
    return asof_join(left, right, left_on=decision_col, right_on=availability_col)


def compute_cross_asset_features(
    primary: pd.DataFrame,
    secondary: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    disabled: list[str] = []
    if secondary is None or secondary.empty:
        return pd.DataFrame(index=primary.index), ["xasset.lagged_return", "xasset.rolling_corr_20"]

    # Align on index timestamps: secondary must only use values available at primary ts
    if "availability_timestamp" in secondary.columns and "availability_timestamp" in primary.columns:
        left = pd.DataFrame(
            {
                "decision_timestamp": primary["availability_timestamp"],
                "p_close": primary["close"].to_numpy(),
            }
        )
        right = pd.DataFrame(
            {
                "availability_timestamp": secondary["availability_timestamp"],
                "s_close": secondary["close"].to_numpy(),
            }
        )
        joined = asof_join(left, right)
        s_close = pd.Series(joined["s_close"].to_numpy(), index=primary.index)
    else:
        # Index-aligned causal: use only secondary values strictly before primary bar end
        s_close = secondary["close"].reindex(primary.index).shift(1)

    out = pd.DataFrame(
        {
            "xasset.lagged_return": lagged_return(s_close, 1),
            "xasset.rolling_corr_20": causal_rolling_correlation(primary["close"], s_close, 20),
        },
        index=primary.index,
    )
    return out, disabled
