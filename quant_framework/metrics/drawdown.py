"""Drawdown analytics."""

from __future__ import annotations

import pandas as pd


def equity_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak.replace(0, pd.NA) - 1.0).fillna(0.0)


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(equity_drawdown(equity).min() * 100.0)


def drawdown_duration_bars(equity: pd.Series) -> int:
    """Longest consecutive stretch below peak (in bars)."""
    dd = equity_drawdown(equity)
    max_run = 0
    run = 0
    for v in dd:
        if v < -1e-12:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run
