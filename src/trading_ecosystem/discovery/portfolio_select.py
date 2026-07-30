from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _daily_returns(equity_curve: list[tuple]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    df = pd.DataFrame(equity_curve, columns=["ts", "equity"]).sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # Collapse to daily last equity
    daily = df.set_index("ts")["equity"].resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    return rets


def correlation(a: list[tuple], b: list[tuple]) -> float:
    ra = _daily_returns(a)
    rb = _daily_returns(b)
    if ra.empty or rb.empty:
        return 0.0
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(joined) < 5:
        return 0.0
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if corr is None or (isinstance(corr, float) and np.isnan(corr)):
        return 0.0
    return float(corr)


def select_portfolio(
    evaluated: list[dict[str, Any]],
    *,
    max_size: int = 6,
    max_correlation: float = 0.65,
) -> list[dict[str, Any]]:
    """Greedy pick highest-score passers with pairwise return correlation cap."""
    passed = [
        e
        for e in evaluated
        if e.get("fitness", {}).get("passed") and e.get("full_equity_curve") is not None
    ]
    passed.sort(key=lambda e: e.get("fitness", {}).get("score", -1e9), reverse=True)
    selected: list[dict[str, Any]] = []
    for cand in passed:
        if len(selected) >= max_size:
            break
        ok = True
        for prev in selected:
            c = correlation(cand["full_equity_curve"], prev["full_equity_curve"])
            if abs(c) > max_correlation:
                ok = False
                break
        if ok:
            selected.append(cand)
    # If correlation filter emptied the book, fall back to top scorers
    if not selected and passed:
        selected = passed[: max(1, min(max_size, len(passed)))]
    return selected


def select_sector_champions(
    sectors: dict[str, list[dict[str, Any]]],
    *,
    rank: int = 1,
    require_passed: bool = True,
) -> list[dict[str, Any]]:
    """
    One book: take the best algorithm at `rank` for every sector/symbol.

    Example: rank=1 → #1 champion of SPY + #1 of QQQ + … all sectors.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for sym in sorted(sectors.keys()):
        rows = sectors.get(sym) or []
        idx = rank - 1
        if idx >= len(rows):
            continue
        row = rows[idx]
        fit = row.get("fitness") or {}
        if require_passed and not fit.get("passed"):
            continue
        cand = row.get("candidate") or {}
        sid = str(cand.get("strategy_id") or "")
        if sid and sid in seen_ids:
            continue
        if sid:
            seen_ids.add(sid)
        selected.append(row)
    return selected
