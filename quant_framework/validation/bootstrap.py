"""Bootstrap helpers for validation metrics (serial-dependence preserving)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.prop_breach import PropBreachEstimate, estimate_prop_breach_probability


def block_bootstrap_mean(
    series: pd.Series,
    *,
    block_size: int = 5,
    n_simulations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via block bootstrap."""
    obs = series.dropna().astype(float).to_numpy()
    n = len(obs)
    if n < block_size:
        m = float(obs.mean()) if n else 0.0
        return m, m, m
    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _ in range(n_simulations):
        path: list[float] = []
        while len(path) < n:
            start = int(rng.integers(0, max(1, n - block_size + 1)))
            path.extend(obs[start : start + block_size].tolist())
        means.append(float(np.mean(path[:n])))
    arr = np.asarray(means)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def prop_breach_from_daily(
    daily_pnl_pct: pd.Series,
    **kwargs: object,
) -> PropBreachEstimate:
    return estimate_prop_breach_probability(daily_pnl_pct, **kwargs)  # type: ignore[arg-type]
