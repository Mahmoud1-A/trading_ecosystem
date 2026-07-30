"""Deflated Sharpe Ratio and Probability of Backtest Overfitting helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sharpe_ratio(returns: list[float] | np.ndarray, *, periods_per_year: float = 252.0) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_returns: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """
    Bailey & López de Prado style DSR (approximation).
    Returns DSR in [0,1]-ish probability that SR is significant given multiple tests.
    """
    n_trials = max(1, int(n_trials))
    n_returns = max(2, int(n_returns))
    sr = float(observed_sharpe)
    # Expected max SR under null ~ sqrt(2 log n_trials) scaled (rough)
    euler = 0.5772156649
    exp_max = math.sqrt(max(1e-12, 2.0 * math.log(n_trials))) * (
        1.0 - euler / (2.0 * math.log(n_trials) + 1e-12) + euler / (2.0 * (math.log(n_trials) ** 2) + 1e-12)
    )
    # Variance of SR estimator
    sr_var = (
        1.0
        + 0.5 * sr * sr
        - skew * sr
        + ((kurtosis - 3.0) / 4.0) * sr * sr
    ) / max(n_returns - 1, 1)
    sr_std = math.sqrt(max(sr_var, 1e-12))
    # Deflated: Φ((SR - E[max SR])/σ)
    z = (sr - exp_max) / sr_std
    # logistic approx to normal CDF for portability
    dsr = 1.0 / (1.0 + math.exp(-1.702 * z))
    return {
        "sharpe": sr,
        "dsr": float(dsr),
        "expected_max_sharpe_null": float(exp_max),
        "n_trials": n_trials,
        "n_returns": n_returns,
        "z": float(z),
    }


def probability_of_backtest_overfitting(
    trial_scores: list[float],
    *,
    n_slices: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Lightweight CSCV-style PBO proxy.
    Split trials' score vectors is unavailable; we approximate using rank instability:
    randomly partition the list of trial scores into IS/OOS halves repeatedly and
    measure how often the IS-best underperforms the median OOS.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray([float(x) for x in trial_scores if x is not None and np.isfinite(x)], dtype=float)
    n = len(scores)
    if n < 8:
        return {"pbo": None, "n_trials": n, "reason": "insufficient_trials"}
    n_slices = max(4, min(n_slices, n // 2))
    failures = 0
    rounds = 0
    for _ in range(200):
        perm = rng.permutation(n)
        half = n // 2
        if half < 2:
            break
        is_idx = perm[:half]
        oos_idx = perm[half : 2 * half]
        is_best = int(is_idx[np.argmax(scores[is_idx])])
        # Map is_best global index performance on OOS set: use relative rank of that score among OOS
        # Proxy: if the IS-best score itself is below median of OOS scores → count as overfit signal
        oos_med = float(np.median(scores[oos_idx]))
        if float(scores[is_best]) < oos_med:
            failures += 1
        rounds += 1
    pbo = failures / rounds if rounds else None
    return {"pbo": pbo, "n_trials": n, "rounds": rounds, "failures": failures}


def equity_to_returns(equity_curve: list[tuple]) -> list[float]:
    if len(equity_curve) < 2:
        return []
    out: list[float] = []
    prev = float(equity_curve[0][1])
    for _, eq in equity_curve[1:]:
        e = float(eq)
        if prev > 0:
            out.append(e / prev - 1.0)
        prev = e
    return out
