"""Prop breach probability via block bootstrap (Phase 4 surface)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropBreachEstimate:
    breach_probability: float
    ci_low: float
    ci_high: float
    n_simulations: int
    sample_size: int
    assumptions: str

    def as_dict(self) -> dict[str, object]:
        return {
            "breach_probability": self.breach_probability,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_simulations": self.n_simulations,
            "sample_size": self.sample_size,
            "assumptions": self.assumptions,
        }


def estimate_prop_breach_probability(
    daily_pnl_pct: pd.Series,
    *,
    hard_daily_limit: float = -0.015,
    n_simulations: int = 2000,
    block_size: int = 5,
    seed: int = 42,
) -> PropBreachEstimate:
    """
    Block-bootstrap daily PnL to estimate probability of breaching prop daily limit.

    Never reports zero future breach probability from historical non-breach alone.
    """
    obs = daily_pnl_pct.dropna().astype(float).to_numpy()
    n = len(obs)
    if n < block_size:
        return PropBreachEstimate(
            breach_probability=1.0,
            ci_low=0.5,
            ci_high=1.0,
            n_simulations=0,
            sample_size=n,
            assumptions="insufficient sample; conservative upper bound returned",
        )

    rng = np.random.default_rng(seed)
    breaches = 0
    sim_days = n
    for _ in range(n_simulations):
        path: list[float] = []
        while len(path) < sim_days:
            start = int(rng.integers(0, max(1, n - block_size + 1)))
            block = obs[start : start + block_size]
            path.extend(block.tolist())
        path = path[:sim_days]
        cum = np.cumsum(path)
        if float(np.min(cum)) <= hard_daily_limit:
            breaches += 1

    p = breaches / n_simulations
    # Wilson-ish interval
    se = np.sqrt(p * (1 - p) / max(n_simulations, 1))
    ci_low = max(0.0, p - 1.96 * se)
    ci_high = min(1.0, p + 1.96 * se)

    return PropBreachEstimate(
        breach_probability=float(p),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_simulations=n_simulations,
        sample_size=n,
        assumptions=(
            f"block_bootstrap block_size={block_size} hard_daily_limit={hard_daily_limit}; "
            "preserves short-range serial dependence"
        ),
    )
