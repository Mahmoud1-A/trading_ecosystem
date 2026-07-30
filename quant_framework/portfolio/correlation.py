"""Correlation and overlap utilities for portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from portfolio.constraints import MemberMeta


@dataclass(frozen=True)
class CorrelationReport:
    pairwise: dict[tuple[str, str], float]
    mean_correlation: float
    max_correlation: float

    def as_dict(self) -> dict[str, object]:
        return {
            "pairwise": {f"{a}|{b}": v for (a, b), v in self.pairwise.items()},
            "mean_correlation": self.mean_correlation,
            "max_correlation": self.max_correlation,
        }


def pnl_correlation(series_a: Sequence[float], series_b: Sequence[float]) -> float:
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_correlation_matrix(
    pnl_by_id: Mapping[str, Sequence[float]],
) -> dict[tuple[str, str], float]:
    ids = list(pnl_by_id.keys())
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            out[(a, b)] = pnl_correlation(pnl_by_id[a], pnl_by_id[b])
    return out


def correlation_penalty(
    weights: Mapping[str, float],
    pairwise: Mapping[tuple[str, str], float],
    *,
    threshold: float = 0.5,
) -> float:
    """Penalize co-weighted correlated pairs."""
    penalty = 0.0
    ids = list(weights.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            corr = pairwise.get((a, b), pairwise.get((b, a), 0.0))
            if corr > threshold:
                penalty += corr * weights[a] * weights[b]
    return float(penalty)


def symbol_overlap(a: MemberMeta, b: MemberMeta) -> float:
    sa, sb = set(a.symbols), set(b.symbols)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
