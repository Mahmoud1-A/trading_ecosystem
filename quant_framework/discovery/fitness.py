"""Robust OOS fitness — training metrics never promote candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class FitnessError(ValueError):
    pass


@dataclass(frozen=True)
class FoldOOSMetrics:
    """Out-of-sample metrics for a single validation fold."""

    fold_id: int
    expectancy: float
    sharpe: float
    profit_factor: float
    calmar: float
    max_drawdown: float
    drawdown_duration: float = 0.0
    worst_day: float = 0.0
    turnover: float = 0.0
    prop_breach_prob: float = 0.0
    regime_entropy: float = 1.0
    n_trades: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "expectancy": self.expectancy,
            "sharpe": self.sharpe,
            "profit_factor": self.profit_factor,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "worst_day": self.worst_day,
            "turnover": self.turnover,
            "prop_breach_prob": self.prop_breach_prob,
            "regime_entropy": self.regime_entropy,
            "n_trades": self.n_trades,
        }


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    ranking_source: str
    components: dict[str, float]
    fold_scores: tuple[float, ...]
    rejected: bool = False
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitness": self.fitness,
            "ranking_source": self.ranking_source,
            "components": dict(self.components),
            "fold_scores": list(self.fold_scores),
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


RANKING_SOURCE_OOS = "validation_oos"
RANKING_SOURCE_IS = "training_is"


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(np.median(np.asarray(xs, dtype=float)))


def _trimmed_mean(xs: Sequence[float], trim: float = 0.1) -> float:
    arr = np.sort(np.asarray(xs, dtype=float))
    if len(arr) == 0:
        return 0.0
    k = int(len(arr) * trim)
    if len(arr) - 2 * k <= 0:
        return float(np.mean(arr))
    return float(np.mean(arr[k : len(arr) - k]))


def _lower_confidence_bound(xs: Sequence[float], z: float = 1.0) -> float:
    arr = np.asarray(xs, dtype=float)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    return float(np.mean(arr) - z * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _percentile(xs: Sequence[float], q: float = 25.0) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=float), q))


@dataclass
class RobustFitness:
    """
    Aggregate fold-level OOS evidence into a single ranking score.

    One exceptional fold cannot dominate: medians, trimmed means, and lower
    confidence bounds are preferred over raw means.
    """

    complexity_penalty: float = 0.05
    drawdown_penalty: float = 1.0
    turnover_penalty: float = 0.1
    similarity_penalty: float = 0.5
    multiple_testing_penalty: float = 0.0
    min_folds: int = 1

    def score(
        self,
        folds: Sequence[FoldOOSMetrics],
        *,
        ranking_source: str = RANKING_SOURCE_OOS,
        complexity: float = 0.0,
        behavioral_similarity: float = 0.0,
        cost_sensitivity: float = 0.0,
        parameter_instability: float = 0.0,
        train_metrics: Mapping[str, float] | None = None,
    ) -> FitnessResult:
        if ranking_source != RANKING_SOURCE_OOS:
            raise FitnessError(
                f"ranking_source must be {RANKING_SOURCE_OOS!r}; "
                f"got {ranking_source!r}. Training metrics cannot promote."
            )
        if train_metrics:
            # Explicitly ignore IS metrics for ranking — presence is diagnostic only
            pass
        if len(folds) < self.min_folds:
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components={},
                fold_scores=(),
                rejected=True,
                rejection_reason="insufficient_oos_folds",
            )

        expectancies = [f.expectancy for f in folds]
        sharpes = [f.sharpe for f in folds]
        pfs = [f.profit_factor for f in folds]
        calmars = [f.calmar for f in folds]
        dds = [f.max_drawdown for f in folds]
        turnovers = [f.turnover for f in folds]
        breaches = [f.prop_breach_prob for f in folds]
        regimes = [f.regime_entropy for f in folds]

        fold_scores = tuple(
            0.35 * f.expectancy
            + 0.25 * f.sharpe
            + 0.15 * min(f.profit_factor, 3.0)
            + 0.15 * f.calmar
            - self.drawdown_penalty * abs(f.max_drawdown)
            - 0.5 * f.prop_breach_prob
            for f in folds
        )

        positive = {
            "median_oos_expectancy": _median(expectancies),
            "oos_dsr_proxy": _lower_confidence_bound(sharpes),
            "oos_profit_factor": _median(pfs),
            "oos_calmar": _median(calmars),
            "fold_stability": 1.0 / (1.0 + float(np.std(fold_scores)) if fold_scores else 1.0),
            "regime_breadth": _median(regimes),
            "low_prop_breach": 1.0 - _median(breaches),
            "worst_fold_score": float(min(fold_scores)) if fold_scores else 0.0,
            "pct25_fold_score": _percentile(fold_scores, 25),
            "trimmed_mean_fold": _trimmed_mean(fold_scores),
        }
        negative = {
            "max_drawdown": _median([abs(x) for x in dds]),
            "turnover": _median(turnovers),
            "complexity": complexity,
            "behavioral_similarity": behavioral_similarity,
            "cost_sensitivity": cost_sensitivity,
            "parameter_instability": parameter_instability,
            "multiple_testing_penalty": self.multiple_testing_penalty,
        }

        fitness = (
            0.40 * positive["trimmed_mean_fold"]
            + 0.20 * positive["pct25_fold_score"]
            + 0.15 * positive["worst_fold_score"]
            + 0.10 * positive["fold_stability"]
            + 0.10 * positive["low_prop_breach"]
            + 0.05 * positive["regime_breadth"]
            - self.complexity_penalty * negative["complexity"]
            - self.turnover_penalty * negative["turnover"]
            - self.similarity_penalty * negative["behavioral_similarity"]
            - 0.25 * negative["cost_sensitivity"]
            - 0.25 * negative["parameter_instability"]
            - negative["multiple_testing_penalty"]
        )

        components = {**{f"pos_{k}": v for k, v in positive.items()}, **{f"neg_{k}": v for k, v in negative.items()}}
        return FitnessResult(
            fitness=float(fitness),
            ranking_source=ranking_source,
            components=components,
            fold_scores=fold_scores,
        )


def assert_oos_only_promotion(
    *,
    train_score: float | None,
    oos_fitness: float,
    promote_threshold: float,
) -> bool:
    """Return True only when OOS fitness clears the bar; train_score is ignored."""
    _ = train_score  # never used for promotion
    return oos_fitness >= promote_threshold
