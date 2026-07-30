"""
Probability of Backtest Overfitting over the full registry trial population (Phase 5.5).

Implements combinatorially symmetric cross-validation (CSCV, Bailey et al.). The
registry path returns :class:`PBOStatus.INSUFFICIENT_DATA` — never a misleading
``0.0`` — when there are too few folds, trials, or observations, and it rejects
Top-N-only trial inputs because a shortlist hides the true search intensity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from metrics.trial_population import (
    TopNOnlyInputError,
    TrialPopulation,
    TrialSelection,
    build_trial_population,
    estimate_effective_trials,
)

CALCULATION_VERSION = "pbo_v2"

MIN_SPLITS = 4
MIN_TRIALS = 2
MIN_OBSERVATIONS_PER_SPLIT = 2


class PBOStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class PBOResult:
    """PBO result with full provenance for the registry."""

    status: PBOStatus
    pbo: float | None
    n_combinations: int
    n_splits: int
    total_trials: int
    scored_trials: int
    rejected_trials: int
    failed_trials: int
    effective_independent_trials: float
    mean_correlation: float | None
    n_clusters: int | None
    n_observations: int
    sample_size: int
    median_logit: float | None
    performance_degradation: float | None
    assumptions: str
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""
    trial_selection: str = TrialSelection.ALL_TRIALS.value
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "pbo": self.pbo,
            "n_combinations": self.n_combinations,
            "n_splits": self.n_splits,
            "total_trials": self.total_trials,
            "scored_trials": self.scored_trials,
            "rejected_trials": self.rejected_trials,
            "failed_trials": self.failed_trials,
            "effective_independent_trials": self.effective_independent_trials,
            "mean_correlation": self.mean_correlation,
            "n_clusters": self.n_clusters,
            "n_observations": self.n_observations,
            "sample_size": self.sample_size,
            "median_logit": self.median_logit,
            "performance_degradation": self.performance_degradation,
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
            "trial_selection": self.trial_selection,
            "meta": dict(self.meta),
        }


def _sharpe(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0, mean / std, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_pbo(
    performance_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    population: TrialPopulation | None = None,
    n_splits: int = 8,
    min_splits: int = MIN_SPLITS,
    min_trials: int = MIN_TRIALS,
    correlation_threshold: float = 0.95,
) -> PBOResult:
    """
    CSCV probability of backtest overfitting.

    ``performance_matrix`` has shape ``(n_observations, n_trials)`` and holds
    per-period returns for every trial in the population — including trials that
    were later rejected. Observations are split into ``n_splits`` contiguous
    blocks; every balanced combination of blocks forms an in-sample set and its
    complement the out-of-sample set. PBO is the fraction of combinations where
    the in-sample winner lands in the bottom half out-of-sample.

    Raises :class:`TopNOnlyInputError` for a Top-N-only population.
    """
    matrix = np.asarray(performance_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("performance_matrix must be 2-D (n_observations, n_trials)")
    n_obs, n_trials = matrix.shape

    if population is None:
        population = build_trial_population(
            _sharpe(matrix).tolist(), selection=TrialSelection.ALL_TRIALS
        )
    population.require_all_trials()

    independence = estimate_effective_trials(
        population.total_trials,
        correlation_matrix=population.correlation_matrix,
        correlation_threshold=correlation_threshold,
    )
    assumptions = (
        "combinatorially symmetric cross-validation (Bailey et al.); contiguous blocks; "
        "per-trial Sharpe as the ranking statistic; PBO = P(logit(OOS rank) <= 0); "
        f"n_splits={n_splits}; min_splits={min_splits}; min_trials={min_trials}; "
        f"min_observations_per_split={MIN_OBSERVATIONS_PER_SPLIT}; "
        f"version={CALCULATION_VERSION}"
    )

    def _bad(reason: str, status: PBOStatus = PBOStatus.INSUFFICIENT_DATA) -> PBOResult:
        return PBOResult(
            status=status,
            pbo=None,
            n_combinations=0,
            n_splits=int(n_splits),
            total_trials=population.total_trials,
            scored_trials=population.scored_trials,
            rejected_trials=population.rejected_trials,
            failed_trials=population.failed_trials,
            effective_independent_trials=independence.effective_independent_trials,
            mean_correlation=independence.mean_correlation,
            n_clusters=independence.n_clusters,
            n_observations=int(n_obs),
            sample_size=int(n_trials),
            median_logit=None,
            performance_degradation=None,
            assumptions=assumptions,
            reason=reason,
            trial_selection=population.selection.value,
        )

    if n_splits % 2 != 0:
        return _bad(f"n_splits must be even for symmetric CV, got {n_splits}", PBOStatus.INVALID_INPUT)
    if n_splits < min_splits:
        return _bad(f"need >= {min_splits} splits, got {n_splits}")
    if n_trials < min_trials:
        return _bad(f"need >= {min_trials} trials, got {n_trials}")
    if n_obs < n_splits * MIN_OBSERVATIONS_PER_SPLIT:
        return _bad(
            f"need >= {n_splits * MIN_OBSERVATIONS_PER_SPLIT} observations for "
            f"{n_splits} splits, got {n_obs}"
        )

    blocks = [b for b in np.array_split(np.arange(n_obs), n_splits) if len(b) > 0]
    if len(blocks) < n_splits:
        return _bad(f"could not form {n_splits} non-empty blocks from {n_obs} observations")

    half = n_splits // 2
    logits: list[float] = []
    degradations: list[float] = []
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = tuple(b for b in range(n_splits) if b not in is_blocks)
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in oos_blocks])
        is_perf = _sharpe(matrix[is_rows, :])
        oos_perf = _sharpe(matrix[oos_rows, :])

        best = int(np.argmax(is_perf))
        # Relative rank of the IS winner among OOS results (1 = worst, N = best)
        order = np.argsort(np.argsort(oos_perf)) + 1
        w = float(order[best]) / float(n_trials + 1)
        w = min(max(w, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(w / (1.0 - w)))
        degradations.append(float(oos_perf[best] - is_perf[best]))

    pbo = float(np.mean([1.0 if lg <= 0.0 else 0.0 for lg in logits]))
    return PBOResult(
        status=PBOStatus.OK,
        pbo=pbo,
        n_combinations=len(logits),
        n_splits=int(n_splits),
        total_trials=population.total_trials,
        scored_trials=population.scored_trials,
        rejected_trials=population.rejected_trials,
        failed_trials=population.failed_trials,
        effective_independent_trials=independence.effective_independent_trials,
        mean_correlation=independence.mean_correlation,
        n_clusters=independence.n_clusters,
        n_observations=int(n_obs),
        sample_size=int(n_trials),
        median_logit=float(np.median(logits)),
        performance_degradation=float(np.mean(degradations)),
        assumptions=assumptions,
        trial_selection=population.selection.value,
        meta={"independence": independence.as_dict()},
    )


def pbo_from_is_oos(
    is_performance: Sequence[float] | np.ndarray,
    oos_performance: Sequence[float] | np.ndarray,
    *,
    population: TrialPopulation | None = None,
) -> PBOResult:
    """
    Single-split PBO from paired IS/OOS trial scores.

    A single split cannot support CSCV, so this always reports
    INSUFFICIENT_DATA for the CSCV estimate while still recording the observed
    rank degradation of the in-sample winner in ``meta``.
    """
    is_arr = np.asarray(is_performance, dtype=float)
    oos_arr = np.asarray(oos_performance, dtype=float)
    if is_arr.shape != oos_arr.shape:
        raise ValueError("IS and OOS arrays must have equal length")
    if population is None:
        population = build_trial_population(
            oos_arr.tolist(), selection=TrialSelection.ALL_TRIALS
        )
    population.require_all_trials()
    independence = estimate_effective_trials(
        population.total_trials, correlation_matrix=population.correlation_matrix
    )

    n = int(len(is_arr))
    meta: dict[str, Any] = {"independence": independence.as_dict()}
    if n >= 2:
        best = int(np.argmax(is_arr))
        order = np.argsort(np.argsort(oos_arr)) + 1
        w = min(max(float(order[best]) / float(n + 1), 1e-9), 1.0 - 1e-9)
        meta["is_winner_oos_rank"] = int(order[best])
        meta["is_winner_logit"] = math.log(w / (1.0 - w))
        meta["is_winner_degradation"] = float(oos_arr[best] - is_arr[best])

    return PBOResult(
        status=PBOStatus.INSUFFICIENT_DATA,
        pbo=None,
        n_combinations=1,
        n_splits=1,
        total_trials=population.total_trials,
        scored_trials=population.scored_trials,
        rejected_trials=population.rejected_trials,
        failed_trials=population.failed_trials,
        effective_independent_trials=independence.effective_independent_trials,
        mean_correlation=independence.mean_correlation,
        n_clusters=independence.n_clusters,
        n_observations=n,
        sample_size=n,
        median_logit=meta.get("is_winner_logit"),
        performance_degradation=meta.get("is_winner_degradation"),
        assumptions=(
            "single IS/OOS split; CSCV requires >= "
            f"{MIN_SPLITS} folds so no PBO estimate is reported; "
            f"version={CALCULATION_VERSION}"
        ),
        reason=f"a single IS/OOS split cannot support CSCV (needs >= {MIN_SPLITS} folds)",
        trial_selection=population.selection.value,
        meta=meta,
    )


def probability_backtest_overfitting(
    is_performance: np.ndarray,
    oos_performance: np.ndarray,
) -> float:
    """
    Legacy scalar entry point kept for backwards compatibility.

    Prefer :func:`compute_pbo`, which reports INSUFFICIENT_DATA and full provenance
    rather than collapsing an unusable input to a single number.
    """
    if len(is_performance) < 2 or len(oos_performance) < 2:
        return 1.0
    if len(is_performance) != len(oos_performance):
        raise ValueError("IS and OOS arrays must have equal length")
    best_is = int(np.argmax(is_performance))
    oos_rank = int(np.argsort(oos_performance)[::-1].tolist().index(best_is))
    n = len(is_performance)
    median_oos = float(np.median(oos_performance))
    if oos_performance[best_is] < median_oos:
        return float(np.mean(oos_performance[best_is] < oos_performance))
    return float(max(0.0, oos_rank / max(n - 1, 1)))


__all__ = [
    "CALCULATION_VERSION",
    "MIN_SPLITS",
    "MIN_TRIALS",
    "PBOResult",
    "PBOStatus",
    "TopNOnlyInputError",
    "compute_pbo",
    "pbo_from_is_oos",
    "probability_backtest_overfitting",
]
