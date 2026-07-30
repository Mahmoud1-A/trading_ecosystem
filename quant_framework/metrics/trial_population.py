"""
Trial population accounting for multiple-testing adjustments (Phase 5.5).

DSR and PBO are only meaningful against the *complete* trial population. Feeding
them a Top-N shortlist understates the search intensity and inflates both
statistics, so the population must be declared explicitly and any Top-N-only
input is rejected rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

CALCULATION_VERSION = "trial_population_v1"


class TrialSelection(str, Enum):
    """How the supplied trial scores were selected from the search."""

    ALL_TRIALS = "ALL_TRIALS"
    TOP_N = "TOP_N"
    UNKNOWN = "UNKNOWN"


class TopNOnlyInputError(ValueError):
    """Raised when a multiple-testing adjustment is fed a Top-N shortlist."""


@dataclass(frozen=True)
class TrialPopulation:
    """
    Complete trial ledger summary required by DSR / PBO.

    ``scores`` holds the ranking score of every trial that produced one, including
    rejected candidates. ``failed_trials`` counts trials that errored or never
    produced a score — they still consumed a test and must inflate the adjustment.
    """

    scores: tuple[float, ...]
    total_trials: int
    rejected_trials: int = 0
    failed_trials: int = 0
    selection: TrialSelection = TrialSelection.ALL_TRIALS
    correlation_matrix: tuple[tuple[float, ...], ...] | None = None
    calculation_version: str = CALCULATION_VERSION

    def __post_init__(self) -> None:
        if self.total_trials < len(self.scores):
            raise ValueError(
                f"total_trials={self.total_trials} cannot be fewer than "
                f"{len(self.scores)} scored trials"
            )

    @property
    def scored_trials(self) -> int:
        return len(self.scores)

    def require_all_trials(self) -> None:
        """Guard entry point: refuse Top-N shortlists and undeclared populations."""
        if self.selection is TrialSelection.TOP_N:
            raise TopNOnlyInputError(
                "Top-N-only trial input rejected: DSR/PBO require the full trial "
                "population including rejected and failed trials."
            )
        if self.selection is TrialSelection.UNKNOWN:
            raise TopNOnlyInputError(
                "Trial selection is UNKNOWN: declare TrialSelection.ALL_TRIALS after "
                "verifying that rejected and failed trials are included."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_trials": self.total_trials,
            "scored_trials": self.scored_trials,
            "rejected_trials": self.rejected_trials,
            "failed_trials": self.failed_trials,
            "selection": self.selection.value,
            "calculation_version": self.calculation_version,
        }


def build_trial_population(
    scores: Iterable[float],
    *,
    total_trials: int | None = None,
    rejected_trials: int = 0,
    failed_trials: int = 0,
    selection: TrialSelection | str = TrialSelection.ALL_TRIALS,
    correlation_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> TrialPopulation:
    """Construct a :class:`TrialPopulation`, defaulting ``total_trials`` sensibly."""
    vals = tuple(float(s) for s in scores if np.isfinite(float(s)))
    sel = TrialSelection(selection) if not isinstance(selection, TrialSelection) else selection
    total = int(total_trials) if total_trials is not None else len(vals) + int(failed_trials)
    corr = None
    if correlation_matrix is not None:
        arr = np.asarray(correlation_matrix, dtype=float)
        corr = tuple(tuple(float(x) for x in row) for row in arr)
    return TrialPopulation(
        scores=vals,
        total_trials=total,
        rejected_trials=int(rejected_trials),
        failed_trials=int(failed_trials),
        selection=sel,
        correlation_matrix=corr,
    )


@dataclass(frozen=True)
class IndependenceEstimate:
    """Effective number of independent trials given cross-trial correlation."""

    effective_independent_trials: float
    mean_correlation: float | None
    n_clusters: int | None
    method: str
    cluster_labels: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective_independent_trials": self.effective_independent_trials,
            "mean_correlation": self.mean_correlation,
            "n_clusters": self.n_clusters,
            "method": self.method,
            "cluster_labels": list(self.cluster_labels),
        }


def estimate_effective_trials(
    total_trials: int,
    *,
    correlation_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
    correlation_threshold: float = 0.95,
) -> IndependenceEstimate:
    """
    Effective independent trial count.

    With no correlation information every trial is treated as independent (the
    conservative choice for a deflation penalty). With a correlation matrix,
    trials are greedily clustered at ``correlation_threshold`` and the effective
    count is deflated by the average within-population correlation:
    ``n_eff = 1 + (n - 1) * (1 - mean_rho)``, floored at the cluster count.
    """
    n = max(int(total_trials), 0)
    if correlation_matrix is None or n <= 1:
        return IndependenceEstimate(
            effective_independent_trials=float(n),
            mean_correlation=None,
            n_clusters=None,
            method="assume_independent",
        )

    corr = np.asarray(correlation_matrix, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation_matrix must be square")
    m = corr.shape[0]
    off = corr[~np.eye(m, dtype=bool)]
    mean_rho = float(np.clip(np.nanmean(off), -1.0, 1.0)) if off.size else 0.0

    labels = [-1] * m
    cluster = 0
    for i in range(m):
        if labels[i] != -1:
            continue
        labels[i] = cluster
        for j in range(i + 1, m):
            if labels[j] == -1 and abs(float(corr[i, j])) >= correlation_threshold:
                labels[j] = cluster
        cluster += 1

    deflated = 1.0 + (n - 1) * (1.0 - max(0.0, mean_rho))
    n_eff = float(max(float(cluster), min(float(n), deflated)))
    return IndependenceEstimate(
        effective_independent_trials=n_eff,
        mean_correlation=mean_rho,
        n_clusters=int(cluster),
        method=f"correlation_deflated(threshold={correlation_threshold})",
        cluster_labels=tuple(labels),
    )
