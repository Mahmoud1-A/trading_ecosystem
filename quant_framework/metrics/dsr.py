"""
Deflated Sharpe Ratio over the full registry trial population (Phase 5.5).

The registry path (:func:`compute_deflated_sharpe`) never reports a misleading
``0.0`` when the inputs are inadequate: it returns
:class:`DSRStatus.INSUFFICIENT_DATA` with the reason, the sample size, and the
assumptions used. It requires the *complete* trial population — rejected and
failed trials included — and rejects Top-N shortlists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from metrics.trial_population import (
    IndependenceEstimate,
    TopNOnlyInputError,
    TrialPopulation,
    TrialSelection,
    build_trial_population,
    estimate_effective_trials,
)

CALCULATION_VERSION = "dsr_v2"

# Bailey & Lopez de Prado require enough trials and enough return observations
# before the expected-maximum-Sharpe correction is meaningful.
MIN_TRIALS = 2
MIN_OBSERVATIONS = 20

_EULER_MASCHERONI = 0.5772156649015329


class DSRStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe result with full provenance for the registry."""

    status: DSRStatus
    deflated_sharpe: float | None
    observed_sharpe: float | None
    expected_max_sharpe: float | None
    total_trials: int
    scored_trials: int
    rejected_trials: int
    failed_trials: int
    effective_independent_trials: float
    trial_sharpe_variance: float | None
    mean_correlation: float | None
    n_clusters: int | None
    n_observations: int
    sample_size: int
    skew: float | None
    kurtosis: float | None
    assumptions: str
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""
    trial_selection: str = TrialSelection.ALL_TRIALS.value
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "deflated_sharpe": self.deflated_sharpe,
            "observed_sharpe": self.observed_sharpe,
            "expected_max_sharpe": self.expected_max_sharpe,
            "total_trials": self.total_trials,
            "scored_trials": self.scored_trials,
            "rejected_trials": self.rejected_trials,
            "failed_trials": self.failed_trials,
            "effective_independent_trials": self.effective_independent_trials,
            "trial_sharpe_variance": self.trial_sharpe_variance,
            "mean_correlation": self.mean_correlation,
            "n_clusters": self.n_clusters,
            "n_observations": self.n_observations,
            "sample_size": self.sample_size,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
            "trial_selection": self.trial_selection,
            "meta": dict(self.meta),
        }


def _normal_cdf(z: float) -> float:
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _normal_ppf(p: float) -> float:
    """Acklam's rational approximation to the standard normal quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def expected_max_sharpe(n_trials: float, trial_sharpe_std: float) -> float:
    """
    Expected maximum Sharpe from ``n_trials`` independent draws.

    Uses the Bailey & Lopez de Prado Gumbel approximation
    ``E[max] = sigma * ((1-gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N*e)))``.
    """
    n = float(n_trials)
    if n <= 1 or trial_sharpe_std <= 0:
        return 0.0
    z1 = _normal_ppf(1.0 - 1.0 / n)
    z2 = _normal_ppf(1.0 - 1.0 / (n * math.e))
    return float(trial_sharpe_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def _insufficient(
    reason: str,
    *,
    population: TrialPopulation,
    observed_sharpe: float | None,
    n_observations: int,
    assumptions: str,
    trial_variance: float | None = None,
    independence: IndependenceEstimate | None = None,
    status: DSRStatus = DSRStatus.INSUFFICIENT_DATA,
) -> DSRResult:
    return DSRResult(
        status=status,
        deflated_sharpe=None,
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=None,
        total_trials=population.total_trials,
        scored_trials=population.scored_trials,
        rejected_trials=population.rejected_trials,
        failed_trials=population.failed_trials,
        effective_independent_trials=(
            independence.effective_independent_trials
            if independence is not None
            else float(population.total_trials)
        ),
        trial_sharpe_variance=trial_variance,
        mean_correlation=independence.mean_correlation if independence else None,
        n_clusters=independence.n_clusters if independence else None,
        n_observations=int(n_observations),
        sample_size=population.scored_trials,
        skew=None,
        kurtosis=None,
        assumptions=assumptions,
        reason=reason,
        trial_selection=population.selection.value,
    )


def compute_deflated_sharpe(
    observed_sharpe: float,
    population: TrialPopulation,
    *,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    min_trials: int = MIN_TRIALS,
    min_observations: int = MIN_OBSERVATIONS,
    correlation_threshold: float = 0.95,
) -> DSRResult:
    """
    Deflated Sharpe over the complete trial population.

    ``observed_sharpe`` and the trial scores must be on the same (per-period)
    scale. Failed trials inflate ``total_trials`` and therefore the expected
    maximum Sharpe, which lowers the deflated Sharpe — that is the point of
    recording them.

    Raises :class:`TopNOnlyInputError` when the population is a Top-N shortlist.
    """
    population.require_all_trials()
    assumptions = (
        "Bailey & Lopez de Prado deflated Sharpe; expected maximum Sharpe from the "
        "Gumbel approximation over effective independent trials; trial Sharpe variance "
        "estimated from all scored trials including rejected ones; failed trials counted "
        f"in total_trials; non-normality adjusted via skew/kurtosis; min_trials={min_trials}; "
        f"min_observations={min_observations}; version={CALCULATION_VERSION}"
    )

    scores = np.asarray(population.scores, dtype=float)
    independence = estimate_effective_trials(
        population.total_trials,
        correlation_matrix=population.correlation_matrix,
        correlation_threshold=correlation_threshold,
    )

    if population.total_trials < min_trials or len(scores) < min_trials:
        return _insufficient(
            f"need >= {min_trials} trials, got total_trials={population.total_trials} "
            f"scored={len(scores)}",
            population=population,
            observed_sharpe=float(observed_sharpe),
            n_observations=n_observations,
            assumptions=assumptions,
            independence=independence,
        )
    if int(n_observations) < min_observations:
        return _insufficient(
            f"need >= {min_observations} return observations, got {int(n_observations)}",
            population=population,
            observed_sharpe=float(observed_sharpe),
            n_observations=n_observations,
            assumptions=assumptions,
            independence=independence,
        )

    trial_variance = float(np.var(scores, ddof=1)) if len(scores) > 1 else 0.0
    if not np.isfinite(trial_variance) or trial_variance <= 0:
        return _insufficient(
            "trial Sharpe variance is zero or undefined; the trial population carries "
            "no information about search intensity",
            population=population,
            observed_sharpe=float(observed_sharpe),
            n_observations=n_observations,
            assumptions=assumptions,
            trial_variance=trial_variance,
            independence=independence,
        )

    trial_std = math.sqrt(trial_variance)
    e_max = expected_max_sharpe(independence.effective_independent_trials, trial_std)

    n = int(n_observations)
    sr = float(observed_sharpe)
    denom = 1.0 - float(skew) * sr + ((float(kurtosis) - 1.0) / 4.0) * sr * sr
    if denom <= 0 or n < 2:
        return _insufficient(
            "Sharpe standard error is undefined for the supplied skew/kurtosis",
            population=population,
            observed_sharpe=sr,
            n_observations=n,
            assumptions=assumptions,
            trial_variance=trial_variance,
            independence=independence,
            status=DSRStatus.INVALID_INPUT,
        )
    se = math.sqrt(denom / (n - 1))
    dsr = _normal_cdf((sr - e_max) / se) if se > 0 else 0.0

    return DSRResult(
        status=DSRStatus.OK,
        deflated_sharpe=float(dsr),
        observed_sharpe=sr,
        expected_max_sharpe=float(e_max),
        total_trials=population.total_trials,
        scored_trials=population.scored_trials,
        rejected_trials=population.rejected_trials,
        failed_trials=population.failed_trials,
        effective_independent_trials=independence.effective_independent_trials,
        trial_sharpe_variance=trial_variance,
        mean_correlation=independence.mean_correlation,
        n_clusters=independence.n_clusters,
        n_observations=n,
        sample_size=population.scored_trials,
        skew=float(skew),
        kurtosis=float(kurtosis),
        assumptions=assumptions,
        trial_selection=population.selection.value,
        meta={"independence": independence.as_dict()},
    )


def deflated_sharpe_from_scores(
    observed_sharpe: float,
    scores: Sequence[float],
    *,
    n_observations: int,
    total_trials: int | None = None,
    rejected_trials: int = 0,
    failed_trials: int = 0,
    selection: TrialSelection | str = TrialSelection.ALL_TRIALS,
    correlation_matrix: Sequence[Sequence[float]] | None = None,
    **kwargs: Any,
) -> DSRResult:
    """Convenience wrapper building the population from raw trial scores."""
    population = build_trial_population(
        scores,
        total_trials=total_trials,
        rejected_trials=rejected_trials,
        failed_trials=failed_trials,
        selection=selection,
        correlation_matrix=correlation_matrix,
    )
    return compute_deflated_sharpe(
        observed_sharpe, population, n_observations=n_observations, **kwargs
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    trial_sharpe_variance: float,
    n_observations: int,
) -> float:
    """
    Legacy scalar entry point kept for backwards compatibility.

    Prefer :func:`compute_deflated_sharpe`, which reports INSUFFICIENT_DATA instead
    of collapsing an unusable input to ``0.0``.
    """
    if n_trials < 2 or n_observations < 2 or trial_sharpe_variance <= 0:
        return 0.0
    std_sr = math.sqrt(trial_sharpe_variance)
    e_max = expected_max_sharpe(n_trials, std_sr)
    z = (observed_sharpe - e_max) / std_sr if std_sr > 0 else 0.0
    return _normal_cdf(z)


__all__ = [
    "CALCULATION_VERSION",
    "DSRResult",
    "DSRStatus",
    "MIN_OBSERVATIONS",
    "MIN_TRIALS",
    "TopNOnlyInputError",
    "compute_deflated_sharpe",
    "deflated_sharpe_from_scores",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
]
