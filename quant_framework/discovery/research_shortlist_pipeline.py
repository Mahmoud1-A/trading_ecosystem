"""Phase 3C: DSR/PBO → Behavioral Clustering → Research Shortlist.

Honest statistics only from completed Full WFO (+ Stress/Robustness provenance).
Never fabricates DSR/PBO; insufficient inputs surface as explicit reject reasons.
Vault / Paper / Live remain blocked after this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from discovery.behavioral_dedup import (
    BehaviorCluster,
    BehaviorSignature,
    BehavioralDeduper,
    behavioral_similarity,
)
from discovery.candidate import StrategyCandidate
from discovery.evaluator import EvaluationRecord
from discovery.fitness import FoldOOSMetrics
from metrics.dsr import DSRResult, DSRStatus, compute_deflated_sharpe
from metrics.pbo import PBOResult, PBOStatus, compute_pbo
from metrics.trial_population import (
    TrialPopulation,
    TrialSelection,
    build_trial_population,
)
from registry.experiment_registry import ExperimentRegistry

# Status / decision constants (sequence-ordered campaign events).
STATISTICS_TESTED = "STATISTICS_TESTED"
STATISTICS_NOT_ENTERED = "STATISTICS_NOT_ENTERED"
DSR_PASSED = "DSR_PASSED"
DSR_FAILED = "DSR_FAILED"
DSR_INSUFFICIENT_DATA = "DSR_INSUFFICIENT_DATA"
PBO_PASSED = "PBO_PASSED"
PBO_FAILED = "PBO_FAILED"
PBO_INSUFFICIENT_DATA = "PBO_INSUFFICIENT_DATA"
STATISTICALLY_PASSED = "STATISTICALLY_PASSED"
STATISTICALLY_REJECTED = "STATISTICALLY_REJECTED"
BEHAVIORALLY_CLUSTERED = "BEHAVIORALLY_CLUSTERED"
CLUSTERING_NOT_ENTERED = "CLUSTERING_NOT_ENTERED"
RESEARCH_SHORTLISTED = "RESEARCH_SHORTLISTED"
SHORTLIST_REJECTED = "SHORTLIST_REJECTED"
SHORTLIST_NOT_ENTERED = "SHORTLIST_NOT_ENTERED"

# Explicit reject / missing reasons (never silent).
STATISTICS_REQUIRES_FULL_WFO = "STATISTICS_REQUIRES_FULL_WFO"
STATISTICS_REQUIRES_STRESS_PASSED = "STATISTICS_REQUIRES_STRESS_PASSED"
STATISTICS_REQUIRES_ROBUSTNESS_PASSED = "STATISTICS_REQUIRES_ROBUSTNESS_PASSED"
STATISTICS_MISSING_OOS_ARTIFACTS = "STATISTICS_MISSING_OOS_ARTIFACTS"
DSR_BELOW_MINIMUM = "DSR_BELOW_MINIMUM"
PBO_ABOVE_MAXIMUM = "PBO_ABOVE_MAXIMUM"
PBO_MATRIX_INSUFFICIENT = "PBO_MATRIX_INSUFFICIENT"
CLUSTER_NO_GATE_PASSER = "CLUSTER_NO_GATE_PASSER"
NOT_CLUSTER_REPRESENTATIVE = "NOT_CLUSTER_REPRESENTATIVE"
GATE_SCORE_QUALIFIED_FAILED = "GATE_SCORE_QUALIFIED_FAILED"
GATE_STRESS_FAILED = "GATE_STRESS_FAILED"
GATE_ROBUSTNESS_FAILED = "GATE_ROBUSTNESS_FAILED"
GATE_DSR_FAILED = "GATE_DSR_FAILED"
GATE_PBO_FAILED = "GATE_PBO_FAILED"
NO_ROBUSTNESS_PASSED_FOR_STATISTICS = "NO_ROBUSTNESS_PASSED_FOR_STATISTICS"
NO_BEHAVIORAL_SIGNATURE = "NO_BEHAVIORAL_SIGNATURE"

PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST = (
    "MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST"
)

RESEARCH_SHORTLISTED_DOES_NOT_MEAN = (
    "finalist",
    "promoted",
    "vault eligible",
    "paper eligible",
    "live eligible",
)


@dataclass(frozen=True)
class ResearchShortlistConfig:
    min_dsr: float = 0.95
    max_pbo: float = 0.50
    pbo_n_splits: int = 4
    behavioral_similarity_threshold: float = 0.85
    min_oos_observations_for_dsr: int = 20
    min_signature_bins: int = 16


@dataclass
class CandidateStatisticsSummary:
    candidate_id: str
    family_id: str
    generation: int
    observed_sharpe: float | None
    n_observations: int
    dsr_status: str
    dsr_value: float | None
    dsr_reason: str
    pbo_status: str
    pbo_value: float | None
    pbo_reason: str
    final_decision: str
    final_reason: str
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    dsr_payload: dict[str, Any] = field(default_factory=dict)
    pbo_payload: dict[str, Any] = field(default_factory=dict)
    trial_population_refs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "generation": self.generation,
            "observed_sharpe": self.observed_sharpe,
            "n_observations": self.n_observations,
            "dsr_status": self.dsr_status,
            "dsr_value": self.dsr_value,
            "dsr_reason": self.dsr_reason,
            "pbo_status": self.pbo_status,
            "pbo_value": self.pbo_value,
            "pbo_reason": self.pbo_reason,
            "final_decision": self.final_decision,
            "final_reason": self.final_reason,
            "artifact_refs": dict(self.artifact_refs),
            "dsr": dict(self.dsr_payload),
            "pbo": dict(self.pbo_payload),
            "trial_population": dict(self.trial_population_refs),
        }


@dataclass
class CampaignStatisticsAccounting:
    candidates_robustness_passed: int = 0
    candidates_statistics_entered: int = 0
    candidates_statistics_passed: int = 0
    candidates_statistics_failed: int = 0
    candidates_statistics_insufficient: int = 0
    candidates_statistics_not_entered: int = 0
    population_total_trials: int = 0
    population_scored_trials: int = 0
    pbo_matrix_shape: tuple[int, int] | None = None
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates_robustness_passed": self.candidates_robustness_passed,
            "candidates_statistics_entered": self.candidates_statistics_entered,
            "candidates_statistics_passed": self.candidates_statistics_passed,
            "candidates_statistics_failed": self.candidates_statistics_failed,
            "candidates_statistics_insufficient": self.candidates_statistics_insufficient,
            "candidates_statistics_not_entered": self.candidates_statistics_not_entered,
            "population_total_trials": self.population_total_trials,
            "population_scored_trials": self.population_scored_trials,
            "pbo_matrix_shape": (
                list(self.pbo_matrix_shape) if self.pbo_matrix_shape is not None else None
            ),
            "stop_reason": self.stop_reason,
        }


@dataclass
class ResearchShortlistEntry:
    candidate_id: str
    family_id: str
    cluster_id: str
    generation: int
    fitness: float | None
    dsr_value: float | None
    pbo_value: float | None
    gates_passed: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "cluster_id": self.cluster_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "dsr_value": self.dsr_value,
            "pbo_value": self.pbo_value,
            "gates_passed": list(self.gates_passed),
            "provenance": dict(self.provenance),
            "vault_eligible": False,
            "paper_eligible": False,
            "live_eligible": False,
            "finalist": False,
        }


@dataclass
class ShortlistRejectRecord:
    candidate_id: str
    family_id: str
    cluster_id: str | None
    reason: str
    missing_gates: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "cluster_id": self.cluster_id,
            "reason": self.reason,
            "missing_gates": list(self.missing_gates),
            "artifact_refs": dict(self.artifact_refs),
        }


@dataclass
class CampaignClusteringAccounting:
    candidates_clustered: int = 0
    cluster_count: int = 0
    shortlist_count: int = 0
    reject_count: int = 0
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates_clustered": self.candidates_clustered,
            "cluster_count": self.cluster_count,
            "shortlist_count": self.shortlist_count,
            "reject_count": self.reject_count,
            "stop_reason": self.stop_reason,
        }


@dataclass
class ResearchShortlistPhaseResult:
    statistics_summaries: list[CandidateStatisticsSummary]
    statistics_accounting: CampaignStatisticsAccounting
    clusters: list[dict[str, Any]]
    signatures: list[dict[str, Any]]
    clustering_accounting: CampaignClusteringAccounting
    research_shortlist: list[ResearchShortlistEntry]
    shortlist_rejects: list[ShortlistRejectRecord]
    population_stats: dict[str, Any]
    reproducible_fingerprint_payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_statistics_summaries": [s.as_dict() for s in self.statistics_summaries],
            "statistics_accounting": self.statistics_accounting.as_dict(),
            "clusters": list(self.clusters),
            "behavioral_signatures": list(self.signatures),
            "clustering_accounting": self.clustering_accounting.as_dict(),
            "research_shortlist": [e.as_dict() for e in self.research_shortlist],
            "shortlist_rejects": [r.as_dict() for r in self.shortlist_rejects],
            "population_stats": dict(self.population_stats),
        }


def _baseline_artifacts(rec: EvaluationRecord) -> dict[str, Any]:
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    raw = meta.get("baseline_wfo_artifacts")
    return dict(raw) if isinstance(raw, dict) else {}


def _full_wfo_completed(rec: EvaluationRecord) -> bool:
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    if meta.get("is_full_wfo_completion") is True:
        return True
    train = rec.train_metrics or {}
    baseline = _baseline_artifacts(rec)
    signal = str(
        baseline.get("signal_source")
        or train.get("signal_source")
        or ""
    )
    is_full = bool(
        baseline.get("is_full_event_wfo", train.get("is_full_event_wfo", False))
    )
    completed = int(
        baseline.get("wfo_completed_folds")
        or train.get("wfo_completed_folds")
        or len(rec.oos_folds)
        or 0
    )
    return signal == "candidate_dsl_trees" and is_full and completed > 0


def extract_oos_return_series(rec: EvaluationRecord) -> list[float]:
    """OOS return observations from completed Full WFO fold / trade artifacts."""
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    returns: list[float] = []
    if isinstance(trades, list) and trades:
        for t in trades:
            if not isinstance(t, dict):
                continue
            for key in ("net_pnl", "realized_pnl", "pnl", "return"):
                if key in t and t[key] is not None:
                    try:
                        returns.append(float(t[key]))
                        break
                    except (TypeError, ValueError):
                        continue
    if len(returns) >= 2:
        return returns
    # Fold-level OOS expectancy is a real WFO artifact — not a fabricated statistic.
    fold_rets = [float(f.expectancy) for f in rec.oos_folds]
    return fold_rets


def extract_trade_timing_vector(
    rec: EvaluationRecord, *, n_bins: int = 16
) -> tuple[float, ...]:
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    counts = [0.0] * n_bins
    if isinstance(trades, list) and trades:
        for i, t in enumerate(trades):
            if not isinstance(t, dict):
                continue
            # Prefer explicit timestamps; else distribute by fold_id / sequence.
            hour = None
            for key in ("exit_time", "entry_time", "timestamp"):
                val = t.get(key)
                if val is None:
                    continue
                try:
                    # ISO / epoch / hour int
                    if isinstance(val, (int, float)):
                        hour = int(val) % 24
                    else:
                        s = str(val)
                        if "T" in s and len(s) >= 13:
                            hour = int(s[11:13])
                        else:
                            hour = abs(hash(s)) % 24
                    break
                except (TypeError, ValueError):
                    continue
            if hour is None:
                fold = int(t.get("fold_id") or 0)
                hour = (fold * 3 + i) % 24
            bin_i = int((hour / 24.0) * n_bins) % n_bins
            counts[bin_i] += 1.0
        total = sum(counts) or 1.0
        return tuple(c / total for c in counts)
    # Fold trade-count timing proxy from OOS folds.
    for i, f in enumerate(rec.oos_folds):
        counts[i % n_bins] += float(max(0, int(f.n_trades)))
    total = sum(counts) or 1.0
    return tuple(c / total for c in counts)


def extract_exposure_vector(rec: EvaluationRecord, *, n_bins: int = 8) -> tuple[float, ...]:
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    if isinstance(trades, list) and trades:
        sizes: list[float] = []
        for t in trades:
            if not isinstance(t, dict):
                continue
            size = None
            for key in ("qty", "quantity", "size", "exposure", "notional"):
                if t.get(key) is not None:
                    try:
                        size = abs(float(t[key]))
                        break
                    except (TypeError, ValueError):
                        continue
            if size is None:
                pnl = t.get("net_pnl") or t.get("realized_pnl") or t.get("pnl")
                try:
                    size = abs(float(pnl)) if pnl is not None else 1.0
                except (TypeError, ValueError):
                    size = 1.0
            sizes.append(size)
        if sizes:
            arr = np.asarray(sizes, dtype=float)
            # Histogram of exposure magnitudes (normalized).
            hist, _ = np.histogram(arr, bins=n_bins)
            total = float(hist.sum()) or 1.0
            return tuple(float(x) / total for x in hist.tolist())
    # Fold turnover / trade-count exposure proxy.
    vec = []
    for f in rec.oos_folds:
        turnover = float(getattr(f, "turnover", 0.0) or 0.0)
        trades_n = float(max(0, int(f.n_trades)))
        vec.append(turnover if turnover > 0 else trades_n)
    while len(vec) < n_bins:
        vec = vec + vec if vec else [0.0]
    total = sum(vec[:n_bins]) or 1.0
    return tuple(float(v) / total for v in vec[:n_bins])


def build_behavioral_signature(
    candidate: StrategyCandidate,
    rec: EvaluationRecord,
    *,
    n_bins: int = 16,
) -> BehaviorSignature | None:
    """Build signature from OOS returns, trade timing, exposure (Full WFO artifacts)."""
    if not _full_wfo_completed(rec):
        return None
    returns = extract_oos_return_series(rec)
    if len(returns) < 1:
        return None
    timing = extract_trade_timing_vector(rec, n_bins=n_bins)
    exposure = extract_exposure_vector(rec, n_bins=max(4, n_bins // 2))
    # Signal vector = OOS returns (padded) + timing + exposure for correlation.
    signal = list(returns)
    while len(signal) < n_bins:
        signal = signal + signal if signal else [0.0]
    signal = signal[:n_bins]
    # daily_pnl slot carries timing∥exposure blend for pairwise correlation.
    pnl = list(timing[: n_bins // 2]) + list(exposure[: n_bins - n_bins // 2])
    while len(pnl) < n_bins:
        pnl.append(0.0)
    fit = rec.fitness.fitness if rec.fitness else float("-inf")
    return BehaviorSignature(
        candidate_id=candidate.candidate_id,
        signal_vector=tuple(float(x) for x in signal),
        daily_pnl=tuple(float(x) for x in pnl[:n_bins]),
        feature_ids=candidate.feature_ids,
        complexity=candidate.complexity_score,
        fitness=float(fit) if fit is not None else float("-inf"),
    )


def _observed_sharpe_from_folds(folds: Sequence[FoldOOSMetrics]) -> float | None:
    if not folds:
        return None
    sharpes = [float(f.sharpe) for f in folds]
    return float(np.median(np.asarray(sharpes, dtype=float)))


def _skew_kurtosis(returns: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 3:
        return 0.0, 3.0
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    if s < 1e-12:
        return 0.0, 3.0
    z = (arr - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # raw kurtosis (normal ≈ 3)
    return skew, kurt


def build_full_wfo_trial_population(
    records: Sequence[EvaluationRecord],
) -> tuple[TrialPopulation, list[str], list[list[float]]]:
    """
    Full-WFO completed trial ledger for DSR/PBO.

    Includes rejected Full WFO trials (search intensity). Never Top-N only.
    Returns (population, ordered_candidate_ids, per-trial return series).
    """
    scores: list[float] = []
    ids: list[str] = []
    series_list: list[list[float]] = []
    rejected = 0
    failed = 0
    for rec in records:
        if not _full_wfo_completed(rec):
            continue
        rets = extract_oos_return_series(rec)
        if len(rets) < 1:
            continue
        sharpe = _observed_sharpe_from_folds(rec.oos_folds)
        if sharpe is None and rec.fitness is not None and not rec.fitness.rejected:
            sharpe = float(rec.fitness.fitness)
        if sharpe is None:
            # Still count toward intensity when WFO completed but no score.
            failed += 1
            continue
        scores.append(float(sharpe))
        ids.append(rec.candidate_id)
        series_list.append([float(x) for x in rets])
        if rec.rejection_reason or (rec.fitness is not None and rec.fitness.rejected):
            rejected += 1
    population = build_trial_population(
        scores,
        total_trials=len(scores) + failed,
        rejected_trials=rejected,
        failed_trials=failed,
        selection=TrialSelection.ALL_TRIALS,
    )
    return population, ids, series_list


def align_performance_matrix(
    series_list: Sequence[Sequence[float]],
) -> np.ndarray | None:
    """Align per-trial OOS return series into (n_obs, n_trials); None if unusable."""
    if len(series_list) < 2:
        return None
    lengths = [len(s) for s in series_list]
    n_obs = int(min(lengths))
    if n_obs < 2:
        return None
    matrix = np.column_stack(
        [np.asarray(s[:n_obs], dtype=float) for s in series_list]
    )
    return matrix


def evaluate_candidate_dsr_pbo(
    *,
    rec: EvaluationRecord,
    population: TrialPopulation,
    performance_matrix: np.ndarray | None,
    trial_column_index: int | None,
    config: ResearchShortlistConfig,
) -> tuple[DSRResult, PBOResult | None, dict[str, Any]]:
    """Real DSR (+ optional PBO) — never fabricates OK on insufficient data."""
    returns = extract_oos_return_series(rec)
    n_obs = len(returns)
    observed = _observed_sharpe_from_folds(rec.oos_folds)
    meta: dict[str, Any] = {
        "n_observations_source": "closed_trades_or_oos_fold_expectancy",
        "n_observations": n_obs,
        "full_wfo_completed": _full_wfo_completed(rec),
    }
    if observed is None:
        dsr = compute_deflated_sharpe(
            0.0,
            population,
            n_observations=n_obs,
            min_observations=config.min_oos_observations_for_dsr,
        )
        # Force insufficient when no observed sharpe artifact.
        if dsr.status == DSRStatus.OK:
            from metrics.dsr import DSRResult as _DR

            dsr = _DR(
                status=DSRStatus.INSUFFICIENT_DATA,
                deflated_sharpe=None,
                observed_sharpe=None,
                expected_max_sharpe=None,
                total_trials=population.total_trials,
                scored_trials=population.scored_trials,
                rejected_trials=population.rejected_trials,
                failed_trials=population.failed_trials,
                effective_independent_trials=float(population.total_trials),
                trial_sharpe_variance=None,
                mean_correlation=None,
                n_clusters=None,
                n_observations=n_obs,
                sample_size=population.scored_trials,
                skew=None,
                kurtosis=None,
                assumptions=dsr.assumptions,
                reason=STATISTICS_MISSING_OOS_ARTIFACTS,
                trial_selection=population.selection.value,
            )
        pbo = None
        return dsr, pbo, meta

    skew, kurt = _skew_kurtosis(returns)
    dsr = compute_deflated_sharpe(
        float(observed),
        population,
        n_observations=n_obs,
        skew=skew,
        kurtosis=kurt,
        min_observations=config.min_oos_observations_for_dsr,
    )
    pbo: PBOResult | None = None
    if performance_matrix is None:
        pbo = PBOResult(
            status=PBOStatus.INSUFFICIENT_DATA,
            pbo=None,
            n_combinations=0,
            n_splits=int(config.pbo_n_splits),
            total_trials=population.total_trials,
            scored_trials=population.scored_trials,
            rejected_trials=population.rejected_trials,
            failed_trials=population.failed_trials,
            effective_independent_trials=float(population.total_trials),
            mean_correlation=None,
            n_clusters=None,
            n_observations=n_obs,
            sample_size=population.scored_trials,
            median_logit=None,
            performance_degradation=None,
            assumptions="PBO requires aligned Full WFO performance matrix",
            reason=PBO_MATRIX_INSUFFICIENT,
            trial_selection=population.selection.value,
        )
    else:
        pbo = compute_pbo(
            performance_matrix,
            population=population,
            n_splits=int(config.pbo_n_splits),
        )
        meta["pbo_matrix_shape"] = list(performance_matrix.shape)
        meta["trial_column_index"] = trial_column_index
    return dsr, pbo, meta


def _dsr_gate(dsr: DSRResult, *, min_dsr: float) -> tuple[str, str]:
    if dsr.status == DSRStatus.INSUFFICIENT_DATA:
        return DSR_INSUFFICIENT_DATA, dsr.reason or DSR_INSUFFICIENT_DATA
    if dsr.status != DSRStatus.OK or dsr.deflated_sharpe is None:
        return DSR_FAILED, dsr.reason or dsr.status.value
    if float(dsr.deflated_sharpe) < float(min_dsr):
        return DSR_FAILED, f"{DSR_BELOW_MINIMUM}:dsr={dsr.deflated_sharpe:.6f}<{min_dsr}"
    return DSR_PASSED, "dsr_above_minimum"


def _pbo_gate(pbo: PBOResult | None, *, max_pbo: float) -> tuple[str, str]:
    if pbo is None:
        return PBO_INSUFFICIENT_DATA, PBO_MATRIX_INSUFFICIENT
    if pbo.status == PBOStatus.INSUFFICIENT_DATA:
        return PBO_INSUFFICIENT_DATA, pbo.reason or PBO_INSUFFICIENT_DATA
    if pbo.status != PBOStatus.OK or pbo.pbo is None:
        return PBO_FAILED, pbo.reason or pbo.status.value
    if float(pbo.pbo) > float(max_pbo):
        return PBO_FAILED, f"{PBO_ABOVE_MAXIMUM}:pbo={pbo.pbo:.6f}>{max_pbo}"
    return PBO_PASSED, "pbo_below_maximum"


def hard_gate_check(
    *,
    score_qualified: bool,
    stress_passed: bool,
    robustness_passed: bool,
    dsr_decision: str,
    pbo_decision: str,
) -> tuple[bool, list[str], str | None]:
    """Return (passed, gates_passed, reject_reason)."""
    passed_gates: list[str] = []
    if not score_qualified:
        return False, passed_gates, GATE_SCORE_QUALIFIED_FAILED
    passed_gates.append("SCORE_QUALIFIED")
    if not stress_passed:
        return False, passed_gates, GATE_STRESS_FAILED
    passed_gates.append("STRESS_PASSED")
    if not robustness_passed:
        return False, passed_gates, GATE_ROBUSTNESS_FAILED
    passed_gates.append("ROBUSTNESS_PASSED")
    if dsr_decision != DSR_PASSED:
        return False, passed_gates, GATE_DSR_FAILED if dsr_decision == DSR_FAILED else dsr_decision
    passed_gates.append("DSR_PASSED")
    if pbo_decision != PBO_PASSED:
        return False, passed_gates, GATE_PBO_FAILED if pbo_decision == PBO_FAILED else pbo_decision
    passed_gates.append("PBO_PASSED")
    return True, passed_gates, None


def select_best_per_cluster(
    *,
    clusters: Sequence[BehaviorCluster],
    candidates: dict[str, StrategyCandidate],
    records: dict[str, EvaluationRecord],
    stats_by_id: dict[str, CandidateStatisticsSummary],
    score_qualified_ids: set[str],
    stress_passed_ids: set[str],
    robustness_passed_ids: set[str],
) -> tuple[list[ResearchShortlistEntry], list[ShortlistRejectRecord]]:
    """Best gate-passing candidate per cluster (fitness, then lower complexity)."""
    shortlist: list[ResearchShortlistEntry] = []
    rejects: list[ShortlistRejectRecord] = []

    for cl in clusters:
        eligible: list[tuple[float, float, str]] = []
        cluster_rejects: list[ShortlistRejectRecord] = []
        for mid in cl.member_ids:
            cand = candidates.get(mid)
            rec = records.get(mid)
            st = stats_by_id.get(mid)
            if cand is None or rec is None or st is None:
                cluster_rejects.append(
                    ShortlistRejectRecord(
                        candidate_id=mid,
                        family_id="",
                        cluster_id=cl.cluster_id,
                        reason=STATISTICS_MISSING_OOS_ARTIFACTS,
                        missing_gates=["STATISTICS"],
                    )
                )
                continue
            ok, gates, reason = hard_gate_check(
                score_qualified=mid in score_qualified_ids,
                stress_passed=mid in stress_passed_ids,
                robustness_passed=mid in robustness_passed_ids,
                dsr_decision=(
                    DSR_PASSED
                    if st.dsr_status == DSR_PASSED
                    else (
                        DSR_INSUFFICIENT_DATA
                        if st.dsr_status == DSR_INSUFFICIENT_DATA
                        else DSR_FAILED
                    )
                ),
                pbo_decision=(
                    PBO_PASSED
                    if st.pbo_status == PBO_PASSED
                    else (
                        PBO_INSUFFICIENT_DATA
                        if st.pbo_status == PBO_INSUFFICIENT_DATA
                        else PBO_FAILED
                    )
                ),
            )
            if not ok:
                cluster_rejects.append(
                    ShortlistRejectRecord(
                        candidate_id=mid,
                        family_id=st.family_id,
                        cluster_id=cl.cluster_id,
                        reason=reason or SHORTLIST_REJECTED,
                        missing_gates=[g for g in (
                            "SCORE_QUALIFIED",
                            "STRESS_PASSED",
                            "ROBUSTNESS_PASSED",
                            "DSR_PASSED",
                            "PBO_PASSED",
                        ) if g not in gates],
                        artifact_refs={"statistics": st.as_dict()},
                    )
                )
                continue
            fit = float(rec.fitness.fitness) if rec.fitness else float("-inf")
            eligible.append((fit, -float(cand.complexity_score), mid))

        if not eligible:
            rejects.extend(cluster_rejects)
            rejects.append(
                ShortlistRejectRecord(
                    candidate_id=cl.representative_id,
                    family_id="",
                    cluster_id=cl.cluster_id,
                    reason=CLUSTER_NO_GATE_PASSER,
                    missing_gates=["CLUSTER_GATE_PASSER"],
                    artifact_refs={"member_ids": list(cl.member_ids)},
                )
            )
            continue

        eligible.sort(reverse=True)
        best_id = eligible[0][2]
        st = stats_by_id[best_id]
        rec = records[best_id]
        cand = candidates[best_id]
        _, gates, _ = hard_gate_check(
            score_qualified=True,
            stress_passed=True,
            robustness_passed=True,
            dsr_decision=DSR_PASSED,
            pbo_decision=PBO_PASSED,
        )
        shortlist.append(
            ResearchShortlistEntry(
                candidate_id=best_id,
                family_id=st.family_id,
                cluster_id=cl.cluster_id,
                generation=int(cand.generation),
                fitness=float(rec.fitness.fitness) if rec.fitness else None,
                dsr_value=st.dsr_value,
                pbo_value=st.pbo_value,
                gates_passed=gates,
                provenance={
                    "cluster_member_ids": list(cl.member_ids),
                    "statistics": st.as_dict(),
                    "full_wfo": True,
                    "stress_passed": True,
                    "robustness_passed": True,
                },
            )
        )
        for r in cluster_rejects:
            rejects.append(r)
        for _fit, _neg_c, mid in eligible[1:]:
            rejects.append(
                ShortlistRejectRecord(
                    candidate_id=mid,
                    family_id=stats_by_id[mid].family_id,
                    cluster_id=cl.cluster_id,
                    reason=NOT_CLUSTER_REPRESENTATIVE,
                    missing_gates=[],
                    artifact_refs={"selected_representative": best_id},
                )
            )
    return shortlist, rejects


__all__ = [
    "BEHAVIORALLY_CLUSTERED",
    "CLUSTERING_NOT_ENTERED",
    "CLUSTER_NO_GATE_PASSER",
    "CampaignClusteringAccounting",
    "CampaignStatisticsAccounting",
    "CandidateStatisticsSummary",
    "DSR_FAILED",
    "DSR_INSUFFICIENT_DATA",
    "DSR_PASSED",
    "GATE_DSR_FAILED",
    "GATE_PBO_FAILED",
    "GATE_ROBUSTNESS_FAILED",
    "GATE_SCORE_QUALIFIED_FAILED",
    "GATE_STRESS_FAILED",
    "NOT_CLUSTER_REPRESENTATIVE",
    "NO_BEHAVIORAL_SIGNATURE",
    "NO_ROBUSTNESS_PASSED_FOR_STATISTICS",
    "PBO_ABOVE_MAXIMUM",
    "PBO_FAILED",
    "PBO_INSUFFICIENT_DATA",
    "PBO_MATRIX_INSUFFICIENT",
    "PBO_PASSED",
    "PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST",
    "RESEARCH_SHORTLISTED",
    "RESEARCH_SHORTLISTED_DOES_NOT_MEAN",
    "ResearchShortlistConfig",
    "ResearchShortlistEntry",
    "ResearchShortlistPhaseResult",
    "SHORTLIST_NOT_ENTERED",
    "SHORTLIST_REJECTED",
    "STATISTICS_MISSING_OOS_ARTIFACTS",
    "STATISTICS_NOT_ENTERED",
    "STATISTICS_REQUIRES_FULL_WFO",
    "STATISTICS_REQUIRES_ROBUSTNESS_PASSED",
    "STATISTICS_REQUIRES_STRESS_PASSED",
    "STATISTICS_TESTED",
    "STATISTICALLY_PASSED",
    "STATISTICALLY_REJECTED",
    "ShortlistRejectRecord",
    "align_performance_matrix",
    "build_behavioral_signature",
    "build_full_wfo_trial_population",
    "evaluate_candidate_dsr_pbo",
    "extract_exposure_vector",
    "extract_oos_return_series",
    "extract_trade_timing_vector",
    "hard_gate_check",
    "select_best_per_cluster",
    "BehavioralDeduper",
    "behavioral_similarity",
]
