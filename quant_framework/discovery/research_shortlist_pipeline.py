"""Phase 3C: DSR/PBO → Behavioral Clustering → Research Shortlist.

Honest statistics only from completed Full WFO (+ Stress/Robustness provenance).
Never fabricates DSR/PBO; insufficient inputs surface as explicit reject reasons.
Vault / Paper / Live remain blocked after this stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
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
STATISTICS_INSUFFICIENT_DATA = "STATISTICS_INSUFFICIENT_DATA"
PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY = "PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY"
PBO_ALIGNMENT_INSUFFICIENT_PERIODS = "PBO_ALIGNMENT_INSUFFICIENT_PERIODS"
PBO_ALIGNMENT_TOO_MANY_MISSING = "PBO_ALIGNMENT_TOO_MANY_MISSING"
TIMING_DATA_UNAVAILABLE = "timing_data_unavailable"
EXPOSURE_DATA_UNAVAILABLE = "exposure_data_unavailable"

# Explicit observation types for DSR (never mixed silently).
OBS_NORMALIZED_EQUITY_BAR_RETURN = "normalized_equity_bar_return"
OBS_NORMALIZED_TRADE_RETURN = "normalized_trade_return"
OBS_INSUFFICIENT = "insufficient"

# Documented normalization methods.
NORM_EQUITY_BAR = "equity_or_bar_return_field"
NORM_TRADE_RETURN_FIELD = "explicit_normalized_trade_return_field"
NORM_PNL_OVER_NOTIONAL = "net_pnl_divided_by_notional_or_qty_exposure"
NORM_NONE = "none"

# PBO partition / fill policy labels.
PARTITION_WFO_VALIDATION_PERIOD = "wfo_validation_period"
PARTITION_TRADING_DAY = "trading_day"
FILL_ZERO_WHEN_NO_TRADE = "zero_return_when_no_trade_in_common_period"

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


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _full_wfo_completed(rec: EvaluationRecord) -> bool:
    """Independent Full-WFO proof — never trusts ``is_full_wfo_completion`` alone.

    The summary flag may be recorded for provenance, but integrity requires:
    signal_source, is_full_event_wfo, completed folds, baseline candidate_id match,
    and actual OOS folds or immutable baseline evidence.
    """
    train = rec.train_metrics or {}
    baseline = _baseline_artifacts(rec)
    signal = str(
        baseline.get("signal_source") or train.get("signal_source") or ""
    )
    is_full = bool(
        baseline.get("is_full_event_wfo", train.get("is_full_event_wfo", False))
    )
    completed = int(
        baseline.get("wfo_completed_folds")
        or train.get("wfo_completed_folds")
        or 0
    )
    if completed <= 0:
        completed = len(rec.oos_folds)
    baseline_cid = str(baseline.get("candidate_id") or "").strip()
    if baseline_cid and baseline_cid != str(rec.candidate_id):
        return False
    # Require baseline candidate_id when immutable baseline artifacts exist.
    has_baseline_blob = bool(baseline)
    if has_baseline_blob and not baseline_cid:
        return False
    if has_baseline_blob and baseline_cid != str(rec.candidate_id):
        return False
    has_oos_folds = len(rec.oos_folds) > 0
    has_baseline_evidence = bool(baseline.get("closed_trades")) or bool(
        baseline.get("oos_ranges")
    )
    if not (has_oos_folds or has_baseline_evidence):
        return False
    return signal == "candidate_dsl_trees" and is_full and completed > 0


@dataclass(frozen=True)
class OOSStatisticsSeries:
    """Coherent OOS observation series for DSR — one explicit observation type."""

    values: tuple[float, ...]
    observation_type: str
    normalization_method: str
    timestamps: tuple[str, ...]
    observation_count: int
    frequency: str
    source_artifact: str
    input_fingerprint: str
    reason: str | None = None

    @property
    def is_sufficient(self) -> bool:
        return (
            self.observation_type != OBS_INSUFFICIENT
            and self.observation_count >= 1
            and len(self.values) == self.observation_count
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_type": self.observation_type,
            "normalization_method": self.normalization_method,
            "source_timestamps": list(self.timestamps),
            "observation_count": self.observation_count,
            "frequency": self.frequency,
            "source_artifact": self.source_artifact,
            "input_fingerprint": self.input_fingerprint,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AlignedPerformanceMatrix:
    """Timestamp/partition-aligned PBO matrix — rows are common market periods."""

    matrix: np.ndarray | None
    common_time_index: tuple[str, ...]
    partition_frequency: str
    candidate_ids: tuple[str, ...]
    missing_counts: dict[str, int]
    fill_policy: str
    alignment_fingerprint: str
    reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.matrix is not None and self.reason is None

    def as_dict(self) -> dict[str, Any]:
        shape = list(self.matrix.shape) if self.matrix is not None else None
        return {
            "common_time_index": list(self.common_time_index),
            "partition_frequency": self.partition_frequency,
            "candidate_ids": list(self.candidate_ids),
            "matrix_shape": shape,
            "missing_counts": dict(self.missing_counts),
            "fill_policy": self.fill_policy,
            "alignment_fingerprint": self.alignment_fingerprint,
            "reason": self.reason,
        }


def _parse_timestamp(val: Any) -> datetime | None:
    """Parse a real timestamp; never hash malformed strings into artificial hours."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        # Epoch seconds / ms heuristic.
        x = float(val)
        if x > 1e12:
            x = x / 1000.0
        if x < 1e8:
            return None
        try:
            return datetime.utcfromtimestamp(x)
        except (OSError, OverflowError, ValueError):
            return None
    s = str(val).strip()
    if not s:
        return None
    # ISO-like: YYYY-MM-DD[ T]HH:MM[:SS]
    iso = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s[:19], fmt) if len(s) >= 10 else None
        except ValueError:
            continue
    return None


def _trade_timestamp_str(trade: dict[str, Any]) -> str | None:
    for key in ("exit_time", "entry_time", "timestamp"):
        dt = _parse_timestamp(trade.get(key))
        if dt is not None:
            return dt.isoformat(sep="T", timespec="seconds")
    return None


def _trade_hour(trade: dict[str, Any]) -> int | None:
    for key in ("exit_time", "entry_time", "timestamp"):
        dt = _parse_timestamp(trade.get(key))
        if dt is not None:
            return int(dt.hour)
    return None


def _trade_partition_keys(trade: dict[str, Any]) -> dict[str, str]:
    """Return available calendar/partition keys for a trade (no ordinal inventing)."""
    keys: dict[str, str] = {}
    if trade.get("fold_id") is not None:
        try:
            keys[PARTITION_WFO_VALIDATION_PERIOD] = f"fold:{int(trade['fold_id'])}"
        except (TypeError, ValueError):
            pass
    for key in ("exit_time", "entry_time", "timestamp"):
        dt = _parse_timestamp(trade.get(key))
        if dt is not None:
            keys[PARTITION_TRADING_DAY] = dt.strftime("%Y-%m-%d")
            break
    return keys


def _normalized_trade_return(trade: dict[str, Any]) -> tuple[float, str] | None:
    """Extract a documented normalized trade return — never raw monetary PnL alone."""
    for key in (
        "normalized_equity_return",
        "equity_return",
        "bar_return",
        "normalized_bar_return",
    ):
        if trade.get(key) is not None:
            try:
                return float(trade[key]), NORM_EQUITY_BAR
            except (TypeError, ValueError):
                continue
    for key in (
        "net_return",
        "normalized_return",
        "trade_return",
        "normalized_net_return",
    ):
        if trade.get(key) is not None:
            try:
                return float(trade[key]), NORM_TRADE_RETURN_FIELD
            except (TypeError, ValueError):
                continue
    # Explicit ``return`` only when paired with documented capital / exposure.
    if trade.get("return") is not None and (
        trade.get("normalization") in {"capital", "notional", "risk", "qty", "exposure"}
        or trade.get("normalized") is True
        or trade.get("return_is_normalized") is True
    ):
        try:
            return float(trade["return"]), NORM_TRADE_RETURN_FIELD
        except (TypeError, ValueError):
            pass
    # Documented PnL / exposure normalization.
    pnl = None
    for key in ("net_pnl", "realized_pnl", "pnl"):
        if trade.get(key) is not None:
            try:
                pnl = float(trade[key])
                break
            except (TypeError, ValueError):
                continue
    if pnl is None:
        return None
    exposure = None
    for key in ("notional", "risk_exposure", "position_exposure", "capital", "qty", "quantity"):
        if trade.get(key) is not None:
            try:
                cand_exp = abs(float(trade[key]))
                if cand_exp > 0:
                    exposure = cand_exp
                    break
            except (TypeError, ValueError):
                continue
    if exposure is not None and exposure > 0:
        return float(pnl) / float(exposure), NORM_PNL_OVER_NOTIONAL
    return None


def extract_oos_statistics_series(rec: EvaluationRecord) -> OOSStatisticsSeries:
    """Explicit OOS statistics series extractor for DSR.

    Preferred order:
    1. timestamped normalized equity/bar returns;
    2. timestamped normalized net trade returns;
    3. otherwise STATISTICS_INSUFFICIENT_DATA.

    Never silently mixes raw PnL, unlabeled returns, or fold expectancy.
    """
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    equity_vals: list[float] = []
    equity_ts: list[str] = []
    trade_vals: list[float] = []
    trade_ts: list[str] = []
    trade_norm = NORM_NONE

    if isinstance(trades, list):
        for t in trades:
            if not isinstance(t, dict):
                continue
            got = _normalized_trade_return(t)
            if got is None:
                continue
            val, method = got
            ts = _trade_timestamp_str(t) or ""
            if method == NORM_EQUITY_BAR:
                equity_vals.append(val)
                equity_ts.append(ts)
            else:
                trade_vals.append(val)
                trade_ts.append(ts)
                trade_norm = method

    def _pack(
        values: list[float],
        timestamps: list[str],
        *,
        observation_type: str,
        normalization_method: str,
        source_artifact: str,
        frequency: str,
    ) -> OOSStatisticsSeries:
        fp = _stable_fingerprint(
            {
                "observation_type": observation_type,
                "normalization_method": normalization_method,
                "values": values,
                "timestamps": timestamps,
                "candidate_id": rec.candidate_id,
                "source_artifact": source_artifact,
            }
        )
        return OOSStatisticsSeries(
            values=tuple(float(v) for v in values),
            observation_type=observation_type,
            normalization_method=normalization_method,
            timestamps=tuple(timestamps),
            observation_count=len(values),
            frequency=frequency,
            source_artifact=source_artifact,
            input_fingerprint=fp,
            reason=None,
        )

    if len(equity_vals) >= 1 and all(equity_ts):
        return _pack(
            equity_vals,
            equity_ts,
            observation_type=OBS_NORMALIZED_EQUITY_BAR_RETURN,
            normalization_method=NORM_EQUITY_BAR,
            source_artifact="baseline_wfo_artifacts.closed_trades",
            frequency="trade_event",
        )
    if len(equity_vals) >= 1:
        # Equity/bar fields present but missing timestamps → still typed, timed when possible.
        return _pack(
            equity_vals,
            equity_ts,
            observation_type=OBS_NORMALIZED_EQUITY_BAR_RETURN,
            normalization_method=NORM_EQUITY_BAR,
            source_artifact="baseline_wfo_artifacts.closed_trades",
            frequency="trade_event_partial_timestamps",
        )
    if len(trade_vals) >= 1 and all(trade_ts):
        return _pack(
            trade_vals,
            trade_ts,
            observation_type=OBS_NORMALIZED_TRADE_RETURN,
            normalization_method=trade_norm,
            source_artifact="baseline_wfo_artifacts.closed_trades",
            frequency="trade_event",
        )
    if len(trade_vals) >= 1:
        return _pack(
            trade_vals,
            trade_ts,
            observation_type=OBS_NORMALIZED_TRADE_RETURN,
            normalization_method=trade_norm,
            source_artifact="baseline_wfo_artifacts.closed_trades",
            frequency="trade_event_partial_timestamps",
        )

    # Fold expectancy is NOT a homogeneous return observation — refuse.
    reason = STATISTICS_INSUFFICIENT_DATA
    if isinstance(trades, list) and trades:
        reason = (
            f"{STATISTICS_INSUFFICIENT_DATA}:raw_or_unnormalized_pnl_without_"
            "documented_capital_or_risk_exposure"
        )
    elif rec.oos_folds:
        reason = (
            f"{STATISTICS_INSUFFICIENT_DATA}:fold_expectancy_is_not_a_return_observation"
        )
    else:
        reason = f"{STATISTICS_INSUFFICIENT_DATA}:{STATISTICS_MISSING_OOS_ARTIFACTS}"

    fp = _stable_fingerprint(
        {
            "observation_type": OBS_INSUFFICIENT,
            "candidate_id": rec.candidate_id,
            "reason": reason,
            "n_folds": len(rec.oos_folds),
            "n_trades": len(trades) if isinstance(trades, list) else 0,
        }
    )
    return OOSStatisticsSeries(
        values=(),
        observation_type=OBS_INSUFFICIENT,
        normalization_method=NORM_NONE,
        timestamps=(),
        observation_count=0,
        frequency="none",
        source_artifact="none",
        input_fingerprint=fp,
        reason=reason,
    )


def extract_oos_return_series(rec: EvaluationRecord) -> list[float]:
    """Backward-compatible values-only view of ``extract_oos_statistics_series``."""
    series = extract_oos_statistics_series(rec)
    if not series.is_sufficient:
        return []
    return list(series.values)


def extract_trade_timing_vector(
    rec: EvaluationRecord, *, n_bins: int = 16
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Hour-of-day timing histogram from *parsed* timestamps only.

    Returns (vector_or_empty, meta). Does not fabricate bins from ``hash()``,
    malformed timestamp strings, fold_id, or candidate sequence.
    """
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    counts = [0.0] * n_bins
    parsed = 0
    skipped_invalid = 0
    if isinstance(trades, list) and trades:
        for t in trades:
            if not isinstance(t, dict):
                continue
            hour = _trade_hour(t)
            if hour is None:
                # Invalid / missing timestamps: do not invent timing evidence.
                skipped_invalid += 1
                continue
            bin_i = int((hour / 24.0) * n_bins) % n_bins
            counts[bin_i] += 1.0
            parsed += 1
    meta: dict[str, Any] = {
        "available": parsed > 0,
        "parsed_timestamps": parsed,
        "skipped_invalid_or_missing": skipped_invalid,
        "status": "ok" if parsed > 0 else TIMING_DATA_UNAVAILABLE,
    }
    if parsed <= 0:
        return tuple(), meta
    total = sum(counts) or 1.0
    return tuple(c / total for c in counts), meta


def extract_exposure_vector(
    rec: EvaluationRecord, *, n_bins: int = 8
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Exposure histogram from quantity / notional / risk — never absolute PnL."""
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    sizes: list[float] = []
    skipped_pnl_only = 0
    if isinstance(trades, list) and trades:
        for t in trades:
            if not isinstance(t, dict):
                continue
            size = None
            for key in (
                "qty",
                "quantity",
                "size",
                "exposure",
                "notional",
                "risk_exposure",
                "position_exposure",
            ):
                if t.get(key) is not None:
                    try:
                        cand = abs(float(t[key]))
                        if cand > 0:
                            size = cand
                            break
                    except (TypeError, ValueError):
                        continue
            if size is None:
                # PnL is not exposure — record gap explicitly.
                if any(t.get(k) is not None for k in ("net_pnl", "realized_pnl", "pnl")):
                    skipped_pnl_only += 1
                continue
            sizes.append(size)
    meta: dict[str, Any] = {
        "available": len(sizes) > 0,
        "n_exposure_observations": len(sizes),
        "skipped_pnl_used_as_exposure": skipped_pnl_only,
        "status": "ok" if sizes else EXPOSURE_DATA_UNAVAILABLE,
    }
    if not sizes:
        return tuple(), meta
    arr = np.asarray(sizes, dtype=float)
    hist, _ = np.histogram(arr, bins=n_bins)
    total = float(hist.sum()) or 1.0
    return tuple(float(x) / total for x in hist.tolist()), meta


def extract_daily_oos_pnl_vector(
    rec: EvaluationRecord, *, n_bins: int = 16
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Time-aligned daily OOS PnL (or normalized daily return) for ``daily_pnl``."""
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    by_day: dict[str, float] = {}
    used_return = False
    if isinstance(trades, list):
        for t in trades:
            if not isinstance(t, dict):
                continue
            day = None
            for key in ("exit_time", "entry_time", "timestamp"):
                dt = _parse_timestamp(t.get(key))
                if dt is not None:
                    day = dt.strftime("%Y-%m-%d")
                    break
            if day is None:
                continue
            got = _normalized_trade_return(t)
            if got is not None:
                by_day[day] = by_day.get(day, 0.0) + float(got[0])
                used_return = True
                continue
            pnl = None
            for key in ("net_pnl", "realized_pnl", "pnl"):
                if t.get(key) is not None:
                    try:
                        pnl = float(t[key])
                        break
                    except (TypeError, ValueError):
                        continue
            if pnl is not None:
                by_day[day] = by_day.get(day, 0.0) + float(pnl)
    meta: dict[str, Any] = {
        "available": len(by_day) > 0,
        "n_days": len(by_day),
        "value_kind": (
            "normalized_daily_return" if used_return and by_day else "daily_oos_pnl"
        ),
        "common_days": sorted(by_day.keys()),
    }
    if not by_day:
        meta["status"] = "daily_pnl_unavailable"
        return tuple(), meta
    ordered = [by_day[d] for d in sorted(by_day.keys())]
    # Pad/truncate to n_bins for correlation stability while preserving time order.
    if len(ordered) < n_bins:
        ordered = ordered + [0.0] * (n_bins - len(ordered))
    meta["status"] = "ok"
    return tuple(float(x) for x in ordered[:n_bins]), meta


def build_behavioral_signature(
    candidate: StrategyCandidate,
    rec: EvaluationRecord,
    *,
    n_bins: int = 16,
) -> BehaviorSignature | None:
    """Build signature from time-aligned daily OOS PnL + timing + exposure.

    Required: Full WFO + daily OOS PnL/returns vector.
    Timing/exposure may be unavailable → documented reduced signature.
    Never uses builtin ``hash()``; never treats PnL as exposure.
    """
    if not _full_wfo_completed(rec):
        return None
    daily, daily_meta = extract_daily_oos_pnl_vector(rec, n_bins=n_bins)
    if not daily_meta.get("available"):
        return None
    timing, timing_meta = extract_trade_timing_vector(rec, n_bins=n_bins)
    exposure, exposure_meta = extract_exposure_vector(
        rec, n_bins=max(4, n_bins // 2)
    )
    stats = extract_oos_statistics_series(rec)
    signal: list[float]
    if stats.is_sufficient:
        signal = list(stats.values)
    else:
        signal = list(daily)
    while len(signal) < n_bins:
        signal = signal + signal if signal else [0.0]
    signal = signal[:n_bins]

    availability = {
        "daily_pnl": True,
        "timing": bool(timing_meta.get("available")),
        "exposure": bool(exposure_meta.get("available")),
        "normalized_returns": stats.is_sufficient,
    }
    if availability["timing"] and availability["exposure"]:
        weights = {"daily_pnl": 0.4, "timing": 0.2, "exposure": 0.2, "features": 0.2}
        kind = "full"
    elif availability["timing"] or availability["exposure"]:
        weights = {"daily_pnl": 0.6, "timing": 0.15, "exposure": 0.05, "features": 0.2}
        kind = "reduced_missing_timing_or_exposure"
    else:
        weights = {"daily_pnl": 0.8, "timing": 0.0, "exposure": 0.0, "features": 0.2}
        kind = "reduced_daily_pnl_only"

    fit = rec.fitness.fitness if rec.fitness else float("-inf")
    return BehaviorSignature(
        candidate_id=candidate.candidate_id,
        signal_vector=tuple(float(x) for x in signal),
        daily_pnl=tuple(float(x) for x in daily),
        feature_ids=candidate.feature_ids,
        complexity=candidate.complexity_score,
        fitness=float(fit) if fit is not None else float("-inf"),
        timing_vector=tuple(float(x) for x in timing),
        exposure_vector=tuple(float(x) for x in exposure),
        component_availability=tuple(sorted(availability.items())),
        component_weights=tuple(sorted(weights.items())),
        signature_kind=kind,
    )


def _sharpe_from_returns(returns: Sequence[float]) -> float | None:
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 2:
        return None
    s = float(arr.std(ddof=1))
    if s < 1e-12:
        return 0.0
    return float(arr.mean() / s)


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
    Trial scores are Sharpe of the same normalized OOS series used for DSR.
    """
    scores: list[float] = []
    ids: list[str] = []
    series_list: list[list[float]] = []
    rejected = 0
    failed = 0
    for rec in records:
        if not _full_wfo_completed(rec):
            continue
        series = extract_oos_statistics_series(rec)
        if not series.is_sufficient:
            failed += 1
            continue
        rets = list(series.values)
        sharpe = _sharpe_from_returns(rets)
        if sharpe is None:
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


def _period_series_for_candidate(
    rec: EvaluationRecord,
) -> tuple[dict[str, float], str | None]:
    """Map partition key → period performance for one candidate.

    Prefers WFO validation fold periods; else trading-day aggregation.
    Returns ({}, None) when only trade-sequence ordering would be possible.
    """
    baseline = _baseline_artifacts(rec)
    trades = baseline.get("closed_trades")
    if not isinstance(trades, list) or not trades:
        return {}, None

    fold_map: dict[str, float] = {}
    day_map: dict[str, float] = {}
    any_partition = False
    for t in trades:
        if not isinstance(t, dict):
            continue
        keys = _trade_partition_keys(t)
        if not keys:
            continue
        any_partition = True
        value = None
        got = _normalized_trade_return(t)
        if got is not None:
            value = float(got[0])
        else:
            for pk in ("net_pnl", "realized_pnl", "pnl"):
                if t.get(pk) is not None:
                    try:
                        value = float(t[pk])
                        break
                    except (TypeError, ValueError):
                        continue
        if value is None:
            continue
        if PARTITION_WFO_VALIDATION_PERIOD in keys:
            k = keys[PARTITION_WFO_VALIDATION_PERIOD]
            fold_map[k] = fold_map.get(k, 0.0) + value
        if PARTITION_TRADING_DAY in keys:
            k = keys[PARTITION_TRADING_DAY]
            day_map[k] = day_map.get(k, 0.0) + value

    if not any_partition:
        return {}, None
    if fold_map:
        return fold_map, PARTITION_WFO_VALIDATION_PERIOD
    if day_map:
        return day_map, PARTITION_TRADING_DAY
    return {}, None


def align_performance_matrix(
    records: Sequence[EvaluationRecord] | Sequence[Sequence[float]] | None = None,
    *,
    candidate_ids: Sequence[str] | None = None,
    fill_policy: str = FILL_ZERO_WHEN_NO_TRADE,
    min_periods: int = 2,
    max_missing_fraction: float = 0.75,
    series_list: Sequence[Sequence[float]] | None = None,
) -> np.ndarray | AlignedPerformanceMatrix | None:
    """Align candidates onto a shared market-time partition index.

    Rows are common calendar / WFO validation periods — never trade ordinals.
    Legacy ``series_list``-only callers receive ``None`` (trade-sequence refuse).
    """
    # Legacy path: bare float sequences are trade-ordinal — refuse.
    if series_list is not None and records is None:
        return None
    if records is not None and records and not isinstance(records[0], EvaluationRecord):
        # Old API: align_performance_matrix(list_of_float_series)
        return None
    if records is None:
        return None

    recs = [r for r in records if isinstance(r, EvaluationRecord)]
    if candidate_ids is not None:
        want = list(candidate_ids)
        by_id = {r.candidate_id: r for r in recs}
        ordered_recs = [by_id[c] for c in want if c in by_id]
        ids = [r.candidate_id for r in ordered_recs]
    else:
        ordered_recs = list(recs)
        ids = [r.candidate_id for r in ordered_recs]

    if len(ordered_recs) < 2:
        return AlignedPerformanceMatrix(
            matrix=None,
            common_time_index=(),
            partition_frequency="",
            candidate_ids=tuple(ids),
            missing_counts={},
            fill_policy=fill_policy,
            alignment_fingerprint=_stable_fingerprint({"reason": PBO_MATRIX_INSUFFICIENT}),
            reason=PBO_MATRIX_INSUFFICIENT,
        )

    per_cand: list[dict[str, float]] = []
    freqs: list[str] = []
    for rec in ordered_recs:
        series, freq = _period_series_for_candidate(rec)
        if freq is None or not series:
            return AlignedPerformanceMatrix(
                matrix=None,
                common_time_index=(),
                partition_frequency="",
                candidate_ids=tuple(ids),
                missing_counts={},
                fill_policy=fill_policy,
                alignment_fingerprint=_stable_fingerprint(
                    {"reason": PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY, "ids": ids}
                ),
                reason=PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY,
            )
        per_cand.append(series)
        freqs.append(freq)

    # Prefer a single common frequency; if mixed, use trading day when available.
    if all(f == PARTITION_WFO_VALIDATION_PERIOD for f in freqs):
        partition_frequency = PARTITION_WFO_VALIDATION_PERIOD
    elif all(f == PARTITION_TRADING_DAY for f in freqs):
        partition_frequency = PARTITION_TRADING_DAY
    else:
        # Re-extract trading-day maps for consistency across candidates.
        partition_frequency = PARTITION_TRADING_DAY
        per_cand = []
        for rec in ordered_recs:
            baseline = _baseline_artifacts(rec)
            trades = baseline.get("closed_trades")
            day_map: dict[str, float] = {}
            if isinstance(trades, list):
                for t in trades:
                    if not isinstance(t, dict):
                        continue
                    keys = _trade_partition_keys(t)
                    if PARTITION_TRADING_DAY not in keys:
                        continue
                    value = None
                    got = _normalized_trade_return(t)
                    if got is not None:
                        value = float(got[0])
                    else:
                        for pk in ("net_pnl", "realized_pnl", "pnl"):
                            if t.get(pk) is not None:
                                try:
                                    value = float(t[pk])
                                    break
                                except (TypeError, ValueError):
                                    continue
                    if value is None:
                        continue
                    k = keys[PARTITION_TRADING_DAY]
                    day_map[k] = day_map.get(k, 0.0) + value
            if not day_map:
                return AlignedPerformanceMatrix(
                    matrix=None,
                    common_time_index=(),
                    partition_frequency=partition_frequency,
                    candidate_ids=tuple(ids),
                    missing_counts={},
                    fill_policy=fill_policy,
                    alignment_fingerprint=_stable_fingerprint(
                        {"reason": PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY}
                    ),
                    reason=PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY,
                )
            per_cand.append(day_map)

    # Shared sorted partition index = sorted union of all candidate periods.
    index_set: set[str] = set()
    for m in per_cand:
        index_set.update(m.keys())
    common_index = tuple(sorted(index_set))
    if len(common_index) < int(min_periods):
        return AlignedPerformanceMatrix(
            matrix=None,
            common_time_index=common_index,
            partition_frequency=partition_frequency,
            candidate_ids=tuple(ids),
            missing_counts={},
            fill_policy=fill_policy,
            alignment_fingerprint=_stable_fingerprint(
                {"reason": PBO_ALIGNMENT_INSUFFICIENT_PERIODS, "index": list(common_index)}
            ),
            reason=PBO_ALIGNMENT_INSUFFICIENT_PERIODS,
        )

    missing_counts: dict[str, int] = {}
    columns: list[np.ndarray] = []
    for cid, series in zip(ids, per_cand):
        col = []
        missing = 0
        for period in common_index:
            if period in series:
                col.append(float(series[period]))
            else:
                missing += 1
                if fill_policy == FILL_ZERO_WHEN_NO_TRADE:
                    col.append(0.0)
                else:
                    col.append(float("nan"))
        missing_counts[cid] = missing
        columns.append(np.asarray(col, dtype=float))

    total_cells = len(common_index) * len(ids)
    total_missing = sum(missing_counts.values())
    if total_cells > 0 and (total_missing / total_cells) > float(max_missing_fraction):
        return AlignedPerformanceMatrix(
            matrix=None,
            common_time_index=common_index,
            partition_frequency=partition_frequency,
            candidate_ids=tuple(ids),
            missing_counts=missing_counts,
            fill_policy=fill_policy,
            alignment_fingerprint=_stable_fingerprint(
                {
                    "reason": PBO_ALIGNMENT_TOO_MANY_MISSING,
                    "missing_counts": missing_counts,
                }
            ),
            reason=PBO_ALIGNMENT_TOO_MANY_MISSING,
        )

    matrix = np.column_stack(columns)
    fp = _stable_fingerprint(
        {
            "common_time_index": list(common_index),
            "partition_frequency": partition_frequency,
            "candidate_ids": list(ids),
            "matrix_shape": list(matrix.shape),
            "missing_counts": missing_counts,
            "fill_policy": fill_policy,
            # Content fingerprint via stable hash of rounded values.
            "matrix_digest": [
                [round(float(x), 10) for x in matrix[:, j].tolist()]
                for j in range(matrix.shape[1])
            ],
        }
    )
    return AlignedPerformanceMatrix(
        matrix=matrix,
        common_time_index=common_index,
        partition_frequency=partition_frequency,
        candidate_ids=tuple(ids),
        missing_counts=missing_counts,
        fill_policy=fill_policy,
        alignment_fingerprint=fp,
        reason=None,
    )


def evaluate_candidate_dsr_pbo(
    *,
    rec: EvaluationRecord,
    population: TrialPopulation,
    performance_matrix: np.ndarray | AlignedPerformanceMatrix | None,
    trial_column_index: int | None,
    config: ResearchShortlistConfig,
) -> tuple[DSRResult, PBOResult | None, dict[str, Any]]:
    """Real DSR (+ optional PBO) — never fabricates OK on insufficient data."""
    series = extract_oos_statistics_series(rec)
    returns = list(series.values) if series.is_sufficient else []
    n_obs = int(series.observation_count)
    observed = _sharpe_from_returns(returns) if series.is_sufficient else None
    meta: dict[str, Any] = {
        **series.as_dict(),
        "n_observations": n_obs,
        "full_wfo_completed": _full_wfo_completed(rec),
    }

    aligned_meta: dict[str, Any] = {}
    matrix_arr: np.ndarray | None = None
    pbo_reason_override: str | None = None
    if isinstance(performance_matrix, AlignedPerformanceMatrix):
        aligned_meta = performance_matrix.as_dict()
        meta["pbo_alignment"] = aligned_meta
        if performance_matrix.is_usable:
            matrix_arr = performance_matrix.matrix
        else:
            pbo_reason_override = performance_matrix.reason or PBO_MATRIX_INSUFFICIENT
    elif isinstance(performance_matrix, np.ndarray):
        matrix_arr = performance_matrix

    if not series.is_sufficient or observed is None:
        dsr = compute_deflated_sharpe(
            0.0,
            population,
            n_observations=n_obs,
            min_observations=config.min_oos_observations_for_dsr,
        )
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
                reason=series.reason or STATISTICS_INSUFFICIENT_DATA,
                trial_selection=population.selection.value,
            )
        elif series.reason and dsr.status == DSRStatus.INSUFFICIENT_DATA:
            from metrics.dsr import DSRResult as _DR

            dsr = _DR(
                status=dsr.status,
                deflated_sharpe=dsr.deflated_sharpe,
                observed_sharpe=None,
                expected_max_sharpe=dsr.expected_max_sharpe,
                total_trials=dsr.total_trials,
                scored_trials=dsr.scored_trials,
                rejected_trials=dsr.rejected_trials,
                failed_trials=dsr.failed_trials,
                effective_independent_trials=dsr.effective_independent_trials,
                trial_sharpe_variance=dsr.trial_sharpe_variance,
                mean_correlation=dsr.mean_correlation,
                n_clusters=dsr.n_clusters,
                n_observations=n_obs,
                sample_size=dsr.sample_size,
                skew=None,
                kurtosis=None,
                assumptions=dsr.assumptions,
                reason=series.reason,
                trial_selection=dsr.trial_selection,
            )
        pbo = None
        if matrix_arr is None:
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
                assumptions="PBO requires time-aligned Full WFO performance matrix",
                reason=pbo_reason_override or PBO_MATRIX_INSUFFICIENT,
                trial_selection=population.selection.value,
            )
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
    if matrix_arr is None:
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
            assumptions="PBO requires time-aligned Full WFO performance matrix",
            reason=pbo_reason_override or PBO_MATRIX_INSUFFICIENT,
            trial_selection=population.selection.value,
        )
    else:
        pbo = compute_pbo(
            matrix_arr,
            population=population,
            n_splits=int(config.pbo_n_splits),
        )
        meta["pbo_matrix_shape"] = list(matrix_arr.shape)
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
    "AlignedPerformanceMatrix",
    "BEHAVIORALLY_CLUSTERED",
    "CLUSTERING_NOT_ENTERED",
    "CLUSTER_NO_GATE_PASSER",
    "CampaignClusteringAccounting",
    "CampaignStatisticsAccounting",
    "CandidateStatisticsSummary",
    "DSR_FAILED",
    "DSR_INSUFFICIENT_DATA",
    "DSR_PASSED",
    "EXPOSURE_DATA_UNAVAILABLE",
    "FILL_ZERO_WHEN_NO_TRADE",
    "GATE_DSR_FAILED",
    "GATE_PBO_FAILED",
    "GATE_ROBUSTNESS_FAILED",
    "GATE_SCORE_QUALIFIED_FAILED",
    "GATE_STRESS_FAILED",
    "NOT_CLUSTER_REPRESENTATIVE",
    "NO_BEHAVIORAL_SIGNATURE",
    "NO_ROBUSTNESS_PASSED_FOR_STATISTICS",
    "OBS_INSUFFICIENT",
    "OBS_NORMALIZED_EQUITY_BAR_RETURN",
    "OBS_NORMALIZED_TRADE_RETURN",
    "OOSStatisticsSeries",
    "PARTITION_TRADING_DAY",
    "PARTITION_WFO_VALIDATION_PERIOD",
    "PBO_ABOVE_MAXIMUM",
    "PBO_ALIGNMENT_INSUFFICIENT_PERIODS",
    "PBO_ALIGNMENT_TOO_MANY_MISSING",
    "PBO_ALIGNMENT_TRADE_SEQUENCE_ONLY",
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
    "STATISTICS_INSUFFICIENT_DATA",
    "STATISTICS_MISSING_OOS_ARTIFACTS",
    "STATISTICS_NOT_ENTERED",
    "STATISTICS_REQUIRES_FULL_WFO",
    "STATISTICS_REQUIRES_ROBUSTNESS_PASSED",
    "STATISTICS_REQUIRES_STRESS_PASSED",
    "STATISTICS_TESTED",
    "STATISTICALLY_PASSED",
    "STATISTICALLY_REJECTED",
    "ShortlistRejectRecord",
    "TIMING_DATA_UNAVAILABLE",
    "align_performance_matrix",
    "build_behavioral_signature",
    "build_full_wfo_trial_population",
    "evaluate_candidate_dsr_pbo",
    "extract_daily_oos_pnl_vector",
    "extract_exposure_vector",
    "extract_oos_return_series",
    "extract_oos_statistics_series",
    "extract_trade_timing_vector",
    "hard_gate_check",
    "select_best_per_cluster",
    "BehavioralDeduper",
    "behavioral_similarity",
    "_full_wfo_completed",
]
