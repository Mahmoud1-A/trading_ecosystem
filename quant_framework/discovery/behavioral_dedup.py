"""Behavioral deduplication — cluster similar candidates, keep robust simplest."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from discovery.candidate import StrategyCandidate
from discovery.evaluator import EvaluationRecord

# Reject mixing raw monetary PnL with normalized returns in one comparison.
DAILY_KIND_NORMALIZED = "normalized_daily_return"
DAILY_KIND_RAW_PNL = "daily_oos_pnl"
MIXED_BEHAVIORAL_VALUE_KINDS = "MIXED_BEHAVIORAL_VALUE_KINDS"

_DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "daily_pnl": 0.4,
    "timing": 0.2,
    "exposure": 0.2,
    "features": 0.2,
}


@dataclass(frozen=True)
class BehaviorSignature:
    candidate_id: str
    signal_vector: tuple[float, ...]
    daily_pnl: tuple[float, ...]
    feature_ids: tuple[str, ...]
    complexity: float
    fitness: float
    timing_vector: tuple[float, ...] = ()
    exposure_vector: tuple[float, ...] = ()
    component_availability: tuple[tuple[str, bool], ...] = ()
    component_weights: tuple[tuple[str, float], ...] = ()
    signature_kind: str = "full"
    daily_index: tuple[str, ...] = ()
    daily_value_kind: str = DAILY_KIND_NORMALIZED
    covered_days: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "signal_vector": list(self.signal_vector),
            "daily_pnl": list(self.daily_pnl),
            "feature_ids": list(self.feature_ids),
            "complexity": self.complexity,
            "fitness": self.fitness,
            "timing_vector": list(self.timing_vector),
            "exposure_vector": list(self.exposure_vector),
            "component_availability": dict(self.component_availability),
            "component_weights": dict(self.component_weights),
            "signature_kind": self.signature_kind,
            "daily_index": list(self.daily_index),
            "daily_value_kind": self.daily_value_kind,
            "covered_days": list(self.covered_days),
        }


@dataclass(frozen=True)
class BehaviorCluster:
    cluster_id: str
    member_ids: tuple[str, ...]
    representative_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_ids": list(self.member_ids),
            "representative_id": self.representative_id,
        }


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; skips NaN pairs (uncovered days)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    x, y = x[mask], y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def shared_daily_calendar_index(
    signatures: Sequence[BehaviorSignature],
) -> tuple[str, ...]:
    """One shared sorted daily OOS index across clustering candidates.

    Built from the union of traded calendar days (daily_index), not from each
    candidate's local first-active-day ordinal.
    """
    days: set[str] = set()
    for sig in signatures:
        days.update(d for d in sig.daily_index if d)
    return tuple(sorted(days))


def align_daily_vector_to_index(
    sig: BehaviorSignature,
    common_index: Sequence[str],
) -> tuple[float, ...]:
    """Map signature daily values onto a shared calendar.

    - traded day → normalized return
    - covered no-trade day → 0.0
    - uncovered day → NaN (missing)
    """
    by_day = {
        d: float(v)
        for d, v in zip(sig.daily_index, sig.daily_pnl)
        if d
    }
    covered = set(sig.covered_days) | set(by_day.keys())
    out: list[float] = []
    for day in common_index:
        if day in by_day:
            out.append(by_day[day])
        elif day in covered:
            out.append(0.0)
        else:
            out.append(float("nan"))
    return tuple(out)


def align_signatures_to_common_calendar(
    signatures: Sequence[BehaviorSignature],
) -> list[BehaviorSignature]:
    """Rewrite daily_pnl/daily_index onto one shared sorted calendar."""
    if not signatures:
        return []
    common = shared_daily_calendar_index(signatures)
    if not common:
        return list(signatures)
    aligned: list[BehaviorSignature] = []
    for sig in signatures:
        vec = align_daily_vector_to_index(sig, common)
        aligned.append(
            replace(sig, daily_pnl=vec, daily_index=tuple(common))
        )
    return aligned


def signature_from_record(
    candidate: StrategyCandidate,
    record: EvaluationRecord,
    *,
    n_bins: int = 16,
) -> BehaviorSignature:
    """Build a behavior signature from OOS fold metrics + feature dependency."""
    # Synthetic signal / pnl vectors from fold evidence (deterministic)
    folds = record.oos_folds
    signal = tuple(float(f.expectancy) for f in folds) + tuple(
        float(f.sharpe) for f in folds
    )
    pnl = tuple(float(f.expectancy) * max(f.n_trades, 1) for f in folds)
    # Pad for stable correlation length
    while len(signal) < n_bins:
        signal = signal + signal
    while len(pnl) < n_bins:
        pnl = pnl + pnl
    fit = record.fitness.fitness if record.fitness else float("-inf")
    return BehaviorSignature(
        candidate_id=candidate.candidate_id,
        signal_vector=signal[:n_bins],
        daily_pnl=pnl[:n_bins],
        feature_ids=candidate.feature_ids,
        complexity=candidate.complexity_score,
        fitness=fit,
        daily_value_kind=DAILY_KIND_RAW_PNL,
    )


def feature_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _declared_weights(sig: BehaviorSignature) -> dict[str, float]:
    if sig.component_weights:
        return {str(k): float(v) for k, v in sig.component_weights}
    return dict(_DEFAULT_COMPONENT_WEIGHTS)


def behavioral_similarity(
    a: BehaviorSignature,
    b: BehaviorSignature,
    *,
    meta_out: dict[str, Any] | None = None,
) -> float:
    """Weighted composite similarity using available documented components.

    Components (when present on both sides):
    - common-calendar OOS return/PnL behavior (``daily_pnl``)
    - entry timing
    - exposure
    - feature overlap

    Unavailable components are excluded and remaining weights renormalized.
    Mixing raw PnL with normalized returns is rejected (similarity 0).
    """
    meta: dict[str, Any] = {
        "components_used": [],
        "renormalized_weights": {},
        "rejected_reason": None,
    }

    kind_a = str(a.daily_value_kind or "")
    kind_b = str(b.daily_value_kind or "")
    if (
        kind_a
        and kind_b
        and kind_a != kind_b
        and {kind_a, kind_b} == {DAILY_KIND_NORMALIZED, DAILY_KIND_RAW_PNL}
    ):
        meta["rejected_reason"] = MIXED_BEHAVIORAL_VALUE_KINDS
        if meta_out is not None:
            meta_out.update(meta)
        return 0.0

    weights = _declared_weights(a)
    # Prefer non-zero declared weights from either side.
    for k, v in _declared_weights(b).items():
        if k not in weights or (weights[k] <= 0 and v > 0):
            weights[k] = v

    scores: dict[str, float] = {}

    # Align daily behavior onto a shared calendar when indices exist.
    if a.daily_pnl and b.daily_pnl:
        if a.daily_index and b.daily_index:
            common = tuple(sorted(set(a.daily_index) | set(b.daily_index)))
            va = align_daily_vector_to_index(a, common)
            vb = align_daily_vector_to_index(b, common)
            scores["daily_pnl"] = max(0.0, _corr(va, vb))
        else:
            scores["daily_pnl"] = max(0.0, _corr(a.daily_pnl, b.daily_pnl))

    if a.timing_vector and b.timing_vector:
        scores["timing"] = max(0.0, _corr(a.timing_vector, b.timing_vector))

    if a.exposure_vector and b.exposure_vector:
        scores["exposure"] = max(0.0, _corr(a.exposure_vector, b.exposure_vector))

    # Feature overlap is always computable from ids.
    scores["features"] = float(feature_jaccard(a.feature_ids, b.feature_ids))

    usable = {
        k: float(scores[k])
        for k in scores
        if float(weights.get(k, 0.0)) > 0.0
    }
    if not usable:
        if meta_out is not None:
            meta_out.update(meta)
        return 0.0

    raw_w = {k: float(weights[k]) for k in usable}
    total_w = sum(raw_w.values())
    if total_w <= 0:
        if meta_out is not None:
            meta_out.update(meta)
        return 0.0
    renorm = {k: raw_w[k] / total_w for k in raw_w}
    sim = float(sum(renorm[k] * usable[k] for k in usable))
    meta["components_used"] = sorted(usable.keys())
    meta["renormalized_weights"] = {k: renorm[k] for k in sorted(renorm)}
    meta["component_scores"] = {k: usable[k] for k in sorted(usable)}
    if meta_out is not None:
        meta_out.update(meta)
    return sim


@dataclass
class BehavioralDeduper:
    similarity_threshold: float = 0.85

    def cluster(self, signatures: Sequence[BehaviorSignature]) -> list[BehaviorCluster]:
        # Align all daily series onto one shared calendar before comparing.
        remaining = align_signatures_to_common_calendar(list(signatures))
        clusters: list[BehaviorCluster] = []
        cid = 0
        while remaining:
            seed = remaining.pop(0)
            members = [seed]
            kept: list[BehaviorSignature] = []
            for other in remaining:
                if behavioral_similarity(seed, other) >= self.similarity_threshold:
                    members.append(other)
                else:
                    kept.append(other)
            remaining = kept
            # Prefer highest fitness, then lowest complexity
            rep = sorted(members, key=lambda s: (-s.fitness, s.complexity))[0]
            clusters.append(
                BehaviorCluster(
                    cluster_id=f"beh_{cid}",
                    member_ids=tuple(m.candidate_id for m in members),
                    representative_id=rep.candidate_id,
                )
            )
            cid += 1
        return clusters

    def representatives(
        self, signatures: Sequence[BehaviorSignature]
    ) -> list[BehaviorSignature]:
        clusters = self.cluster(signatures)
        by_id = {s.candidate_id: s for s in signatures}
        return [by_id[c.representative_id] for c in clusters]
