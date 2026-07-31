"""Behavioral deduplication — cluster similar candidates, keep robust simplest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from discovery.candidate import StrategyCandidate
from discovery.evaluator import EvaluationRecord


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
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


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
    )


def feature_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def behavioral_similarity(a: BehaviorSignature, b: BehaviorSignature) -> float:
    """Composite similarity in [0, 1]."""
    signal_ov = max(0.0, _corr(a.signal_vector, b.signal_vector))
    pnl_ov = max(0.0, _corr(a.daily_pnl, b.daily_pnl))
    feat = feature_jaccard(a.feature_ids, b.feature_ids)
    return float(0.4 * signal_ov + 0.4 * pnl_ov + 0.2 * feat)


@dataclass
class BehavioralDeduper:
    similarity_threshold: float = 0.85

    def cluster(self, signatures: Sequence[BehaviorSignature]) -> list[BehaviorCluster]:
        remaining = list(signatures)
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
