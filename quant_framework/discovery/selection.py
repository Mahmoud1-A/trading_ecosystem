"""Diverse parent selection across structure, features, and behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discovery.behavioral_dedup import BehaviorSignature, behavioral_similarity
from discovery.candidate import StrategyCandidate


@dataclass
class ScoredCandidate:
    candidate: StrategyCandidate
    fitness: float
    signature: BehaviorSignature | None = None


@dataclass
class DiverseSelector:
    """Tournament selection with behavioral / structural diversity pressure."""

    tournament_size: int = 3
    diversity_weight: float = 0.35

    def select_parents(
        self,
        pool: Sequence[ScoredCandidate],
        *,
        n_pairs: int,
        rng: np.random.Generator,
    ) -> list[tuple[StrategyCandidate, StrategyCandidate]]:
        if len(pool) < 2:
            return []
        pairs: list[tuple[StrategyCandidate, StrategyCandidate]] = []
        for _ in range(n_pairs):
            a = self._pick(pool, rng, avoid=None)
            b = self._pick(pool, rng, avoid=a)
            pairs.append((a.candidate, b.candidate))
        return pairs

    def _pick(
        self,
        pool: Sequence[ScoredCandidate],
        rng: np.random.Generator,
        avoid: ScoredCandidate | None,
    ) -> ScoredCandidate:
        eligible = [p for p in pool if avoid is None or p.candidate.candidate_id != avoid.candidate.candidate_id]
        if not eligible:
            eligible = list(pool)
        k = min(self.tournament_size, len(eligible))
        idxs = rng.choice(len(eligible), size=k, replace=False)
        contestants = [eligible[int(i)] for i in idxs]

        def score(c: ScoredCandidate) -> float:
            div = 0.0
            if avoid is not None and c.signature and avoid.signature:
                div = 1.0 - behavioral_similarity(c.signature, avoid.signature)
            # Structural diversity via feature Jaccard complement
            if avoid is not None:
                fa, fb = set(c.candidate.feature_ids), set(avoid.candidate.feature_ids)
                j = len(fa & fb) / len(fa | fb) if (fa or fb) else 1.0
                div = 0.5 * div + 0.5 * (1.0 - j)
            return c.fitness + self.diversity_weight * div

        return max(contestants, key=score)

    def select_elites(
        self, pool: Sequence[ScoredCandidate], *, n: int
    ) -> list[StrategyCandidate]:
        ranked = sorted(pool, key=lambda s: s.fitness, reverse=True)
        return [s.candidate for s in ranked[:n]]
