"""Promotion gates — OOS-only; training metrics cannot promote."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from discovery.fitness import RANKING_SOURCE_OOS, assert_oos_only_promotion
from discovery.freezing import FrozenCandidate
from discovery.stress import StressResult


class PromotionStatus(str, Enum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    HELD = "HELD"


@dataclass(frozen=True)
class PromotionDecision:
    status: PromotionStatus
    frozen_id: str
    candidate_id: str
    reason: str
    oos_fitness: float
    train_score_ignored: float | None
    stress_pass_rate: float
    decided_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "frozen_id": self.frozen_id,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "oos_fitness": self.oos_fitness,
            "train_score_ignored": self.train_score_ignored,
            "stress_pass_rate": self.stress_pass_rate,
            "decided_at": self.decided_at,
        }


@dataclass
class PromotionGate:
    min_oos_fitness: float = 0.0
    min_stress_pass_rate: float = 0.5

    def decide(
        self,
        frozen: FrozenCandidate,
        *,
        stress_results: list[StressResult] | None = None,
        train_score: float | None = None,
    ) -> PromotionDecision:
        if frozen.ranking_source != RANKING_SOURCE_OOS:
            return PromotionDecision(
                status=PromotionStatus.REJECTED,
                frozen_id=frozen.frozen_id,
                candidate_id=frozen.candidate_id,
                reason="ranking_source_not_oos",
                oos_fitness=frozen.oos_fitness,
                train_score_ignored=train_score,
                stress_pass_rate=0.0,
            )

        # Explicitly prove train metrics cannot promote even if huge
        oos_ok = assert_oos_only_promotion(
            train_score=train_score,
            oos_fitness=frozen.oos_fitness,
            promote_threshold=self.min_oos_fitness,
        )
        if not oos_ok:
            return PromotionDecision(
                status=PromotionStatus.REJECTED,
                frozen_id=frozen.frozen_id,
                candidate_id=frozen.candidate_id,
                reason="oos_fitness_below_threshold",
                oos_fitness=frozen.oos_fitness,
                train_score_ignored=train_score,
                stress_pass_rate=0.0,
            )

        stress = stress_results or []
        # Non-executed scenarios (not-applicable single-symbol exclusion,
        # unsupported/unknown scenario names, baseline-trades-unavailable)
        # must never inflate or deflate the promotion pass-rate denominator.
        executed = [s for s in stress if getattr(s, "status", "executed") == "executed"]
        pass_rate = sum(1 for s in executed if s.passed) / len(executed) if executed else 1.0
        if pass_rate < self.min_stress_pass_rate:
            return PromotionDecision(
                status=PromotionStatus.HELD,
                frozen_id=frozen.frozen_id,
                candidate_id=frozen.candidate_id,
                reason="stress_pass_rate_low",
                oos_fitness=frozen.oos_fitness,
                train_score_ignored=train_score,
                stress_pass_rate=pass_rate,
            )

        return PromotionDecision(
            status=PromotionStatus.PROMOTED,
            frozen_id=frozen.frozen_id,
            candidate_id=frozen.candidate_id,
            reason="oos_and_stress_ok",
            oos_fitness=frozen.oos_fitness,
            train_score_ignored=train_score,
            stress_pass_rate=pass_rate,
        )
