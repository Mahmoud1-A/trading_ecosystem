"""Portfolio candidate pool — behaviorally unique finalists for Phase 7 handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from discovery.behavioral_dedup import BehaviorCluster
from discovery.candidate import StrategyCandidate
from discovery.freezing import FrozenCandidate


@dataclass
class PortfolioCandidate:
    candidate: StrategyCandidate
    frozen: FrozenCandidate | None
    fitness: float
    cluster_id: str
    stress_pass_rate: float
    robustness_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "lineage_id": self.candidate.lineage_id,
            "frozen_id": self.frozen.frozen_id if self.frozen else None,
            "fitness": self.fitness,
            "cluster_id": self.cluster_id,
            "stress_pass_rate": self.stress_pass_rate,
            "robustness_ok": self.robustness_ok,
            "complexity": self.candidate.complexity_score,
            "feature_ids": list(self.candidate.feature_ids),
        }


@dataclass
class PortfolioCandidatePool:
    """Retain one representative per behavior cluster."""

    members: list[PortfolioCandidate] = field(default_factory=list)

    def add(self, member: PortfolioCandidate) -> None:
        # Replace weaker member of same cluster
        for i, existing in enumerate(self.members):
            if existing.cluster_id == member.cluster_id:
                if member.fitness > existing.fitness:
                    self.members[i] = member
                return
        self.members.append(member)

    def from_clusters(
        self,
        *,
        clusters: list[BehaviorCluster],
        by_id: dict[str, PortfolioCandidate],
    ) -> PortfolioCandidatePool:
        for cluster in clusters:
            rep = by_id.get(cluster.representative_id)
            if rep is not None:
                self.add(
                    PortfolioCandidate(
                        candidate=rep.candidate,
                        frozen=rep.frozen,
                        fitness=rep.fitness,
                        cluster_id=cluster.cluster_id,
                        stress_pass_rate=rep.stress_pass_rate,
                        robustness_ok=rep.robustness_ok,
                    )
                )
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"members": [m.as_dict() for m in self.members], "size": len(self.members)}
