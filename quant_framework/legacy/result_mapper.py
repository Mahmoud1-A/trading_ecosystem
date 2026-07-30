"""Map reevaluation results — strip legacy status, attach new-pipeline evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from discovery.candidate import StrategyCandidate
from discovery.evaluator import EvaluationRecord
from legacy.candidate_adapter import FORBIDDEN_INHERITED_FIELDS
from legacy.records import LegacyStrategyRecord


@dataclass
class MappedTrial:
    legacy_id: str
    candidate_id: str
    lineage_id: str
    creation_method: str
    reevaluated: bool
    inherited_status: dict[str, Any]
    new_oos_fitness: float | None
    rejection_reason: str | None
    trial_id: str | None
    notes: str = "reevaluated_from_zero"

    def as_dict(self) -> dict[str, Any]:
        return {
            "legacy_id": self.legacy_id,
            "candidate_id": self.candidate_id,
            "lineage_id": self.lineage_id,
            "creation_method": self.creation_method,
            "reevaluated": self.reevaluated,
            "inherited_status_archived_only": dict(self.inherited_status),
            "new_oos_fitness": self.new_oos_fitness,
            "rejection_reason": self.rejection_reason,
            "trial_id": self.trial_id,
            "notes": self.notes,
        }


def map_reevaluation(
    record: LegacyStrategyRecord,
    candidate: StrategyCandidate,
    evaluation: EvaluationRecord,
) -> MappedTrial:
    archived = record.archival_status()
    # Ensure we never treat archived status as live outcome
    for k in FORBIDDEN_INHERITED_FIELDS:
        assert k not in (evaluation.meta or {})
    fit = evaluation.fitness.fitness if evaluation.fitness else None
    return MappedTrial(
        legacy_id=record.legacy_id,
        candidate_id=candidate.candidate_id,
        lineage_id=candidate.lineage_id,
        creation_method=candidate.creation_method.value,
        reevaluated=True,
        inherited_status=archived,
        new_oos_fitness=fit,
        rejection_reason=evaluation.rejection_reason,
        trial_id=evaluation.trial_id,
    )
