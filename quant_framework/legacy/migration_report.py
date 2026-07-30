"""Legacy migration report — archive snapshot vs reevaluation funnel events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from legacy.records import LEGACY_BOOK_SIZE
from legacy.result_mapper import MappedTrial

FUNNEL_STAGES = (
    "GENERATED",
    "PRECHECK_REJECTED",
    "FULLY_EVALUATED",
    "SCORE_QUALIFIED",
    "BEHAVIORALLY_UNIQUE",
    "FINALIST",
)


@dataclass
class MigrationReport:
    imported: int = 0
    reevaluated: int = 0
    precheck_rejected: int = 0
    fully_evaluated: int = 0
    score_qualified: int = 0
    behaviorally_unique: int = 0
    finalists: int = 0
    legacy_book_size_snapshot: int = LEGACY_BOOK_SIZE
    mapped: list[MappedTrial] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def record_mapped(
        self,
        trial: MappedTrial,
        *,
        qualified: bool,
        unique: bool,
        finalist: bool,
    ) -> None:
        self.mapped.append(trial)
        self.imported += 1
        if trial.reevaluated:
            self.reevaluated += 1
        if trial.rejection_reason and "precheck" in (trial.rejection_reason or ""):
            self.precheck_rejected += 1
        else:
            self.fully_evaluated += 1
        if qualified:
            self.score_qualified += 1
        if unique:
            self.behaviorally_unique += 1
        if finalist:
            self.finalists += 1

    def funnel_event_counts(self) -> dict[str, int]:
        return {
            "GENERATED": self.imported,
            "PRECHECK_REJECTED": self.precheck_rejected,
            "FULLY_EVALUATED": self.fully_evaluated,
            "SCORE_QUALIFIED": self.score_qualified,
            "BEHAVIORALLY_UNIQUE": self.behaviorally_unique,
            "FINALIST": self.finalists,
        }

    def as_dict(self) -> dict[str, Any]:
        funnel = self.funnel_event_counts()
        assert "BOOK_SIZE" not in funnel
        assert "legacy_book_size" not in funnel
        return {
            "created_at": self.created_at,
            "snapshots": {"legacy_book_size": self.legacy_book_size_snapshot},
            "funnel_event_counts": funnel,
            "mapped_trials": [m.as_dict() for m in self.mapped],
            "all_reevaluated": self.imported > 0 and self.reevaluated == self.imported,
        }
