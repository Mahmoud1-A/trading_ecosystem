"""Paper-trading status API — research paper eligibility only (not live)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from api.types import CounterKind, TypedCounter


class PaperStatus(str, Enum):
    NONE = "NONE"
    ELIGIBLE = "ELIGIBLE"
    ACTIVE = "ACTIVE"
    HALTED = "HALTED"


@dataclass
class PaperAPI:
    """Track paper eligibility for frozen portfolios / candidates."""

    status_by_id: dict[str, PaperStatus] = field(default_factory=dict)
    promotions_blocked_legacy: int = 0

    def set_status(self, entity_id: str, status: PaperStatus) -> None:
        self.status_by_id[entity_id] = status

    def counters(self) -> list[TypedCounter]:
        counts = {s: 0 for s in PaperStatus}
        for st in self.status_by_id.values():
            counts[st] += 1
        return [
            TypedCounter("paper_eligible", counts[PaperStatus.ELIGIBLE], CounterKind.SNAPSHOT),
            TypedCounter("paper_active", counts[PaperStatus.ACTIVE], CounterKind.SNAPSHOT),
            TypedCounter("paper_halted", counts[PaperStatus.HALTED], CounterKind.SNAPSHOT),
            TypedCounter(
                "legacy_promotions_blocked",
                self.promotions_blocked_legacy,
                CounterKind.CUMULATIVE_EVENT,
                "attempts blocked because legacy promotion is disabled",
            ),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status_by_id": {k: v.value for k, v in self.status_by_id.items()},
            "counters": [c.as_dict() for c in self.counters()],
        }
