"""Legacy strategy records — historical archive only; no inherited status."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegacyStrategyRecord:
    """
    Snapshot of a strategy from the legacy dashboard / discovery engine.

    Status fields are archival metadata and MUST NOT transfer into the new system.
    """

    legacy_id: str
    name: str
    strategy_family: str
    parameters: dict[str, float]
    expression: dict[str, Any] | None = None
    # Archival only — never copied into StrategyCandidate / promotion / vault
    book_member: bool = False
    vault_eligible: bool = False
    paper_status: str | None = None
    legacy_fitness: float | None = None
    promotion_status: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def archival_status(self) -> dict[str, Any]:
        return {
            "book_member": self.book_member,
            "vault_eligible": self.vault_eligible,
            "paper_status": self.paper_status,
            "legacy_fitness": self.legacy_fitness,
            "promotion_status": self.promotion_status,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Canonical size of the historical Book — a snapshot, never a funnel stage
LEGACY_BOOK_SIZE = 29


def make_legacy_book(n: int = LEGACY_BOOK_SIZE) -> list[LegacyStrategyRecord]:
    """Synthetic stand-in for the old 29-member Book (read-only archive)."""
    records: list[LegacyStrategyRecord] = []
    for i in range(n):
        records.append(
            LegacyStrategyRecord(
                legacy_id=f"legacy_book_{i:02d}",
                name=f"BookMember_{i:02d}",
                strategy_family="legacy_mean_reversion",
                parameters={"lookback": float(10 + i), "z_entry": -1.5 - 0.05 * i},
                book_member=True,
                vault_eligible=True,
                paper_status="PAPER" if i % 2 == 0 else "LIVE_PAPER",
                legacy_fitness=1.0 + 0.1 * i,
                promotion_status="BOOK",
            )
        )
    return records
