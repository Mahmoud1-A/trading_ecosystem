"""Retirement — retired strategies cannot automatically return."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from governance.degradation import DecayState


class RetirementError(PermissionError):
    pass


@dataclass
class RetirementRecord:
    strategy_id: str
    lineage_id: str
    retired_at: str
    reason: str
    vault_claim_voided: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "lineage_id": self.lineage_id,
            "retired_at": self.retired_at,
            "reason": self.reason,
            "vault_claim_voided": self.vault_claim_voided,
        }


@dataclass
class RetirementRegistry:
    records: dict[str, RetirementRecord] = field(default_factory=dict)

    def retire(self, *, strategy_id: str, lineage_id: str, reason: str) -> RetirementRecord:
        rec = RetirementRecord(
            strategy_id=strategy_id,
            lineage_id=lineage_id,
            retired_at=datetime.now(tz=timezone.utc).isoformat(),
            reason=reason,
            vault_claim_voided=True,
        )
        self.records[strategy_id] = rec
        return rec

    def is_retired(self, strategy_id: str) -> bool:
        return strategy_id in self.records

    def assert_not_auto_reactivate(self, strategy_id: str, *, new_state: DecayState) -> None:
        if strategy_id in self.records and new_state is not DecayState.RETIRED:
            raise RetirementError(
                f"retired strategy {strategy_id} cannot reactivate automatically; "
                "requires new validation lineage"
            )

    def void_prior_vault_claim(self, strategy_id: str) -> bool:
        rec = self.records.get(strategy_id)
        return bool(rec and rec.vault_claim_voided)
