"""Risk budget utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskBudget:
    target_vol: float = 0.10
    max_member_risk_share: float = 0.35

    def risk_shares(self, weights: Mapping[str, float], vols: Mapping[str, float]) -> dict[str, float]:
        contrib = {cid: weights[cid] * vols.get(cid, 0.1) for cid in weights}
        total = sum(contrib.values()) or 1.0
        return {cid: c / total for cid, c in contrib.items()}

    def within_limits(self, weights: Mapping[str, float], vols: Mapping[str, float]) -> bool:
        shares = self.risk_shares(weights, vols)
        return all(s <= self.max_member_risk_share + 1e-12 for s in shares.values())
