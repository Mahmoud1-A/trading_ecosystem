"""Position / cash / open-order reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brokers.protocol import BrokerProtocol
from runtime.state_store import RuntimeState


@dataclass
class ReconMismatch:
    kind: str
    detail: str
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class ReconciliationReport:
    ok: bool
    mismatches: list[ReconMismatch] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "mismatches": [m.as_dict() for m in self.mismatches]}


@dataclass
class Reconciler:
    broker: BrokerProtocol
    state: RuntimeState
    cash_tolerance: float = 1e-4
    qty_tolerance: float = 1e-6

    def run(self) -> ReconciliationReport:
        mismatches: list[ReconMismatch] = []
        # Positions
        broker_pos = {p.symbol: p for p in self.broker.positions()}
        for sym, expected in self.state.positions.items():
            actual = broker_pos.get(sym)
            exp_qty = float(expected.get("quantity", 0.0))
            if actual is None and abs(exp_qty) > self.qty_tolerance:
                mismatches.append(
                    ReconMismatch("position_missing", sym, exp_qty, None)
                )
            elif actual is not None and abs(actual.quantity - exp_qty) > self.qty_tolerance:
                mismatches.append(
                    ReconMismatch("position_qty", sym, exp_qty, actual.quantity)
                )
        for sym, actual in broker_pos.items():
            if sym not in self.state.positions and abs(actual.quantity) > self.qty_tolerance:
                mismatches.append(
                    ReconMismatch("position_unexpected", sym, 0.0, actual.quantity)
                )

        # Cash
        acct = self.broker.account()
        if abs(acct.cash - self.state.cash) > self.cash_tolerance:
            mismatches.append(
                ReconMismatch("cash", "cash", self.state.cash, acct.cash)
            )

        # Open orders — every mapped id should resolve
        for coid, boid in self.state.broker_order_map.items():
            order = self.broker.get_order(coid)
            if order is None:
                mismatches.append(
                    ReconMismatch("open_order_missing", coid, boid, None)
                )

        return ReconciliationReport(ok=len(mismatches) == 0, mismatches=mismatches)
