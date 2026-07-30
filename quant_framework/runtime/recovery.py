"""Restart recovery — restore state without duplicating positions."""

from __future__ import annotations

from dataclasses import dataclass

from brokers.protocol import BrokerProtocol
from runtime.reconciliation import Reconciler, ReconciliationReport
from runtime.state_store import RuntimeState, StateStore


@dataclass
class RecoveryManager:
    store: StateStore
    broker: BrokerProtocol

    def recover(self) -> tuple[RuntimeState, ReconciliationReport]:
        state = self.store.load()
        if state is None:
            raise FileNotFoundError(f"no runtime state at {self.store.path}")
        # Do not re-submit recorded orders — idempotent map already exists
        report = Reconciler(self.broker, state).run()
        # Sync local position snapshot FROM broker if empty after crash mid-write
        if not state.positions:
            for p in self.broker.positions():
                state.positions[p.symbol] = {"quantity": p.quantity, "avg_price": p.avg_price}
            acct = self.broker.account()
            state.cash = acct.cash
            self.store.save(state)
            report = Reconciler(self.broker, state).run()
        return state, report
