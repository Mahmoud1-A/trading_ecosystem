"""Independent emergency kill switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brokers.protocol import BrokerProtocol
from runtime.state_store import RuntimeState, StateStore


@dataclass
class KillSwitch:
    """
    Independent emergency control — cancels open orders and optionally flattens.

    Engaging the kill switch does not require the signal service or scheduler.
    """

    broker: BrokerProtocol
    state: RuntimeState
    store: StateStore | None = None
    flatten_on_engage: bool = True
    _engaged: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def engaged(self) -> bool:
        return self._engaged or self.state.kill_switch_engaged

    def engage(self, *, reason: str = "manual") -> dict[str, Any]:
        self._engaged = True
        self.state.kill_switch_engaged = True
        cancelled = []
        for coid in list(self.state.client_order_ids):
            order = self.broker.get_order(coid)
            if order is None:
                continue
            if order.status.value in {"FILLED", "THEORETICAL", "CANCELLED", "REJECTED"}:
                continue
            try:
                cancelled.append(self.broker.cancel_order(coid).client_order_id)
            except Exception as exc:  # noqa: BLE001
                cancelled.append(f"error:{coid}:{exc}")

        flattened = []
        if self.flatten_on_engage:
            for order in self.broker.flatten_all():
                flattened.append(order.client_order_id)
                if order.client_order_id not in self.state.client_order_ids:
                    self.state.client_order_ids.append(order.client_order_id)

        event = {
            "event": "kill_switch_engaged",
            "reason": reason,
            "cancelled": cancelled,
            "flattened": flattened,
        }
        self.events.append(event)
        if self.store is not None:
            self.store.append_audit(self.state, event)
            self.store.save(self.state)
        return event

    def test_fire(self) -> dict[str, Any]:
        """Dry engagement used by paper promotion gates."""
        return self.engage(reason="promotion_gate_test")
