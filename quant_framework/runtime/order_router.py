"""Idempotent order router with deterministic client order IDs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from brokers.protocol import BrokerMode, BrokerOrder, BrokerProtocol, ShadowSubmitError
from engine.ids import DeterministicIdFactory
from runtime.signal_service import Signal
from runtime.state_store import RuntimeState, StateStore


class OrderRouterError(RuntimeError):
    pass


@dataclass
class OrderRouter:
    broker: BrokerProtocol
    id_factory: DeterministicIdFactory
    state: RuntimeState
    store: StateStore | None = None
    allow_shadow_submit: bool = False  # shadow must not really submit

    def client_order_id_for(self, signal: Signal) -> str:
        """Stable ID from signal fingerprint — identical signals map to one order."""
        material = f"{self.state.run_id}|{signal.fingerprint()}|{signal.signal_id}"
        digest = hashlib.sha256(material.encode()).hexdigest()[:24]
        return f"coid_{digest}"

    def route(self, signal: Signal) -> BrokerOrder:
        coid = self.client_order_id_for(signal)

        # Duplicate / idempotent guard
        if coid in self.state.client_order_ids:
            existing = self.broker.get_order(coid)
            if existing is not None:
                return existing
            raise OrderRouterError(f"client_order_id {coid} recorded but missing at broker")

        order = BrokerOrder(
            client_order_id=coid,
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            order_type="MARKET",
            meta={"signal_id": signal.signal_id, "mode": self.broker.mode.value},
        )

        if self.broker.mode is BrokerMode.SHADOW and not self.allow_shadow_submit:
            submitted = self.broker.submit_order(order)
            if submitted.meta.get("submitted") is True:
                raise ShadowSubmitError("Shadow mode must not submit real orders")
            self._record(submitted)
            return submitted

        submitted = self.broker.submit_order(order)
        self._record(submitted)
        return submitted

    def _record(self, order: BrokerOrder) -> None:
        self.state.client_order_ids.append(order.client_order_id)
        if order.broker_order_id:
            self.state.broker_order_map[order.client_order_id] = order.broker_order_id
        if self.store is not None:
            self.store.append_audit(
                self.state,
                {
                    "event": "order_routed",
                    "client_order_id": order.client_order_id,
                    "status": order.status.value,
                },
            )
            self.store.save(self.state)
