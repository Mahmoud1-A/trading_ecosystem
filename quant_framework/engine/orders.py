"""Deterministic order lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

import pandas as pd

from config.models import OrderStatus, OrderType, Side
from engine.events import InformationTiming, require_aware


# Legal transitions for the order state machine.
_ALLOWED: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset(
        {OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.SUBMITTED: frozenset(
        {OrderStatus.ACTIVE, OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
    ),
    OrderStatus.ACTIVE: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}

TERMINAL_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
)


class OrderLifecycleError(ValueError):
    """Illegal order state transition."""


@dataclass
class Order:
    """Mutable order with explicit timing and deterministic lifecycle."""

    symbol: str
    contract: str
    side: Side
    quantity: float
    order_type: OrderType
    timing: InformationTiming
    order_id: str = ""
    status: OrderStatus = OrderStatus.CREATED
    limit_price: float | None = None
    stop_price: float | None = None
    stop_limit_price: float | None = None
    filled_quantity: float = 0.0
    remaining_quantity: float | None = None
    parent_signal_id: str | None = None
    reduce_only: bool = False
    latency: timedelta = field(default_factory=lambda: timedelta(0))
    reject_reason: str | None = None
    expire_at: pd.Timestamp | None = None
    linked_stop: float | None = None
    linked_target: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    status_history: list[tuple[OrderStatus, pd.Timestamp]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be > 0")
        if self.contract.strip() == "":
            raise ValueError("Order requires explicit tradable contract")
        if not self.order_id:
            # Stable fallback for unit tests that construct Order directly
            material = (
                f"{self.symbol}|{self.contract}|{self.side.value}|"
                f"{self.quantity}|{self.order_type.value}|{self.timing.decision_timestamp}"
            )
            import hashlib

            self.order_id = "ord_" + hashlib.sha256(material.encode()).hexdigest()[:24]
        if self.remaining_quantity is None:
            self.remaining_quantity = float(self.quantity)
        if not self.status_history:
            created_ts = self.timing.decision_timestamp
            self.status_history = [(OrderStatus.CREATED, created_ts)]
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and self.limit_price is None:
            if self.order_type == OrderType.STOP_LIMIT and self.stop_limit_price is not None:
                self.limit_price = self.stop_limit_price
            elif self.order_type == OrderType.LIMIT:
                raise ValueError("LIMIT order requires limit_price")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires stop_price")

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_buy(self) -> bool:
        return self.side == Side.BUY

    @property
    def activation_timestamp(self) -> pd.Timestamp | None:
        return self.timing.order_activation_timestamp

    def transition(self, new_status: OrderStatus, at: pd.Timestamp, *, reason: str | None = None) -> None:
        at = require_aware(at, name="transition_at")
        allowed = _ALLOWED[self.status]
        if new_status not in allowed:
            raise OrderLifecycleError(
                f"Illegal transition {self.status.value} -> {new_status.value} "
                f"for order {self.order_id}"
            )
        self.status = new_status
        self.status_history.append((new_status, at))
        if reason is not None:
            self.reject_reason = reason

    def submit(self, at: pd.Timestamp, *, latency: timedelta | None = None) -> None:
        lat = self.latency if latency is None else latency
        timing = self.timing.with_submission(at, latency=lat)
        self.timing = timing
        self.latency = lat
        self.transition(OrderStatus.SUBMITTED, at)

    def activate(self, at: pd.Timestamp) -> None:
        if self.timing.order_activation_timestamp is None:
            raise OrderLifecycleError("Cannot activate before submission timing is set")
        if at < self.timing.order_activation_timestamp:
            raise OrderLifecycleError(
                "Cannot activate order before order_activation_timestamp "
                f"(at={at}, activation={self.timing.order_activation_timestamp})"
            )
        self.transition(OrderStatus.ACTIVE, at)

    def reject(self, at: pd.Timestamp, reason: str) -> None:
        self.transition(OrderStatus.REJECTED, at, reason=reason)

    def cancel(self, at: pd.Timestamp, reason: str = "cancelled") -> None:
        self.transition(OrderStatus.CANCELLED, at, reason=reason)

    def expire(self, at: pd.Timestamp, reason: str = "expired") -> None:
        self.transition(OrderStatus.EXPIRED, at, reason=reason)

    def apply_fill_qty(self, qty: float, at: pd.Timestamp) -> OrderStatus:
        if qty <= 0:
            raise ValueError("Fill quantity must be > 0")
        if self.status not in {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}:
            raise OrderLifecycleError(f"Cannot fill order in status {self.status.value}")
        remaining = float(self.remaining_quantity or 0.0)
        if qty - remaining > 1e-12:
            raise OrderLifecycleError(
                f"Fill qty {qty} exceeds remaining {remaining} for order {self.order_id}"
            )
        self.filled_quantity += qty
        self.remaining_quantity = remaining - qty
        if self.remaining_quantity <= 1e-12:
            self.remaining_quantity = 0.0
            self.transition(OrderStatus.FILLED, at)
            return OrderStatus.FILLED
        if self.status == OrderStatus.ACTIVE:
            self.transition(OrderStatus.PARTIALLY_FILLED, at)
        return OrderStatus.PARTIALLY_FILLED

    def snapshot(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "contract": self.contract,
            "side": self.side.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "reduce_only": self.reduce_only,
            "reject_reason": self.reject_reason,
            "parent_signal_id": self.parent_signal_id,
            "decision_timestamp": str(self.timing.decision_timestamp),
            "order_submission_timestamp": (
                str(self.timing.order_submission_timestamp)
                if self.timing.order_submission_timestamp
                else None
            ),
            "order_activation_timestamp": (
                str(self.timing.order_activation_timestamp)
                if self.timing.order_activation_timestamp
                else None
            ),
            "status_history": [(s.value, str(t)) for s, t in self.status_history],
            "meta": dict(self.meta),
        }


def clone_order(order: Order, **changes: Any) -> Order:
    return replace(order, **changes)
