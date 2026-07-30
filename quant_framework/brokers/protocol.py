"""Broker protocol for shadow / paper runtime (research-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from brokers.capabilities import BrokerCapabilities


class BrokerMode(str, Enum):
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    # LIVE intentionally unsupported in this research framework


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    ACCEPTED = "ACCEPTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    THEORETICAL = "THEORETICAL"  # shadow only


@dataclass
class BrokerOrder:
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: str
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.NEW
    broker_order_id: str | None = None
    filled_qty: float = 0.0
    avg_price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "broker_order_id": self.broker_order_id,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "meta": dict(self.meta),
        }


@dataclass
class BrokerFill:
    fill_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: str
    theoretical: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
            "theoretical": self.theoretical,
        }


@dataclass
class BrokerPosition:
    symbol: str
    quantity: float
    avg_price: float

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "quantity": self.quantity, "avg_price": self.avg_price}


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    margin_used: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


class BrokerError(RuntimeError):
    pass


class LiveCredentialsError(BrokerError):
    """Raised when paper adapters are given live trading credentials."""


class ShadowSubmitError(BrokerError):
    """Raised when shadow mode attempts a real order submission."""


@runtime_checkable
class BrokerProtocol(Protocol):
    @property
    def capabilities(self) -> BrokerCapabilities: ...

    @property
    def mode(self) -> BrokerMode: ...

    def submit_order(self, order: BrokerOrder) -> BrokerOrder: ...

    def cancel_order(self, client_order_id: str) -> BrokerOrder: ...

    def get_order(self, client_order_id: str) -> BrokerOrder | None: ...

    def positions(self) -> list[BrokerPosition]: ...

    def account(self) -> AccountSnapshot: ...

    def flatten_all(self) -> list[BrokerOrder]: ...

    def connected(self) -> bool: ...
