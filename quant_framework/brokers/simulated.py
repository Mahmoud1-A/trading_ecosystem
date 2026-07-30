"""In-process simulated broker for shadow expected fills and paper simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from brokers.capabilities import BrokerCapabilities, simulated_capabilities
from brokers.protocol import (
    AccountSnapshot,
    BrokerError,
    BrokerFill,
    BrokerMode,
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    OrderStatus,
    ShadowSubmitError,
)


@dataclass
class SimulatedBroker:
    """
    Research simulator.

    In SHADOW mode, submit_order records theoretical fills and never marks
    orders as live-submitted. In PAPER mode, it accepts paper fills in-process.
    """

    mode: BrokerMode = BrokerMode.SHADOW
    starting_cash: float = 100_000.0
    mid_prices: dict[str, float] = field(default_factory=dict)
    _orders: dict[str, BrokerOrder] = field(default_factory=dict)
    _fills: list[BrokerFill] = field(default_factory=list)
    _positions: dict[str, BrokerPosition] = field(default_factory=dict)
    _cash: float = field(init=False)
    _fill_seq: int = 0
    _connected: bool = True
    _submit_count: int = 0  # real submissions (paper only)

    def __post_init__(self) -> None:
        self._cash = self.starting_cash

    @property
    def capabilities(self) -> BrokerCapabilities:
        return simulated_capabilities()

    def connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False

    def reconnect(self) -> None:
        self._connected = True

    def set_price(self, symbol: str, price: float) -> None:
        self.mid_prices[symbol] = price

    def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        if not self._connected:
            raise BrokerError("broker disconnected")
        if order.client_order_id in self._orders:
            # Idempotent — return existing
            return self._orders[order.client_order_id]

        if self.mode is BrokerMode.SHADOW:
            # Theoretical only — must not count as real submission
            order = BrokerOrder(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                status=OrderStatus.THEORETICAL,
                broker_order_id=f"theo_{order.client_order_id}",
                filled_qty=order.quantity,
                avg_price=self._px(order.symbol),
                meta={**order.meta, "shadow": True, "submitted": False},
            )
            self._orders[order.client_order_id] = order
            self._record_fill(order, theoretical=True)
            return order

        # PAPER path
        self._submit_count += 1
        px = self._px(order.symbol)
        order = BrokerOrder(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            status=OrderStatus.FILLED,
            broker_order_id=f"paper_{order.client_order_id}",
            filled_qty=order.quantity,
            avg_price=px,
            meta={**order.meta, "shadow": False, "submitted": True},
        )
        self._orders[order.client_order_id] = order
        self._record_fill(order, theoretical=False)
        self._apply_position(order)
        return order

    def cancel_order(self, client_order_id: str) -> BrokerOrder:
        order = self._orders.get(client_order_id)
        if order is None:
            raise BrokerError(f"unknown order {client_order_id}")
        if order.status in {OrderStatus.FILLED, OrderStatus.THEORETICAL}:
            return order
        order.status = OrderStatus.CANCELLED
        return order

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        return self._orders.get(client_order_id)

    def positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    def account(self) -> AccountSnapshot:
        pos_val = sum(p.quantity * p.avg_price for p in self._positions.values())
        return AccountSnapshot(equity=self._cash + pos_val, cash=self._cash)

    def flatten_all(self) -> list[BrokerOrder]:
        outs: list[BrokerOrder] = []
        for pos in list(self._positions.values()):
            if abs(pos.quantity) < 1e-12:
                continue
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            oid = f"flatten_{pos.symbol}_{len(self._orders)}"
            order = BrokerOrder(
                client_order_id=oid,
                symbol=pos.symbol,
                side=side,
                quantity=abs(pos.quantity),
                order_type="MARKET",
            )
            outs.append(self.submit_order(order))
        return outs

    def fills(self) -> list[BrokerFill]:
        return list(self._fills)

    def real_submit_count(self) -> int:
        return self._submit_count

    def _px(self, symbol: str) -> float:
        return float(self.mid_prices.get(symbol, 100.0))

    def _record_fill(self, order: BrokerOrder, *, theoretical: bool) -> None:
        self._fill_seq += 1
        self._fills.append(
            BrokerFill(
                fill_id=f"fill_{self._fill_seq}",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.filled_qty,
                price=float(order.avg_price or 0.0),
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                theoretical=theoretical,
            )
        )

    def _apply_position(self, order: BrokerOrder) -> None:
        signed = order.filled_qty if order.side is OrderSide.BUY else -order.filled_qty
        px = float(order.avg_price or 0.0)
        cost = signed * px
        self._cash -= cost
        prev = self._positions.get(order.symbol)
        if prev is None:
            self._positions[order.symbol] = BrokerPosition(order.symbol, signed, px)
            return
        new_qty = prev.quantity + signed
        if abs(new_qty) < 1e-12:
            del self._positions[order.symbol]
            return
        # average price for increases; keep prev avg on reductions
        if prev.quantity * signed > 0:
            avg = (prev.avg_price * abs(prev.quantity) + px * abs(signed)) / abs(new_qty)
        else:
            avg = prev.avg_price if abs(new_qty) <= abs(prev.quantity) else px
        self._positions[order.symbol] = BrokerPosition(order.symbol, new_qty, avg)
