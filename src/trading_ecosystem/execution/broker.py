from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from trading_ecosystem.common.contracts import (
    Fill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    Position,
)


class BrokerGateway(ABC):
    @abstractmethod
    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, broker_order_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def flatten(self, symbol: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sync(self) -> dict[str, Any]:
        """Recovery: pull broker state after restart."""
        raise NotImplementedError


class PaperSimBroker(BrokerGateway):
    """Local simulated broker for tests / offline paper path."""

    def __init__(self, cash: float = 100_000.0, costs: Any | None = None) -> None:
        from trading_ecosystem.execution.costs import ExecutionCosts, load_execution_costs

        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.fills: list[Fill] = []
        self.last_prices: dict[str, float] = {}
        self.costs = costs or load_execution_costs(universe_id="etf")
        if not isinstance(self.costs, ExecutionCosts):
            self.costs = load_execution_costs(universe_id="etf")

    def set_price(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = price

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        oid = str(uuid4())
        mid = self.last_prices.get(intent.symbol, intent.limit_price or 0.0)
        if mid <= 0:
            rec = {
                "id": oid,
                "status": OrderStatus.FAILED.value,
                "intent_id": intent.intent_id,
                "reason": "no_price",
            }
            self.orders[oid] = rec
            return rec

        liq = None
        if isinstance(intent.meta, dict):
            liq = intent.meta.get("liquidity")
        px = self.costs.fill_price(mid, intent.side, liq)
        commission = self.costs.commission(intent.qty, px, liq)
        fill = Fill(
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            price=px,
            ts=intent.ts if intent.ts.tzinfo else datetime.now(timezone.utc),
            commission=commission,
            slippage=abs(px - mid),
        )
        self.fills.append(fill)
        key = f"{intent.strategy_id}:{intent.symbol}"
        pos = self.positions.get(key, Position(symbol=intent.symbol, strategy_id=intent.strategy_id))
        if intent.side == OrderSide.BUY:
            self.cash -= px * intent.qty
            self.cash -= commission
            new_qty = pos.qty + intent.qty
            if pos.qty >= 0 and new_qty > 0:
                pos.avg_price = ((pos.avg_price * pos.qty) + px * intent.qty) / new_qty
            pos.qty = new_qty
        else:
            self.cash += px * intent.qty
            self.cash -= commission
            pos.qty -= intent.qty
            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
                pos.avg_price = 0.0
        if abs(pos.qty) < 1e-12:
            self.positions.pop(key, None)
        else:
            self.positions[key] = pos
        rec = {
            "id": oid,
            "status": OrderStatus.FILLED.value,
            "intent_id": intent.intent_id,
            "fill": fill,
        }
        self.orders[oid] = rec
        return rec

    def cancel(self, broker_order_id: str) -> bool:
        rec = self.orders.get(broker_order_id)
        if not rec or rec["status"] == OrderStatus.FILLED.value:
            return False
        rec["status"] = OrderStatus.CANCELED.value
        return True

    def flatten(self, symbol: str | None = None) -> list[dict[str, Any]]:
        results = []
        for key, pos in list(self.positions.items()):
            if symbol and pos.symbol != symbol:
                continue
            if pos.qty == 0:
                continue
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            intent = OrderIntent(
                strategy_id=pos.strategy_id,
                symbol=pos.symbol,
                side=side,
                qty=abs(pos.qty),
                ts=datetime.now(timezone.utc),
                reduce_only=True,
            )
            results.append(self.place_order(intent))
        return results

    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    def get_account(self) -> dict[str, Any]:
        mtm = sum(
            p.qty * self.last_prices.get(p.symbol, p.avg_price) for p in self.positions.values()
        )
        return {"cash": self.cash, "equity": self.cash + mtm, "positions": len(self.positions)}

    def sync(self) -> dict[str, Any]:
        return {"account": self.get_account(), "positions": [p.model_dump() for p in self.get_positions()]}


class ShadowBroker(BrokerGateway):
    """
    No-submit shadow path: records intended fills vs a quote mid for slippage audit.
    Does not send orders to a real broker.
    """

    def __init__(self, starting_cash: float = 100_000.0) -> None:
        self.cash = starting_cash
        self.last_prices: dict[str, float] = {}
        self.shadow_intents: list[dict[str, Any]] = []
        self.expected_fills: list[dict[str, Any]] = []

    def set_price(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = float(price)

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        mid = self.last_prices.get(intent.symbol) or intent.limit_price or 0.0
        # Pessimistic expected fill: buy +5bps / sell -5bps vs quote
        slip = mid * 0.0005
        px = mid + slip if intent.side == OrderSide.BUY else mid - slip
        rec = {
            "id": str(uuid4()),
            "status": "shadow",
            "intent_id": intent.intent_id,
            "symbol": intent.symbol,
            "side": intent.side.value,
            "qty": intent.qty,
            "quote_mid": mid,
            "expected_fill": px,
            "expected_slippage": abs(px - mid) if mid else None,
            "submitted": False,
        }
        self.shadow_intents.append(rec)
        self.expected_fills.append(rec)
        return rec

    def cancel(self, broker_order_id: str) -> bool:
        return True

    def flatten(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    def get_positions(self) -> list[Position]:
        return []

    def get_account(self) -> dict[str, Any]:
        return {"cash": self.cash, "equity": self.cash, "positions": 0, "mode": "shadow"}

    def sync(self) -> dict[str, Any]:
        return {"account": self.get_account(), "shadow_intents": len(self.shadow_intents)}
