from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from trading_ecosystem.common.contracts import OrderIntent, OrderSide, OrderStatus, Position
from trading_ecosystem.execution.broker import BrokerGateway

logger = logging.getLogger(__name__)


class AlpacaPaperGateway(BrokerGateway):
    """Alpaca Paper Trading adapter. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY."""

    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}
        self._client = None
        self._order_map: dict[str, str] = {}  # intent_id -> broker id
        if self.api_key and self.secret_key:
            self._init_client()
        else:
            logger.warning("Alpaca credentials missing; gateway in dry-run mode")

    def _init_client(self) -> None:
        try:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            logger.info("Alpaca TradingClient initialized (paper=%s)", self.paper)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to init Alpaca client: %s", exc)
            self._client = None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        if not self._client:
            oid = f"dry-{uuid4()}"
            self._order_map[intent.intent_id] = oid
            return {
                "id": oid,
                "status": OrderStatus.SUBMITTED.value,
                "intent_id": intent.intent_id,
                "dry_run": True,
            }

        from alpaca.trading.enums import OrderSide as ASide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = ASide.BUY if intent.side == OrderSide.BUY else ASide.SELL
        req = MarketOrderRequest(
            symbol=intent.symbol,
            qty=intent.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=intent.intent_id[:48],
        )
        try:
            order = self._client.submit_order(req)
        except Exception as exc:  # noqa: BLE001
            # Keep the paper loop alive (e.g. insufficient buying power).
            logger.warning(
                "Alpaca order rejected strategy=%s symbol=%s qty=%s: %s",
                intent.strategy_id,
                intent.symbol,
                intent.qty,
                exc,
            )
            return {
                "id": None,
                "status": OrderStatus.REJECTED.value,
                "intent_id": intent.intent_id,
                "error": str(exc),
                "rejected": True,
            }
        self._order_map[intent.intent_id] = str(order.id)
        return {
            "id": str(order.id),
            "status": OrderStatus.SUBMITTED.value,
            "intent_id": intent.intent_id,
            "raw_status": str(order.status),
        }

    def cancel(self, broker_order_id: str) -> bool:
        if not self._client:
            return False
        try:
            self._client.cancel_order_by_id(broker_order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("cancel failed: %s", exc)
            return False

    def flatten(self, symbol: str | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pos in self.get_positions():
            if symbol and pos.symbol != symbol:
                continue
            if abs(pos.qty) < 1e-12:
                continue
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            intent = OrderIntent(
                strategy_id=pos.strategy_id or "flatten",
                symbol=pos.symbol,
                side=side,
                qty=abs(pos.qty),
                ts=datetime.now(timezone.utc),
                reduce_only=True,
            )
            results.append(self.place_order(intent))
        return results

    def get_positions(self) -> list[Position]:
        if not self._client:
            return []
        out: list[Position] = []
        for p in self._client.get_all_positions():
            out.append(
                Position(
                    symbol=str(p.symbol),
                    strategy_id="broker",
                    qty=float(p.qty),
                    avg_price=float(p.avg_entry_price),
                    unrealized_pnl=float(p.unrealized_pl or 0.0),
                )
            )
        return out

    def get_account(self) -> dict[str, Any]:
        if not self._client:
            return {"configured": False, "equity": None, "cash": None}
        acct = self._client.get_account()
        return {
            "configured": True,
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": str(acct.status),
            "paper": self.paper,
        }

    def sync(self) -> dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "account": self.get_account(),
            "positions": [p.model_dump() for p in self.get_positions()],
            "order_map_size": len(self._order_map),
        }
