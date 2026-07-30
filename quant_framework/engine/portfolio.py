"""Event-driven portfolio accounting (not DataFrame arithmetic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from engine.fills import Fill
from engine.orders import Order
from engine.positions import Position


@dataclass
class TradeReport:
    """Closed-trade attribution report."""

    trade_id: str
    symbol: str
    contract: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    gross_pnl: float
    commission: float
    spread_cost: float
    slippage_cost: float
    financing_cost: float
    rollover_cost: float
    net_pnl: float
    side: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "contract": self.contract,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "gross_pnl": self.gross_pnl,
            "commission": self.commission,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "financing_cost": self.financing_cost,
            "rollover_cost": self.rollover_cost,
            "net_pnl": self.net_pnl,
        }


@dataclass
class Portfolio:
    """
    Sequential portfolio state machine.

    Cash and positions update only through explicit fill events.
    """

    starting_equity: float
    cash: float | None = None
    positions: dict[str, Position] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    trades: list[TradeReport] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    multipliers: dict[str, float] = field(default_factory=dict)
    last_marks: dict[str, float] = field(default_factory=dict)
    _entry_lots: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)
    _trade_seq: int = 0

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = float(self.starting_equity)

    def set_multiplier(self, symbol: str, multiplier: float) -> None:
        self.multipliers[symbol] = float(multiplier)

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, contract="")
        return self.positions[symbol]

    def register_order(self, order: Order) -> None:
        self.orders[order.order_id] = order

    def equity(self) -> float:
        upnl = 0.0
        for sym, pos in self.positions.items():
            mark = self.last_marks.get(sym, pos.avg_price)
            upnl += pos.unrealized_pnl(mark)
        return float(self.cash or 0.0) + upnl

    def mark_to_market(self, ts: pd.Timestamp, marks: dict[str, float]) -> float:
        self.last_marks.update(marks)
        eq = self.equity()
        self.equity_curve.append((ts, eq))
        return eq

    def apply_fill(self, fill: Fill) -> float:
        """Apply fill sequentially; return realized net impact on cash from friction + closed PnL."""
        mult = self.multipliers.get(fill.symbol, 1.0)
        pos = self.get_position(fill.symbol)
        if not pos.contract:
            pos.contract = fill.contract

        # Track entry lots for trade reports
        signed = fill.quantity if fill.side.value == "BUY" else -fill.quantity
        lots = self._entry_lots.setdefault(fill.symbol, [])

        closed_gross = 0.0
        commission = fill.costs.commission
        spread = fill.costs.spread_cost
        slip = fill.costs.slippage_cost
        fin = fill.costs.financing_cost
        roll = fill.costs.rollover_cost
        impact = fill.costs.market_impact_cost
        friction = commission + spread + slip + fin + roll + impact

        if pos.is_flat or (pos.quantity > 0 and signed > 0) or (pos.quantity < 0 and signed < 0):
            lots.append(
                {
                    "qty": fill.quantity,
                    "price": fill.price,
                    "time": fill.fill_timestamp,
                    "side": "LONG" if signed > 0 else "SHORT",
                    "commission": commission,
                    "spread_cost": spread,
                    "slippage_cost": slip,
                    "financing_cost": fin,
                    "rollover_cost": roll,
                }
            )
            pos.apply_fill(fill, multiplier=mult)
            # Opening: friction reduces cash immediately
            assert self.cash is not None
            self.cash -= friction
            self.fills.append(fill)
            self.last_marks[fill.symbol] = fill.price
            return -friction

        # Reducing / closing against FIFO lots
        remaining = fill.quantity
        exit_side = "LONG" if pos.quantity > 0 else "SHORT"
        while remaining > 1e-12 and lots:
            lot = lots[0]
            close_qty = min(lot["qty"], remaining)
            direction = 1.0 if exit_side == "LONG" else -1.0
            gross = (fill.price - float(lot["price"])) * direction * close_qty * mult
            closed_gross += gross
            frac_exit = close_qty / fill.quantity
            # Allocate exit friction + entry friction proportionally for the closed lot
            entry_frac = close_qty / float(lot["qty"]) if lot["qty"] else 1.0
            entry_friction = (
                float(lot["commission"])
                + float(lot["spread_cost"])
                + float(lot["slippage_cost"])
                + float(lot["financing_cost"])
                + float(lot["rollover_cost"])
            ) * entry_frac
            exit_friction = friction * frac_exit
            net = gross - entry_friction - exit_friction
            self._trade_seq += 1
            self.trades.append(
                TradeReport(
                    trade_id=f"T{self._trade_seq:06d}",
                    symbol=fill.symbol,
                    contract=fill.contract,
                    quantity=close_qty,
                    entry_price=float(lot["price"]),
                    exit_price=fill.price,
                    entry_time=lot["time"],
                    exit_time=fill.fill_timestamp,
                    gross_pnl=gross,
                    commission=float(lot["commission"]) * entry_frac + commission * frac_exit,
                    spread_cost=float(lot["spread_cost"]) * entry_frac + spread * frac_exit,
                    slippage_cost=float(lot["slippage_cost"]) * entry_frac + slip * frac_exit,
                    financing_cost=float(lot["financing_cost"]) * entry_frac + fin * frac_exit,
                    rollover_cost=float(lot["rollover_cost"]) * entry_frac + roll * frac_exit,
                    net_pnl=net,
                    side=exit_side,
                )
            )
            # Reduce entry friction already booked at open from double-count:
            # entry friction was deducted from cash at open; exit friction deducted now;
            # cash gets +gross at close.
            lot["qty"] -= close_qty
            if lot["qty"] <= 1e-12:
                lots.pop(0)
            else:
                # Scale remaining entry cost fields
                for k in ("commission", "spread_cost", "slippage_cost", "financing_cost", "rollover_cost"):
                    lot[k] = float(lot[k]) * (1.0 - entry_frac)
            remaining -= close_qty

        pos.apply_fill(fill, multiplier=mult)
        assert self.cash is not None
        self.cash += closed_gross - friction
        # If reversed and leftover opened a new lot inside apply_fill:
        if remaining <= 1e-12 and not pos.is_flat and not lots:
            # Reversal leftover already applied to position; book new lot
            pass
        if not pos.is_flat and abs(pos.quantity) > 1e-12:
            # Ensure lot book matches residual position after reverse
            signed_pos = pos.quantity
            lot_signed = sum(l["qty"] if l["side"] == "LONG" else -l["qty"] for l in lots)
            if abs(lot_signed - signed_pos) > 1e-6:
                lots.clear()
                lots.append(
                    {
                        "qty": abs(pos.quantity),
                        "price": pos.avg_price,
                        "time": fill.fill_timestamp,
                        "side": "LONG" if pos.quantity > 0 else "SHORT",
                        "commission": 0.0,
                        "spread_cost": 0.0,
                        "slippage_cost": 0.0,
                        "financing_cost": 0.0,
                        "rollover_cost": 0.0,
                    }
                )

        self.fills.append(fill)
        self.last_marks[fill.symbol] = fill.price
        return closed_gross - friction
