"""Position tracking for event-driven portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.models import Side
from engine.fills import Fill


@dataclass
class Position:
    """Open (or flat) position on a single symbol/contract."""

    symbol: str
    contract: str
    quantity: float = 0.0  # signed: >0 long, <0 short
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    realized_gross_pnl: float = 0.0
    commission_paid: float = 0.0
    spread_paid: float = 0.0
    slippage_paid: float = 0.0
    financing_paid: float = 0.0
    rollover_paid: float = 0.0
    multiplier: float = 1.0
    opened_at: pd.Timestamp | None = None
    updated_at: pd.Timestamp | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) <= 1e-12

    @property
    def side(self) -> Side | None:
        if self.is_flat:
            return None
        return Side.BUY if self.quantity > 0 else Side.SELL

    def unrealized_pnl(self, mark: float) -> float:
        if self.is_flat:
            return 0.0
        return (mark - self.avg_price) * self.quantity * self.multiplier

    def apply_fill(self, fill: Fill, *, multiplier: float) -> float:
        """
        Apply a fill; return realized gross PnL from any closed quantity
        (before friction; friction is tracked separately on the fill).
        """
        self.multiplier = multiplier
        self.updated_at = fill.fill_timestamp
        signed_qty = fill.quantity if fill.side == Side.BUY else -fill.quantity
        realized = 0.0

        if self.is_flat or (self.quantity > 0 and signed_qty > 0) or (self.quantity < 0 and signed_qty < 0):
            # Increasing / opening
            new_qty = self.quantity + signed_qty
            if self.is_flat:
                self.avg_price = fill.price
                self.opened_at = fill.fill_timestamp
            else:
                self.avg_price = (
                    abs(self.quantity) * self.avg_price + fill.quantity * fill.price
                ) / (abs(self.quantity) + fill.quantity)
            self.quantity = new_qty
        else:
            # Reducing / reversing
            close_qty = min(abs(self.quantity), fill.quantity)
            direction = 1.0 if self.quantity > 0 else -1.0
            realized = (fill.price - self.avg_price) * direction * close_qty * multiplier
            self.realized_gross_pnl += realized
            self.realized_pnl += realized - (
                fill.costs.commission
                + fill.costs.spread_cost
                + fill.costs.slippage_cost
                + fill.costs.financing_cost
                + fill.costs.rollover_cost
                + fill.costs.market_impact_cost
            ) * (close_qty / fill.quantity if fill.quantity else 1.0)

            remaining_pos = abs(self.quantity) - close_qty
            leftover_fill = fill.quantity - close_qty
            if remaining_pos <= 1e-12:
                self.quantity = 0.0
                self.avg_price = 0.0
                self.opened_at = None
                if leftover_fill > 1e-12:
                    # Reverse
                    self.quantity = leftover_fill if fill.side == Side.BUY else -leftover_fill
                    self.avg_price = fill.price
                    self.opened_at = fill.fill_timestamp
            else:
                self.quantity = remaining_pos * direction

        self.commission_paid += fill.costs.commission
        self.spread_paid += fill.costs.spread_cost
        self.slippage_paid += fill.costs.slippage_cost
        self.financing_paid += fill.costs.financing_cost
        self.rollover_paid += fill.costs.rollover_cost
        self.contract = fill.contract
        return realized
