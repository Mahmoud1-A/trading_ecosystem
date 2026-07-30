from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING_RISK = "pending_risk"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    WORKING = "working"
    FILLED = "filled"
    CANCELED = "canceled"
    FAILED = "failed"


class SignalAction(str, Enum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"
    FLATTEN = "flatten"
    HOLD = "hold"


class Bar(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timeframe: str = "1d"

    model_config = {"frozen": True}


class Signal(BaseModel):
    strategy_id: str
    symbol: str
    ts: datetime
    action: SignalAction
    strength: float = 1.0
    stop_price: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class OrderIntent(BaseModel):
    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    ts: datetime
    reduce_only: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Fill(BaseModel):
    fill_id: str = Field(default_factory=lambda: str(uuid4()))
    intent_id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    ts: datetime
    commission: float = 0.0
    slippage: float = 0.0


class Position(BaseModel):
    symbol: str
    strategy_id: str
    qty: float = 0.0
    avg_price: float = 0.0
    stop_price: float | None = None
    unrealized_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.qty * self.avg_price)

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12


class AccountState(BaseModel):
    equity: float
    cash: float
    starting_equity: float
    peak_equity: float
    day_start_equity: float
    trading_day: date
    positions: dict[str, Position] = Field(default_factory=dict)
    halted: bool = False
    daily_halted: bool = False
    realized_pnl: float = 0.0
    # Optional broker buying power (Alpaca). When set, MRM clips new buys to it.
    buying_power: float | None = None

    def position_key(self, strategy_id: str, symbol: str) -> str:
        return f"{strategy_id}:{symbol}"

    def gross_exposure(self) -> float:
        return sum(abs(p.qty * p.avg_price) for p in self.positions.values())

    def daily_drawdown_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity * 100.0)

    def total_drawdown_pct(self) -> float:
        ref = self.peak_equity if self.peak_equity > 0 else self.starting_equity
        if ref <= 0:
            return 0.0
        return max(0.0, (ref - self.equity) / ref * 100.0)


class RiskDecision(BaseModel):
    approved: bool
    intent_id: str
    reason: str
    mode: str = "HARD"
    adjusted_qty: float | None = None
    ts: datetime
    meta: dict[str, Any] = Field(default_factory=dict)
