"""CFD overnight financing accrual."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from config.asset_spec import CFDAssetSpec
from config.models import Side
from engine.fills import CostBreakdown
from engine.portfolio import Portfolio


@dataclass(frozen=True)
class FinancingEvent:
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    side: str
    rate: float
    notional: float
    amount: float  # cash debit (positive reduces equity)
    triple_swap: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": str(self.timestamp),
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side,
            "rate": self.rate,
            "notional": self.notional,
            "amount": self.amount,
            "triple_swap": self.triple_swap,
        }


@dataclass
class CFDFinancingEngine:
    """
    Accrue overnight swap at broker-local rollover time.

    Distinguishes trading-day rollover boundaries from mere elapsed midnight UTC.
    """

    asset: CFDAssetSpec
    rollover_time: time = time(22, 0)
    triple_swap_weekday: int = 2  # Wednesday (Mon=0)
    _last_rollover_boundary: pd.Timestamp | None = None
    events: list[FinancingEvent] | None = None

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = []

    def _zone(self) -> ZoneInfo:
        return self.asset.broker_trading_hours.zoneinfo()

    def _boundary_containing(self, local_ts: pd.Timestamp) -> pd.Timestamp:
        tz = self._zone()
        boundary = pd.Timestamp(datetime.combine(local_ts.date(), self.rollover_time, tzinfo=tz))
        if local_ts < boundary:
            boundary -= pd.Timedelta(days=1)
        return boundary

    def rollover_boundaries_crossed(
        self,
        prev_ts: pd.Timestamp | None,
        curr_ts: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """Return broker-local rollover timestamps crossed in (prev_ts, curr_ts]."""
        tz = self._zone()
        curr_local = curr_ts.tz_convert(tz)
        if prev_ts is None:
            # First mark: no financing yet
            self._last_rollover_boundary = self._boundary_containing(curr_local)
            return []

        prev_local = prev_ts.tz_convert(tz)
        boundaries: list[pd.Timestamp] = []
        # Walk day by day from day after prev to curr
        cursor_date = prev_local.date()
        end_date = curr_local.date()
        # Start from the next potential boundary after prev
        while True:
            boundary = pd.Timestamp(datetime.combine(cursor_date, self.rollover_time, tzinfo=tz))
            if boundary <= prev_local:
                cursor_date = cursor_date + timedelta(days=1)
                continue
            if boundary > curr_local:
                break
            # Only count if the broker session is active that calendar day
            # (weekend gaps: still charge if position held — CFD weekend financing applies)
            boundaries.append(boundary)
            cursor_date = cursor_date + timedelta(days=1)
            if cursor_date > end_date + timedelta(days=1):
                break
        return boundaries

    def accrue(
        self,
        portfolio: Portfolio,
        *,
        prev_ts: pd.Timestamp | None,
        curr_ts: pd.Timestamp,
        mark_price: float,
        symbol: str,
    ) -> list[FinancingEvent]:
        """Apply financing for each crossed rollover; debit portfolio cash."""
        if not isinstance(self.asset, CFDAssetSpec):
            return []
        boundaries = self.rollover_boundaries_crossed(prev_ts, curr_ts)
        if not boundaries:
            return []

        pos = portfolio.get_position(symbol)
        if pos.is_flat:
            return []

        events: list[FinancingEvent] = []
        qty = abs(float(pos.quantity))
        side = "LONG" if pos.quantity > 0 else "SHORT"
        rate = (
            float(self.asset.overnight_swap_long)
            if pos.quantity > 0
            else float(self.asset.overnight_swap_short)
        )
        notional = qty * float(self.asset.contract_size) * float(mark_price)

        for boundary in boundaries:
            local = boundary.tz_convert(self._zone())
            triple = local.weekday() == int(self.triple_swap_weekday)
            mult = 3.0 if triple else 1.0
            # Positive amount = cost paid by trader (reduces cash)
            # Rate sign: negative long swap means payer; we take abs for cost when rate favors broker
            # Convention: overnight_swap_* is fraction of notional charged to the position holder
            # (positive => debit, negative => credit)
            amount = notional * rate * mult
            # If rate is negative for long, it's a credit (amount negative => cash increases)
            assert portfolio.cash is not None
            portfolio.cash -= amount  # debit positive cost; credit if amount negative
            pos.financing_paid += abs(amount) if amount > 0 else amount
            evt = FinancingEvent(
                timestamp=boundary,
                symbol=symbol,
                quantity=qty,
                side=side,
                rate=rate * mult,
                notional=notional,
                amount=amount,
                triple_swap=triple,
            )
            events.append(evt)
            assert self.events is not None
            self.events.append(evt)

        self._last_rollover_boundary = boundaries[-1]
        return events
