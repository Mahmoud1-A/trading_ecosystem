from __future__ import annotations

from datetime import date, datetime, timezone

from trading_ecosystem.common.contracts import AccountState, Fill, OrderSide, Position


class PortfolioBook:
    """Tracks cash, equity, and per-strategy positions (long/short capable)."""

    def __init__(self, starting_equity: float, trading_day: date | None = None) -> None:
        day = trading_day or datetime.now(timezone.utc).date()
        self.state = AccountState(
            equity=starting_equity,
            cash=starting_equity,
            starting_equity=starting_equity,
            peak_equity=starting_equity,
            day_start_equity=starting_equity,
            trading_day=day,
        )
        self.stops: dict[str, float] = {}

    def maybe_roll_day(self, ts: datetime) -> None:
        d = ts.astimezone(timezone.utc).date() if ts.tzinfo else ts.replace(tzinfo=timezone.utc).date()
        if d != self.state.trading_day:
            self.state.trading_day = d
            self.state.day_start_equity = self.state.equity
            self.state.daily_halted = False

    def set_stop(self, strategy_id: str, symbol: str, stop_price: float | None) -> None:
        key = self.state.position_key(strategy_id, symbol)
        if stop_price is None:
            self.stops.pop(key, None)
        else:
            self.stops[key] = stop_price
            if key in self.state.positions:
                self.state.positions[key].stop_price = stop_price

    def mark_to_market(self, prices: dict[str, float]) -> None:
        unrealized = 0.0
        for pos in self.state.positions.values():
            px = prices.get(pos.symbol, pos.avg_price)
            pos.unrealized_pnl = (px - pos.avg_price) * pos.qty
            unrealized += pos.unrealized_pnl
        invested = sum(p.qty * p.avg_price for p in self.state.positions.values())
        # equity = cash + MTM value of positions (cash already reduced on buys)
        self.state.equity = self.state.cash + invested + unrealized
        # Simplify: cash holds free cash; positions marked separately
        mtm_value = sum(p.qty * prices.get(p.symbol, p.avg_price) for p in self.state.positions.values())
        self.state.equity = self.state.cash + mtm_value
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

    def apply_fill(self, fill: Fill, stop_price: float | None = None) -> None:
        self.maybe_roll_day(fill.ts)
        key = self.state.position_key(fill.strategy_id, fill.symbol)
        pos = self.state.positions.get(
            key,
            Position(symbol=fill.symbol, strategy_id=fill.strategy_id),
        )

        if fill.side == OrderSide.BUY:
            new_qty = pos.qty + fill.qty
            if pos.qty >= 0:
                total = pos.avg_price * pos.qty + fill.price * fill.qty
                pos.qty = new_qty
                pos.avg_price = total / pos.qty if pos.qty else 0.0
            else:
                # Cover short partially/fully
                cover = min(fill.qty, abs(pos.qty))
                pnl = (pos.avg_price - fill.price) * cover
                self.state.realized_pnl += pnl
                self.state.cash += pnl
                leftover = fill.qty - cover
                pos.qty += fill.qty
                if pos.qty > 0 and leftover > 0:
                    pos.avg_price = fill.price
                elif pos.qty == 0:
                    pos.avg_price = 0.0
            self.state.cash -= fill.price * fill.qty
            self.state.cash -= fill.commission
        else:
            # SELL
            if pos.qty > 0:
                close_qty = min(fill.qty, pos.qty)
                pnl = (fill.price - pos.avg_price) * close_qty
                self.state.realized_pnl += pnl
                self.state.cash += fill.price * close_qty
                pos.qty -= close_qty
                leftover = fill.qty - close_qty
                if leftover > 0:
                    # Flip short
                    pos.qty = -leftover
                    pos.avg_price = fill.price
                    self.state.cash += fill.price * leftover  # short proceeds
                if pos.qty == 0:
                    pos.avg_price = 0.0
            else:
                # Add short
                total_qty = abs(pos.qty) + fill.qty
                total = pos.avg_price * abs(pos.qty) + fill.price * fill.qty
                pos.qty = -total_qty
                pos.avg_price = total / total_qty if total_qty else 0.0
                self.state.cash += fill.price * fill.qty
            self.state.cash -= fill.commission

        if abs(pos.qty) < 1e-12:
            pos.qty = 0.0
            pos.avg_price = 0.0
            self.state.positions.pop(key, None)
            self.stops.pop(key, None)
        else:
            if stop_price is not None:
                pos.stop_price = stop_price
                self.stops[key] = stop_price
            self.state.positions[key] = pos

        self.mark_to_market({fill.symbol: fill.price})
