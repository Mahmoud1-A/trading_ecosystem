from __future__ import annotations

from trading_ecosystem.common.contracts import Bar, OrderIntent, Position, Signal, SignalAction
from trading_ecosystem.strategies.base import Strategy
from trading_ecosystem.strategies.indicators import atr
from trading_ecosystem.strategies.sizing import long_only_intents


def _in_window(ts, window_code: int) -> bool:
    """window_code: 1-12 = calendar month; 101-104 = Q1-Q4."""
    if 1 <= window_code <= 12:
        return ts.month == window_code
    if 101 <= window_code <= 104:
        quarter = (ts.month - 1) // 3 + 1
        return quarter == (window_code - 100)
    return False


class SeasonalityWindow(Strategy):
    """Stay long only inside a simple calendar window; flat otherwise."""

    def generate_signal(self, bar: Bar, position: Position | None) -> Signal | None:
        window_code = int(self.params.get("window_code", 11))
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_stop_mult", 2.5))
        hist = self.history(bar.symbol)
        if len(hist) < atr_period + 2:
            return None
        a = atr(hist, atr_period)
        if a is None or a <= 0:
            return None

        active = _in_window(bar.ts, window_code)
        in_pos = position is not None and position.qty > 0

        if active and not in_pos:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                ts=bar.ts,
                action=SignalAction.ENTER_LONG,
                stop_price=bar.close - atr_mult * a,
                meta={"window_code": window_code},
            )

        if in_pos:
            stop = position.stop_price or (bar.close - atr_mult * a)
            trail = bar.close - atr_mult * a
            if trail > stop:
                stop = trail
            if (not active) or bar.low <= stop:
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    ts=bar.ts,
                    action=SignalAction.EXIT_LONG,
                    stop_price=stop,
                    meta={"reason": "window_end" if not active else "stop"},
                )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                ts=bar.ts,
                action=SignalAction.HOLD,
                stop_price=stop,
            )
        return None

    def signal_to_intents(
        self,
        signal: Signal,
        bar: Bar,
        position: Position | None,
    ) -> list[OrderIntent]:
        return long_only_intents(self.strategy_id, signal, bar, position, self.params)
