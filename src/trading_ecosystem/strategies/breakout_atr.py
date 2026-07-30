from __future__ import annotations

from trading_ecosystem.common.contracts import Bar, OrderIntent, Position, Signal, SignalAction
from trading_ecosystem.strategies.base import Strategy
from trading_ecosystem.strategies.indicators import atr
from trading_ecosystem.strategies.sizing import long_only_intents


class BreakoutATR(Strategy):
    """Long-only N-day high breakout with ATR stop — single-factor trend."""

    def generate_signal(self, bar: Bar, position: Position | None) -> Signal | None:
        lookback = int(self.params.get("lookback", 20))
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_stop_mult", 2.5))
        hist = self.history(bar.symbol)
        if len(hist) < max(lookback, atr_period) + 2:
            return None

        prior = hist[-(lookback + 1) : -1]
        if len(prior) < lookback:
            return None
        upper = max(b.high for b in prior)
        lower = min(b.low for b in prior)
        a = atr(hist, atr_period)
        if a is None or a <= 0:
            return None

        in_pos = position is not None and position.qty > 0
        if not in_pos and bar.close > upper:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                ts=bar.ts,
                action=SignalAction.ENTER_LONG,
                stop_price=bar.close - atr_mult * a,
                meta={"upper": upper, "atr": a},
            )

        if in_pos:
            stop = position.stop_price or (bar.close - atr_mult * a)
            trail = bar.close - atr_mult * a
            if trail > stop:
                stop = trail
            if bar.low <= stop or bar.close < lower:
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    ts=bar.ts,
                    action=SignalAction.EXIT_LONG,
                    stop_price=stop,
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
