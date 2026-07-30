from __future__ import annotations

from trading_ecosystem.common.contracts import Bar, OrderIntent, Position, Signal, SignalAction
from trading_ecosystem.strategies.base import Strategy
from trading_ecosystem.strategies.indicators import atr, sma, stdev
from trading_ecosystem.strategies.sizing import long_only_intents


class ZScoreRevert(Strategy):
    """Buy when close z-score vs SMA is deeply negative; exit near mean — single-factor MR."""

    def generate_signal(self, bar: Bar, position: Position | None) -> Signal | None:
        lookback = int(self.params.get("lookback", 20))
        entry_z = float(self.params.get("entry_z", 2.0))
        exit_z = float(self.params.get("exit_z", 0.5))
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_stop_mult", 2.5))
        hist = self.history(bar.symbol)
        need = max(lookback, atr_period) + 1
        if len(hist) < need:
            return None

        closes = [b.close for b in hist]
        mean = sma(closes, lookback)
        sd = stdev(closes, lookback)
        a = atr(hist, atr_period)
        if mean is None or sd is None or sd <= 1e-12 or a is None or a <= 0:
            return None
        z = (bar.close - mean) / sd

        in_pos = position is not None and position.qty > 0
        if not in_pos and z <= -entry_z:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                ts=bar.ts,
                action=SignalAction.ENTER_LONG,
                stop_price=bar.close - atr_mult * a,
                meta={"z": z},
            )

        if in_pos:
            stop = position.stop_price or (bar.close - atr_mult * a)
            if bar.low <= stop or z >= -exit_z:
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    ts=bar.ts,
                    action=SignalAction.EXIT_LONG,
                    stop_price=stop,
                    meta={"z": z},
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
