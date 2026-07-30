from __future__ import annotations

from trading_ecosystem.common.contracts import (
    Bar,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    Signal,
    SignalAction,
)
from trading_ecosystem.strategies.base import Strategy


def _atr(bars: list[Bar], period: int) -> float | None:
    if len(bars) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        b = bars[i]
        tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
        trs.append(tr)
    window = trs[-period:]
    return sum(window) / period if window else None


class DonchianATR(Strategy):
    """Classic long-only Donchian breakout with ATR stop — trend following template."""

    def generate_signal(self, bar: Bar, position: Position | None) -> Signal | None:
        channel = int(self.params.get("channel_period", 20))
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_stop_mult", 2.5))
        hist = self.history(bar.symbol)
        if len(hist) < max(channel, atr_period) + 2:
            return None

        # Prior channel excludes current bar (no look-ahead)
        prior = hist[-(channel + 1) : -1]
        if len(prior) < channel:
            return None
        upper = max(b.high for b in prior)
        lower = min(b.low for b in prior)
        atr = _atr(hist, atr_period)
        if atr is None or atr <= 0:
            return None

        in_pos = position is not None and position.qty > 0

        if not in_pos and bar.close > upper:
            stop = bar.close - atr_mult * atr
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                ts=bar.ts,
                action=SignalAction.ENTER_LONG,
                stop_price=stop,
                meta={"upper": upper, "atr": atr},
            )

        if in_pos:
            stop = position.stop_price
            if stop is None:
                stop = bar.close - atr_mult * atr
            # Trailing: raise stop with ATR from close
            trail = bar.close - atr_mult * atr
            if trail > (stop or 0):
                stop = trail
            exit_stop = bar.low <= (stop or 0)
            exit_channel = bar.close < lower
            if exit_stop or exit_channel:
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    ts=bar.ts,
                    action=SignalAction.EXIT_LONG,
                    stop_price=stop,
                    meta={"reason": "stop" if exit_stop else "channel"},
                )
            # Emit hold with updated stop via meta for engine to trail
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
        risk_fraction = float(self.params.get("risk_fraction", 0.0075))

        if signal.action == SignalAction.ENTER_LONG:
            stop = signal.stop_price
            if stop is None or stop >= bar.close:
                return []
            risk_per_share = bar.close - stop
            # Provisional qty; MRM will clip further using account equity
            # Strategy uses unit equity placeholder 100k if not injected
            equity_hint = float(self.params.get("equity_hint", 100_000.0))
            qty = (equity_hint * risk_fraction) / risk_per_share
            qty = max(1.0, float(int(qty)))
            return [
                OrderIntent(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    qty=qty,
                    order_type=OrderType.MARKET,
                    stop_price=stop,
                    ts=bar.ts,
                    meta={"signal": signal.action.value},
                )
            ]

        if signal.action == SignalAction.EXIT_LONG and position and position.qty > 0:
            return [
                OrderIntent(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    qty=abs(position.qty),
                    order_type=OrderType.MARKET,
                    ts=bar.ts,
                    reduce_only=True,
                    meta={"signal": signal.action.value},
                )
            ]

        if signal.action == SignalAction.HOLD and signal.stop_price and position:
            position.stop_price = signal.stop_price
        return []
