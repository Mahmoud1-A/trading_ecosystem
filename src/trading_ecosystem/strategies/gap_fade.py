from __future__ import annotations

from trading_ecosystem.common.contracts import Bar, OrderIntent, Position, Signal, SignalAction
from trading_ecosystem.strategies.base import Strategy
from trading_ecosystem.strategies.indicators import atr
from trading_ecosystem.strategies.sizing import long_only_intents


class GapFade(Strategy):
    """
    Fade a down-gap (Post-Open execution contract).

    Decision uses today's open vs prior close. ATR/stop use history *before* the
    signal bar. Entries are marked fill_mode=next_bar_open so the engine does not
    fill at the same open used in the gap condition (Auction Mode unsupported).
    """

    def __init__(self, strategy_id: str, symbols: list[str], params: dict | None = None) -> None:
        super().__init__(strategy_id, symbols, params)
        self._bars_held: dict[str, int] = {}

    def generate_signal(self, bar: Bar, position: Position | None) -> Signal | None:
        gap_pct = float(self.params.get("gap_pct", 0.015))
        hold_bars = int(self.params.get("hold_bars", 2))
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_stop_mult", 2.0))
        hist = self.history(bar.symbol)
        if len(hist) < atr_period + 2:
            return None

        # hist includes current bar (appended in on_bar); prior bar is [-2]
        prev = hist[-2]
        gap = (bar.open - prev.close) / prev.close if prev.close else 0.0
        # ATR from bars strictly before today — no same-bar look-ahead
        prior = hist[:-1]
        a = atr(prior, atr_period)
        if a is None or a <= 0:
            return None

        in_pos = position is not None and position.qty > 0
        if not in_pos and gap <= -gap_pct:
            self._bars_held[bar.symbol] = 0
            stop = float(bar.open) - atr_mult * a
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                ts=bar.ts,
                action=SignalAction.ENTER_LONG,
                stop_price=stop,
                meta={
                    "gap": gap,
                    "execution_model": "post_open",
                    "fill_mode": "next_bar_open",
                    "auction_mode": "unsupported",
                    "signal_price_ref": "open",
                },
            )

        if in_pos:
            held = self._bars_held.get(bar.symbol, 0) + 1
            self._bars_held[bar.symbol] = held
            stop = position.stop_price or (float(bar.open) - atr_mult * a)
            if bar.low <= stop or held >= hold_bars:
                self._bars_held[bar.symbol] = 0
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    ts=bar.ts,
                    action=SignalAction.EXIT_LONG,
                    stop_price=stop,
                    meta={"held": held, "fill_mode": "same_bar_close"},
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
        params = dict(self.params)
        if (signal.meta or {}).get("signal_price_ref") == "open":
            params["price_ref"] = "open"
        intents = long_only_intents(self.strategy_id, signal, bar, position, params)
        for intent in intents:
            meta = dict(intent.meta or {})
            meta.update(signal.meta or {})
            intent.meta = meta
        return intents
