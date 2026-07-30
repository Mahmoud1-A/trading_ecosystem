from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trading_ecosystem.common.contracts import Bar, Fill, OrderIntent, Position, Signal


class Strategy(ABC):
    # Strategy agents emit intents only - never talk to the broker.

    def __init__(self, strategy_id: str, symbols: list[str], params: dict[str, Any] | None = None) -> None:
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.params = params or {}
        self._history: dict[str, list[Bar]] = {s: [] for s in symbols}

    def on_bar(self, bar: Bar, position: Position | None) -> list[OrderIntent]:
        if bar.symbol not in self._history:
            self._history[bar.symbol] = []
        self._history[bar.symbol].append(bar)
        signal = self.generate_signal(bar, position)
        if signal is None:
            return []
        return self.signal_to_intents(signal, bar, position)

    def on_fill(self, fill: Fill) -> None:
        return None

    @abstractmethod
    def generate_signal(self, bar: Bar, position: Position | None) -> Signal | None:
        raise NotImplementedError

    @abstractmethod
    def signal_to_intents(
        self,
        signal: Signal,
        bar: Bar,
        position: Position | None,
    ) -> list[OrderIntent]:
        raise NotImplementedError

    def history(self, symbol: str) -> list[Bar]:
        return self._history.get(symbol, [])
