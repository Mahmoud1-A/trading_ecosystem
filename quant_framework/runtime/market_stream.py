"""Market stream with stale-data protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class StaleDataError(RuntimeError):
    pass


@dataclass
class MarketBar:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class MarketStream:
    max_staleness_seconds: float = 30.0
    _last_bar: dict[str, MarketBar] = field(default_factory=dict)
    _halted: bool = False
    halt_reason: str | None = None

    @property
    def halted(self) -> bool:
        return self._halted

    def ingest(self, bar: MarketBar, *, now: datetime | None = None) -> MarketBar:
        now = now or datetime.now(tz=timezone.utc)
        age = (now - bar.timestamp).total_seconds()
        if age > self.max_staleness_seconds:
            self._halted = True
            self.halt_reason = f"stale_data age={age:.1f}s > {self.max_staleness_seconds}"
            raise StaleDataError(self.halt_reason)
        self._last_bar[bar.symbol] = bar
        return bar

    def last(self, symbol: str) -> MarketBar | None:
        return self._last_bar.get(symbol)

    def clear_halt(self) -> None:
        self._halted = False
        self.halt_reason = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "halted": self._halted,
            "halt_reason": self.halt_reason,
            "symbols": list(self._last_bar.keys()),
        }
