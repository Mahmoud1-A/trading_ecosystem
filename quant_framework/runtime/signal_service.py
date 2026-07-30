"""Signal service with duplicate-signal prevention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from brokers.protocol import OrderSide


@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    side: OrderSide
    quantity: float
    bar_timestamp: str
    strategy_id: str = "default"

    def fingerprint(self) -> str:
        return f"{self.strategy_id}|{self.symbol}|{self.side.value}|{self.quantity}|{self.bar_timestamp}"


@dataclass
class SignalService:
    _seen: set[str] = field(default_factory=set)
    duplicate_count: int = 0
    emitted: list[Signal] = field(default_factory=list)

    def emit(self, signal: Signal) -> Signal | None:
        fp = signal.fingerprint()
        if fp in self._seen or signal.signal_id in {s.signal_id for s in self.emitted}:
            self.duplicate_count += 1
            return None
        self._seen.add(fp)
        self.emitted.append(signal)
        return signal

    def reset_session(self) -> None:
        self._seen.clear()
