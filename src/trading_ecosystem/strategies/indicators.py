from __future__ import annotations

from trading_ecosystem.common.contracts import Bar


def atr(bars: list[Bar], period: int) -> float | None:
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


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def stdev(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    var = sum((x - mean) ** 2 for x in window) / period
    return var**0.5
