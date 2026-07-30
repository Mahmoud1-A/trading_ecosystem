from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import yfinance as yf

from trading_ecosystem.common.contracts import Bar


def _ensure_utc(ts: pd.Timestamp | datetime) -> datetime:
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts.tz_convert("UTC").to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_bars(
        self,
        symbols: Iterable[str],
        start: str | datetime,
        end: str | datetime | None = None,
        timeframe: str = "1d",
    ) -> list[Bar]:
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    """Free market data provider backed by Yahoo Finance."""

    _INTERVAL_MAP = {
        "1d": "1d",
        "1h": "1h",
        "1H": "1h",
        "4h": "1h",  # yfinance lacks native 4h; resample downstream if needed
        "1m": "1m",
    }

    def fetch_bars(
        self,
        symbols: Iterable[str],
        start: str | datetime,
        end: str | datetime | None = None,
        timeframe: str = "1d",
        *,
        auto_adjust: bool = True,
    ) -> list[Bar]:
        symbol_list = list(symbols)
        if not symbol_list:
            return []

        interval = self._INTERVAL_MAP.get(timeframe, timeframe)
        data = yf.download(
            tickers=symbol_list,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=bool(auto_adjust),
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if data is None or data.empty:
            return []

        bars: list[Bar] = []
        if len(symbol_list) == 1:
            symbol = symbol_list[0]
            frame = data.copy()
            frame.columns = [str(c).lower() for c in frame.columns]
            bars.extend(self._frame_to_bars(symbol, frame, timeframe))
            return bars

        # Multi-ticker: columns are MultiIndex (ticker, field) or (field, ticker)
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            if set(symbol_list).issubset(set(level0)):
                for symbol in symbol_list:
                    if symbol not in level0:
                        continue
                    frame = data[symbol].copy()
                    frame.columns = [str(c).lower() for c in frame.columns]
                    bars.extend(self._frame_to_bars(symbol, frame, timeframe))
            else:
                for symbol in symbol_list:
                    try:
                        frame = data.xs(symbol, axis=1, level=1).copy()
                    except KeyError:
                        continue
                    frame.columns = [str(c).lower() for c in frame.columns]
                    bars.extend(self._frame_to_bars(symbol, frame, timeframe))
        return bars

    def fetch_dual_series(
        self,
        symbols: Iterable[str],
        start: str | datetime,
        end: str | datetime | None = None,
        timeframe: str = "1d",
    ) -> dict[str, list[Bar]]:
        """Fetch total_return (auto_adjust) and raw (no auto_adjust) series."""
        return {
            "total_return": self.fetch_bars(symbols, start, end, timeframe, auto_adjust=True),
            "raw": self.fetch_bars(symbols, start, end, timeframe, auto_adjust=False),
        }
    def _frame_to_bars(self, symbol: str, frame: pd.DataFrame, timeframe: str) -> list[Bar]:
        required = {"open", "high", "low", "close"}
        cols = {c.lower(): c for c in frame.columns}
        if not required.issubset(cols):
            return []
        out: list[Bar] = []
        for idx, row in frame.dropna(subset=["open", "high", "low", "close"]).iterrows():
            vol = float(row["volume"]) if "volume" in frame.columns and pd.notna(row.get("volume")) else 0.0
            out.append(
                Bar(
                    symbol=symbol,
                    ts=_ensure_utc(idx),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=vol,
                    timeframe=timeframe,
                )
            )
        return out
