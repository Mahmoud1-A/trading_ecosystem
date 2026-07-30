from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_ecosystem.common.config import data_dir
from trading_ecosystem.common.contracts import Bar


class ParquetBarStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (data_dir() / "processed")
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, timeframe: str = "1d", *, series: str = "") -> Path:
        safe = symbol.replace("/", "_").upper()
        suffix = f"_{series}" if series and series not in {"", "total_return", "default"} else ""
        return self.root / f"{safe}_{timeframe}{suffix}.parquet"

    def write_bars(self, bars: list[Bar], *, series: str = "") -> dict[str, int]:
        if not bars:
            return {}
        df = pd.DataFrame([b.model_dump() for b in bars])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        counts: dict[str, int] = {}
        for (symbol, timeframe), group in df.groupby(["symbol", "timeframe"], sort=False):
            path = self.path_for(str(symbol), str(timeframe), series=series)
            g = group.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
            if path.exists():
                old = pd.read_parquet(path)
                old["ts"] = pd.to_datetime(old["ts"], utc=True)
                merged = (
                    pd.concat([old, g], ignore_index=True)
                    .sort_values("ts")
                    .drop_duplicates(subset=["ts"], keep="last")
                )
            else:
                merged = g
            merged.to_parquet(path, index=False)
            label = series or "total_return"
            counts[f"{symbol}:{timeframe}:{label}"] = len(merged)
            prov = path.with_suffix(".provenance.json")
            prov.write_text(
                json.dumps(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "series": label,
                        "n_bars": int(len(merged)),
                        "path": str(path),
                        "source": "yfinance",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return counts

    def load_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
        *,
        series: str = "",
    ) -> list[Bar]:
        path = self.path_for(symbol, timeframe, series=series)
        if not path.exists() and series:
            path = self.path_for(symbol, timeframe, series="")
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if start:
            df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
        bars: list[Bar] = []
        for row in df.itertuples(index=False):
            bars.append(
                Bar(
                    symbol=str(row.symbol),
                    ts=row.ts.to_pydatetime(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    timeframe=str(row.timeframe),
                )
            )
        return bars

    def load_panel(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
        *,
        series: str = "",
    ) -> dict[str, list[Bar]]:
        return {
            s: self.load_bars(s, timeframe=timeframe, start=start, end=end, series=series)
            for s in symbols
        }
