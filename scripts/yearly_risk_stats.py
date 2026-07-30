from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml
from trading_ecosystem.discovery.aggregate_book import load_portfolio_book
from trading_ecosystem.discovery.runner import _load_panel
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategy


def main() -> None:
    book = load_portfolio_book()
    members = list(book.get("members") or [])
    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    symbols_cfg = load_yaml(CONFIG_DIR / "symbols.yaml")
    symbols = sorted(
        {
            s
            for m in members
            for s in (m.get("candidate") or {}).get("symbols") or []
        }
    )
    panel = _load_panel(
        symbols,
        start="2018-01-01",
        end=None,
        timeframe=str(symbols_cfg.get("timeframe", "1d")),
    )
    strategies = [
        build_strategy(
            m["candidate"]["class_name"],
            m["candidate"]["strategy_id"],
            m["candidate"]["symbols"],
            m["candidate"]["params"],
        )
        for m in members
    ]
    port = BacktestEngine(
        strategies=strategies,
        risk_manager=MasterRiskManager(),
        starting_equity=float(prop.get("starting_equity", 100_000)),
    ).run({s: panel.get(s, []) for s in symbols})

    df = pd.DataFrame(port.equity_curve, columns=["ts", "equity"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last")
    daily = df.set_index("ts")["equity"].resample("1D").last().dropna()
    rets = daily.pct_change()

    rows: list[dict] = []
    years = sorted(set(daily.index.year))
    prev_year_end = None
    for y in years:
        yd = daily[daily.index.year == y]
        if yd.empty:
            continue
        start_eq = float(yd.iloc[0]) if prev_year_end is None else float(prev_year_end)
        end_eq = float(yd.iloc[-1])
        year_ret = end_eq / start_eq - 1.0
        peak = np.maximum.accumulate(np.concatenate([[start_eq], yd.values.astype(float)]))[1:]
        dd = yd.values.astype(float) / peak - 1.0
        max_dd = float(dd.min()) if len(dd) else 0.0
        yr_rets = rets[rets.index.year == y].dropna()
        worst_day = float(yr_rets.min()) if len(yr_rets) else 0.0
        worst_day_ts = str(yr_rets.idxmin().date()) if len(yr_rets) else None
        rows.append(
            {
                "year": int(y),
                "return": year_ret,
                "max_drawdown": abs(max_dd),
                "worst_daily_return": worst_day,
                "worst_daily_date": worst_day_ts,
                "partial": bool(y == int(daily.index.max().year)),
            }
        )
        prev_year_end = end_eq

    clean = rets.dropna()
    all_worst = float(clean.min()) if len(clean) else None
    all_worst_date = str(clean.idxmin().date()) if len(clean) else None

    out = {
        "yearly": rows,
        "overall_cagr": port.metrics.cagr,
        "overall_max_drawdown": port.metrics.max_drawdown,
        "worst_daily_return_all": all_worst,
        "worst_daily_date_all": all_worst_date,
    }
    path = data_dir() / "processed" / "paper_yearly_risk.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Year | Return | MaxDD(year) | Worst day | Date")
    for r in rows:
        note = "*" if r["partial"] else " "
        print(
            f"{r['year']}{note} | {r['return']*100:7.1f}% | "
            f"{r['max_drawdown']*100:6.2f}% | {r['worst_daily_return']*100:7.2f}% | "
            f"{r['worst_daily_date']}"
        )
    print("---")
    print(f"CAGR {port.metrics.cagr*100:.2f}% | Overall MaxDD {port.metrics.max_drawdown*100:.2f}%")
    print(f"Worst daily ever {all_worst*100:.2f}% on {all_worst_date}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
