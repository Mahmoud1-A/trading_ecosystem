from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    cagr: float
    max_drawdown: float
    mar: float
    profit_factor: float
    total_return: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "mar": self.mar,
            "profit_factor": self.profit_factor,
            "total_return": self.total_return,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }


def compute_metrics(
    equity_curve: list[tuple],
    trade_pnls: list[float],
) -> PerformanceMetrics:
    if not equity_curve:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    df = pd.DataFrame(equity_curve, columns=["ts", "equity"]).sort_values("ts")
    start_eq = float(df["equity"].iloc[0])
    end_eq = float(df["equity"].iloc[-1])
    total_return = (end_eq / start_eq - 1.0) if start_eq else 0.0

    days = max((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / 86400.0, 1.0)
    years = days / 365.25
    cagr = (end_eq / start_eq) ** (1 / years) - 1 if start_eq > 0 and years > 0 else 0.0

    peak = df["equity"].cummax()
    dd = (peak - df["equity"]) / peak.replace(0, np.nan)
    max_dd = float(dd.max()) if len(dd) else 0.0
    mar = (cagr / max_dd) if max_dd > 1e-12 else 0.0

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    n = len(trade_pnls)
    win_rate = len(wins) / n if n else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return PerformanceMetrics(
        cagr=float(cagr),
        max_drawdown=float(max_dd),
        mar=float(mar),
        profit_factor=float(pf) if np.isfinite(pf) else 999.0,
        total_return=float(total_return),
        num_trades=n,
        win_rate=float(win_rate),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
