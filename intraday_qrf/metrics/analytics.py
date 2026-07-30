"""Institutional performance analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceReport:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    win_rate: float
    loss_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    n_trades: int
    n_bars: int
    final_equity: float
    volatility_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_console(self, title: str = "Performance Report") -> str:
        lines = [
            "",
            "=" * 64,
            f"  {title}",
            "=" * 64,
            f"  Total Return %        : {self.total_return_pct:>10.2f}",
            f"  CAGR %                : {self.cagr_pct:>10.2f}",
            f"  Sharpe Ratio          : {self.sharpe:>10.3f}",
            f"  Sortino Ratio         : {self.sortino:>10.3f}",
            f"  Max Drawdown %        : {self.max_drawdown_pct:>10.2f}",
            f"  Volatility % (ann.)   : {self.volatility_pct:>10.2f}",
            f"  Profit Factor         : {self.profit_factor:>10.3f}",
            f"  Expectancy ($/trade)  : {self.expectancy:>10.2f}",
            f"  Win Rate              : {self.win_rate:>10.2%}",
            f"  Loss Rate             : {self.loss_rate:>10.2%}",
            f"  Avg Win / Avg Loss    : {self.win_loss_ratio:>10.3f}",
            f"  Trades                : {self.n_trades:>10d}",
            f"  Bars                  : {self.n_bars:>10d}",
            f"  Final Equity          : {self.final_equity:>10,.2f}",
            "=" * 64,
            "",
        ]
        return "\n".join(lines)


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0 or not np.isfinite(den):
        return default
    val = num / den
    return float(val) if np.isfinite(val) else default


def equity_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak.replace(0, np.nan) - 1.0
    return dd.fillna(0.0)


def max_drawdown_pct(equity: pd.Series) -> float:
    dd = equity_drawdown(equity)
    return float(dd.min() * 100.0) if len(dd) else 0.0


def compute_trade_pnls(trades: pd.DataFrame) -> pd.Series:
    """
    Expect columns: pnl (float). Returns series of per-trade PnL.
    """
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    if "pnl" not in trades.columns:
        raise ValueError("trades DataFrame must include 'pnl' column")
    return trades["pnl"].astype(float)


def compute_metrics(
    equity: pd.Series,
    *,
    trades: pd.DataFrame | None = None,
    bars_per_year: int = 252 * 78,
    risk_free: float = 0.0,
    starting_equity: float | None = None,
) -> PerformanceReport:
    """
    Compute institutional metrics from an equity curve and optional trade blotter.

    Parameters
    ----------
    equity:
        Mark-to-market equity indexed by timestamp.
    trades:
        Optional closed-trade blotter with a `pnl` column.
    bars_per_year:
        Annualization factor for Sharpe / Sortino / vol.
    """
    if equity is None or len(equity) < 2:
        start = float(starting_equity or 0.0)
        return PerformanceReport(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            win_rate=0.0,
            loss_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            n_trades=0,
            n_bars=0 if equity is None else len(equity),
            final_equity=start,
            volatility_pct=0.0,
        )

    eq = equity.astype(float).dropna()
    start_eq = float(starting_equity if starting_equity is not None else eq.iloc[0])
    end_eq = float(eq.iloc[-1])
    total_return = _safe_div(end_eq, start_eq, 1.0) - 1.0

    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    excess = rets - (risk_free / max(bars_per_year, 1))
    vol = float(rets.std(ddof=0)) if len(rets) else 0.0
    downside = excess[excess < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0

    sharpe = _safe_div(float(excess.mean()), vol) * np.sqrt(bars_per_year) if vol > 0 else 0.0
    sortino = (
        _safe_div(float(excess.mean()), downside_std) * np.sqrt(bars_per_year)
        if downside_std > 0
        else 0.0
    )

    # CAGR from bar count
    years = len(eq) / max(bars_per_year, 1)
    if years > 0 and start_eq > 0 and end_eq > 0:
        cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    trade_pnls = compute_trade_pnls(trades) if trades is not None else pd.Series(dtype=float)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float((-losses).sum()) if len(losses) else 0.0
    n_trades = int(len(trade_pnls))
    win_rate = _safe_div(float(len(wins)), float(n_trades))
    loss_rate = _safe_div(float(len(losses)), float(n_trades))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
    win_loss_ratio = _safe_div(avg_win, abs(avg_loss)) if avg_loss != 0 else 0.0
    profit_factor = _safe_div(gross_profit, gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    if not np.isfinite(profit_factor):
        profit_factor = 999.0  # cap for reporting
    expectancy = float(trade_pnls.mean()) if n_trades else 0.0

    return PerformanceReport(
        total_return_pct=total_return * 100.0,
        cagr_pct=cagr * 100.0,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown_pct=max_drawdown_pct(eq),
        profit_factor=float(profit_factor),
        expectancy=expectancy,
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        n_trades=n_trades,
        n_bars=int(len(eq)),
        final_equity=end_eq,
        volatility_pct=vol * np.sqrt(bars_per_year) * 100.0,
    )


def metric_from_report(report: PerformanceReport, name: str) -> float:
    mapping = {
        "sharpe": report.sharpe,
        "sortino": report.sortino,
        "profit_factor": report.profit_factor,
        "expectancy": report.expectancy,
        "total_return_pct": report.total_return_pct,
        "max_drawdown_pct": report.max_drawdown_pct,
    }
    if name not in mapping:
        raise KeyError(f"Unknown metric '{name}'. Choose from {list(mapping)}")
    return float(mapping[name])
