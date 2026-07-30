"""Institutional performance metrics — Phase 4 core suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from metrics.diagnostics import (  # noqa: F401  (re-exported for callers)
    CostToGross,
    DailyLossDistribution,
    ExposureTime,
    GroupedBreakdown,
    GroupStat,
    MetricStatus,
    MonthlyConsistency,
    by_regime,
    by_session,
    cost_to_gross,
    daily_loss_distribution,
    exposure_time,
    monthly_consistency,
)
from metrics.drawdown import drawdown_duration_bars, max_drawdown_pct


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    calmar: float
    mar: float
    max_drawdown_pct: float
    drawdown_duration_bars: int
    profit_factor: float
    expectancy: float
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    tail_ratio: float
    turnover: float
    exposure_time_pct: float
    max_consecutive_losses: int
    worst_day_pct: float
    best_day_pct: float
    cost_to_gross_profit_ratio: float
    n_trades: int
    n_bars: int
    final_equity: float
    volatility_pct: float
    by_regime: dict[str, float] = field(default_factory=dict)
    by_session: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_console(self, title: str = "Performance Report") -> str:
        lines = [
            "",
            "=" * 68,
            f"  {title}",
            "=" * 68,
            f"  Total Return %        : {self.total_return_pct:>10.2f}",
            f"  CAGR %                : {self.cagr_pct:>10.2f}",
            f"  Sharpe Ratio          : {self.sharpe:>10.3f}",
            f"  Sortino Ratio         : {self.sortino:>10.3f}",
            f"  Calmar Ratio          : {self.calmar:>10.3f}",
            f"  MAR Ratio             : {self.mar:>10.3f}",
            f"  Max Drawdown %        : {self.max_drawdown_pct:>10.2f}",
            f"  DD Duration (bars)    : {self.drawdown_duration_bars:>10d}",
            f"  Profit Factor         : {self.profit_factor:>10.3f}",
            f"  Expectancy ($)        : {self.expectancy:>10.2f}",
            f"  Win Rate              : {self.win_rate:>10.2%}",
            f"  Win/Loss Ratio        : {self.win_loss_ratio:>10.3f}",
            f"  Tail Ratio            : {self.tail_ratio:>10.3f}",
            f"  Exposure Time %       : {self.exposure_time_pct:>10.2f}",
            f"  Cost/Gross Profit     : {self.cost_to_gross_profit_ratio:>10.3f}",
            f"  Trades                : {self.n_trades:>10d}",
            f"  Final Equity          : {self.final_equity:>10,.2f}",
            "=" * 68,
            "",
        ]
        return "\n".join(lines)


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0 or not np.isfinite(den):
        return default
    val = num / den
    return float(val) if np.isfinite(val) else default


def _consecutive_losses(pnls: pd.Series) -> int:
    max_run = 0
    run = 0
    for p in pnls:
        if p < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def compute_metrics(
    equity: pd.Series,
    *,
    trades: pd.DataFrame | None = None,
    bars_per_year: int = 252 * 78,
    starting_equity: float | None = None,
    exposure_mask: pd.Series | None = None,
    daily_equity: pd.Series | None = None,
) -> PerformanceMetrics:
    """Compute metrics from equity curve and optional trade blotter."""
    if equity is None or len(equity) < 2:
        start = float(starting_equity or 0.0)
        return PerformanceMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            mar=0.0,
            max_drawdown_pct=0.0,
            drawdown_duration_bars=0,
            profit_factor=0.0,
            expectancy=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            tail_ratio=0.0,
            turnover=0.0,
            exposure_time_pct=0.0,
            max_consecutive_losses=0,
            worst_day_pct=0.0,
            best_day_pct=0.0,
            cost_to_gross_profit_ratio=0.0,
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
    vol = float(rets.std(ddof=0)) if len(rets) else 0.0
    downside = rets[rets < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0

    sharpe = _safe_div(float(rets.mean()), vol) * np.sqrt(bars_per_year) if vol > 0 else 0.0
    sortino = _safe_div(float(rets.mean()), downside_std) * np.sqrt(bars_per_year) if downside_std > 0 else 0.0

    years = len(eq) / max(bars_per_year, 1)
    cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0 if years > 0 and start_eq > 0 and end_eq > 0 else 0.0
    mdd = max_drawdown_pct(eq)
    calmar = _safe_div(cagr * 100.0, abs(mdd)) if mdd != 0 else 0.0
    mar = calmar

    trade_pnls = pd.Series(dtype=float)
    gross_profit = gross_loss = 0.0
    wins = losses = pd.Series(dtype=float)
    n_trades = 0
    cost_total = 0.0
    if trades is not None and not trades.empty and "net_pnl" in trades.columns:
        trade_pnls = trades["net_pnl"].astype(float)
        n_trades = len(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float((-losses).sum()) if len(losses) else 0.0
        if "commission" in trades.columns:
            cost_total += float(trades["commission"].sum())
        if "spread_cost" in trades.columns:
            cost_total += float(trades["spread_cost"].sum())
        if "slippage_cost" in trades.columns:
            cost_total += float(trades["slippage_cost"].sum())

    win_rate = _safe_div(float(len(wins)), float(n_trades))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    win_loss_ratio = _safe_div(avg_win, abs(avg_loss)) if avg_loss != 0 else 0.0
    profit_factor = _safe_div(gross_profit, gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    expectancy = float(trade_pnls.mean()) if n_trades else 0.0

    # Tail ratio: 95th / 5th percentile of returns
    if len(rets) >= 20:
        p95 = float(np.percentile(rets, 95))
        p05 = float(np.percentile(rets, 5))
        tail_ratio = _safe_div(abs(p95), abs(p05)) if p05 != 0 else 0.0
    else:
        tail_ratio = 0.0

    exposure_time_pct = 0.0
    if exposure_mask is not None and len(exposure_mask) > 0:
        exposure_time_pct = float(exposure_mask.astype(bool).mean()) * 100.0

    turnover = float(n_trades) / max(len(eq), 1) * bars_per_year

    if daily_equity is not None:
        daily = daily_equity
    elif isinstance(eq.index, pd.DatetimeIndex):
        daily = eq.resample("D").last().dropna()
    else:
        daily = eq
    daily_rets = daily.pct_change().dropna() if len(daily) > 1 else pd.Series(dtype=float)
    worst_day = float(daily_rets.min() * 100.0) if len(daily_rets) else 0.0
    best_day = float(daily_rets.max() * 100.0) if len(daily_rets) else 0.0

    cost_ratio = _safe_div(cost_total, gross_profit) if gross_profit > 0 else 0.0

    # Zero closed trades: do not report mark-to-market equity-path drawdown or
    # annualized ratios as if they came from completed trading activity.
    if n_trades == 0:
        return PerformanceMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            mar=0.0,
            max_drawdown_pct=0.0,
            drawdown_duration_bars=0,
            profit_factor=0.0,
            expectancy=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            tail_ratio=0.0,
            turnover=0.0,
            exposure_time_pct=exposure_time_pct,
            max_consecutive_losses=0,
            worst_day_pct=0.0,
            best_day_pct=0.0,
            cost_to_gross_profit_ratio=0.0,
            n_trades=0,
            n_bars=int(len(eq)),
            final_equity=end_eq,
            volatility_pct=0.0,
        )

    return PerformanceMetrics(
        total_return_pct=total_return * 100.0,
        cagr_pct=cagr * 100.0,
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        mar=float(mar),
        max_drawdown_pct=mdd,
        drawdown_duration_bars=drawdown_duration_bars(eq),
        profit_factor=float(min(profit_factor, 999.0)),
        expectancy=expectancy,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        tail_ratio=tail_ratio,
        turnover=turnover,
        exposure_time_pct=exposure_time_pct,
        max_consecutive_losses=_consecutive_losses(trade_pnls),
        worst_day_pct=worst_day,
        best_day_pct=best_day,
        cost_to_gross_profit_ratio=cost_ratio,
        n_trades=n_trades,
        n_bars=int(len(eq)),
        final_equity=end_eq,
        volatility_pct=vol * np.sqrt(bars_per_year) * 100.0,
    )
