"""
Diagnostic metric breakdowns with explicit availability status (Phase 5.5).

Every helper here reports *why* a result is missing instead of silently returning
an empty mapping. A breakdown whose labels are absent returns
``MetricStatus.UNAVAILABLE_MISSING_LABELS``; a breakdown with too few
observations returns ``MetricStatus.INSUFFICIENT_DATA``. Callers must therefore
distinguish "no exposure to this regime" from "regime labels were never supplied".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

CALCULATION_VERSION = "diagnostics_v1"

# Minimum observations required before a breakdown is reported as OK.
MIN_GROUP_OBSERVATIONS = 2
MIN_DAILY_OBSERVATIONS = 5
MIN_MONTHS = 2


class MetricStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNAVAILABLE_MISSING_LABELS = "UNAVAILABLE_MISSING_LABELS"
    UNAVAILABLE_MISSING_ATTRIBUTION = "UNAVAILABLE_MISSING_ATTRIBUTION"


@dataclass(frozen=True)
class DailyLossDistribution:
    """Distribution of daily losses used for prop-limit risk sizing."""

    status: MetricStatus
    sample_size: int
    n_losing_days: int
    mean_loss: float | None
    median_loss: float | None
    worst_loss: float | None
    p95_loss: float | None
    p99_loss: float | None
    std_loss: float | None
    loss_day_fraction: float | None
    assumptions: str
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "n_losing_days": self.n_losing_days,
            "mean_loss": self.mean_loss,
            "median_loss": self.median_loss,
            "worst_loss": self.worst_loss,
            "p95_loss": self.p95_loss,
            "p99_loss": self.p99_loss,
            "std_loss": self.std_loss,
            "loss_day_fraction": self.loss_day_fraction,
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MonthlyConsistency:
    """Month-by-month stability of returns."""

    status: MetricStatus
    sample_size: int
    n_months: int
    positive_months: int
    negative_months: int
    positive_month_fraction: float | None
    best_month: float | None
    worst_month: float | None
    mean_month: float | None
    std_month: float | None
    monthly_returns: dict[str, float] = field(default_factory=dict)
    assumptions: str = ""
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "n_months": self.n_months,
            "positive_months": self.positive_months,
            "negative_months": self.negative_months,
            "positive_month_fraction": self.positive_month_fraction,
            "best_month": self.best_month,
            "worst_month": self.worst_month,
            "mean_month": self.mean_month,
            "std_month": self.std_month,
            "monthly_returns": dict(self.monthly_returns),
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroupStat:
    """Per-label summary. ``status`` is per-group so thin groups are visible."""

    label: str
    status: MetricStatus
    sample_size: int
    total_return: float | None
    mean_return: float | None
    std_return: float | None
    sharpe: float | None
    win_rate: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status.value,
            "sample_size": self.sample_size,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
        }


@dataclass(frozen=True)
class GroupedBreakdown:
    """Breakdown of returns by a categorical label (regime, session, ...)."""

    status: MetricStatus
    label_name: str
    sample_size: int
    n_labels: int
    groups: dict[str, GroupStat] = field(default_factory=dict)
    unlabelled_observations: int = 0
    assumptions: str = ""
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.status == MetricStatus.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "label_name": self.label_name,
            "sample_size": self.sample_size,
            "n_labels": self.n_labels,
            "groups": {k: v.as_dict() for k, v in self.groups.items()},
            "unlabelled_observations": self.unlabelled_observations,
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExposureTime:
    """Fraction of marked bars during which the book held a position."""

    status: MetricStatus
    sample_size: int
    bars_in_market: int
    exposure_time_pct: float | None
    per_symbol_pct: dict[str, float] = field(default_factory=dict)
    assumptions: str = ""
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "bars_in_market": self.bars_in_market,
            "exposure_time_pct": self.exposure_time_pct,
            "per_symbol_pct": dict(self.per_symbol_pct),
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CostToGross:
    """Friction drag expressed against gross profit and gross PnL."""

    status: MetricStatus
    sample_size: int
    gross_profit: float
    gross_pnl: float
    total_cost: float
    cost_to_gross_profit: float | None
    cost_to_abs_gross_pnl: float | None
    components: dict[str, float] = field(default_factory=dict)
    assumptions: str = ""
    calculation_version: str = CALCULATION_VERSION
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "gross_profit": self.gross_profit,
            "gross_pnl": self.gross_pnl,
            "total_cost": self.total_cost,
            "cost_to_gross_profit": self.cost_to_gross_profit,
            "cost_to_abs_gross_pnl": self.cost_to_abs_gross_pnl,
            "components": dict(self.components),
            "assumptions": self.assumptions,
            "calculation_version": self.calculation_version,
            "reason": self.reason,
        }


COST_COLUMNS = (
    "commission",
    "spread_cost",
    "slippage_cost",
    "financing_cost",
    "rollover_cost",
    "market_impact_cost",
)


def _clean(series: pd.Series | Iterable[float] | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def daily_loss_distribution(
    daily_returns: pd.Series | Iterable[float] | None,
    *,
    min_observations: int = MIN_DAILY_OBSERVATIONS,
) -> DailyLossDistribution:
    """
    Summarise the left tail of the daily return distribution.

    ``daily_returns`` are per-day fractional returns (not percent). Percentiles are
    reported on the loss magnitude, so ``p95_loss`` is the 95th percentile of losses.
    """
    obs = _clean(daily_returns)
    n = int(len(obs))
    assumptions = (
        "daily fractional returns; percentiles taken on loss magnitudes of losing days only; "
        f"min_observations={min_observations}"
    )
    if n < min_observations:
        return DailyLossDistribution(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=n,
            n_losing_days=int((obs < 0).sum()) if n else 0,
            mean_loss=None,
            median_loss=None,
            worst_loss=None,
            p95_loss=None,
            p99_loss=None,
            std_loss=None,
            loss_day_fraction=None,
            assumptions=assumptions,
            reason=f"need >= {min_observations} daily observations, got {n}",
        )

    losses = -obs[obs < 0]
    n_loss = int(len(losses))
    if n_loss == 0:
        return DailyLossDistribution(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=n,
            n_losing_days=0,
            mean_loss=None,
            median_loss=None,
            worst_loss=None,
            p95_loss=None,
            p99_loss=None,
            std_loss=None,
            loss_day_fraction=0.0,
            assumptions=assumptions,
            reason="no losing days observed; loss distribution is unidentified",
        )

    return DailyLossDistribution(
        status=MetricStatus.OK,
        sample_size=n,
        n_losing_days=n_loss,
        mean_loss=float(losses.mean()),
        median_loss=float(losses.median()),
        worst_loss=float(losses.max()),
        p95_loss=float(np.percentile(losses.to_numpy(), 95)),
        p99_loss=float(np.percentile(losses.to_numpy(), 99)),
        std_loss=float(losses.std(ddof=0)),
        loss_day_fraction=float(n_loss) / float(n),
        assumptions=assumptions,
    )


def monthly_consistency(
    equity: pd.Series | None = None,
    *,
    daily_returns: pd.Series | None = None,
    min_months: int = MIN_MONTHS,
) -> MonthlyConsistency:
    """
    Compound returns per calendar month and report month-level stability.

    Provide either a datetime-indexed ``equity`` curve or datetime-indexed
    ``daily_returns``. Months are labelled ``YYYY-MM``.
    """
    assumptions = (
        "calendar-month compounding of the supplied series; "
        f"min_months={min_months}; partial first/last months are included as-is"
    )
    if daily_returns is not None:
        rets = _clean(daily_returns)
    elif equity is not None:
        eq = _clean(equity)
        rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    else:
        return MonthlyConsistency(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=0,
            n_months=0,
            positive_months=0,
            negative_months=0,
            positive_month_fraction=None,
            best_month=None,
            worst_month=None,
            mean_month=None,
            std_month=None,
            assumptions=assumptions,
            reason="neither equity nor daily_returns supplied",
        )

    if not isinstance(rets.index, pd.DatetimeIndex):
        return MonthlyConsistency(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=int(len(rets)),
            n_months=0,
            positive_months=0,
            negative_months=0,
            positive_month_fraction=None,
            best_month=None,
            worst_month=None,
            mean_month=None,
            std_month=None,
            assumptions=assumptions,
            reason="a DatetimeIndex is required to group by calendar month",
        )

    grouped = (1.0 + rets).groupby(rets.index.strftime("%Y-%m")).prod() - 1.0
    n_months = int(len(grouped))
    if n_months < min_months:
        return MonthlyConsistency(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=int(len(rets)),
            n_months=n_months,
            positive_months=int((grouped > 0).sum()),
            negative_months=int((grouped < 0).sum()),
            positive_month_fraction=None,
            best_month=None,
            worst_month=None,
            mean_month=None,
            std_month=None,
            monthly_returns={str(k): float(v) for k, v in grouped.items()},
            assumptions=assumptions,
            reason=f"need >= {min_months} calendar months, got {n_months}",
        )

    return MonthlyConsistency(
        status=MetricStatus.OK,
        sample_size=int(len(rets)),
        n_months=n_months,
        positive_months=int((grouped > 0).sum()),
        negative_months=int((grouped < 0).sum()),
        positive_month_fraction=float((grouped > 0).sum()) / float(n_months),
        best_month=float(grouped.max()),
        worst_month=float(grouped.min()),
        mean_month=float(grouped.mean()),
        std_month=float(grouped.std(ddof=0)),
        monthly_returns={str(k): float(v) for k, v in grouped.items()},
        assumptions=assumptions,
    )


def _grouped_breakdown(
    returns: pd.Series | None,
    labels: pd.Series | Mapping[Any, str] | None,
    *,
    label_name: str,
    min_observations: int,
) -> GroupedBreakdown:
    assumptions = (
        f"per-bar returns grouped by {label_name} label; "
        f"groups with < {min_observations} observations are marked INSUFFICIENT_DATA; "
        "Sharpe is per-bar and not annualised"
    )
    rets = _clean(returns)
    if labels is None:
        return GroupedBreakdown(
            status=MetricStatus.UNAVAILABLE_MISSING_LABELS,
            label_name=label_name,
            sample_size=int(len(rets)),
            n_labels=0,
            assumptions=assumptions,
            reason=f"no {label_name} labels supplied; breakdown is unavailable, not empty",
        )

    lab = labels if isinstance(labels, pd.Series) else pd.Series(labels)
    if len(rets) == 0:
        return GroupedBreakdown(
            status=MetricStatus.INSUFFICIENT_DATA,
            label_name=label_name,
            sample_size=0,
            n_labels=0,
            assumptions=assumptions,
            reason="no return observations supplied",
        )

    aligned = lab.reindex(rets.index) if lab.index.equals(rets.index) is False else lab
    aligned = aligned.where(aligned.notna(), other=None)
    mask = aligned.notna() & (aligned.astype(str).str.strip() != "")
    unlabelled = int((~mask).sum())
    if not bool(mask.any()):
        return GroupedBreakdown(
            status=MetricStatus.UNAVAILABLE_MISSING_LABELS,
            label_name=label_name,
            sample_size=int(len(rets)),
            n_labels=0,
            unlabelled_observations=unlabelled,
            assumptions=assumptions,
            reason=(
                f"every {label_name} label is missing/blank; breakdown is unavailable, not empty"
            ),
        )

    groups: dict[str, GroupStat] = {}
    for key, chunk in rets[mask].groupby(aligned[mask].astype(str)):
        n = int(len(chunk))
        if n < min_observations:
            groups[str(key)] = GroupStat(
                label=str(key),
                status=MetricStatus.INSUFFICIENT_DATA,
                sample_size=n,
                total_return=float((1.0 + chunk).prod() - 1.0),
                mean_return=float(chunk.mean()),
                std_return=None,
                sharpe=None,
                win_rate=None,
            )
            continue
        std = float(chunk.std(ddof=0))
        groups[str(key)] = GroupStat(
            label=str(key),
            status=MetricStatus.OK,
            sample_size=n,
            total_return=float((1.0 + chunk).prod() - 1.0),
            mean_return=float(chunk.mean()),
            std_return=std,
            sharpe=float(chunk.mean() / std) if std > 0 else None,
            win_rate=float((chunk > 0).mean()),
        )

    return GroupedBreakdown(
        status=MetricStatus.OK,
        label_name=label_name,
        sample_size=int(len(rets)),
        n_labels=len(groups),
        groups=groups,
        unlabelled_observations=unlabelled,
        assumptions=assumptions,
    )


def by_regime(
    returns: pd.Series | None,
    regime_labels: pd.Series | Mapping[Any, str] | None,
    *,
    min_observations: int = MIN_GROUP_OBSERVATIONS,
) -> GroupedBreakdown:
    """Performance split by regime label; explicitly unavailable when labels are absent."""
    return _grouped_breakdown(
        returns, regime_labels, label_name="regime", min_observations=min_observations
    )


def by_session(
    returns: pd.Series | None,
    session_labels: pd.Series | Mapping[Any, str] | None,
    *,
    min_observations: int = MIN_GROUP_OBSERVATIONS,
) -> GroupedBreakdown:
    """Performance split by session label; explicitly unavailable when labels are absent."""
    return _grouped_breakdown(
        returns, session_labels, label_name="session", min_observations=min_observations
    )


def exposure_time(portfolio: Any, *, symbol: str | None = None) -> ExposureTime:
    """
    Exposure time reconstructed from portfolio state.

    Replays the portfolio's fills against its own mark timestamps, so exposure is
    derived from realised position state rather than from a signal mask.
    """
    assumptions = (
        "position quantity replayed from fills at each mark timestamp; "
        "a bar counts as exposed when any tracked symbol is non-flat at that mark"
    )
    curve = list(getattr(portfolio, "equity_curve", []) or [])
    if not curve:
        return ExposureTime(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=0,
            bars_in_market=0,
            exposure_time_pct=None,
            assumptions=assumptions,
            reason="portfolio has no marked equity curve",
        )

    fills = sorted(
        list(getattr(portfolio, "fills", []) or []), key=lambda f: f.fill_timestamp
    )
    qty: dict[str, float] = {}
    per_symbol_bars: dict[str, int] = {}
    cursor = 0
    exposed_bars = 0
    for ts, _eq in curve:
        while cursor < len(fills) and fills[cursor].fill_timestamp <= ts:
            f = fills[cursor]
            signed = f.quantity if f.side.value == "BUY" else -f.quantity
            qty[f.symbol] = qty.get(f.symbol, 0.0) + signed
            cursor += 1
        tracked = {s: q for s, q in qty.items() if symbol is None or s == symbol}
        any_open = False
        for sym, q in tracked.items():
            if abs(q) > 1e-12:
                any_open = True
                per_symbol_bars[sym] = per_symbol_bars.get(sym, 0) + 1
        if any_open:
            exposed_bars += 1

    n = len(curve)
    return ExposureTime(
        status=MetricStatus.OK,
        sample_size=n,
        bars_in_market=exposed_bars,
        exposure_time_pct=100.0 * exposed_bars / n,
        per_symbol_pct={s: 100.0 * b / n for s, b in sorted(per_symbol_bars.items())},
        assumptions=assumptions,
    )


def cost_to_gross(attribution: pd.DataFrame | Iterable[Mapping[str, Any]] | None) -> CostToGross:
    """
    Friction-to-gross ratios from a trade cost attribution blotter.

    ``attribution`` is a trade-level frame (or iterable of mappings) carrying
    ``gross_pnl`` and any of the per-component cost columns.
    """
    assumptions = (
        "trade-level attribution; total_cost sums "
        + ", ".join(COST_COLUMNS)
        + "; cost_to_gross_profit uses gross profit of winning trades only"
    )
    if attribution is None:
        return CostToGross(
            status=MetricStatus.UNAVAILABLE_MISSING_ATTRIBUTION,
            sample_size=0,
            gross_profit=0.0,
            gross_pnl=0.0,
            total_cost=0.0,
            cost_to_gross_profit=None,
            cost_to_abs_gross_pnl=None,
            assumptions=assumptions,
            reason="no cost attribution supplied",
        )

    df = attribution if isinstance(attribution, pd.DataFrame) else pd.DataFrame(list(attribution))
    if df.empty:
        return CostToGross(
            status=MetricStatus.INSUFFICIENT_DATA,
            sample_size=0,
            gross_profit=0.0,
            gross_pnl=0.0,
            total_cost=0.0,
            cost_to_gross_profit=None,
            cost_to_abs_gross_pnl=None,
            assumptions=assumptions,
            reason="attribution blotter is empty",
        )
    if "gross_pnl" not in df.columns:
        return CostToGross(
            status=MetricStatus.UNAVAILABLE_MISSING_ATTRIBUTION,
            sample_size=int(len(df)),
            gross_profit=0.0,
            gross_pnl=0.0,
            total_cost=0.0,
            cost_to_gross_profit=None,
            cost_to_abs_gross_pnl=None,
            assumptions=assumptions,
            reason="attribution is missing the gross_pnl column",
        )

    present = [c for c in COST_COLUMNS if c in df.columns]
    if not present:
        return CostToGross(
            status=MetricStatus.UNAVAILABLE_MISSING_ATTRIBUTION,
            sample_size=int(len(df)),
            gross_profit=0.0,
            gross_pnl=float(df["gross_pnl"].astype(float).sum()),
            total_cost=0.0,
            cost_to_gross_profit=None,
            cost_to_abs_gross_pnl=None,
            assumptions=assumptions,
            reason="attribution carries no cost component columns",
        )

    components = {c: float(df[c].astype(float).sum()) for c in present}
    total_cost = float(sum(components.values()))
    gross = df["gross_pnl"].astype(float)
    gross_pnl = float(gross.sum())
    gross_profit = float(gross[gross > 0].sum())
    return CostToGross(
        status=MetricStatus.OK,
        sample_size=int(len(df)),
        gross_profit=gross_profit,
        gross_pnl=gross_pnl,
        total_cost=total_cost,
        cost_to_gross_profit=(total_cost / gross_profit) if gross_profit > 0 else None,
        cost_to_abs_gross_pnl=(total_cost / abs(gross_pnl)) if abs(gross_pnl) > 0 else None,
        components=components,
        assumptions=assumptions,
    )
