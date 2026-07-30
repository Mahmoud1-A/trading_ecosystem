from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_ecosystem.backtest.engine import BacktestResult


@dataclass
class FitnessResult:
    passed: bool
    score: float
    reason: str
    oos_cagr: float
    oos_max_dd: float
    oos_mar: float
    oos_trades: int
    positive_oos_folds: int
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    hit_target_cagr: bool = False
    halted_or_prop_breach: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "oos_cagr": self.oos_cagr,
            "oos_max_dd": self.oos_max_dd,
            "oos_mar": self.oos_mar,
            "oos_trades": self.oos_trades,
            "positive_oos_folds": self.positive_oos_folds,
            "hit_target_cagr": self.hit_target_cagr,
            "halted_or_prop_breach": self.halted_or_prop_breach,
            "fold_metrics": self.fold_metrics,
        }


def _halted(result: BacktestResult) -> bool:
    for d in result.audit:
        reason = getattr(d, "reason", "") or ""
        if "halt" in reason or "dd_breach" in reason or "total_dd" in reason or "daily_dd" in reason:
            if not getattr(d, "approved", True):
                return True
    return False


def evaluate_folds(
    fold_results: list[tuple[str, str, BacktestResult]],
    *,
    max_total_dd_pct: float,
    max_daily_dd_pct: float,
    target_cagr: float,
    min_trades: int,
    min_positive_oos_folds: int,
) -> FitnessResult:
    """Score candidate under HARD prop DD caps. Research validation folds only.

    Allowed kinds: ``validate`` (internal validation) and ``oos`` (yearly research OOS).
    Never uses ``vault`` / ``vault_year`` (sealed Validation Vault = promotion-only).
    Legacy ``holdout`` is ignored — research slices must be named validate/oos.
    """
    # Immutable Validation Vault must NEVER enter discovery fitness / Eligible gate.
    oos = [(n, k, r) for n, k, r in fold_results if k in {"validate", "oos"}]
    if not oos:
        return FitnessResult(
            passed=False,
            score=-1e9,
            reason="no_oos_folds",
            oos_cagr=0.0,
            oos_max_dd=0.0,
            oos_mar=0.0,
            oos_trades=0,
            positive_oos_folds=0,
        )

    fold_metrics: list[dict[str, Any]] = []
    cagrs: list[float] = []
    dds: list[float] = []
    mars: list[float] = []
    trades = 0
    positive = 0
    prop_breach = False
    max_total_frac = max_total_dd_pct / 100.0

    for name, kind, result in oos:
        m = result.metrics
        halted = _halted(result)
        breach = halted or m.max_drawdown >= max_total_frac - 1e-12
        if breach:
            prop_breach = True
        fold_metrics.append(
            {
                "name": name,
                "kind": kind,
                "cagr": m.cagr,
                "max_drawdown": m.max_drawdown,
                "mar": m.mar,
                "num_trades": m.num_trades,
                "total_return": m.total_return,
                "prop_breach": breach,
            }
        )
        cagrs.append(m.cagr)
        dds.append(m.max_drawdown)
        mars.append(m.mar)
        trades += m.num_trades
        if m.total_return > 0 and not breach:
            positive += 1

    oos_cagr = float(sum(cagrs) / len(cagrs)) if cagrs else 0.0
    oos_max_dd = float(max(dds)) if dds else 0.0
    oos_mar = float(sum(mars) / len(mars)) if mars else 0.0
    hit = oos_cagr >= target_cagr

    if prop_breach:
        return FitnessResult(
            passed=False,
            score=-1e6 + oos_cagr,
            reason=f"prop_dd_breach:max_dd={oos_max_dd:.4f}>={max_total_frac:.4f}",
            oos_cagr=oos_cagr,
            oos_max_dd=oos_max_dd,
            oos_mar=oos_mar,
            oos_trades=trades,
            positive_oos_folds=positive,
            fold_metrics=fold_metrics,
            hit_target_cagr=hit,
            halted_or_prop_breach=True,
        )

    if trades < min_trades:
        return FitnessResult(
            passed=False,
            score=-1e5 + oos_cagr,
            reason=f"insufficient_trades:{trades}<{min_trades}",
            oos_cagr=oos_cagr,
            oos_max_dd=oos_max_dd,
            oos_mar=oos_mar,
            oos_trades=trades,
            positive_oos_folds=positive,
            fold_metrics=fold_metrics,
            hit_target_cagr=hit,
        )

    if positive < min_positive_oos_folds:
        return FitnessResult(
            passed=False,
            score=-1e4 + oos_cagr,
            reason=f"unstable_oos:positive_folds={positive}<{min_positive_oos_folds}",
            oos_cagr=oos_cagr,
            oos_max_dd=oos_max_dd,
            oos_mar=oos_mar,
            oos_trades=trades,
            positive_oos_folds=positive,
            fold_metrics=fold_metrics,
            hit_target_cagr=hit,
        )

    # Prefer target hit, then CAGR, then MAR, penalize DD
    score = oos_cagr * 100.0 + oos_mar * 5.0 - oos_max_dd * 50.0
    if hit:
        score += 50.0
    # Soft awareness of daily cap for reporting only (already gated via MRM path)
    _ = max_daily_dd_pct

    return FitnessResult(
        passed=True,
        score=score,
        reason="ok",
        oos_cagr=oos_cagr,
        oos_max_dd=oos_max_dd,
        oos_mar=oos_mar,
        oos_trades=trades,
        positive_oos_folds=positive,
        fold_metrics=fold_metrics,
        hit_target_cagr=hit,
        halted_or_prop_breach=False,
    )
