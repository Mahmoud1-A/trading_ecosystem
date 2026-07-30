from __future__ import annotations

from datetime import datetime, timezone

from trading_ecosystem.backtest.engine import BacktestResult
from trading_ecosystem.backtest.metrics import PerformanceMetrics
from trading_ecosystem.common.contracts import RiskDecision
from trading_ecosystem.discovery.fitness import evaluate_folds


def _result(
    cagr: float,
    max_dd: float,
    trades: int,
    total_return: float,
    *,
    halt_reject: bool = False,
) -> BacktestResult:
    audit = []
    if halt_reject:
        audit.append(
            RiskDecision(
                approved=False,
                intent_id="x",
                reason="total_dd_breach:12.0>=10.0",
                ts=datetime.now(timezone.utc),
            )
        )
    return BacktestResult(
        metrics=PerformanceMetrics(
            cagr=cagr,
            max_drawdown=max_dd,
            mar=cagr / max_dd if max_dd else 0.0,
            profit_factor=1.5,
            total_return=total_return,
            num_trades=trades,
            win_rate=0.5,
            avg_win=100.0,
            avg_loss=-50.0,
        ),
        equity_curve=[],
        fills=[],
        risk_rejects=1 if halt_reject else 0,
        audit=audit,
    )


def test_fitness_rejects_prop_dd_even_with_high_cagr():
    folds = [
        ("holdout", "holdout", _result(cagr=0.45, max_dd=0.15, trades=20, total_return=1.0)),
    ]
    fit = evaluate_folds(
        folds,
        max_total_dd_pct=10.0,
        max_daily_dd_pct=5.0,
        target_cagr=0.30,
        min_trades=8,
        min_positive_oos_folds=1,
    )
    assert fit.passed is False
    assert fit.halted_or_prop_breach is True
    assert "prop_dd_breach" in fit.reason


def test_fitness_rejects_halt_audit():
    folds = [
        ("validate", "validate", _result(0.2, 0.05, 12, 0.3, halt_reject=True)),
    ]
    fit = evaluate_folds(
        folds,
        max_total_dd_pct=10.0,
        max_daily_dd_pct=5.0,
        target_cagr=0.30,
        min_trades=8,
        min_positive_oos_folds=1,
    )
    assert fit.passed is False
    assert fit.halted_or_prop_breach is True


def test_fitness_passes_stable_under_prop():
    folds = [
        ("validate", "validate", _result(0.12, 0.04, 10, 0.2)),
        ("holdout", "holdout", _result(0.10, 0.03, 10, 0.15)),
    ]
    fit = evaluate_folds(
        folds,
        max_total_dd_pct=10.0,
        max_daily_dd_pct=5.0,
        target_cagr=0.30,
        min_trades=8,
        min_positive_oos_folds=1,
    )
    assert fit.passed is True
    assert fit.hit_target_cagr is False
    assert fit.score > 0
