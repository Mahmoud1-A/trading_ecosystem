from trading_ecosystem.discovery.multiple_testing import deflated_sharpe_ratio, probability_of_backtest_overfitting
from trading_ecosystem.discovery.walk_forward import build_folds
from trading_ecosystem.common.contracts import Bar
from datetime import datetime, timedelta, timezone


def test_dsr_and_pbo_smoke():
    dsr = deflated_sharpe_ratio(1.2, n_trials=100, n_returns=250)
    assert 0.0 <= dsr["dsr"] <= 1.0
    pbo = probability_of_backtest_overfitting(list(range(30)))
    assert pbo["pbo"] is not None


def test_vault_fold_sealed():
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    bars = [
        Bar(
            symbol="SPY",
            ts=start + timedelta(days=i),
            open=100,
            high=101,
            low=99,
            close=100.5,
            volume=1e6,
            timeframe="1d",
        )
        for i in range(800)
    ]
    folds = build_folds({"SPY": bars}, vault_ratio=0.2, seal_vault=True, use_yearly_rolls=False)
    kinds = {f.kind for f in folds}
    assert "vault" in kinds
    assert "validate" in kinds
    assert "holdout" not in kinds
    vault = next(f for f in folds if f.kind == "vault")
    validate = next(f for f in folds if f.kind == "validate")
    assert validate.end < vault.start


def test_fitness_ignores_vault_and_legacy_holdout():
    from trading_ecosystem.backtest.metrics import PerformanceMetrics
    from trading_ecosystem.backtest.engine import BacktestResult
    from trading_ecosystem.discovery.fitness import evaluate_folds

    def _res(cagr: float, dd: float = 0.05, trades: int = 20) -> BacktestResult:
        m = PerformanceMetrics(
            cagr=cagr,
            max_drawdown=dd,
            mar=cagr / dd if dd else 0.0,
            profit_factor=1.5,
            total_return=0.15,
            num_trades=trades,
            win_rate=0.5,
            avg_win=0.01,
            avg_loss=0.01,
        )
        return BacktestResult(metrics=m, equity_curve=[], fills=[], risk_rejects=0, audit=[])

    # Vault rows must not affect Eligible
    fit = evaluate_folds(
        [
            ("validate", "validate", _res(0.2)),
            ("vault", "vault", _res(9.0)),  # would dominate if leaked
            ("holdout", "holdout", _res(9.0)),  # legacy name ignored
        ],
        max_total_dd_pct=10.0,
        max_daily_dd_pct=5.0,
        target_cagr=0.3,
        min_trades=8,
        min_positive_oos_folds=1,
    )
    assert fit.passed is True
    assert abs(fit.oos_cagr - 0.2) < 1e-9
