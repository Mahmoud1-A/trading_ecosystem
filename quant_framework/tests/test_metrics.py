"""Phase 4 acceptance tests — metrics vs known mathematical examples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metrics import compute_metrics, estimate_prop_breach_probability, max_drawdown_pct
from metrics.dsr import deflated_sharpe_ratio
from metrics.pbo import probability_backtest_overfitting


class TestSharpeAndSortino:
    def _with_trade(self, equity: pd.Series):
        return pd.DataFrame({"net_pnl": [float(equity.iloc[-1] - equity.iloc[0])]})

    def test_sharpe_known_constant_return(self) -> None:
        rets = np.array([0.01, 0.012, 0.008, 0.011])
        equity = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + rets]))
        m = compute_metrics(equity, trades=self._with_trade(equity), bars_per_year=252)
        std = float(rets.std(ddof=0))
        expected = (float(rets.mean()) / std) * math.sqrt(252)
        assert m.sharpe == pytest.approx(expected, rel=1e-6)
        assert m.total_return_pct > 0

    def test_sharpe_hand_calculated(self) -> None:
        # Simple returns: 1%, -0.5%, 1%, -0.5% repeating
        rets = np.array([0.01, -0.005, 0.01, -0.005, 0.01, -0.005, 0.01, -0.005])
        equity = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + rets]))
        mean = float(rets.mean())
        std = float(rets.std(ddof=0))
        expected_sharpe = (mean / std) * math.sqrt(252)
        m = compute_metrics(equity, trades=self._with_trade(equity), bars_per_year=252)
        assert m.sharpe == pytest.approx(expected_sharpe, rel=1e-6)

    def test_sortino_uses_downside_only(self) -> None:
        rets = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.015])
        equity = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + rets]))
        downside = rets[rets < 0]
        ds_std = float(downside.std(ddof=0))
        expected = (float(rets.mean()) / ds_std) * math.sqrt(252)
        m = compute_metrics(equity, trades=self._with_trade(equity), bars_per_year=252)
        assert m.sortino == pytest.approx(expected, rel=1e-6)


class TestDrawdown:
    def test_max_drawdown_hand_calculated(self) -> None:
        equity = pd.Series([100.0, 110.0, 99.0, 105.0])
        # Peak 110, trough 99 => dd = 99/110 - 1 = -0.10
        assert max_drawdown_pct(equity) == pytest.approx(-10.0, rel=1e-6)
        # Without closed trades, metrics must not manufacture equity-path DD/Calmar
        m0 = compute_metrics(equity, starting_equity=100.0, bars_per_year=252)
        assert m0.n_trades == 0
        assert m0.max_drawdown_pct == 0.0
        assert m0.calmar == 0.0
        trades = pd.DataFrame({"net_pnl": [-5.0]})
        m = compute_metrics(equity, trades=trades, starting_equity=100.0, bars_per_year=252)
        assert m.max_drawdown_pct == pytest.approx(-10.0, rel=1e-6)

    def test_calmar(self) -> None:
        # Monotone growth — MDD ~ 0, calmar handled gracefully
        equity = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
        m = compute_metrics(equity, bars_per_year=252)
        assert m.max_drawdown_pct == pytest.approx(0.0, abs=1e-9)


class TestTradeMetrics:
    def test_profit_factor_and_expectancy(self) -> None:
        equity = pd.Series([100_000.0, 100_500.0, 100_200.0, 100_800.0])
        trades = pd.DataFrame(
            {
                "net_pnl": [500.0, -300.0, 600.0],
                "gross_pnl": [520.0, -280.0, 620.0],
                "commission": [20.0, 20.0, 20.0],
                "spread_cost": [0.0, 0.0, 0.0],
                "slippage_cost": [0.0, 0.0, 0.0],
            }
        )
        m = compute_metrics(equity, trades=trades, bars_per_year=252)
        # PF = (500+600)/(300) = 3.667
        assert m.profit_factor == pytest.approx(1100.0 / 300.0, rel=1e-6)
        assert m.expectancy == pytest.approx(800.0 / 3, rel=1e-6)
        assert m.win_rate == pytest.approx(2 / 3, rel=1e-6)
        assert m.n_trades == 3


class TestPropBreach:
    def test_breach_probability_nonzero_with_losses(self) -> None:
        daily = pd.Series([-0.005, -0.004, -0.006, -0.003, -0.007])
        est = estimate_prop_breach_probability(daily, hard_daily_limit=-0.015, n_simulations=500, seed=1)
        assert 0.0 <= est.breach_probability <= 1.0
        assert est.n_simulations == 500
        assert est.sample_size == 5

    def test_insufficient_sample_not_zero_probability(self) -> None:
        est = estimate_prop_breach_probability(pd.Series([-0.001]), n_simulations=100, seed=1)
        assert est.breach_probability >= 0.5


class TestDSRAndPBO:
    """Legacy scalar helpers retained; full registry path covered in test_dsr_pbo.py."""

    def test_legacy_dsr_collapses_insufficient_history_to_zero(self) -> None:
        assert deflated_sharpe_ratio(1.5, n_trials=1, trial_sharpe_variance=0.1, n_observations=100) == 0.0

    def test_legacy_pbo_simplified(self) -> None:
        is_perf = np.array([1.0, 2.0, 3.0, 0.5])
        oos_perf = np.array([0.5, 1.0, 0.2, 2.5])
        pbo = probability_backtest_overfitting(is_perf, oos_perf)
        assert 0.0 <= pbo <= 1.0
