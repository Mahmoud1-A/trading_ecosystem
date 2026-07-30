"""Regression: known-trading probe must open and close ≥1 OOS trade under risk FSM."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.asset_spec import default_es_futures
from config.cost_model import CostModel
from config.prop_profile import default_prop_profile
from config.walk_forward_config import WalkForwardConfig
from discovery.event_wfo_backend import (
    EventDrivenDiscoveryBackend,
    known_trading_signal_factory,
)
from engine.event_execution import ExecutionConfig
from validation.event_driven_wfo import EventWFOContext, run_event_driven_wfo


TZ = ZoneInfo("America/Chicago")


def _zero_cost() -> CostModel:
    return CostModel(
        version="known_trading_zero",
        commission_per_contract=0.0,
        minimum_commission=0.0,
        fixed_spread_ticks=0.0,
        dynamic_spread_enabled=False,
        slippage_ticks_mean=0.0,
        slippage_ticks_std=0.0,
        volatility_dependent_slippage=False,
        time_of_day_slippage=False,
        liquidity_dependent_slippage=False,
        overnight_swap_enabled=False,
        participation_rate_cap=1.0,
    )


def _bars(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="5min", tz=TZ)
    px = 100.0 + np.linspace(0, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": px,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "volume": 10_000.0,
            "contract": "ESH24",
        }
    )


class TestKnownTradingPipeline:
    def test_known_trading_closes_oos_trade_with_full_symbol_exposure(self) -> None:
        """Buy 1 lot (=max_symbol_exposure) then FLAT must produce a closed trade."""
        cfg = WalkForwardConfig(
            train_window_days=2,
            validation_window_days=1,
            step_forward_days=1,
            bars_per_day=10,
            purge_gap_bars=0,
            embargo_gap_bars=0,
            max_folds=2,
            param_grid={"entry_bar": [2], "hold_bars": [3]},
            optimize_metric="sharpe",
        )
        profile = default_prop_profile()
        assert float(profile.max_symbol_exposure) == pytest.approx(1.0)
        ctx = EventWFOContext(
            asset=default_es_futures(),
            cost_model=_zero_cost(),
            prop_profile=profile,
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=11),
            freq="5min",
            run_id="known_trading",
        )
        result = run_event_driven_wfo(
            _bars(80),
            cfg,
            signal_fn_factory=known_trading_signal_factory,
            context=ctx,
            ranking_metric="sharpe",
        )
        assert result.folds
        oos_trades = 0
        for fold in result.folds:
            assert fold.validation_run is not None
            funnel = fold.validation_run.trade_funnel
            assert funnel["entry_true_count"] >= 1
            assert funnel["exit_true_count"] >= 1
            assert funnel["fills"] >= 2
            assert funnel["positions_closed"] >= 1
            assert "exposure_cap" not in funnel.get("reject_reasons", [])
            oos_trades += len(fold.validation_run.trades)
            assert fold.validation_run.net_metrics["n_trades"] >= 1
        assert oos_trades >= 1

    def test_discovery_backend_session_flatten_closes_trade(self) -> None:
        """INTRADAY_ONLY + DSL entry must close via session/window-end flatten."""
        from discovery.candidate import build_candidate
        from discovery.expression_tree import constant_node, feature_node, op_node
        from discovery.grammar import GRAMMAR_VERSION
        from discovery.operators import OperatorId
        from discovery.types import CreationMethod, ValueType

        bars = _bars(780)
        backend = EventDrivenDiscoveryBackend(
            bars=bars,
            wfo_config=WalkForwardConfig(
                train_window_days=2,
                validation_window_days=1,
                step_forward_days=1,
                purge_gap_bars=1,
                embargo_gap_bars=1,
                bars_per_day=78,
                max_folds=2,
                param_grid={},
            ),
            require_real_bars=False,
            intraday_only=True,
        )
        # Near-always-true entry on non-null returns
        ret = feature_node("price.simple_return_1", ValueType.RETURN)
        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(OperatorId.GREATER_THAN, ret, constant_node(-1.0)),
        )
        exit_tree = op_node(
            OperatorId.EXIT_SIGNAL,
            op_node(OperatorId.LESS_THAN, ret, constant_node(-0.5)),
        )
        cand = build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            strategy_family="dsl_session_flatten",
            creation_method=CreationMethod.RANDOM,
            grammar_version=GRAMMAR_VERSION,
            feature_set_version="feature_set_v1_phase6b",
            random_seed=9,
        )
        folds, diag = backend.evaluate(cand)
        assert folds
        assert diag.get("intraday_only") is True
        assert diag.get("signal_source") == "candidate_dsl_trees"
        total = sum(f.n_trades for f in folds)
        assert total >= 1, f"expected closed trades via session flatten, funnels={diag['fold_trade_funnels']}"
        oos_funnels = [f for f in diag["fold_trade_funnels"] if f.get("phase") == "validation_oos"]
        assert any(f.get("positions_closed", 0) >= 1 or f.get("exit_true_count", 0) >= 1 for f in oos_funnels)
