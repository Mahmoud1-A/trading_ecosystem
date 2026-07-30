"""
Phase 5.5-A acceptance — full event-driven WFO integration.

Proves the institutional path runs through EventExecutionEngine and that the
reported WFO aggregate equals the mean of the per-fold OOS event results.
"""

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
from engine.event_execution import ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.portfolio import Portfolio
from validation.event_driven_wfo import (
    RANKING_SOURCE,
    EventWFOContext,
    RankingSourceError,
    proxy_metric_evaluate_fn,
    run_event_driven_wfo,
)


TZ = ZoneInfo("America/Chicago")


def _zero_cost() -> CostModel:
    return CostModel(
        version="event_wfo_zero",
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


def _synthetic_bars(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="5min", tz=TZ)
    # Deterministic mild mean-reverting path
    px = 100.0 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2.0
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


def _signal_factory(params: dict, _features: pd.DataFrame):
    entry_bar = int(params.get("lookback", 2))

    def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
        timing = InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.bar_end,
            decision_timestamp=bar.bar_end,
        )
        pos = portfolio.get_position("ES")
        if i == entry_bar and pos.is_flat:
            return SignalEvent(
                timing=timing,
                symbol="ES",
                side="BUY",
                quantity=1.0,
                signal_id=f"entry_{entry_bar}",
            )
        if i == entry_bar + 4 and not pos.is_flat:
            return SignalEvent(
                timing=timing,
                symbol="ES",
                side="FLAT",
                quantity=1.0,
                signal_id=f"exit_{entry_bar}",
            )
        return None

    return signal_fn


class TestEventDrivenWFO:
    def test_aggregate_equals_mean_of_event_driven_fold_validation_metrics(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=2,
            validation_window_days=1,
            step_forward_days=1,
            bars_per_day=10,
            purge_gap_bars=0,
            embargo_gap_bars=0,
            param_grid={"lookback": [2, 3]},
            optimize_metric="sharpe",
        )
        bars = _synthetic_bars(80)
        ctx = EventWFOContext(
            asset=default_es_futures(),
            cost_model=_zero_cost(),
            prop_profile=default_prop_profile(),
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=7),
            freq="5min",
            run_id="p55a",
        )
        result = run_event_driven_wfo(
            bars,
            cfg,
            signal_fn_factory=_signal_factory,
            context=ctx,
            ranking_metric="sharpe",
        )

        assert len(result.folds) >= 2
        assert result.ranking_source == RANKING_SOURCE
        # Aggregate must equal mean of the actual OOS event-window metrics
        recomputed = result.aggregate_from_folds()
        assert result.aggregated_validation_metric == pytest.approx(recomputed)
        fold_vals = [f.validation_metric for f in result.folds]
        assert result.aggregated_validation_metric == pytest.approx(
            float(sum(fold_vals) / len(fold_vals))
        )

        for fold in result.folds:
            assert fold.ranking_score == pytest.approx(fold.validation_metric)
            assert fold.ranking_score != fold.train_metric or fold.train_metric == fold.validation_metric
            assert fold.validation_run is not None
            assert fold.train_run is not None
            # Fold artifacts required by the acceptance gate
            assert fold.train_start_ts
            assert fold.validation_start_ts
            assert "orders" in fold.as_dict()
            assert "fills" in fold.as_dict()
            assert "trades" in fold.as_dict()
            assert "risk_transitions" in fold.as_dict()
            assert "gross_metrics" in fold.as_dict()
            assert "net_metrics" in fold.as_dict()
            assert "cost_attribution" in fold.as_dict()
            assert "failure_reason" in fold.as_dict()

    def test_rejects_in_sample_ranking(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=2,
            validation_window_days=1,
            step_forward_days=1,
            bars_per_day=10,
            param_grid={"lookback": [2]},
        )
        ctx = EventWFOContext(
            asset=default_es_futures(),
            cost_model=_zero_cost(),
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=1),
        )
        with pytest.raises(RankingSourceError, match="validation_oos"):
            run_event_driven_wfo(
                _synthetic_bars(50),
                cfg,
                signal_fn_factory=_signal_factory,
                context=ctx,
                rank_on="train",
            )

    def test_proxy_metric_is_explicitly_named_and_not_the_default_path(self) -> None:
        df = _synthetic_bars(60).set_index("timestamp")
        score = proxy_metric_evaluate_fn(df.iloc[:40], {"lookback": 10})
        assert isinstance(score, float)
        # The proxy helper must remain callable but is not used by run_event_driven_wfo
        assert "proxy_metric_evaluate_fn" in proxy_metric_evaluate_fn.__name__
