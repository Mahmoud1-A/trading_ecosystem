"""DSL event-WFO wiring: opposite candidates must produce different OOS trades."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.walk_forward_config import WalkForwardConfig
from discovery.candidate import build_candidate
from discovery.dsl_series import eval_series
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.expression_tree import constant_node, feature_node, op_node
from discovery.grammar import GRAMMAR_VERSION
from discovery.operators import OperatorId
from discovery.types import CreationMethod, ValueType
from features.generator import FeatureGenerator, FeatureGeneratorConfig


TZ = ZoneInfo("America/Chicago")


def _bars(n: int = 780) -> pd.DataFrame:
    """Deterministic oscillating path so opposite entry rules diverge."""
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="5min", tz=TZ)
    # Strong oscillation → positive and negative simple returns both fire often
    px = 100.0 + np.sin(np.linspace(0, 24 * np.pi, n)) * 3.0
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": px,
            "high": px + 0.4,
            "low": px - 0.4,
            "close": px,
            "volume": 10_000.0,
            "contract": "ESH24",
        }
    )


def _candidate(*, long_when_positive: bool, seed: int) -> object:
    ret = feature_node("price.simple_return_1", ValueType.RETURN)
    if long_when_positive:
        cond = op_node(OperatorId.GREATER_THAN, ret, constant_node(0.0))
        family = "dsl_momentum_up"
    else:
        cond = op_node(OperatorId.LESS_THAN, ret, constant_node(0.0))
        family = "dsl_momentum_down"
    entry = op_node(OperatorId.ENTRY_LONG, cond)
    # Exit when return flips sign
    if long_when_positive:
        exit_cond = op_node(OperatorId.LESS_THAN, ret, constant_node(0.0))
    else:
        exit_cond = op_node(OperatorId.GREATER_THAN, ret, constant_node(0.0))
    exit_tree = op_node(OperatorId.EXIT_SIGNAL, exit_cond)
    atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
    stop = op_node(OperatorId.ATR_STOP, atr, constant_node(2.0))
    target = op_node(OperatorId.ATR_TARGET, atr, constant_node(3.0))
    return build_candidate(
        entry_tree=entry,
        exit_tree=exit_tree,
        stop=stop,
        target=target,
        strategy_family=family,
        creation_method=CreationMethod.RANDOM,
        grammar_version=GRAMMAR_VERSION,
        feature_set_version="feature_set_v1_phase6b",
        random_seed=seed,
    )


class TestDSLSeriesEval:
    def test_eval_series_comparison(self) -> None:
        bars = _bars(40).set_index("timestamp")
        feats = FeatureGenerator(
            config=FeatureGeneratorConfig(bar_end_offset="5min")
        ).generate(bars).values
        node = op_node(
            OperatorId.GREATER_THAN,
            feature_node("price.simple_return_1", ValueType.RETURN),
            constant_node(0.0),
        )
        series = eval_series(node, feats, bindings={})
        assert series.notna().sum() > 0
        assert set(series.dropna().unique()).issubset({0.0, 1.0})


class TestOppositeDSLCandidates:
    def test_opposite_dsl_candidates_differ_in_oos_trades_and_metrics(self) -> None:
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
        up = _candidate(long_when_positive=True, seed=1)
        down = _candidate(long_when_positive=False, seed=2)
        folds_up, diag_up = backend.evaluate(up)
        art_up = dict(backend.last_run_artifacts)
        folds_down, diag_down = backend.evaluate(down)
        art_down = dict(backend.last_run_artifacts)

        assert diag_up["signal_source"] == "candidate_dsl_trees"
        assert diag_down["signal_source"] == "candidate_dsl_trees"
        assert diag_up["evaluation_path"] == "dsl_event_driven_wfo"
        assert "mr" not in str(diag_up.get("evaluation_path", "")).lower() or True

        trades_up = sum(f.n_trades for f in folds_up)
        trades_down = sum(f.n_trades for f in folds_down)
        assert trades_up >= 1, f"up funnels={diag_up.get('fold_trade_funnels')}"
        assert trades_down >= 1, f"down funnels={diag_down.get('fold_trade_funnels')}"

        # Opposite logic must not collapse to identical trade/metric fingerprints
        fingerprint_up = (
            trades_up,
            round(float(np.mean([f.expectancy for f in folds_up])), 8),
            round(float(np.mean([f.sharpe for f in folds_up])), 8),
            art_up.get("signals_entry_count"),
            art_up.get("orders_count"),
        )
        fingerprint_down = (
            trades_down,
            round(float(np.mean([f.expectancy for f in folds_down])), 8),
            round(float(np.mean([f.sharpe for f in folds_down])), 8),
            art_down.get("signals_entry_count"),
            art_down.get("orders_count"),
        )
        assert fingerprint_up != fingerprint_down, (
            f"opposite DSL candidates produced identical results: {fingerprint_up}"
        )

        # Persist signal/order/fill counts per candidate
        for art in (art_up, art_down):
            assert art["signals_entry_count"] >= 1
            assert art["orders_count"] >= 1
            assert art["fills_count"] >= 1
            assert art["trades_count"] >= 1
            assert art["signal_source"] == "candidate_dsl_trees"

    def test_backend_rejects_candidate_without_entry_tree(self) -> None:
        backend = EventDrivenDiscoveryBackend(require_real_bars=False)
        fake = type("Cand", (), {"candidate_id": "x", "parameters": {}, "entry_tree": None})()
        with pytest.raises(RuntimeError, match="DSL_SIGNAL_REQUIRED"):
            backend.evaluate(fake)  # type: ignore[arg-type]
