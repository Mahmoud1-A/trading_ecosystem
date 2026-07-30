"""Phase 5 acceptance tests — rolling walk-forward + purge/embargo."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.walk_forward_config import WalkForwardConfig
from validation import assert_no_overlap, expand_param_grid, generate_rolling_folds, run_walk_forward
from validation.purge_embargo import apply_purge_embargo


TZ = ZoneInfo("America/Chicago")


def _bars(n: int = 2000) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz=TZ)
    rng = np.random.default_rng(0)
    px = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": px,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "volume": 1000.0,
        }
    )


class TestPurgeEmbargo:
    def test_no_train_validation_overlap(self) -> None:
        s = apply_purge_embargo(
            train_end=100,
            validation_start=100,
            validation_end=150,
            purge_gap_bars=5,
            embargo_gap_bars=5,
            n_bars=200,
        )
        # Fix train_start for absolute window
        from dataclasses import replace

        s = replace(s, train_start=0)
        assert_no_overlap(s)
        assert s.train_end == 95
        assert s.validation_start == 105

    def test_embargo_cannot_consume_validation(self) -> None:
        with pytest.raises(ValueError, match="Embargo"):
            apply_purge_embargo(
                train_end=100,
                validation_start=100,
                validation_end=103,
                purge_gap_bars=0,
                embargo_gap_bars=10,
                n_bars=200,
            )


class TestRollingWFO:
    def test_not_single_7030_split(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=5,
            validation_window_days=2,
            step_forward_days=2,
            bars_per_day=78,
            purge_gap_bars=1,
            embargo_gap_bars=1,
        )
        bars = _bars(5 * 78 + 2 * 78 + 2 * 78 + 50)
        folds = generate_rolling_folds(pd.DatetimeIndex(bars["timestamp"]), cfg)
        assert len(folds) >= 2
        # Windows roll forward
        assert folds[1].slice.train_start > folds[0].slice.train_start

    def test_timestamps_ordered_and_no_leakage(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=3,
            validation_window_days=1,
            step_forward_days=1,
            bars_per_day=78,
            purge_gap_bars=2,
            embargo_gap_bars=2,
            param_grid={"lookback": [10, 20]},
        )
        bars = _bars(3 * 78 + 1 * 78 + 1 * 78 + 100)
        folds = generate_rolling_folds(pd.DatetimeIndex(bars["timestamp"]), cfg)
        for f in folds:
            assert_no_overlap(f.slice)
            assert f.train_end_ts < f.validation_start_ts

    def test_run_walk_forward_persists_all_params(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=3,
            validation_window_days=1,
            step_forward_days=1,
            bars_per_day=78,
            param_grid={"lookback": [10, 15], "z_entry": [1.5, 2.0]},
        )
        bars = _bars(3 * 78 + 1 * 78 + 1 * 78 + 80)
        persisted: list[dict] = []

        def evaluate(window: pd.DataFrame, params: dict) -> float:
            return float(params.get("lookback", 0)) + float(params.get("z_entry", 0))

        result = run_walk_forward(
            bars,
            cfg,
            evaluate_fn=evaluate,
            persist_trial_fn=persisted.append,
        )
        assert len(result.folds) >= 1
        assert len(result.evaluated_parameter_sets) > 0
        assert len(persisted) == len(result.evaluated_parameter_sets)
        # Grid size * folds (train) + folds (validation)
        grid = expand_param_grid(cfg.param_grid)
        assert len(grid) == 4

    def test_param_grid_expansion(self) -> None:
        grid = expand_param_grid({"a": [1, 2], "b": [3]})
        assert grid == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
