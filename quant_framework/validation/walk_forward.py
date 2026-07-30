"""Rolling walk-forward optimization framework."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from config.walk_forward_config import WalkForwardConfig
from validation.candidate_lineage import CandidateLineage
from validation.purge_embargo import WindowSlice, assert_no_overlap, timestamps_ordered

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_start_ts: pd.Timestamp
    train_end_ts: pd.Timestamp
    validation_start_ts: pd.Timestamp
    validation_end_ts: pd.Timestamp
    slice: WindowSlice


@dataclass
class FoldResult:
    fold_id: int
    best_params: dict[str, Any]
    train_metric: float
    validation_metric: float
    all_trials: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    aggregated_validation_metric: float
    config: WalkForwardConfig
    evaluated_parameter_sets: list[dict[str, Any]] = field(default_factory=list)


def _days_to_bars(days: int, bars_per_day: int) -> int:
    return int(days) * int(bars_per_day)


def generate_rolling_folds(
    timestamps: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[FoldSpec]:
    """
    Generate rolling train/validation windows with purge and embargo.

    Does NOT implement a single 70/30 split.
    """
    if not timestamps_ordered(timestamps):
        raise ValueError("Timestamps must be strictly ordered ascending")
    n = len(timestamps)
    train_bars = _days_to_bars(config.train_window_days, config.bars_per_day)
    val_bars = _days_to_bars(config.validation_window_days, config.bars_per_day)
    step_bars = _days_to_bars(config.step_forward_days, config.bars_per_day)

    if train_bars + val_bars + config.purge_gap_bars + config.embargo_gap_bars > n:
        raise ValueError(
            f"Insufficient bars ({n}) for train={train_bars} val={val_bars} "
            f"purge={config.purge_gap_bars} embargo={config.embargo_gap_bars}"
        )

    folds: list[FoldSpec] = []
    fold_id = 0
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_bars
        # validation begins after train (+ purge handled inside)
        val_start_raw = train_end
        val_end = val_start_raw + val_bars + config.embargo_gap_bars
        if val_end > n:
            break

        # apply_purge_embargo expects absolute indices with train starting at 0 in relative
        # Recompute absolute:
        purged_train_end = train_end - config.purge_gap_bars
        if purged_train_end <= train_start:
            raise ValueError("purge_gap_bars consumes entire train window")
        emb_end = train_end + config.embargo_gap_bars
        val_start = emb_end
        val_end = val_start + val_bars
        if val_end > n:
            break

        if config.max_folds is not None and len(folds) >= int(config.max_folds):
            break

        slice_ = WindowSlice(
            train_start=train_start,
            train_end=purged_train_end,
            validation_start=val_start,
            validation_end=val_end,
            purge_start=purged_train_end,
            purge_end=train_end,
            embargo_start=train_end,
            embargo_end=emb_end,
        )
        assert_no_overlap(slice_)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start_ts=pd.Timestamp(timestamps[slice_.train_start]),
                train_end_ts=pd.Timestamp(timestamps[slice_.train_end - 1]),
                validation_start_ts=pd.Timestamp(timestamps[slice_.validation_start]),
                validation_end_ts=pd.Timestamp(timestamps[slice_.validation_end - 1]),
                slice=slice_,
            )
        )
        fold_id += 1
        start += step_bars
        if start + train_bars + val_bars + config.embargo_gap_bars > n:
            break

    if not folds:
        raise ValueError("No walk-forward folds generated")
    return folds


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


MetricFn = Callable[[pd.DataFrame, dict[str, Any]], float]


def run_walk_forward(
    bars: pd.DataFrame,
    config: WalkForwardConfig,
    *,
    evaluate_fn: MetricFn,
    persist_trial_fn: Callable[[dict[str, Any]], None] | None = None,
) -> WalkForwardResult:
    """
    Rolling WFO:

    1. Train on historical window (optimize over param_grid)
    2. Validate on next window with best params
    3. Step forward and repeat
    4. Aggregate validation metrics
    5. Persist all evaluated parameter sets
    """
    work = bars.copy()
    if "timestamp" in work.columns:
        work = work.set_index("timestamp")
    if work.index.tz is None:
        raise ValueError("WFO requires timezone-aware timestamps")
    if not timestamps_ordered(work.index):
        work = work.sort_index()

    folds = generate_rolling_folds(pd.DatetimeIndex(work.index), config)
    grid = expand_param_grid(config.param_grid)
    fold_results: list[FoldResult] = []
    all_params: list[dict[str, Any]] = []

    for fold in folds:
        train_df = work.iloc[fold.slice.train_start : fold.slice.train_end]
        val_df = work.iloc[fold.slice.validation_start : fold.slice.validation_end]
        trials: list[dict[str, Any]] = []
        best_params: dict[str, Any] = {}
        best_train = float("-inf")

        for params in grid:
            metric = float(evaluate_fn(train_df, params))
            trial = {
                "fold_id": fold.fold_id,
                "phase": "train",
                "params": params,
                "metric": metric,
                "train_start": str(fold.train_start_ts),
                "train_end": str(fold.train_end_ts),
            }
            trials.append(trial)
            all_params.append(trial)
            if persist_trial_fn is not None:
                persist_trial_fn(trial)
            if metric > best_train:
                best_train = metric
                best_params = params

        val_metric = float(evaluate_fn(val_df, best_params))
        val_trial = {
            "fold_id": fold.fold_id,
            "phase": "validation",
            "params": best_params,
            "metric": val_metric,
            "validation_start": str(fold.validation_start_ts),
            "validation_end": str(fold.validation_end_ts),
        }
        trials.append(val_trial)
        all_params.append(val_trial)
        if persist_trial_fn is not None:
            persist_trial_fn(val_trial)

        fold_results.append(
            FoldResult(
                fold_id=fold.fold_id,
                best_params=best_params,
                train_metric=best_train,
                validation_metric=val_metric,
                all_trials=trials,
            )
        )
        logger.info(
            "WFO fold=%s train=%.4f val=%.4f params=%s",
            fold.fold_id,
            best_train,
            val_metric,
            best_params,
        )

    agg = float(sum(f.validation_metric for f in fold_results) / len(fold_results))
    return WalkForwardResult(
        folds=fold_results,
        aggregated_validation_metric=agg,
        config=config,
        evaluated_parameter_sets=all_params,
    )
