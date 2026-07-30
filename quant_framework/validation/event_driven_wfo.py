"""
Event-driven walk-forward optimization — the institutional evaluation path.

Every fold runs the full pipeline:

    Strategy -> features -> signals -> orders -> fills -> friction -> portfolio
    -> PropRiskFSM -> fold metrics -> registry

There is no vectorized shortcut on the default path: parameter evaluation goes
through :class:`~engine.event_execution.EventExecutionEngine`, so timing, friction,
partial fills and prop-risk gating all apply. A cheap proxy metric is still
available, but only through the explicitly named
:func:`proxy_metric_evaluate_fn` helper so that its use is never implicit.

Candidate ranking uses validation (out-of-sample) results only. In-sample metrics
are recorded for diagnostics and parameter selection inside a fold, and
:attr:`EventDrivenWFOResult.ranking_source` records that fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from config.asset_spec import CFDAssetSpec, FuturesAssetSpec
from config.cost_model import CostModel
from config.prop_profile import PropProfile
from config.walk_forward_config import WalkForwardConfig
from engine.event_execution import (
    EventExecutionEngine,
    ExecutionConfig,
    ExecutionResult,
    bars_from_frame,
)
from engine.events import BarEvent, SignalEvent
from engine.ids import DeterministicIdFactory
from engine.portfolio import Portfolio
from metrics.diagnostics import cost_to_gross, exposure_time
from metrics.performance import compute_metrics
from registry.experiment_registry import ExperimentRegistry, TrialStatus
from registry.hashing import sha256_json
from risk.fsm import PropRiskFSM
from validation.walk_forward import FoldSpec, expand_param_grid, generate_rolling_folds

logger = logging.getLogger(__name__)

RANKING_SOURCE = "validation_oos"
CALCULATION_VERSION = "event_wfo_v1"

SignalFn = Callable[[BarEvent, Portfolio, int], SignalEvent | list[SignalEvent] | None]
SignalFnFactory = Callable[[dict[str, Any], pd.DataFrame], SignalFn]

COST_FIELDS = (
    "commission",
    "spread_cost",
    "slippage_cost",
    "financing_cost",
    "rollover_cost",
    "market_impact_cost",
)


class RankingSourceError(ValueError):
    """Raised when a caller tries to rank candidates on in-sample results."""


@dataclass
class WindowRun:
    """Full event-driven result for one (fold, params, phase) window."""

    fold_id: int
    phase: str
    params: dict[str, Any]
    start_ts: str
    end_ts: str
    n_bars: int
    metric: float
    gross_metrics: dict[str, Any]
    net_metrics: dict[str, Any]
    cost_attribution: dict[str, Any]
    orders: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    risk_transitions: list[dict[str, Any]] = field(default_factory=list)
    rejected_signals: list[dict[str, Any]] = field(default_factory=list)
    financing_events: list[dict[str, Any]] = field(default_factory=list)
    rollover_decisions: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "phase": self.phase,
            "params": dict(self.params),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "n_bars": self.n_bars,
            "metric": self.metric,
            "gross_metrics": dict(self.gross_metrics),
            "net_metrics": dict(self.net_metrics),
            "cost_attribution": dict(self.cost_attribution),
            "orders": list(self.orders),
            "fills": list(self.fills),
            "trades": list(self.trades),
            "risk_transitions": list(self.risk_transitions),
            "rejected_signals": list(self.rejected_signals),
            "financing_events": list(self.financing_events),
            "rollover_decisions": list(self.rollover_decisions),
            "equity_curve": list(self.equity_curve),
            "failure_reason": self.failure_reason,
        }


@dataclass
class EventFoldResult:
    """Persisted record of one walk-forward fold."""

    fold_id: int
    best_params: dict[str, Any]
    train_metric: float
    validation_metric: float
    train_start_ts: str
    train_end_ts: str
    validation_start_ts: str
    validation_end_ts: str
    purge_gap_bars: int
    embargo_gap_bars: int
    train_run: WindowRun | None
    validation_run: WindowRun | None
    all_runs: list[WindowRun] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def ranking_score(self) -> float:
        """Fold ranking score — validation (OOS) only, never the train metric."""
        return float(self.validation_metric)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "best_params": dict(self.best_params),
            "train_metric": self.train_metric,
            "validation_metric": self.validation_metric,
            "ranking_score": self.ranking_score,
            "ranking_source": RANKING_SOURCE,
            "train_start_ts": self.train_start_ts,
            "train_end_ts": self.train_end_ts,
            "validation_start_ts": self.validation_start_ts,
            "validation_end_ts": self.validation_end_ts,
            "purge_gap_bars": self.purge_gap_bars,
            "embargo_gap_bars": self.embargo_gap_bars,
            "train_run": self.train_run.as_dict() if self.train_run else None,
            "validation_run": self.validation_run.as_dict() if self.validation_run else None,
            "orders": list(self.validation_run.orders) if self.validation_run else [],
            "fills": list(self.validation_run.fills) if self.validation_run else [],
            "trades": list(self.validation_run.trades) if self.validation_run else [],
            "risk_transitions": (
                list(self.validation_run.risk_transitions) if self.validation_run else []
            ),
            "gross_metrics": dict(self.validation_run.gross_metrics) if self.validation_run else {},
            "net_metrics": dict(self.validation_run.net_metrics) if self.validation_run else {},
            "cost_attribution": (
                dict(self.validation_run.cost_attribution) if self.validation_run else {}
            ),
            "failure_reason": self.failure_reason,
        }


@dataclass
class EventDrivenWFOResult:
    folds: list[EventFoldResult]
    aggregated_validation_metric: float
    config: WalkForwardConfig
    ranking_metric: str
    ranking_source: str = RANKING_SOURCE
    calculation_version: str = CALCULATION_VERSION
    all_runs: list[WindowRun] = field(default_factory=list)
    trial_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fold_ranking_scores(self) -> list[float]:
        return [f.ranking_score for f in self.folds]

    def aggregate_from_folds(self) -> float:
        """Recompute the aggregate directly from per-fold OOS results."""
        if not self.folds:
            return 0.0
        return float(sum(self.fold_ranking_scores) / len(self.folds))

    def as_dict(self) -> dict[str, Any]:
        return {
            "aggregated_validation_metric": self.aggregated_validation_metric,
            "ranking_metric": self.ranking_metric,
            "ranking_source": self.ranking_source,
            "calculation_version": self.calculation_version,
            "folds": [f.as_dict() for f in self.folds],
        }


def proxy_metric_evaluate_fn(window: pd.DataFrame, params: dict[str, Any]) -> float:
    """
    Explicitly named cheap proxy metric — NOT the institutional path.

    Kept only for fast smoke runs and grid pre-screening. It ignores execution
    timing, friction, sizing and prop-risk gating, so a score produced here must
    never be presented as a backtest result or used for candidate ranking.
    """
    lookback = int(params.get("lookback", 20))
    if len(window) < lookback + 5:
        return float("-inf")
    close = window["close"].astype(float)
    z = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std(ddof=0)
    return float(z.dropna().abs().mean())


def _friction_total(costs: dict[str, Any]) -> float:
    return float(sum(float(costs.get(k, 0.0) or 0.0) for k in COST_FIELDS))


def _cost_attribution(result: ExecutionResult) -> dict[str, Any]:
    totals = {k: 0.0 for k in COST_FIELDS}
    for fill in result.fills:
        cost_dict = fill.costs.as_dict()
        for k in COST_FIELDS:
            totals[k] += float(cost_dict.get(k, 0.0) or 0.0)
    financing_cash = float(sum(e.amount for e in result.financing_events))
    totals["financing_cost"] += financing_cash
    trades = pd.DataFrame([t.as_dict() for t in result.portfolio.trades])
    ratio = cost_to_gross(trades if not trades.empty else None)
    return {
        **totals,
        "total_friction": float(sum(totals.values())),
        "financing_cash_debits": financing_cash,
        "cost_to_gross": ratio.as_dict(),
    }


def _equity_series(portfolio: Portfolio) -> pd.Series:
    if not portfolio.equity_curve:
        return pd.Series(dtype=float)
    return pd.Series(
        [float(e) for _, e in portfolio.equity_curve],
        index=pd.DatetimeIndex([t for t, _ in portfolio.equity_curve]),
        name="equity",
    )


def _gross_equity_series(portfolio: Portfolio, result: ExecutionResult) -> pd.Series:
    """
    Equity with all friction added back, aligned to the same mark timestamps.

    Gross is the counterfactual "no friction" curve, so gross vs net isolates the
    cost drag rather than a different trading path.
    """
    net = _equity_series(portfolio)
    if net.empty:
        return net
    events: list[tuple[pd.Timestamp, float]] = [
        (f.fill_timestamp, _friction_total(f.costs.as_dict())) for f in result.fills
    ]
    events.extend((e.timestamp, float(e.amount)) for e in result.financing_events)
    events.sort(key=lambda kv: kv[0])
    cumulative = 0.0
    cursor = 0
    out: list[float] = []
    for ts, value in zip(net.index, net.to_numpy(), strict=True):
        while cursor < len(events) and events[cursor][0] <= ts:
            cumulative += events[cursor][1]
            cursor += 1
        out.append(float(value) + cumulative)
    return pd.Series(out, index=net.index, name="gross_equity")


def _metric_value(metrics: Any, name: str) -> float:
    value = getattr(metrics, name, None)
    if value is None:
        raise ValueError(f"Unknown ranking metric {name!r}")
    val = float(value)
    return val if np.isfinite(val) else 0.0


@dataclass
class EventWFOContext:
    """Everything the event path needs to build an engine for one window."""

    asset: FuturesAssetSpec | CFDAssetSpec
    cost_model: CostModel
    prop_profile: PropProfile | None = None
    starting_equity: float = 100_000.0
    exec_config: ExecutionConfig | None = None
    bars_per_year: int = 252 * 78
    freq: str = "5min"
    run_id: str = "event_wfo"

    @property
    def symbol(self) -> str:
        return (
            self.asset.symbol
            if isinstance(self.asset, FuturesAssetSpec)
            else self.asset.broker_symbol
        )

    @property
    def contract(self) -> str:
        return (
            self.asset.active_contract
            if isinstance(self.asset, FuturesAssetSpec)
            else self.asset.broker_symbol
        )


def run_event_window(
    window: pd.DataFrame,
    params: dict[str, Any],
    *,
    context: EventWFOContext,
    signal_fn_factory: SignalFnFactory,
    fold_id: int,
    phase: str,
    ranking_metric: str = "sharpe",
) -> WindowRun:
    """
    Run one window through the full event-driven pipeline and score it.

    Deterministic: the engine receives a :class:`DeterministicIdFactory` keyed by
    run/candidate/fold, so identical inputs produce identical order and fill IDs.
    """
    candidate_id = sha256_json({"params": params})[:16]
    start_ts = str(window.index[0]) if len(window) else ""
    end_ts = str(window.index[-1]) if len(window) else ""

    def _empty(reason: str) -> WindowRun:
        return WindowRun(
            fold_id=fold_id,
            phase=phase,
            params=dict(params),
            start_ts=start_ts,
            end_ts=end_ts,
            n_bars=int(len(window)),
            metric=0.0,
            gross_metrics={},
            net_metrics={},
            cost_attribution={},
            failure_reason=reason,
        )

    if len(window) < 2:
        return _empty("insufficient_bars")

    risk = (
        PropRiskFSM(context.prop_profile, starting_equity=context.starting_equity)
        if context.prop_profile is not None
        else None
    )
    engine = EventExecutionEngine(
        context.asset,
        context.cost_model,
        starting_equity=context.starting_equity,
        exec_config=context.exec_config or ExecutionConfig(),
        risk_manager=risk,
        id_factory=DeterministicIdFactory(
            run_id=context.run_id, candidate_id=f"{candidate_id}:{phase}", fold_id=fold_id
        ),
    )
    frame = window.reset_index()
    bars = bars_from_frame(
        frame, symbol=context.symbol, contract=context.contract, freq=context.freq
    )
    signal_fn = signal_fn_factory(params, window)
    result = engine.run(bars, signal_fn)

    net_equity = _equity_series(result.portfolio)
    gross_equity = _gross_equity_series(result.portfolio, result)
    trades = pd.DataFrame([t.as_dict() for t in result.portfolio.trades])
    net = compute_metrics(
        net_equity,
        trades=trades if not trades.empty else None,
        bars_per_year=context.bars_per_year,
        starting_equity=context.starting_equity,
    )
    gross = compute_metrics(
        gross_equity,
        trades=trades if not trades.empty else None,
        bars_per_year=context.bars_per_year,
        starting_equity=context.starting_equity,
    )
    net_dict = net.to_dict()
    net_dict["exposure_time"] = exposure_time(result.portfolio).as_dict()

    risk_transitions = [
        e.as_dict() for e in (risk.events if risk is not None else [])
    ]
    failure = None if result.portfolio.trades else "no_closed_trades"

    return WindowRun(
        fold_id=fold_id,
        phase=phase,
        params=dict(params),
        start_ts=start_ts,
        end_ts=end_ts,
        n_bars=int(len(window)),
        metric=_metric_value(net, ranking_metric),
        gross_metrics=gross.to_dict(),
        net_metrics=net_dict,
        cost_attribution=_cost_attribution(result),
        orders=[o.snapshot() for o in result.orders],
        fills=[f.as_dict() for f in result.fills],
        trades=[t.as_dict() for t in result.portfolio.trades],
        risk_transitions=risk_transitions,
        rejected_signals=list(result.rejected_signals),
        financing_events=[e.as_dict() for e in result.financing_events],
        rollover_decisions=[d.as_dict() for d in result.rollover_decisions],
        equity_curve=[(str(t), float(e)) for t, e in result.portfolio.equity_curve],
        failure_reason=failure,
    )


def run_event_driven_wfo(
    bars: pd.DataFrame,
    config: WalkForwardConfig,
    *,
    signal_fn_factory: SignalFnFactory,
    context: EventWFOContext,
    ranking_metric: str = "sharpe",
    registry: ExperimentRegistry | None = None,
    registry_context: dict[str, Any] | None = None,
    rank_on: str = RANKING_SOURCE,
) -> EventDrivenWFOResult:
    """
    Rolling walk-forward over the event-driven institutional path.

    For every fold and parameter set the train window is run through the engine to
    select parameters; the selected parameters are then run on the purged/embargoed
    validation window. The fold's ranking score is the validation metric only.
    """
    if rank_on != RANKING_SOURCE:
        raise RankingSourceError(
            f"Candidate ranking must use {RANKING_SOURCE!r}; in-sample ranking is not allowed "
            f"(got {rank_on!r})"
        )

    work = bars.copy()
    if "timestamp" in work.columns:
        work = work.set_index("timestamp")
    if work.index.tz is None:
        raise ValueError("Event-driven WFO requires timezone-aware timestamps")
    work = work.sort_index()

    folds: list[FoldSpec] = generate_rolling_folds(pd.DatetimeIndex(work.index), config)
    grid = expand_param_grid(config.param_grid)

    fold_results: list[EventFoldResult] = []
    all_runs: list[WindowRun] = []
    trial_records: list[dict[str, Any]] = []

    for fold in folds:
        train_df = work.iloc[fold.slice.train_start : fold.slice.train_end]
        val_df = work.iloc[fold.slice.validation_start : fold.slice.validation_end]

        best_run: WindowRun | None = None
        for params in grid:
            run = run_event_window(
                train_df,
                params,
                context=context,
                signal_fn_factory=signal_fn_factory,
                fold_id=fold.fold_id,
                phase="train",
                ranking_metric=ranking_metric,
            )
            all_runs.append(run)
            if best_run is None or run.metric > best_run.metric:
                best_run = run

        assert best_run is not None
        val_run = run_event_window(
            val_df,
            best_run.params,
            context=context,
            signal_fn_factory=signal_fn_factory,
            fold_id=fold.fold_id,
            phase="validation",
            ranking_metric=ranking_metric,
        )
        all_runs.append(val_run)

        fold_result = EventFoldResult(
            fold_id=fold.fold_id,
            best_params=dict(best_run.params),
            train_metric=best_run.metric,
            validation_metric=val_run.metric,
            train_start_ts=str(fold.train_start_ts),
            train_end_ts=str(fold.train_end_ts),
            validation_start_ts=str(fold.validation_start_ts),
            validation_end_ts=str(fold.validation_end_ts),
            purge_gap_bars=int(config.purge_gap_bars),
            embargo_gap_bars=int(config.embargo_gap_bars),
            train_run=best_run,
            validation_run=val_run,
            all_runs=[r for r in all_runs if r.fold_id == fold.fold_id],
            failure_reason=val_run.failure_reason,
        )
        fold_results.append(fold_result)
        logger.info(
            "Event WFO fold=%s train=%.6f val=%.6f params=%s",
            fold.fold_id,
            best_run.metric,
            val_run.metric,
            best_run.params,
        )

        if registry is not None:
            trial_records.append(
                _persist_fold(
                    registry,
                    fold_result,
                    context=context,
                    ranking_metric=ranking_metric,
                    registry_context=registry_context or {},
                )
            )

    aggregated = (
        float(sum(f.validation_metric for f in fold_results) / len(fold_results))
        if fold_results
        else 0.0
    )
    return EventDrivenWFOResult(
        folds=fold_results,
        aggregated_validation_metric=aggregated,
        config=config,
        ranking_metric=ranking_metric,
        all_runs=all_runs,
        trial_records=trial_records,
    )


def _persist_fold(
    registry: ExperimentRegistry,
    fold: EventFoldResult,
    *,
    context: EventWFOContext,
    ranking_metric: str,
    registry_context: dict[str, Any],
) -> dict[str, Any]:
    """Persist a fold as a registry trial, including its failure reason."""
    val = fold.validation_run
    record = registry.create_trial(
        candidate_id=registry_context.get("candidate_id", f"fold{fold.fold_id}"),
        lineage_id=registry_context.get("lineage_id", "event_wfo"),
        strategy_family=registry_context.get("strategy_family", "event_driven"),
        parameters=dict(fold.best_params),
        config_snapshot=registry_context.get("config_snapshot", {}),
        system_version=registry_context.get("system_version", "unknown"),
        data_hash=registry_context.get("data_hash", "unknown"),
        random_seed=int(
            (context.exec_config or ExecutionConfig()).random_seed
        ),
        cost_model_version=context.cost_model.version,
        code_hash=registry_context.get("code_hash", "unknown"),
        gross_metrics=dict(val.gross_metrics) if val else {},
        net_metrics=dict(val.net_metrics) if val else {},
        ranking_score=fold.ranking_score,
        rejection_reason=fold.failure_reason,
        train_window={"start": fold.train_start_ts, "end": fold.train_end_ts},
        validation_window={
            "start": fold.validation_start_ts,
            "end": fold.validation_end_ts,
        },
        execution_assumptions={
            "ranking_metric": ranking_metric,
            "ranking_source": RANKING_SOURCE,
            "purge_gap_bars": fold.purge_gap_bars,
            "embargo_gap_bars": fold.embargo_gap_bars,
            "engine": "EventExecutionEngine",
            "calculation_version": CALCULATION_VERSION,
        },
        trial_status=TrialStatus.COMPLETED,
        fold_records=[fold.as_dict()],
        ranking_source=RANKING_SOURCE,
    )
    return record.as_dict()


__all__ = [
    "CALCULATION_VERSION",
    "RANKING_SOURCE",
    "EventDrivenWFOResult",
    "EventFoldResult",
    "EventWFOContext",
    "RankingSourceError",
    "WindowRun",
    "proxy_metric_evaluate_fn",
    "run_event_driven_wfo",
    "run_event_window",
]
