"""Compile StrategyCandidate DSL trees into EventExecutionEngine signal_fn factories.

Alpha Miner must evaluate ENTRY_LONG / EXIT_SIGNAL / stop / target / regime_gates —
never the mean-reversion parameter stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from discovery.candidate import StrategyCandidate
from discovery.dsl_series import (
    time_exit_bars,
    unwrap_boolean_intent,
    unwrap_distance,
)
from discovery.expression_tree import ExprNode
from discovery.operators import OperatorId
from discovery.types import NodeKind
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.portfolio import Portfolio
from features.generator import FeatureGenerator, FeatureGeneratorConfig


SignalFn = Callable[[BarEvent, Portfolio, int], SignalEvent | list[SignalEvent] | None]
SignalFnFactory = Callable[[dict[str, Any], pd.DataFrame], SignalFn]


@dataclass
class CompiledDSLSignals:
    """Per-bar boolean/distance series for one WFO window."""

    entry_long: pd.Series
    entry_short: pd.Series
    exit_signal: pd.Series
    session_exit: pd.Series
    regime_ok: pd.Series
    stop_distance: pd.Series | None = None
    target_distance: pd.Series | None = None
    time_exit_max_bars: float | None = None
    feature_row_count: int = 0
    entry_true_count: int = 0
    exit_true_count: int = 0


@dataclass
class DSLCompileStats:
    """Aggregated compile diagnostics for artifacts."""

    feature_ids: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    last_window_bars: int = 0
    last_entry_true: int = 0
    last_exit_true: int = 0


def _root_op(node: ExprNode | None) -> OperatorId | None:
    if node is None or node.kind is not NodeKind.OPERATOR:
        return None
    try:
        return OperatorId(node.name)
    except ValueError:
        return None


def _prepare_bars(window: pd.DataFrame) -> pd.DataFrame:
    work = window.copy()
    if "timestamp" in work.columns:
        work = work.set_index("timestamp")
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("DSL signal factory requires DatetimeIndex or timestamp column")
    if work.index.tz is None:
        raise ValueError("DSL signal factory requires timezone-aware bars")
    return work.sort_index()


def _feature_frame_for_window(
    bars: pd.DataFrame,
    feature_ids: tuple[str, ...],
    *,
    bar_end_offset: str,
) -> tuple[pd.DataFrame, list[str]]:
    fg = FeatureGenerator(config=FeatureGeneratorConfig(bar_end_offset=bar_end_offset))
    # Generate all eligible features then select — avoids dropping needed ids
    # when a subset request would empty the frame.
    full = fg.generate(bars).values
    missing: list[str] = []
    cols: dict[str, pd.Series] = {}
    for fid in feature_ids:
        if fid in full.columns:
            cols[fid] = full[fid].astype(float)
        else:
            missing.append(fid)
            cols[fid] = pd.Series(np.nan, index=bars.index, dtype=float)
    if not feature_ids:
        # Still provide price/vol staples so simple trees can run in tests
        for fid in ("price.rolling_z_20", "price.simple_return_1", "vol.atr_14"):
            if fid in full.columns:
                cols[fid] = full[fid].astype(float)
    frame = pd.DataFrame(cols, index=bars.index) if cols else pd.DataFrame(index=bars.index)
    return frame, missing


def compile_candidate_dsl(
    candidate: StrategyCandidate,
    features: pd.DataFrame,
    *,
    bindings: dict[str, float] | None = None,
) -> CompiledDSLSignals:
    """Compile entry/exit/stop/target/regime trees over a feature frame."""
    bindings = {**dict(candidate.parameters), **(bindings or {})}
    index = features.index

    entry_long = pd.Series(False, index=index)
    entry_short = pd.Series(False, index=index)
    root = _root_op(candidate.entry_tree)
    intent = unwrap_boolean_intent(candidate.entry_tree, features, bindings)
    if root is OperatorId.ENTRY_SHORT:
        entry_short = intent
    else:
        # ENTRY_LONG or bare boolean condition treated as long entry
        entry_long = intent

    exit_signal = unwrap_boolean_intent(candidate.exit_tree, features, bindings)
    session_exit = pd.Series(False, index=index)
    if candidate.exit_tree is not None and _root_op(candidate.exit_tree) is OperatorId.SESSION_EXIT:
        session_exit = exit_signal

    regime_ok = pd.Series(True, index=index)
    for gate in candidate.regime_gates:
        regime_ok = regime_ok & unwrap_boolean_intent(gate, features, bindings)

    stop_distance = unwrap_distance(candidate.stop, features, bindings)
    target_distance = unwrap_distance(candidate.target, features, bindings)
    t_exit = time_exit_bars(candidate.exit_tree, bindings)

    entry_long = entry_long & regime_ok
    entry_short = entry_short & regime_ok

    return CompiledDSLSignals(
        entry_long=entry_long.fillna(False),
        entry_short=entry_short.fillna(False),
        exit_signal=exit_signal.fillna(False),
        session_exit=session_exit.fillna(False),
        regime_ok=regime_ok.fillna(True),
        stop_distance=stop_distance,
        target_distance=target_distance,
        time_exit_max_bars=t_exit,
        feature_row_count=len(features),
        entry_true_count=int(entry_long.fillna(False).sum() + entry_short.fillna(False).sum()),
        exit_true_count=int(exit_signal.fillna(False).sum()),
    )


def make_dsl_signal_fn_factory(
    candidate: StrategyCandidate,
    *,
    symbol: str,
    quantity: float = 1.0,
    bar_end_offset: str = "5min",
    intraday_only: bool = False,
    stats: DSLCompileStats | None = None,
) -> SignalFnFactory:
    """
    Build a ``signal_fn_factory(params, window)`` that executes the candidate DSL.

    Closed over the candidate trees — ``params`` only override parameter bindings.
    """
    compile_stats = stats if stats is not None else DSLCompileStats()
    compile_stats.feature_ids = list(candidate.feature_ids)

    def factory(params: dict[str, Any], window: pd.DataFrame) -> SignalFn:
        bars = _prepare_bars(window)
        bindings = {**dict(candidate.parameters), **{k: float(v) for k, v in params.items()}}
        features, missing = _feature_frame_for_window(
            bars,
            candidate.feature_ids,
            bar_end_offset=bar_end_offset,
        )
        compile_stats.missing_features = list(missing)
        compile_stats.last_window_bars = len(bars)

        compiled = compile_candidate_dsl(candidate, features, bindings=bindings)
        compile_stats.last_entry_true = compiled.entry_true_count
        compile_stats.last_exit_true = compiled.exit_true_count

        # Align to positional index used by EventExecutionEngine
        entry_long = compiled.entry_long.reindex(bars.index).fillna(False).to_numpy(dtype=bool)
        entry_short = compiled.entry_short.reindex(bars.index).fillna(False).to_numpy(dtype=bool)
        exit_sig = compiled.exit_signal.reindex(bars.index).fillna(False).to_numpy(dtype=bool)
        session_exit = compiled.session_exit.reindex(bars.index).fillna(False).to_numpy(dtype=bool)
        stop_dist = (
            compiled.stop_distance.reindex(bars.index).astype(float).to_numpy()
            if compiled.stop_distance is not None
            else None
        )
        target_dist = (
            compiled.target_distance.reindex(bars.index).astype(float).to_numpy()
            if compiled.target_distance is not None
            else None
        )
        closes = bars["close"].astype(float).to_numpy() if "close" in bars.columns else np.zeros(len(bars))
        session_close_flags = (
            bars["is_session_close"].astype(bool).to_numpy()
            if "is_session_close" in bars.columns
            else np.zeros(len(bars), dtype=bool)
        )
        if intraday_only and len(session_close_flags):
            # Recompute terminal flatten for this window slice
            from discovery.event_wfo_backend import _session_close_flags_for_window

            session_close_flags = np.asarray(
                _session_close_flags_for_window(bars), dtype=bool
            )

        time_exit_max = compiled.time_exit_max_bars
        entry_bar_index = {"i": -1}

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i < 0 or i >= len(closes):
                return None
            pos = portfolio.get_position(bar.symbol if bar.symbol else symbol)
            qty_open = 0.0 if pos is None or pos.is_flat else float(pos.quantity)
            timing = InformationTiming(
                source_timestamp=bar.timestamp,
                availability_timestamp=bar.bar_end,
                decision_timestamp=bar.bar_end,
            )
            sym = bar.symbol or symbol

            # Session / intraday flatten
            if qty_open != 0.0 and (
                bar.is_session_close
                or (i < len(session_close_flags) and bool(session_close_flags[i]))
                or (i < len(session_exit) and bool(session_exit[i]))
            ):
                return SignalEvent(
                    timing=timing,
                    symbol=sym,
                    side="FLAT",
                    quantity=abs(qty_open),
                    reason="dsl_session_exit",
                )

            # Time exit
            if (
                qty_open != 0.0
                and time_exit_max is not None
                and entry_bar_index["i"] >= 0
                and (i - entry_bar_index["i"]) >= int(time_exit_max)
            ):
                return SignalEvent(
                    timing=timing,
                    symbol=sym,
                    side="FLAT",
                    quantity=abs(qty_open),
                    reason="dsl_time_exit",
                )

            # Exit signal
            if qty_open != 0.0 and i < len(exit_sig) and bool(exit_sig[i]):
                return SignalEvent(
                    timing=timing,
                    symbol=sym,
                    side="FLAT",
                    quantity=abs(qty_open),
                    reason="dsl_exit_signal",
                )

            stop_px = None
            target_px = None
            px = float(closes[i]) if i < len(closes) else float(bar.close)

            def _brackets(side: str) -> tuple[float | None, float | None]:
                s = t = None
                if stop_dist is not None and i < len(stop_dist) and np.isfinite(stop_dist[i]):
                    d = float(stop_dist[i])
                    s = px - d if side == "BUY" else px + d
                if target_dist is not None and i < len(target_dist) and np.isfinite(target_dist[i]):
                    d = float(target_dist[i])
                    t = px + d if side == "BUY" else px - d
                return s, t

            # Entries only when flat
            if qty_open == 0.0 and i < len(entry_long) and bool(entry_long[i]):
                stop_px, target_px = _brackets("BUY")
                entry_bar_index["i"] = i
                return SignalEvent(
                    timing=timing,
                    symbol=sym,
                    side="BUY",
                    quantity=float(quantity),
                    stop_price=stop_px,
                    target_price=target_px,
                    reason="dsl_entry_long",
                )
            if qty_open == 0.0 and i < len(entry_short) and bool(entry_short[i]):
                stop_px, target_px = _brackets("SELL")
                entry_bar_index["i"] = i
                return SignalEvent(
                    timing=timing,
                    symbol=sym,
                    side="SELL",
                    quantity=float(quantity),
                    stop_price=stop_px,
                    target_price=target_px,
                    reason="dsl_entry_short",
                )
            return None

        return signal_fn

    return factory


__all__ = [
    "CompiledDSLSignals",
    "DSLCompileStats",
    "compile_candidate_dsl",
    "make_dsl_signal_fn_factory",
]
