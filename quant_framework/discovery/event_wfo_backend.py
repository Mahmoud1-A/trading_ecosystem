"""Institutional event-driven WFO backend for discovery candidates.

This is the control-plane / Phase-12 path for true FINALIST promotion.
It never uses :func:`validation.event_driven_wfo.proxy_metric_evaluate_fn`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.asset_spec import CFDAssetSpec, FuturesAssetSpec
from config.cost_model import CostModel, default_cfd_cost_model
from config.system_config import default_system_config
from config.walk_forward_config import WalkForwardConfig
from data.catalog.silver_bar_resolution import REAL_DATA_BACKEND_UNAVAILABLE
from discovery.candidate import StrategyCandidate
from discovery.fitness import FoldOOSMetrics
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.portfolio import Portfolio
from validation.event_driven_wfo import EventWFOContext, run_event_driven_wfo


TZ = ZoneInfo("America/Chicago")


def _synthetic_bars(n: int = 780, seed: int = 0) -> pd.DataFrame:
    """Timezone-aware bars sufficient for a small rolling WFO."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz=TZ)
    close = 4800 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(100, 2000, n).astype(float),
        }
    )


def mark_intraday_session_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark first/last bar of each UTC calendar day for INTRADAY_ONLY flatten.

    Always marks the final row of *this* frame as session-close so WFO validation
    slices (cut from a pre-marked full series) still flatten open lots at window end.
    """
    work = frame.copy()
    if "timestamp" in work.columns:
        ts = pd.to_datetime(work["timestamp"], utc=True)
    elif isinstance(work.index, pd.DatetimeIndex):
        ts = pd.to_datetime(work.index, utc=True)
    else:
        n = len(work)
        work["is_session_open"] = [False] * n
        work["is_session_close"] = [False] * n
        if n:
            work.iloc[0, work.columns.get_loc("is_session_open")] = True
            work.iloc[-1, work.columns.get_loc("is_session_close")] = True
        return work
    day = pd.Series(ts).dt.floor("D")
    # Align boolean masks to work's index
    open_mask = (day != day.shift(1)).fillna(True).to_numpy()
    close_mask = (day != day.shift(-1)).fillna(True).to_numpy()
    work["is_session_open"] = open_mask
    work["is_session_close"] = close_mask
    work.iloc[0, work.columns.get_loc("is_session_open")] = True
    work.iloc[-1, work.columns.get_loc("is_session_close")] = True
    return work


def _session_close_flags_for_window(window: pd.DataFrame) -> list[bool]:
    """Per-window session-close flags (recomputed; safe for WFO slices)."""
    marked = mark_intraday_session_boundaries(window)
    return marked["is_session_close"].astype(bool).tolist()


def mr_param_stub_signal_factory(params: dict[str, Any], window: pd.DataFrame):
    """TEST FIXTURE ONLY — z-score MR on close. Never used by Alpha Miner evaluate()."""
    lookback = int(params.get("lookback", 20))
    z_entry = abs(float(params.get("z_entry", 2.0)))
    if z_entry < 1e-9:
        z_entry = 2.0
    closes = window["close"].astype(float).tolist() if "close" in window.columns else []
    if "is_session_close" in window.columns:
        session_close = _session_close_flags_for_window(window)
        if len(session_close) < len(closes):
            session_close = session_close + [False] * (len(closes) - len(session_close))
    else:
        session_close = [False] * len(closes)

    def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
        pos = portfolio.get_position(bar.symbol)
        qty = 0.0 if pos is None or pos.is_flat else float(pos.quantity)
        timing = InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.bar_end,
            decision_timestamp=bar.bar_end,
        )
        if (bar.is_session_close or (i < len(session_close) and session_close[i])) and qty != 0:
            return SignalEvent(
                timing=timing,
                symbol=bar.symbol,
                side="FLAT",
                quantity=abs(qty),
                reason="intraday_session_close_flatten",
            )
        if i < lookback or i >= len(closes):
            return None
        slice_ = closes[max(0, i - lookback) : i]
        if len(slice_) < 2:
            return None
        mu = float(np.mean(slice_))
        sd = float(np.std(slice_)) or 1e-9
        z = (closes[i] - mu) / sd
        if z <= -z_entry and qty <= 0:
            return SignalEvent(
                timing=timing,
                symbol=bar.symbol,
                side="BUY",
                quantity=1.0,
                reason="mr_entry",
            )
        if z >= z_entry and qty >= 0:
            return SignalEvent(
                timing=timing,
                symbol=bar.symbol,
                side="SELL",
                quantity=1.0,
                reason="mr_entry_short",
            )
        if abs(z) < 0.25 and qty != 0:
            return SignalEvent(
                timing=timing,
                symbol=bar.symbol,
                side="FLAT",
                quantity=abs(qty),
                reason="mr_exit",
            )
        return None

    return signal_fn


# Back-compat alias for older tests that imported the private name.
_signal_factory_from_params = mr_param_stub_signal_factory


def known_trading_signal_factory(params: dict[str, Any], window: pd.DataFrame):
    """Deterministic strategy that must open and close at least one trade.

    Buys on ``entry_bar``, flats ``hold_bars`` later. Used as a pipeline probe
    so zero-trade regressions cannot hide behind weak alpha logic.
    """
    entry_bar = int(params.get("entry_bar", params.get("lookback", 5)))
    hold_bars = int(params.get("hold_bars", 3))
    qty = float(params.get("quantity", 1.0))

    def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
        pos = portfolio.get_position(bar.symbol)
        open_qty = 0.0 if pos is None or pos.is_flat else float(pos.quantity)
        timing = InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.bar_end,
            decision_timestamp=bar.bar_end,
        )
        if i == entry_bar and open_qty == 0.0:
            return SignalEvent(
                timing=timing,
                symbol=bar.symbol,
                side="BUY",
                quantity=qty,
                reason="known_trading_entry",
            )
        if i == entry_bar + hold_bars and open_qty != 0.0:
            return SignalEvent(
                timing=timing,
                symbol=bar.symbol,
                side="FLAT",
                quantity=abs(open_qty),
                reason="known_trading_exit",
            )
        return None

    return signal_fn


def trade_funnel_from_window_run(val: Any, signal_stats: dict[str, int] | None = None) -> dict[str, Any]:
    """Per-fold execution funnel: signals → orders → fills → positions → closes."""
    orders = list(getattr(val, "orders", None) or [])
    fills = list(getattr(val, "fills", None) or [])
    trades = list(getattr(val, "trades", None) or [])
    rejected_signals = list(getattr(val, "rejected_signals", None) or [])
    rejected_orders = [
        o
        for o in orders
        if str(o.get("status", "")).upper() in {"REJECTED", "CANCELLED"}
        or o.get("reject_reason")
    ]
    orders_submitted = sum(1 for o in orders if not o.get("reject_reason"))
    stats = signal_stats or {}
    return {
        "entry_true_count": int(stats.get("entry_true_count", 0)),
        "exit_true_count": int(stats.get("exit_true_count", 0)),
        "orders_submitted": int(orders_submitted),
        "fills": len(fills),
        "positions_opened": sum(
            1 for o in orders if not o.get("reduce_only") and not o.get("reject_reason")
        ),
        "positions_closed": len(trades),
        "rejected_orders": len(rejected_orders),
        "rejected_signals": len(rejected_signals),
        "reject_reasons": sorted(
            {str(o.get("reject_reason")) for o in orders if o.get("reject_reason")}
        ),
    }


def fold_oos_metrics_from_net(*, fold_id: int, net: dict[str, Any], fallback_metric: float) -> FoldOOSMetrics:
    required=("expectancy","sharpe","profit_factor","calmar","max_drawdown_pct","drawdown_duration_bars","worst_day_pct","turnover","n_trades")
    missing=[k for k in required if k not in net]
    if missing: raise RuntimeError("PERFORMANCE_METRIC_CONTRACT_MISMATCH: missing "+",".join(missing))
    n_trades = int(net["n_trades"])
    sharpe=float(net["sharpe"]); sharpe=sharpe if np.isfinite(sharpe) else float(fallback_metric)
    # Defense in depth: zero closed trades must not surface manufactured equity-path DD.
    if n_trades == 0:
        return FoldOOSMetrics(
            fold_id=fold_id,
            expectancy=0.0,
            sharpe=0.0,
            profit_factor=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            drawdown_duration=0.0,
            worst_day=0.0,
            turnover=0.0,
            prop_breach_prob=float(net.get("prop_breach_prob", 0.0) or 0.0),
            regime_entropy=float(net.get("regime_entropy", 1.0) or 1.0),
            n_trades=0,
        )
    return FoldOOSMetrics(fold_id=fold_id, expectancy=float(net["expectancy"]), sharpe=sharpe, profit_factor=float(net["profit_factor"]), calmar=float(net["calmar"]), max_drawdown=float(net["max_drawdown_pct"])/100.0, drawdown_duration=float(net["drawdown_duration_bars"]), worst_day=float(net["worst_day_pct"])/100.0, turnover=float(net["turnover"]), prop_breach_prob=float(net.get("prop_breach_prob",0.0) or 0.0), regime_entropy=float(net.get("regime_entropy",1.0) or 1.0), n_trades=n_trades)


@dataclass
class EventDrivenDiscoveryBackend:
    """
    Full event-driven WFO backend for StrategyCandidate evaluation.

    ``is_full_event_wfo=True`` — required for dashboard FINALIST promotion.
    When ``require_real_bars=True``, synthetic bars are forbidden.
    """

    bars: pd.DataFrame | None = None
    wfo_config: WalkForwardConfig | None = None
    backend_kind: str = "event_driven_wfo"
    is_full_event_wfo: bool = True
    require_real_bars: bool = False
    asset_class: Literal["futures", "cfd"] = "futures"
    intraday_only: bool = False
    artifact_dir: Path | str | None = None
    silver_resolution: dict[str, Any] = field(default_factory=dict)
    _context: EventWFOContext | None = field(default=None, repr=False)
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)
    available_feature_ids: frozenset[str] | None = None
    dataset_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.bars is None:
            if self.require_real_bars:
                raise RuntimeError(
                    f"{REAL_DATA_BACKEND_UNAVAILABLE}: bars not provided for real-data WFO"
                )
            self.bars = _synthetic_bars()
        if self.intraday_only and self.bars is not None:
            self.bars = mark_intraday_session_boundaries(self.bars)
        if self.wfo_config is None:
            self.wfo_config = WalkForwardConfig(
                train_window_days=2,
                validation_window_days=1,
                step_forward_days=1,
                purge_gap_bars=1,
                embargo_gap_bars=1,
                bars_per_day=78,
                max_folds=3,
                param_grid={"lookback": [10, 15], "z_entry": [1.5, 2.0]},
            )
        if self._context is None:
            cfg = default_system_config(self.asset_class)
            asset = cfg.asset
            cost = cfg.cost_model
            if self.asset_class == "cfd":
                assert isinstance(asset, CFDAssetSpec)
                # cost_v1 CFD with overnight financing disabled for INTRADAY_ONLY
                base = default_cfd_cost_model()
                cost = CostModel(
                    version="cost_v1",
                    commission_per_contract=base.commission_per_contract,
                    minimum_commission=base.minimum_commission,
                    fixed_spread_ticks=base.fixed_spread_ticks,
                    dynamic_spread_enabled=False,  # prefer observed BID/ASK on bars
                    dynamic_spread_atr_mult=base.dynamic_spread_atr_mult,
                    slippage_ticks_mean=base.slippage_ticks_mean,
                    slippage_ticks_std=base.slippage_ticks_std,
                    volatility_dependent_slippage=base.volatility_dependent_slippage,
                    volatility_slippage_mult=base.volatility_slippage_mult,
                    time_of_day_slippage=base.time_of_day_slippage,
                    open_close_slippage_mult=base.open_close_slippage_mult,
                    liquidity_dependent_slippage=base.liquidity_dependent_slippage,
                    liquidity_volume_ref=base.liquidity_volume_ref,
                    overnight_swap_enabled=not self.intraday_only,
                    futures_rollover_cost_ticks=0.0,
                    market_impact_enabled=base.market_impact_enabled,
                    market_impact_coeff=base.market_impact_coeff,
                    participation_rate_cap=base.participation_rate_cap,
                )
            else:
                assert isinstance(asset, FuturesAssetSpec)
            tf = str(self.silver_resolution.get("timeframe") or "")
            freq = "1min" if tf in {"1min", "1m"} else "5min"
            bpd = 1440 if freq == "1min" else 288
            self._context = EventWFOContext(
                asset=asset,
                cost_model=cost,
                prop_profile=cfg.prop_profile,
                starting_equity=100_000.0,
                freq=freq,
                bars_per_year=252 * bpd,
            )
        if self.available_feature_ids is None and self.bars is not None:
            try:
                from features.generator import FeatureGenerator

                fg = FeatureGenerator()
                frame = self.bars
                if "timestamp" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
                    work = frame.set_index(pd.to_datetime(frame["timestamp"], utc=True))
                else:
                    work = frame.copy()
                    if work.index.tz is None and isinstance(work.index, pd.DatetimeIndex):
                        work.index = work.index.tz_localize("UTC")
                ff = fg.generate(work)
                self.available_feature_ids = frozenset(ff.feature_ids)
                self.dataset_capabilities = tuple(
                    sorted(str(c) for c in (ff.meta or {}).get("capabilities", []))
                )
            except Exception:  # noqa: BLE001 — leave None; evaluator skips check
                self.available_feature_ids = None

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
        assert self.bars is not None and self.wfo_config is not None and self._context is not None
        if self.require_real_bars and self.bars is None:
            raise RuntimeError(f"{REAL_DATA_BACKEND_UNAVAILABLE}: empty bars")
        if not hasattr(candidate, "entry_tree") or candidate.entry_tree is None:
            raise RuntimeError(
                "DSL_SIGNAL_REQUIRED: EventDrivenDiscoveryBackend.evaluate requires "
                "candidate.entry_tree — MR param stub is not permitted"
            )

        from discovery.dsl_signal_adapter import DSLCompileStats, make_dsl_signal_fn_factory

        # Single-point grid from candidate parameters (DSL bindings), not MR stub keys.
        raw_params = dict(getattr(candidate, "parameters", {}) or {})
        param_grid = {k: [float(v)] for k, v in raw_params.items()} if raw_params else {}
        cfg = WalkForwardConfig(
            train_window_days=self.wfo_config.train_window_days,
            validation_window_days=self.wfo_config.validation_window_days,
            step_forward_days=self.wfo_config.step_forward_days,
            purge_gap_bars=self.wfo_config.purge_gap_bars,
            embargo_gap_bars=self.wfo_config.embargo_gap_bars,
            bars_per_day=self.wfo_config.bars_per_day,
            max_folds=self.wfo_config.max_folds,
            param_grid=param_grid,
            optimize_metric=self.wfo_config.optimize_metric,
        )
        dsl_stats = DSLCompileStats()
        signal_fn_factory = make_dsl_signal_fn_factory(
            candidate,
            symbol=str(self._context.symbol),
            bar_end_offset=str(self._context.freq),
            intraday_only=bool(self.intraday_only),
            stats=dsl_stats,
        )
        result = run_event_driven_wfo(
            self.bars,
            cfg,
            signal_fn_factory=signal_fn_factory,
            context=self._context,
            ranking_metric="sharpe",
        )
        folds: list[FoldOOSMetrics] = []
        order_count = 0
        fill_count = 0
        trade_count = 0
        signal_entry_count = 0
        signal_exit_count = 0
        training_ranges: list[dict[str, str]] = []
        oos_ranges: list[dict[str, str]] = []
        fold_funnels: list[dict[str, Any]] = []
        for fr in result.folds:
            val = fr.validation_run
            net = val.net_metrics if val else {}
            if val:
                order_count += len(val.orders)
                fill_count += len(val.fills)
                trade_count += len(val.trades)
                funnel = dict(val.trade_funnel) if val.trade_funnel else trade_funnel_from_window_run(val)
                signal_entry_count += int(funnel.get("entry_true_count", 0))
                signal_exit_count += int(funnel.get("exit_true_count", 0))
                fold_funnels.append({"fold_id": fr.fold_id, "phase": "validation_oos", **funnel})
            if fr.train_run:
                order_count += len(fr.train_run.orders)
                fill_count += len(fr.train_run.fills)
                trade_count += len(fr.train_run.trades)
                train_funnel = (
                    dict(fr.train_run.trade_funnel)
                    if fr.train_run.trade_funnel
                    else trade_funnel_from_window_run(fr.train_run)
                )
                fold_funnels.append({"fold_id": fr.fold_id, "phase": "train", **train_funnel})
            training_ranges.append(
                {"start": fr.train_start_ts, "end": fr.train_end_ts, "fold_id": str(fr.fold_id)}
            )
            oos_ranges.append(
                {
                    "start": fr.validation_start_ts,
                    "end": fr.validation_end_ts,
                    "fold_id": str(fr.fold_id),
                }
            )
            if val is None:
                raise RuntimeError(f"PERFORMANCE_METRIC_CONTRACT_MISMATCH: fold {fr.fold_id} has no validation run")
            folds.append(fold_oos_metrics_from_net(fold_id=fr.fold_id, net=net, fallback_metric=fr.validation_metric))

        artifact_payload = {
            "candidate_id": candidate.candidate_id,
            "backend_kind": self.backend_kind,
            "evaluation_path": "dsl_event_driven_wfo",
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "silver_resolution": dict(self.silver_resolution),
            "training_ranges": training_ranges,
            "oos_ranges": oos_ranges,
            "orders_count": order_count,
            "fills_count": fill_count,
            "trades_count": trade_count,
            "signals_entry_count": signal_entry_count,
            "signals_exit_count": signal_exit_count,
            "dsl_feature_ids": list(dsl_stats.feature_ids),
            "dsl_missing_features": list(dsl_stats.missing_features),
            "fold_trade_funnels": fold_funnels,
            "folds": [f.as_dict() for f in result.folds],
        }
        self.last_run_artifacts = artifact_payload
        if self.artifact_dir is not None:
            art = Path(self.artifact_dir) / "wfo_artifacts"
            art.mkdir(parents=True, exist_ok=True)
            out = art / f"{candidate.candidate_id}_wfo.json"
            out.write_text(json.dumps(artifact_payload, indent=2, default=str), encoding="utf-8")

        train_diag: dict[str, Any] = {
            "sharpe": float(
                np.mean([f.train_metric for f in result.folds]) if result.folds else 0.0
            ),
            "wfo_fold_count": len(result.folds),
            "wfo_completed_folds": len(result.folds),
            "wfo_terminal_status": "COMPLETED" if result.folds else "FAILED",
            "wfo_folds": [
                {
                    "fold_id": f.fold_id,
                    "train_start": f.train_start_ts,
                    "train_end": f.train_end_ts,
                    "oos_start": f.validation_start_ts,
                    "oos_end": f.validation_end_ts,
                    "train_metric": f.train_metric,
                    "oos_metric": f.validation_metric,
                    "phase_oos": "validation_oos",
                    "orders": len(f.validation_run.orders) if f.validation_run else 0,
                    "fills": len(f.validation_run.fills) if f.validation_run else 0,
                    "trades": len(f.validation_run.trades) if f.validation_run else 0,
                    "trade_funnel": (
                        dict(f.validation_run.trade_funnel) if f.validation_run else {}
                    ),
                }
                for f in result.folds
            ],
            "fold_trade_funnels": fold_funnels,
            "training_ranges": training_ranges,
            "oos_ranges": oos_ranges,
            "orders_count": order_count,
            "fills_count": fill_count,
            "trades_count": trade_count,
            "signals_entry_count": signal_entry_count,
            "signals_exit_count": signal_exit_count,
            "signal_source": "candidate_dsl_trees",
            "dsl_feature_ids": list(dsl_stats.feature_ids),
            "dsl_missing_features": list(dsl_stats.missing_features),
            "is_full_event_wfo": True,
            "backend_kind": self.backend_kind,
            "proxy_metric_used": False,
            "evaluation_path": "dsl_event_driven_wfo",
            "silver_resolution": dict(self.silver_resolution),
            "intraday_only": self.intraday_only,
            "observed_bid_ask_spread": bool(
                self.bars is not None
                and "bid" in self.bars.columns
                and "ask" in self.bars.columns
            ),
        }
        return folds, train_diag
