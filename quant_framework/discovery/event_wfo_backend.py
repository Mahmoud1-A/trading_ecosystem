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
    """Mark first/last bar of each UTC calendar day for INTRADAY_ONLY flatten."""
    work = frame.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True)
    day = ts.dt.floor("D")
    work["is_session_open"] = day != day.shift(1)
    work["is_session_close"] = day != day.shift(-1)
    work.loc[work.index[0], "is_session_open"] = True
    work.loc[work.index[-1], "is_session_close"] = True
    return work


def _signal_factory_from_params(params: dict[str, Any], window: pd.DataFrame):
    lookback = int(params.get("lookback", 20))
    z_entry = float(params.get("z_entry", 2.0))
    closes = window["close"].astype(float).tolist() if "close" in window.columns else []
    # Align session-close flags if present on the window index order
    session_close = (
        window["is_session_close"].astype(bool).tolist()
        if "is_session_close" in window.columns
        else [False] * len(closes)
    )

    def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
        pos = portfolio.get_position(bar.symbol)
        qty = 0.0 if pos is None or pos.is_flat else float(pos.quantity)
        timing = InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.bar_end,
            decision_timestamp=bar.bar_end,
        )
        # Prohibit overnight positions — flatten at session close
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

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
        assert self.bars is not None and self.wfo_config is not None and self._context is not None
        if self.require_real_bars and self.bars is None:
            raise RuntimeError(f"{REAL_DATA_BACKEND_UNAVAILABLE}: empty bars")
        params = {
            "lookback": int(candidate.parameters.get("lookback", 15)),
            "z_entry": float(candidate.parameters.get("z_entry", 2.0)),
        }
        cfg = WalkForwardConfig(
            train_window_days=self.wfo_config.train_window_days,
            validation_window_days=self.wfo_config.validation_window_days,
            step_forward_days=self.wfo_config.step_forward_days,
            purge_gap_bars=self.wfo_config.purge_gap_bars,
            embargo_gap_bars=self.wfo_config.embargo_gap_bars,
            bars_per_day=self.wfo_config.bars_per_day,
            max_folds=self.wfo_config.max_folds,
            param_grid={k: [v] for k, v in params.items()},
            optimize_metric=self.wfo_config.optimize_metric,
        )
        result = run_event_driven_wfo(
            self.bars,
            cfg,
            signal_fn_factory=_signal_factory_from_params,
            context=self._context,
            ranking_metric="sharpe",
        )
        folds: list[FoldOOSMetrics] = []
        order_count = 0
        fill_count = 0
        trade_count = 0
        training_ranges: list[dict[str, str]] = []
        oos_ranges: list[dict[str, str]] = []
        for fr in result.folds:
            val = fr.validation_run
            net = val.net_metrics if val else {}
            if val:
                order_count += len(val.orders)
                fill_count += len(val.fills)
                trade_count += len(val.trades)
            if fr.train_run:
                order_count += len(fr.train_run.orders)
                fill_count += len(fr.train_run.fills)
                trade_count += len(fr.train_run.trades)
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
            folds.append(
                FoldOOSMetrics(
                    fold_id=fr.fold_id,
                    expectancy=float(net.get("expectancy", fr.validation_metric)),
                    sharpe=float(net.get("sharpe", fr.validation_metric)),
                    profit_factor=float(net.get("profit_factor", 1.0)),
                    calmar=float(net.get("calmar", 0.0)),
                    max_drawdown=float(net.get("max_drawdown", 0.0)),
                    drawdown_duration=float(net.get("drawdown_duration", 0.0) or 0.0),
                    worst_day=float(net.get("worst_day", 0.0) or 0.0),
                    turnover=float(net.get("turnover", 0.0) or 0.0),
                    prop_breach_prob=float(net.get("prop_breach_prob", 0.0) or 0.0),
                    regime_entropy=float(net.get("regime_entropy", 1.0) or 1.0),
                    n_trades=int(net.get("n_trades", 0) or 0),
                )
            )

        artifact_payload = {
            "candidate_id": candidate.candidate_id,
            "backend_kind": self.backend_kind,
            "is_full_event_wfo": True,
            "silver_resolution": dict(self.silver_resolution),
            "training_ranges": training_ranges,
            "oos_ranges": oos_ranges,
            "orders_count": order_count,
            "fills_count": fill_count,
            "trades_count": trade_count,
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
                }
                for f in result.folds
            ],
            "training_ranges": training_ranges,
            "oos_ranges": oos_ranges,
            "orders_count": order_count,
            "fills_count": fill_count,
            "trades_count": trade_count,
            "is_full_event_wfo": True,
            "backend_kind": self.backend_kind,
            "proxy_metric_used": False,
            "evaluation_path": self.backend_kind,
            "silver_resolution": dict(self.silver_resolution),
            "intraday_only": self.intraday_only,
            "observed_bid_ask_spread": bool(
                self.bars is not None
                and "bid" in self.bars.columns
                and "ask" in self.bars.columns
            ),
        }
        return folds, train_diag
