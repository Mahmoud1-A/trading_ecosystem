"""Validation run: prove event_driven_wfo produces non-zero OOS closed trades.

Does not touch Dukascopy acquisition or immutable datasets.
"""

from __future__ import annotations

import json
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

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
        version="validate_oos_zero",
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


def _bars(n: int = 780) -> pd.DataFrame:
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


def main() -> None:
    bars = _bars()
    # 1) Known-trading probe
    cfg = WalkForwardConfig(
        train_window_days=2,
        validation_window_days=1,
        step_forward_days=1,
        bars_per_day=10,
        purge_gap_bars=0,
        embargo_gap_bars=0,
        max_folds=2,
        param_grid={"entry_bar": [2], "hold_bars": [3]},
    )
    ctx = EventWFOContext(
        asset=default_es_futures(),
        cost_model=_zero_cost(),
        prop_profile=default_prop_profile(),
        starting_equity=100_000.0,
        exec_config=ExecutionConfig(latency=timedelta(0), random_seed=7),
        freq="5min",
        run_id="validate_oos_trades",
    )
    known = run_event_driven_wfo(
        bars.iloc[:80],
        cfg,
        signal_fn_factory=known_trading_signal_factory,
        context=ctx,
    )
    known_oos = sum(len(f.validation_run.trades) for f in known.folds if f.validation_run)
    known_funnels = [
        {"fold_id": f.fold_id, **(f.validation_run.trade_funnel if f.validation_run else {})}
        for f in known.folds
    ]

    # 2) Discovery MR stub + intraday session flatten (Alpha Miner path)
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
            param_grid={"lookback": [5], "z_entry": [0.01]},
        ),
        require_real_bars=False,
        intraday_only=True,
    )
    cand = type(
        "Cand",
        (),
        {"candidate_id": "cand_validate_mr", "parameters": {"lookback": 5, "z_entry": -2.0}},
    )()
    folds, diag = backend.evaluate(cand)  # type: ignore[arg-type]
    mr_oos = sum(f.n_trades for f in folds)

    report = {
        "broken_stages_fixed": [
            "PropRiskFSM.evaluate_order: reduce-only exits blocked by exposure_cap",
            "event_execution.run: last-bar FLAT stranded (no_next_bar_for_execution)",
            "session flags on WFO OOS slices not recomputed for window-end flatten",
        ],
        "known_trading": {
            "oos_closed_trades": known_oos,
            "funnels": known_funnels,
            "ok": known_oos >= 1,
        },
        "discovery_mr_intraday": {
            "oos_closed_trades": mr_oos,
            "fold_n_trades": [f.n_trades for f in folds],
            "funnels": diag.get("fold_trade_funnels"),
            "ok": mr_oos >= 1,
        },
        "zero_trade_metrics_hygiene": "n_trades==0 forces MaxDD/Calmar/annualized to 0",
    }
    print(json.dumps(report, indent=2, default=str))
    if known_oos < 1 or mr_oos < 1:
        raise SystemExit("VALIDATION_FAILED: expected non-zero OOS trades")
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
