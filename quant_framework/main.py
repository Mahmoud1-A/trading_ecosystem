"""Quant Framework entrypoint — research/backtest demo (Phase 5).

Not live-trading ready.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha import MeanReversionStrategy, run_strategy_backtest
from config import default_system_config
from config.strategy_config import StrategyParams
from config.walk_forward_config import WalkForwardConfig
from data.calendars import CME_EQUITY_RTH
from data.validator import compute_data_hash, validate_market_data
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from metrics import compute_metrics
from registry import ExperimentRegistry, hash_directory_py, persist_run_artifacts
from reports import print_performance_report, save_equity_curve_chart
from risk import PropRiskFSM
from validation import CandidateLineage, ValidationVault, run_walk_forward


def _synthetic_intraday(n_days: int = 25, *, seed: int = 42) -> pd.DataFrame:
    """Generate RTH-like 5m bars with mild mean reversion for demos."""
    rng = np.random.default_rng(seed)
    tz = CME_EQUITY_RTH.zoneinfo()
    bars_per_day = 78
    idx: list[pd.Timestamp] = []
    day0 = pd.Timestamp("2024-01-02 08:30", tz=tz)
    d = 0
    while len(idx) // bars_per_day < n_days:
        day = day0 + pd.Timedelta(days=d)
        d += 1
        if day.weekday() >= 5:
            continue
        for b in range(bars_per_day):
            idx.append(day + pd.Timedelta(minutes=5 * b))

    n = len(idx)
    px = np.empty(n)
    px[0] = 4800.0
    for i in range(1, n):
        px[i] = px[i - 1] + 0.08 * (4800 - px[i - 1]) + rng.normal(0, 0.6)
    noise = np.abs(rng.normal(0, 0.4, n))
    close = px
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.integers(800, 5000, n).astype(float)
    return pd.DataFrame(
        {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def main() -> int:
    cfg = default_system_config("futures")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"quant_framework {cfg.system_version}")
    print("mode=research/backtest only (NOT live-trading ready)")
    print(f"strategy={cfg.strategy.family}")
    print(f"asset={cfg.asset.symbol}")

    raw = _synthetic_intraday(25, seed=cfg.random_seed)
    validated = validate_market_data(
        raw,
        require_timezone=True,
        detect_missing=False,
        calendar="CME",
        freq="5min",
    )
    bars = validated.frame
    data_hash = validated.data_hash
    print(f"bars={len(bars)} data_hash={data_hash[:16]}…")

    # Split: research body vs final vault (vault never used in optimization)
    vault_n = max(200, len(bars) // 10)
    research = bars.iloc[:-vault_n].reset_index(drop=True)
    vault_df = bars.iloc[-vault_n:].reset_index(drop=True)
    vault = ValidationVault(
        vault_version="vault_v1",
        data=vault_df,
        data_hash=compute_data_hash(vault_df),
        store_path=out / "vault",
    )

    # Lightweight WFO evaluate_fn (feature z-score proxy — fast for demo)
    def evaluate_fn(window: pd.DataFrame, params: dict) -> float:
        lookback = int(params.get("lookback", 20))
        if len(window) < lookback + 5:
            return -999.0
        close = window["close"].astype(float)
        z = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std(ddof=0)
        # Pseudo-metric: mean reversion opportunity magnitude
        return float((-z.dropna().abs()).mean() * -1)  # higher |z| mean => more opp; invert for max

    wfo_cfg = WalkForwardConfig(
        train_window_days=5,
        validation_window_days=2,
        step_forward_days=2,
        purge_gap_bars=2,
        embargo_gap_bars=2,
        bars_per_day=78,
        param_grid={"lookback": [15, 20], "z_entry": [1.5, 2.0], "z_exit": [0.15, 0.25]},
    )
    registry = ExperimentRegistry(out / "registry")
    code_hash = hash_directory_py(ROOT)

    def persist_trial(trial: dict) -> None:
        lineage = CandidateLineage.create(
            strategy_family=cfg.strategy.family,
            parameters=trial["params"],
            config_snapshot=cfg.to_snapshot(),
            code_hash=code_hash,
            data_hash=data_hash,
            cost_model_version=cfg.cost_model.version,
        )
        rejected = None if trial.get("phase") == "validation" else None
        registry.create_trial(
            candidate_id=lineage.candidate_id,
            lineage_id=lineage.lineage_id,
            strategy_family=cfg.strategy.family,
            parameters=trial["params"],
            config_snapshot=cfg.to_snapshot(),
            system_version=cfg.system_version,
            data_hash=data_hash,
            random_seed=cfg.random_seed,
            cost_model_version=cfg.cost_model.version,
            code_hash=code_hash,
            net_metrics={"metric": trial["metric"], "phase": trial["phase"]},
            ranking_score=float(trial["metric"]),
            rejection_reason=rejected,
            train_window={"start": trial.get("train_start", ""), "end": trial.get("train_end", "")}
            if trial.get("phase") == "train"
            else None,
            validation_window={
                "start": trial.get("validation_start", ""),
                "end": trial.get("validation_end", ""),
            }
            if trial.get("phase") == "validation"
            else None,
            cwd=ROOT.parent,
        )

    wfo = run_walk_forward(research, wfo_cfg, evaluate_fn=evaluate_fn, persist_trial_fn=persist_trial)
    print(f"wfo_folds={len(wfo.folds)} aggregated_val_metric={wfo.aggregated_validation_metric:.6f}")
    best = wfo.folds[-1].best_params if wfo.folds else {}
    print(f"last_fold_best_params={best}")

    # Full event-driven backtest on research data with frozen params
    strat_params = StrategyParams(
        lookback=int(best.get("lookback", cfg.strategy.lookback)),
        z_entry=float(best.get("z_entry", cfg.strategy.z_entry)),
        z_exit=float(best.get("z_exit", cfg.strategy.z_exit)),
        use_regime_filter=False,
    )
    strategy = MeanReversionStrategy(strat_params, regime_params=cfg.regime, symbol="ES")
    from config.strategy_config import SizingParams
    from config.prop_profile import PropProfile
    from config.cost_model import CostModel

    demo_sizing = SizingParams(
        risk_per_trade_pct=0.01,
        max_lots=1.0,
        min_lot=1.0,
        lot_step=1.0,
        atr_stop_mult=1.5,
    )
    demo_prop = PropProfile(
        name="demo",
        prop_hard_daily_loss_limit=-0.05,
        prop_total_drawdown_limit=-0.20,
        internal_soft_daily_limit=-0.03,
        max_symbol_exposure=5.0,
        max_portfolio_exposure=10.0,
        max_positions=5,
    )
    demo_cost = CostModel(
        version="demo_cost_v1",
        commission_per_contract=2.5,
        minimum_commission=2.5,
        fixed_spread_ticks=1.0,
        dynamic_spread_enabled=False,
        slippage_ticks_mean=0.25,
        slippage_ticks_std=0.0,
        volatility_dependent_slippage=False,
        time_of_day_slippage=False,
        liquidity_dependent_slippage=False,
        overnight_swap_enabled=False,
        participation_rate_cap=1.0,
    )
    risk = PropRiskFSM(demo_prop, starting_equity=cfg.starting_equity)
    engine = EventExecutionEngine(
        cfg.asset,
        demo_cost,
        starting_equity=cfg.starting_equity,
        exec_config=ExecutionConfig(random_seed=cfg.random_seed, intrabar_policy=cfg.intrabar_policy),
        risk_manager=risk,
    )
    run = run_strategy_backtest(engine, strategy, research, sizing_params=demo_sizing)

    eq_pairs = run.execution.portfolio.equity_curve
    if eq_pairs:
        equity = pd.Series(
            [e for _, e in eq_pairs],
            index=pd.DatetimeIndex([t for t, _ in eq_pairs]),
            name="equity",
        )
    else:
        equity = pd.Series([cfg.starting_equity], name="equity")

    trades = pd.DataFrame([t.as_dict() for t in run.execution.portfolio.trades])
    metrics = compute_metrics(
        equity,
        trades=trades if not trades.empty else None,
        bars_per_year=252 * 78,
        starting_equity=cfg.starting_equity,
    )
    print_performance_report(
        metrics,
        title="Research Backtest (pre-vault)",
        extra={
            "data_hash": data_hash[:16] + "…",
            "fills": len(run.execution.fills),
            "orders": len(run.execution.orders),
            "wfo_folds": len(wfo.folds),
            "registry_trials": len(registry.all_trials()),
        },
    )

    chart_path = out / "equity_curve.png"
    save_equity_curve_chart(equity, chart_path, title="Quant Framework Equity Curve")
    print(f"equity_chart={chart_path}")

    lineage = CandidateLineage.create(
        strategy_family=strat_params.family,
        parameters=strat_params.model_dump(),
        config_snapshot=cfg.to_snapshot(),
        code_hash=code_hash,
        data_hash=data_hash,
        cost_model_version=cfg.cost_model.version,
    )

    def vault_eval(df: pd.DataFrame) -> dict:
        # Freeze candidate: evaluate once on vault only
        return {"n_bars": len(df), "vault_hash": compute_data_hash(df), "status": "evaluated"}

    vault_result = vault.evaluate_frozen_candidate(lineage, evaluate_fn=vault_eval)
    print(f"vault_access={vault_result['vault_version']} access_count={vault_result['access_count']}")

    persist_run_artifacts(
        out / "runs" / lineage.candidate_id[:8],
        equity=equity,
        trades=trades if not trades.empty else None,
        fills=[f.as_dict() for f in run.execution.fills],
        orders=[o.snapshot() for o in run.execution.orders],
        risk_events=run.execution.risk_events,
        metrics=metrics.to_dict(),
        config_snapshot=cfg.to_snapshot(),
    )
    print(f"artifacts={out}")
    print("Phase 5 demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
