"""Validation: Alpha Miner DSL wiring produces divergent OOS trade counts.

Creates a deterministic validation run ID and evaluates multiple opposite DSL
candidates through EventDrivenDiscoveryBackend (no MR stub, no Dukascopy writes).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.walk_forward_config import WalkForwardConfig
from discovery.candidate import build_candidate
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.expression_tree import constant_node, feature_node, op_node
from discovery.grammar import GRAMMAR_VERSION
from discovery.operators import OperatorId
from discovery.types import CreationMethod, ValueType

TZ = ZoneInfo("America/Chicago")


def _bars(n: int = 780) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="5min", tz=TZ)
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


def _cand(long_when_positive: bool, seed: int):
    ret = feature_node("price.simple_return_1", ValueType.RETURN)
    if long_when_positive:
        cond = op_node(OperatorId.GREATER_THAN, ret, constant_node(0.0))
        family = "dsl_momentum_up"
    else:
        cond = op_node(OperatorId.LESS_THAN, ret, constant_node(0.0))
        family = "dsl_momentum_down"
    entry = op_node(OperatorId.ENTRY_LONG, cond)
    exit_cond = (
        op_node(OperatorId.LESS_THAN, ret, constant_node(0.0))
        if long_when_positive
        else op_node(OperatorId.GREATER_THAN, ret, constant_node(0.0))
    )
    exit_tree = op_node(OperatorId.EXIT_SIGNAL, exit_cond)
    atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
    stop = op_node(OperatorId.ATR_STOP, atr, constant_node(2.0))
    return build_candidate(
        entry_tree=entry,
        exit_tree=exit_tree,
        stop=stop,
        target=op_node(OperatorId.ATR_TARGET, atr, constant_node(3.0)),
        strategy_family=family,
        creation_method=CreationMethod.RANDOM,
        grammar_version=GRAMMAR_VERSION,
        feature_set_version="feature_set_v1_phase6b",
        random_seed=seed,
    )


def main() -> None:
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    art = Path("artifacts") / "dsl_wiring_validation" / run_id
    art.mkdir(parents=True, exist_ok=True)

    backend = EventDrivenDiscoveryBackend(
        bars=_bars(780),
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
        artifact_dir=art,
    )

    # Seed template + two opposite DSL candidates
    from discovery.generator import CandidateGenerator

    seed_cand = CandidateGenerator(
        feature_set_version="feature_set_v1_phase6b"
    ).seed_template_mean_reversion(seed=7)

    candidates = [
        ("momentum_up", _cand(True, 11)),
        ("momentum_down", _cand(False, 12)),
        ("seed_mr_template", seed_cand),
    ]
    rows = []
    for label, cand in candidates:
        folds, diag = backend.evaluate(cand)
        row = {
            "label": label,
            "candidate_id": cand.candidate_id,
            "family": cand.strategy_family,
            "signal_source": diag.get("signal_source"),
            "evaluation_path": diag.get("evaluation_path"),
            "oos_trades": sum(f.n_trades for f in folds),
            "oos_expectancy_mean": float(np.mean([f.expectancy for f in folds])),
            "oos_sharpe_mean": float(np.mean([f.sharpe for f in folds])),
            "signals_entry_count": diag.get("signals_entry_count"),
            "signals_exit_count": diag.get("signals_exit_count"),
            "orders_count": diag.get("orders_count"),
            "fills_count": diag.get("fills_count"),
            "trades_count": diag.get("trades_count"),
            "fold_trade_funnels": diag.get("fold_trade_funnels"),
        }
        rows.append(row)

    trade_counts = [r["oos_trades"] for r in rows]
    paths = {r["evaluation_path"] for r in rows}
    sources = {r["signal_source"] for r in rows}
    ok = (
        all(t >= 1 for t in trade_counts)
        and len(set(trade_counts)) >= 2
        and paths == {"dsl_event_driven_wfo"}
        and sources == {"candidate_dsl_trees"}
    )
    report = {
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "ok": ok,
        "candidates": rows,
        "distinct_oos_trade_counts": sorted(set(trade_counts)),
        "notes": "MR param stub not used; signal_source=candidate_dsl_trees",
    }
    (art / "validation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    if not ok:
        raise SystemExit("VALIDATION_FAILED")
    print(f"VALIDATION_OK run_id={run_id}")


if __name__ == "__main__":
    main()
