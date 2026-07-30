"""Validation: generated DSL candidate trades on bounded real 2024 Silver BID/ASK."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.walk_forward_config import WalkForwardConfig
from data.acquisition.multiyear_orchestrator import PROTECTED_YEARLY_2024_ID
from data.catalog.catalog import DatasetCatalog
from data.catalog.silver_bar_resolution import resolve_bid_ask_bars
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.generator import CandidateGenerator

ARCHIVE = Path("data_import/dukascopy_archive/dataset_catalog")


def main() -> None:
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    art = Path("artifacts") / "real_2024_dsl_validation" / run_id
    art.mkdir(parents=True, exist_ok=True)

    entry = DatasetCatalog(ARCHIVE).get(PROTECTED_YEARLY_2024_ID)
    assert entry is not None
    resolved = resolve_bid_ask_bars(entry, timeframe="1m", max_rows=12_000)
    backend = EventDrivenDiscoveryBackend(
        bars=resolved.frame,
        require_real_bars=True,
        asset_class="cfd",
        intraday_only=True,
        artifact_dir=art,
        silver_resolution=resolved.as_dict(),
        wfo_config=WalkForwardConfig(
            train_window_days=2,
            validation_window_days=1,
            step_forward_days=1,
            purge_gap_bars=1,
            embargo_gap_bars=1,
            bars_per_day=1440,
            max_folds=2,
            param_grid={},
        ),
    )
    gen = CandidateGenerator()
    cand = gen.generate(seed=41)
    folds, diag = backend.evaluate(cand)
    oos_funnels = [
        f for f in (diag.get("fold_trade_funnels") or []) if f.get("phase") == "validation_oos"
    ]
    oos_trades = sum(f.n_trades for f in folds)
    report = {
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_id": entry.dataset_id,
        "rows": len(resolved.frame),
        "span": [
            str(resolved.frame["timestamp"].iloc[0]),
            str(resolved.frame["timestamp"].iloc[-1]),
        ],
        "candidate_id": cand.candidate_id,
        "family": cand.strategy_family,
        "signal_source": diag.get("signal_source"),
        "evaluation_path": diag.get("evaluation_path"),
        "oos_trades": oos_trades,
        "fold_n_trades": [f.n_trades for f in folds],
        "fold_max_drawdown": [f.max_drawdown for f in folds],
        "fold_calmar": [f.calmar for f in folds],
        "oos_funnels": oos_funnels,
        "ok": oos_trades >= 1
        and diag.get("signal_source") == "candidate_dsl_trees"
        and all(int(f.get("entry_true_count", 0)) > 0 for f in oos_funnels),
    }
    (art / "validation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    if not report["ok"]:
        raise SystemExit("VALIDATION_FAILED")
    print(f"VALIDATION_OK run_id={run_id}")


if __name__ == "__main__":
    main()
