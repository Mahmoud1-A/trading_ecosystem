"""Diagnose one DSL candidate on real 2024 Silver BID/ASK (bounded slice)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config.walk_forward_config import WalkForwardConfig
from data.acquisition.multiyear_orchestrator import PROTECTED_YEARLY_2024_ID
from data.catalog.catalog import DatasetCatalog
from data.catalog.silver_bar_resolution import resolve_bid_ask_bars
from discovery.dsl_series import eval_series, unwrap_boolean_intent
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.expression_tree import ExprNode
from discovery.generator import CandidateGenerator
from features.generator import FeatureGenerator, FeatureGeneratorConfig

ARCHIVE = Path("data_import/dukascopy_archive/dataset_catalog")


def _first_zero_stage(funnel: dict) -> str | None:
    order = [
        ("entry_true_count", "entry_true_count"),
        ("orders_submitted", "orders_submitted"),
        ("fills", "fills"),
        ("positions_opened", "positions_opened"),
        ("positions_closed", "positions_closed"),
    ]
    for key, label in order:
        if int(funnel.get(key, 0) or 0) == 0:
            return label
    return None


def _inspect_dsl_features(candidate, bars: pd.DataFrame) -> dict:
    work = bars.copy()
    if "timestamp" in work.columns:
        work = work.set_index("timestamp")
    fg = FeatureGenerator(config=FeatureGeneratorConfig(bar_end_offset="1min"))
    feats = fg.generate(work).values
    bindings = dict(candidate.parameters)
    out: dict = {"feature_ids": list(candidate.feature_ids), "bindings": bindings}
    for fid in candidate.feature_ids:
        if fid not in feats.columns:
            out[fid] = {"missing": True}
            continue
        s = feats[fid].astype(float)
        out[fid] = {
            "missing": False,
            "n": int(len(s)),
            "nan": int(s.isna().sum()),
            "finite": int(np.isfinite(s.to_numpy(dtype=float)).sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "p50": float(s.median()) if s.notna().any() else None,
        }
    entry_bool = unwrap_boolean_intent(candidate.entry_tree, feats, bindings)
    exit_bool = unwrap_boolean_intent(candidate.exit_tree, feats, bindings)
    out["entry_true_bars"] = int(entry_bool.fillna(False).sum())
    out["exit_true_bars"] = int(exit_bool.fillna(False).sum())
    out["entry_nan_bars"] = int(entry_bool.isna().sum()) if hasattr(entry_bool, "isna") else 0
    # threshold peek for seed template
    if "price.rolling_z_20" in feats.columns and "z_entry" in bindings:
        z = feats["price.rolling_z_20"].astype(float)
        thr = float(bindings["z_entry"])
        out["z_vs_entry"] = {
            "z_entry": thr,
            "bars_z_lt_z_entry": int((z < thr).fillna(False).sum()),
            "bars_z_finite": int(z.notna().sum()),
        }
    return out


def main() -> None:
    cat = DatasetCatalog(ARCHIVE)
    entry = cat.get(PROTECTED_YEARLY_2024_ID)
    assert entry is not None
    # ~8 trading days of 1min — enough for 2 folds without full-year campaign
    resolved = resolve_bid_ask_bars(entry, timeframe="1m", max_rows=12_000)
    frame = resolved.frame
    print("=== DATA ===")
    print(
        json.dumps(
            {
                "dataset_id": entry.dataset_id,
                "rows": len(frame),
                "span": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
                "has_bid_ask": bool("bid" in frame.columns and "ask" in frame.columns),
                "intraday_only": bool(entry.intraday_only),
                "asset_class": entry.asset_class,
            },
            indent=2,
        )
    )

    backend = EventDrivenDiscoveryBackend(
        bars=frame,
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
        require_real_bars=True,
        asset_class="cfd",
        intraday_only=True,
        silver_resolution={"dataset_id": entry.dataset_id, "row_count": len(frame)},
    )

    # Prefer seed template (known DSL), then one generated candidate
    gen = CandidateGenerator(feature_set_version="feature_set_v1_phase6b")
    candidates = [
        ("seed_template", gen.seed_template_mean_reversion(seed=42)),
        ("generated", gen.generate(seed=42)),
    ]

    report = {"dataset_id": entry.dataset_id, "rows": len(frame), "candidates": []}

    for label, cand in candidates:
        print(f"\n=== CANDIDATE {label} id={cand.candidate_id} family={cand.strategy_family} ===")
        feat_diag = _inspect_dsl_features(cand, frame)
        print("DSL/feature inspect:", json.dumps(feat_diag, indent=2, default=str)[:2000])

        folds, diag = backend.evaluate(cand)
        art = dict(backend.last_run_artifacts)
        oos_funnels = [f for f in (diag.get("fold_trade_funnels") or []) if f.get("phase") == "validation_oos"]
        print("signal_source:", diag.get("signal_source"))
        print("evaluation_path:", diag.get("evaluation_path"))
        print("totals:", {k: diag.get(k) for k in (
            "signals_entry_count", "signals_exit_count", "orders_count", "fills_count", "trades_count"
        )})
        print("fold n_trades:", [f.n_trades for f in folds])
        print("fold max_dd:", [f.max_drawdown for f in folds])
        print("fold calmar:", [f.calmar for f in folds])

        for funnel in oos_funnels:
            # enrich reject reason counts from artifacts if present
            print(f"\n--- fold {funnel.get('fold_id')} OOS funnel ---")
            print(json.dumps(funnel, indent=2, default=str))
            zero = _first_zero_stage(funnel)
            print("FIRST_ZERO_STAGE:", zero or "none (pipeline complete)")

        # Rejected orders detail from last artifacts fold dicts
        rejected_detail = []
        for fr in art.get("folds") or []:
            if fr.get("phase") and fr.get("phase") != "validation":
                pass
            val = fr.get("validation_run") or fr  # as_dict structure
            # EventFoldResult.as_dict embeds validation differently
        for fr in (backend.last_run_artifacts.get("folds") or []):
            # folds from EventFoldResult.as_dict — look for trade_funnel / orders
            orders = fr.get("orders") or []
            if not orders and "validation_run" in fr:
                orders = (fr.get("validation_run") or {}).get("orders") or []
            for o in orders:
                if o.get("reject_reason"):
                    rejected_detail.append(
                        {"fold_id": fr.get("fold_id"), "reason": o.get("reject_reason"), "side": o.get("side")}
                    )

        # Parse fold as_dict more carefully
        from discovery.event_wfo_backend import trade_funnel_from_window_run  # noqa: F401

        # Pull rejected from wfo fold records in result via re-read of artifact folds keys
        fold_dicts = art.get("folds") or []
        reject_counter: Counter[str] = Counter()
        forced = 0
        for fd in fold_dicts:
            # EventFoldResult.as_dict has nested validation fields at top level sometimes
            for o in fd.get("orders") or []:
                if o.get("reject_reason"):
                    reject_counter[str(o["reject_reason"])] += 1
                if (o.get("meta") or {}).get("window_end_flatten") or o.get("parent_signal_id") and False:
                    pass
                reason = str((o.get("meta") or {}).get("reason", ""))
            for o in fd.get("orders") or []:
                meta = o.get("meta") or {}
                if meta.get("window_end_flatten"):
                    forced += 1
            # Also check trades/orders from nested structure
            for key in ("validation_orders",):
                pass

        cand_row = {
            "label": label,
            "candidate_id": cand.candidate_id,
            "family": cand.strategy_family,
            "feature_diag": feat_diag,
            "diag_totals": {k: diag.get(k) for k in (
                "signals_entry_count", "signals_exit_count", "orders_count", "fills_count", "trades_count",
                "signal_source", "evaluation_path",
            )},
            "oos_funnels": oos_funnels,
            "fold_n_trades": [f.n_trades for f in folds],
            "fold_max_drawdown": [f.max_drawdown for f in folds],
            "fold_calmar": [f.calmar for f in folds],
            "reject_reason_counts": dict(reject_counter),
            "first_zero_stages": [_first_zero_stage(f) for f in oos_funnels],
            "oos_trades_total": sum(f.n_trades for f in folds),
        }
        report["candidates"].append(cand_row)

        if sum(f.n_trades for f in folds) >= 1:
            print(f"\nSUCCESS: {label} has non-zero OOS trades")
            break
        else:
            print(f"\nFAIL: {label} still zero OOS trades — stopping at first zero stages above")

    out = Path("artifacts/real_2024_funnel_diag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\nWrote", out)
    ok = any(c["oos_trades_total"] >= 1 for c in report["candidates"])
    print("VALIDATION", "OK" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
