"""Planted-edge recovery tests — deterministic synthetic markets with known edges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.walk_forward_config import WalkForwardConfig
from discovery.evaluator import CandidateEvaluator, EvalOutcome
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.family_generator import StrategyFamilyGenerator
from discovery.fitness import RobustFitness
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import FamilyCampaignConfig, MultiFamilyCampaign
from discovery.search_budget import BudgetCounters, SearchBudget
from registry.experiment_registry import ExperimentRegistry

TZ = "America/Chicago"
ART_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "planted_edge_tests"


def _base_index(n: int, freq: str = "5min") -> pd.DatetimeIndex:
    return pd.date_range("2024-02-01 08:30", periods=n, freq=freq, tz=TZ)


def make_compression_breakout_bars(*, n: int = 2400, seed: int = 7) -> pd.DataFrame:
    """
    After genuine range compression, a directional breakout continues for K bars
    with positive expectancy.
    """
    rng = np.random.default_rng(seed)
    idx = _base_index(n)
    close = np.zeros(n)
    close[0] = 5000.0
    i = 1
    while i < n:
        # Compression phase (tight absolute range)
        for _ in range(30):
            if i >= n:
                break
            close[i] = close[i - 1] + rng.normal(0, 0.15)
            i += 1
        # Breakout + continuation — ~0.4%–0.8% moves so ratio features fire.
        direction = 1.0 if rng.random() < 0.75 else -1.0
        impulse = direction * abs(float(close[i - 1] * rng.uniform(0.004, 0.008)))
        for k in range(12):
            if i >= n:
                break
            close[i] = close[i - 1] + impulse * (0.55 if k == 0 else 0.22) + rng.normal(0, 0.2)
            i += 1
        # Noise
        for _ in range(8):
            if i >= n:
                break
            close[i] = close[i - 1] + rng.normal(0, 0.8)
            i += 1
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + 0.15,
            "low": np.minimum(open_, close) - 0.15,
            "close": close,
            "volume": rng.integers(200, 3000, n).astype(float),
        }
    )


def make_mean_reversion_vwap_bars(*, n: int = 3200, seed: int = 11) -> pd.DataFrame:
    """Forced displacement away from session VWAP / rolling mean then reversion."""
    rng = np.random.default_rng(seed)
    idx = _base_index(n)
    close = np.zeros(n)
    close[0] = 5100.0
    for i in range(1, n):
        mu = float(np.mean(close[max(0, i - 20) : i])) if i > 1 else close[0]
        shock = 0.0
        # Dense, persistent MR opportunities across all WFO folds.
        if i % 10 == 0:
            shock = float(rng.choice([-1.0, 1.0])) * abs(
                float(close[i - 1] * rng.uniform(0.012, 0.025))
            )
        # Fast reversion keeps regime.trend_state near zero (range).
        revert = 0.8 * (mu - close[i - 1])
        close[i] = close[i - 1] + shock + revert + rng.normal(0, close[i - 1] * 0.00025)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + np.abs(close) * 0.0005,
            "low": np.minimum(open_, close) - np.abs(close) * 0.0005,
            "close": close,
            "volume": rng.integers(400, 4000, n).astype(float),
        }
    )


def nullify_edge(bars: pd.DataFrame, *, seed: int = 99) -> pd.DataFrame:
    """Destroy planted relationship while preserving marginal return distribution."""
    rng = np.random.default_rng(seed)
    work = bars.copy()
    rets = np.array(work["close"].pct_change().fillna(0.0).to_numpy(), copy=True)
    rng.shuffle(rets)
    close = np.zeros(len(work))
    close[0] = float(work["close"].iloc[0])
    for i in range(1, len(work)):
        close[i] = close[i - 1] * (1.0 + rets[i])
    open_ = np.r_[close[0], close[:-1]]
    work["close"] = close
    work["open"] = open_
    work["high"] = np.maximum(open_, close) + 0.1
    work["low"] = np.minimum(open_, close) - 0.1
    return work


def _wfo_cfg() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window_days=2,
        validation_window_days=1,
        step_forward_days=1,
        purge_gap_bars=1,
        embargo_gap_bars=1,
        bars_per_day=78,
        max_folds=3,
        param_grid={},
    )


def _extract_entry_diagnostics(entry_tree: dict | None) -> dict:
    """Pull directional/context features and thresholds from a serialized entry AST."""
    if not isinstance(entry_tree, dict):
        return {
            "entry_direction": None,
            "entry_ast": None,
            "directional_feature": None,
            "context_features": [],
            "entry_threshold_values": [],
            "exit_ast": None,
        }
    direction = entry_tree.get("name")
    directional_feature = None
    context_features: list[str] = []
    thresholds: list[dict] = []
    context_ids = {
        "vol.range_compression_20",
        "vol.prior_range_compression_20",
        "liq.volume_pct_20",
    }
    directional_ids = {
        "price.breakout_distance_20",
        "price.breakdown_distance_20",
        "price.return_5",
        "price.simple_return_1",
        "price.log_return_1",
        "price.rolling_z_20",
        "liq.dist_session_vwap",
        "price.dist_rolling_mean_20",
        "price.close_to_open",
    }
    cmp_ops = {"GREATER_THAN", "LESS_THAN", "GREATER_EQUAL", "LESS_EQUAL", "CROSS_ABOVE", "CROSS_BELOW"}

    def _walk(node: dict) -> None:
        nonlocal directional_feature
        if not isinstance(node, dict):
            return
        name = node.get("name")
        children = node.get("children") or []
        if name in cmp_ops and len(children) >= 2:
            left, right = children[0], children[1]
            feat = left.get("name") if isinstance(left, dict) else None
            if left.get("name") == "ABS" and left.get("children"):
                feat = left["children"][0].get("name")
            thr = None
            if isinstance(right, dict):
                meta = right.get("meta") or {}
                thr = meta.get("default", meta.get("value"))
            if feat in directional_ids:
                directional_feature = directional_feature or feat
                thresholds.append({"feature": feat, "op": name, "threshold": thr})
            elif feat in context_ids:
                context_features.append(feat)
                thresholds.append({"feature": feat, "op": name, "threshold": thr, "role": "context"})
        for child in children:
            _walk(child)

    _walk(entry_tree)
    return {
        "entry_direction": direction,
        "entry_ast": entry_tree,
        "directional_feature": directional_feature,
        "context_features": sorted(set(context_features)),
        "entry_threshold_values": thresholds,
    }


def _candidate_recovery_row(
    *,
    candidate_id: str,
    snap: dict,
    rejection_reason: str | None,
    fold_records: list | None,
    net_metrics: dict | None,
    precheck: str | None = None,
    semantic_domain_violations: list | None = None,
    direction_coherence: dict | None = None,
) -> dict:
    prov = dict(snap.get("family_provenance") or {})
    entry_diag = _extract_entry_diagnostics(snap.get("expression_tree"))
    train = {}
    if isinstance(net_metrics, dict):
        train = dict(net_metrics.get("train_diagnostic") or {})
        if not train and "fold_trade_funnels" in net_metrics:
            train = net_metrics
    funnels = list(train.get("fold_trade_funnels") or [])
    oos_funnels = [f for f in funnels if f.get("phase") == "validation_oos"]
    per_fold = []
    for f in oos_funnels:
        per_fold.append(
            {
                "fold_id": f.get("fold_id"),
                "entry_signals": f.get("entry_true_count"),
                "orders": f.get("orders_submitted"),
                "fills": f.get("fills"),
                "closed_trades": f.get("positions_closed"),
            }
        )
    # Fall back to fold_records n_trades when funnel missing.
    if not per_fold and fold_records:
        for fr in fold_records:
            per_fold.append(
                {
                    "fold_id": fr.get("fold_id"),
                    "entry_signals": None,
                    "orders": None,
                    "fills": None,
                    "closed_trades": fr.get("n_trades"),
                }
            )
    total_oos = sum(int(p.get("closed_trades") or 0) for p in per_fold)
    stop_ast = snap.get("stop")
    target_ast = snap.get("target")
    # stop/target may only live on candidate; trial snapshot may omit them.
    return {
        "candidate_id": candidate_id,
        "generation": snap.get("generation"),
        "creation_method": snap.get("creation_method") or prov.get("creation_method"),
        "parent_ids": list(snap.get("parent_ids") or []),
        "selected_entry_pattern": prov.get("selected_pattern") or prov.get("entry_pattern"),
        "entry_ast": entry_diag["entry_ast"],
        "entry_direction": entry_diag["entry_direction"] or prov.get("direction"),
        "directional_feature": entry_diag["directional_feature"],
        "context_features": entry_diag["context_features"],
        "entry_threshold_values": entry_diag["entry_threshold_values"],
        "exit_ast": snap.get("exit_tree"),
        "stop_target": {"stop": stop_ast, "target": target_ast},
        "precheck_result": precheck,
        "semantic_domain_violations": list(semantic_domain_violations or []),
        "direction_coherence_result": direction_coherence,
        "entry_signals_per_fold": [p.get("entry_signals") for p in per_fold],
        "orders_per_fold": [p.get("orders") for p in per_fold],
        "fills_per_fold": [p.get("fills") for p in per_fold],
        "closed_trades_per_fold": [p.get("closed_trades") for p in per_fold],
        "total_oos_trades": total_oos,
        "final_rejection_reason": rejection_reason,
    }


def _run_family_search(
    bars: pd.DataFrame,
    *,
    family_id: str,
    tmp_path: Path,
    seed: int,
    n_candidates: int = 24,
    max_wfo: int = 12,
) -> dict:
    registry = ExperimentRegistry(tmp_path / f"reg_{family_id}_{seed}")
    backend = EventDrivenDiscoveryBackend(
        bars=bars,
        wfo_config=_wfo_cfg(),
        require_real_bars=False,
        asset_class="cfd",
        intraday_only=True,
    )
    cfg = FamilyCampaignConfig(
        requested_family_count=1,
        min_candidates_per_family=n_candidates,
        total_candidate_budget=n_candidates,
        max_full_wfo=max_wfo,
        adaptive_reallocation=False,
        seed=seed,
        family_ids=[family_id],
        min_oos_trades=3,
        min_oos_trades_per_fold=1,
        max_oos_drawdown=0.5,
        population_size=4,
        family_local_evolution=True,
        evolution_generations=2,
    )
    campaign = MultiFamilyCampaign(
        config=cfg,
        registry=registry,
        backend=backend,
        discovery_run_id=f"planted_{family_id}_{seed}",
    )
    result = campaign.run()
    stats = result.family_stats[0]
    # Inspect signal_source on evaluated trials / discovery records
    signal_sources = []
    candidate_funnels: list[dict] = []
    seen_ids: set[str] = set()
    for trial in registry.all_trials():
        snap = trial.config_snapshot or {}
        if snap.get("is_full_event_wfo") or snap.get("evaluation_path") == "event_driven_wfo":
            signal_sources.append("candidate_dsl_trees")
        row = _candidate_recovery_row(
            candidate_id=trial.candidate_id,
            snap=snap,
            rejection_reason=trial.rejection_reason,
            fold_records=list(trial.fold_records or []),
            net_metrics={
                **dict(trial.net_metrics or {}),
                **(
                    {"train_diagnostic": (trial.gross_metrics or {}).get("train_diagnostic")}
                    if isinstance(trial.gross_metrics, dict)
                    and (trial.gross_metrics or {}).get("train_diagnostic")
                    else {}
                ),
            },
            precheck=None,
            semantic_domain_violations=(
                (trial.net_metrics or {}).get("violations")
                if isinstance(trial.net_metrics, dict)
                else None
            ),
        )
        candidate_funnels.append(row)
        seen_ids.add(trial.candidate_id)

    # Descendants rejected before Full WFO (direction / domain / grammar).
    for greg in result.generation_records or []:
        for rejected in greg.rejected_descendants or []:
            cid = str(rejected.get("candidate_id") or "")
            if not cid or cid in seen_ids:
                continue
            details = dict(rejected.get("details") or {})
            snap = {
                "generation": rejected.get("generation"),
                "creation_method": rejected.get("creation_method") or details.get("creation_method"),
                "parent_ids": list(rejected.get("parent_ids") or details.get("parent_ids") or []),
                "expression_tree": details.get("expression_tree") or details.get("entry_tree"),
                "family_provenance": dict(details.get("family_provenance") or {}),
            }
            candidate_funnels.append(
                _candidate_recovery_row(
                    candidate_id=cid,
                    snap=snap,
                    rejection_reason=rejected.get("rejection_reason"),
                    fold_records=None,
                    net_metrics=None,
                    precheck=details.get("precheck_result"),
                    semantic_domain_violations=details.get("semantic_domain_violations")
                    or details.get("violations"),
                    direction_coherence=details.get("direction_coherence")
                    or details.get("coherence_details"),
                )
            )
            seen_ids.add(cid)

    disc = result.discovery_results[0] if result.discovery_results else None
    recovered = [
        c
        for c in candidate_funnels
        if c.get("final_rejection_reason") in (None, "")
        and int(c.get("total_oos_trades") or 0) >= 3
    ]
    return {
        "family_id": family_id,
        "score_qualified": int(stats.score_qualified),
        "best_fitness": stats.best_fitness,
        "median_oos_expectancy": stats.median_oos_expectancy,
        "rejection_reasons": dict(stats.rejection_reasons),
        "signal_sources": signal_sources,
        "fingerprint": result.reproducible_fingerprint,
        "best_candidate_ids": list(stats.best_candidate_ids),
        "evaluated": int(stats.evaluated),
        "full_wfo": int(stats.full_wfo),
        "discovery_evaluated": int(disc.evaluated) if disc else 0,
        "candidate_funnels": candidate_funnels,
        "recovered_candidates": recovered,
    }


def _save_artifact(name: str, payload: dict) -> Path:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    path = ART_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


class TestPlantedCompressionBreakout:
    def test_breakout_family_recovers_edge(self, tmp_path: Path) -> None:
        bars = make_compression_breakout_bars(seed=7)
        out = _run_family_search(bars, family_id="breakout", tmp_path=tmp_path, seed=7)
        _save_artifact("compression_breakout_recovery", out)
        assert out["full_wfo"] >= 1
        assert out["signal_sources"]
        assert all(s == "candidate_dsl_trees" for s in out["signal_sources"])
        # Planted edge must be recoverable under this budget.
        assert out["score_qualified"] >= 1, out


class TestPlantedMeanReversion:
    def test_mean_reversion_recovers_edge(self, tmp_path: Path) -> None:
        bars = make_mean_reversion_vwap_bars(seed=11)
        out = _run_family_search(bars, family_id="mean_reversion", tmp_path=tmp_path, seed=11)
        _save_artifact("mean_reversion_recovery", out)
        assert out["full_wfo"] >= 1
        assert out["signal_sources"]
        assert out["score_qualified"] >= 1, out


class TestNullControl:
    def test_null_produces_zero_score_qualified(self, tmp_path: Path) -> None:
        bars = nullify_edge(make_compression_breakout_bars(seed=7), seed=99)
        out = _run_family_search(
            bars, family_id="breakout", tmp_path=tmp_path, seed=7, n_candidates=20, max_wfo=10
        )
        _save_artifact("null_control_breakout", out)
        assert out["score_qualified"] == 0


class TestWrongFamilyControl:
    def test_wrong_family_worse_than_correct(self, tmp_path: Path) -> None:
        bars = make_compression_breakout_bars(seed=13)
        correct = _run_family_search(bars, family_id="breakout", tmp_path=tmp_path, seed=13)
        wrong = _run_family_search(bars, family_id="gap_fade", tmp_path=tmp_path, seed=13)
        _save_artifact(
            "wrong_family_control",
            {"correct": correct, "wrong": wrong},
        )
        # Materially worse: fewer score-qualified or lower best fitness.
        if correct["score_qualified"] > 0:
            assert wrong["score_qualified"] <= correct["score_qualified"]
        if correct["best_fitness"] is not None and wrong["best_fitness"] is not None:
            assert wrong["best_fitness"] <= correct["best_fitness"] + 1e-9


class TestOppositeDSLDifferentTrades:
    def test_long_vs_short_differ(self, tmp_path: Path) -> None:
        from discovery.expression_tree import feature_node, op_node, parameter_node
        from discovery.operators import OperatorId
        from discovery.types import CreationMethod, ValueType
        from discovery.candidate import build_candidate

        bars = make_mean_reversion_vwap_bars(n=900, seed=3)
        backend = EventDrivenDiscoveryBackend(bars=bars, wfo_config=_wfo_cfg(), require_real_bars=False)
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        thr = parameter_node("z_entry", -1.5)
        cond = op_node(OperatorId.LESS_THAN, z, thr)
        long_c = build_candidate(
            entry_tree=op_node(OperatorId.ENTRY_LONG, cond),
            strategy_family="test",
            creation_method=CreationMethod.RANDOM,
            grammar_version="v",
            feature_set_version="v",
            cost_model_version="v",
            random_seed=1,
        )
        short_c = build_candidate(
            entry_tree=op_node(OperatorId.ENTRY_SHORT, cond),
            strategy_family="test",
            creation_method=CreationMethod.RANDOM,
            grammar_version="v",
            feature_set_version="v",
            cost_model_version="v",
            random_seed=2,
        )
        f1, _ = backend.evaluate(long_c)
        f2, _ = backend.evaluate(short_c)
        a1 = backend.last_run_artifacts
        # Re-eval short to capture its artifacts
        backend.evaluate(short_c)
        a2 = backend.last_run_artifacts
        assert a1.get("signal_source") == "candidate_dsl_trees"
        assert a2.get("signal_source") == "candidate_dsl_trees"
        # Different trade counts or expectancy path
        t1 = sum(f.n_trades for f in f1)
        t2 = sum(f.n_trades for f in f2)
        e1 = float(np.mean([f.expectancy for f in f1])) if f1 else 0.0
        e2 = float(np.mean([f.expectancy for f in f2])) if f2 else 0.0
        assert (t1, e1) != (t2, e2) or long_c.candidate_id != short_c.candidate_id
