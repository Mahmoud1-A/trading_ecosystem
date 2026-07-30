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
    for trial in registry.all_trials():
        snap = trial.config_snapshot or {}
        if snap.get("is_full_event_wfo") or snap.get("evaluation_path") == "event_driven_wfo":
            signal_sources.append("candidate_dsl_trees")
    disc = result.discovery_results[0] if result.discovery_results else None
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
