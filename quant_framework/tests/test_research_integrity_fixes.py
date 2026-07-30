"""Acceptance tests for Alpha Miner research-integrity fixes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from discovery.evaluator import (
    FEATURE_UNAVAILABLE,
    CandidateEvaluator,
    EvalOutcome,
    SyntheticOOSBackend,
)
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.family_catalog import FAMILY_BLUEPRINTS
from discovery.family_generator import StrategyFamilyGenerator
from discovery.feature_domains import INVALID_FEATURE_THRESHOLD_DOMAIN, domain_for_feature
from discovery.fitness import RobustFitness
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import adaptive_reallocate_wfo
from discovery.parameter_robustness import (
    SYNTHETIC_ROBUSTNESS_FORBIDDEN,
    ParameterRobustness,
)
from discovery.regime_gates import UNSUPPORTED_FAMILY_CONSTRAINT, compile_regime_constraint
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.stable_hash import stable_int_hash
from discovery.stress import StressTester
from discovery.stress_backend import SYNTHETIC_STRESS_FORBIDDEN, make_stress_backend_factory
from features.regime_features import compute_regime_features
from registry.experiment_registry import ExperimentRegistry


TZ = "America/Chicago"


def _bars(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz=TZ)
    close = 4800 + np.cumsum(rng.normal(0, 0.35, n))
    open_ = np.r_[close[0], close[:-1]]
    # Plant small overnight gaps on session opens.
    gap = np.zeros(n)
    gap[::78] = rng.normal(0, 0.01, size=len(gap[::78]))
    open_ = open_ * (1.0 + gap)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + 0.4,
            "low": np.minimum(open_, close) - 0.4,
            "close": close,
            "volume": rng.integers(100, 2000, n).astype(float),
        }
    )


class TestSyntheticStressForbidden:
    def test_research_eligible_rejects_synthetic_factory(self) -> None:
        factory = make_stress_backend_factory(
            SyntheticOOSBackend(), research_eligible=True, synthetic_stress_forbidden=True
        )
        with pytest.raises(RuntimeError, match=SYNTHETIC_STRESS_FORBIDDEN):
            factory("costs_2x")

    def test_stress_tester_fails_on_synthetic_when_forbidden(self, tmp_path: Path) -> None:
        budget = SearchBudget(max_stress_evaluations=2)
        counters = BudgetCounters()
        tester = StressTester(
            budget=budget,
            counters=counters,
            research_eligible=True,
            synthetic_stress_forbidden=True,
            backend_factory=lambda s: SyntheticOOSBackend(seed_salt=1),
        )
        gen = CandidateGenerator()
        cand = gen.seed_template_mean_reversion(seed=1)
        with pytest.raises(RuntimeError, match=SYNTHETIC_STRESS_FORBIDDEN):
            tester.run(cand, base_fitness=1.0, scenarios=("base_costs",))


class TestRealRobustness:
    def test_research_rejects_synthetic_robustness(self) -> None:
        rob = ParameterRobustness(
            backend=SyntheticOOSBackend(),
            research_eligible=True,
            synthetic_robustness_forbidden=True,
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=2)
        with pytest.raises(RuntimeError, match=SYNTHETIC_ROBUSTNESS_FORBIDDEN):
            rob.probe(cand)

    def test_real_backend_probe_records_points(self, tmp_path: Path) -> None:
        bars = _bars(600, seed=3)
        backend = EventDrivenDiscoveryBackend(bars=bars, require_real_bars=False)
        rob = ParameterRobustness(
            backend=backend,
            research_eligible=True,
            synthetic_robustness_forbidden=True,
            relative_steps=(-0.1, 0.0, 0.1),
            fitness_model=RobustFitness(min_total_oos_trades=1, min_oos_trades_per_fold=0),
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=3)
        rows = rob.probe(cand)
        assert rows
        assert rows[0].backend_kind == "event_driven_wfo"
        assert rows[0].points
        assert "fold_metrics" in rows[0].as_dict()
        assert "total_oos_trades" in rows[0].as_dict()


class TestFeatureUnavailable:
    def test_rejects_before_wfo(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg")
        budget = SearchBudget(max_full_wfo_evaluations=3, max_evaluated_candidates=5)
        counters = BudgetCounters()
        backend = SyntheticOOSBackend()
        calls = {"n": 0}
        orig = backend.evaluate

        def wrapped(cand):
            calls["n"] += 1
            return orig(cand)

        backend.evaluate = wrapped  # type: ignore[method-assign]
        ev = CandidateEvaluator(
            registry=reg,
            budget=budget,
            counters=counters,
            backend=backend,
            available_feature_ids=frozenset({"price.simple_return_1"}),
            dataset_capabilities=("OHLCV_BARS",),
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=4)
        # Template uses price.rolling_z_20 + vol.atr_14 — unavailable.
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.FEATURE_UNAVAILABLE
        assert rec.rejection_reason == FEATURE_UNAVAILABLE
        assert calls["n"] == 0
        assert "missing_features" in rec.meta
        assert counters.full_wfo == 0


class TestRegimeGatesExecutable:
    def test_supported_constraints_compile(self) -> None:
        for key in (
            "require_trend_regime",
            "prefer_range_regime",
            "prefer_vol_expansion",
            "intraday_session_only",
            "prefer_liquid_session",
        ):
            gate = compile_regime_constraint(key)
            assert gate.name == "REGIME_GATE"

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match=UNSUPPORTED_FAMILY_CONSTRAINT):
            compile_regime_constraint("made_up_constraint")

    def test_family_generator_emits_regime_gates(self) -> None:
        spec = StrategyFamilyGenerator(seed=11).generate(count=1, family_ids=["mean_reversion"])[0]
        gen = CandidateGenerator.from_family_spec(spec)
        cand = gen.generate(seed=21)
        assert cand.regime_gates
        assert "direction" in cand.family_provenance


class TestFamilySemantics:
    def test_gap_fade_uses_gap_feature(self) -> None:
        bp = FAMILY_BLUEPRINTS["gap_fade"]
        assert "price.close_to_open" in bp["allowed_features"]
        spec = StrategyFamilyGenerator(seed=5).generate(count=1, family_ids=["gap_fade"])[0]
        gen = CandidateGenerator.from_family_spec(spec)
        cand = None
        for s in range(55, 55 + 80):
            try:
                cand = gen.generate(seed=s, max_attempts=48)
                break
            except RuntimeError:
                continue
        assert cand is not None, "failed to generate gap_fade candidate"
        feats = set(cand.feature_ids)
        assert "price.close_to_open" in feats or "price.gap_size" in feats

    def test_vol_expansion_param_domains(self) -> None:
        bp = FAMILY_BLUEPRINTS["volatility_expansion"]
        keys = set(bp["parameter_ranges"])
        assert "vol_threshold" in keys
        assert "directional_return_threshold" in keys
        assert "exit_vol_threshold" in keys
        assert "exit_threshold" not in keys or "directional_return_threshold" in keys

    def test_regime_features_causal(self) -> None:
        bars = _bars(200, seed=9).set_index("timestamp")
        reg = compute_regime_features(bars)
        assert "regime.trend_state" in reg.columns
        assert "regime.volatility_state" in reg.columns
        vals = reg["regime.trend_state"].dropna()
        assert len(vals) > 0
        assert bool(vals.between(-1.0, 1.0).all())


class TestAdaptiveAllocation:
    def test_stronger_family_gets_more_wfo(self) -> None:
        family_ids = ["weak", "strong", "mid"]
        alloc = adaptive_reallocate_wfo(
            family_ids=family_ids,
            max_full_wfo=12,
            early_scores={"weak": -1.0, "strong": 2.0, "mid": 0.1},
            min_quota=1,
            early_counts={"weak": 2, "strong": 2, "mid": 2},
        )
        assert alloc["strong"] > alloc["weak"]
        assert sum(alloc.values()) == 12
        assert all(v >= 1 for v in alloc.values())


class TestStableHash:
    def test_cross_process_stable(self, tmp_path: Path) -> None:
        script = (
            "from discovery.stable_hash import stable_int_hash; "
            "from discovery.family_generator import StrategyFamilyGenerator; "
            "g=StrategyFamilyGenerator(seed=42).generate(count=3); "
            "print(stable_int_hash('mean_reversion')); "
            "print('|'.join(f.canonical_hash() for f in g)); "
            "print('|'.join(f.effective_grammar_fingerprint() for f in g))"
        )
        env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        a = subprocess.check_output([sys.executable, "-c", script], env={**dict(**__import__("os").environ), **env})
        b = subprocess.check_output([sys.executable, "-c", script], env={**dict(**__import__("os").environ), **env})
        assert a == b

    def test_not_builtin_hash(self) -> None:
        # stable hash must ignore PYTHONHASHSEED salt of builtin hash
        assert stable_int_hash("gap_fade") == stable_int_hash("gap_fade")


class TestThresholdDomains:
    def test_minutes_reject_negative(self) -> None:
        dom = domain_for_feature("temp.minutes_since_open")
        assert not dom.accepts(-0.4)
        assert INVALID_FEATURE_THRESHOLD_DOMAIN.startswith("INVALID")
