"""Focused tests for the Phase 2 cost-controlled repair pass on fix/alpha-budget-drain.

Covers exactly the three Phase 2 blockers:
  1. Feature-capability resolution must fail closed for research-eligible execution.
  2. Semantic threshold-domain validation must run inside CandidateEvaluator for
     every executable tree, regardless of candidate creation path.
  3. Real Stress scenario semantics (removed_best_day / removed_best_trades /
     symbol_exclusion / unknown scenarios / signal-source integrity).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from discovery.candidate import build_candidate
from discovery.crossover import Crossover
from discovery.evaluator import (
    FEATURE_CAPABILITY_RESOLUTION_FAILED,
    PERMISSIVE_NON_RESEARCH_FALLBACK,
    CandidateEvaluator,
    EvalOutcome,
    SyntheticOOSBackend,
)
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.expression_tree import constant_node, feature_node, op_node, parameter_node
from discovery.feature_domains import INVALID_FEATURE_THRESHOLD_DOMAIN
from discovery.fitness import FoldOOSMetrics, RobustFitness
from discovery.generator import CandidateGenerator
from discovery.mutation import Mutator
from discovery.operators import OperatorId
from discovery.promotion import PromotionGate
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.stress import StressResult, StressTester
from discovery.stress_backend import (
    NOT_APPLICABLE_SINGLE_SYMBOL,
    STRESS_BACKEND_KIND,
    STRESS_BASELINE_TRADES_UNAVAILABLE,
    UNSUPPORTED_STRESS_SCENARIO,
    StressScenarioStatus,
    build_stress_backend_for_scenario,
)
from discovery.types import CreationMethod, ValueType
from registry.experiment_registry import ExperimentRegistry


TZ = "America/Chicago"


def _bars(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz=TZ)
    close = 4800 + np.cumsum(rng.normal(0, 0.35, n))
    open_ = np.r_[close[0], close[:-1]]
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


def _and_with_balanced_gate(
    violating_cond,
    *,
    balance_feature_id: str,
    balance_value_type,
    threshold: float,
    combine: OperatorId = OperatorId.AND,
):
    """Wrap a violating comparison with a balanced gate so the structural
    precheck's constant-signal probe (which samples feature values from
    typical ranges and would otherwise see an always-true/always-false entry
    and reject it as PRECHECK_FAILED before threshold validation even runs)
    sees a genuinely mixed-truth entry instead. Uses a feature distinct from
    the one inside ``violating_cond`` so the two conditions are not spuriously
    correlated via a shared binding.

    ``combine`` must be AND when ``violating_cond`` probes as always-TRUE
    (AND collapses to the balanced condition) and OR when it probes as
    always-FALSE (OR collapses to the balanced condition).
    """
    balance_feat = feature_node(balance_feature_id, balance_value_type)
    balanced_cond = op_node(OperatorId.GREATER_THAN, balance_feat, constant_node(threshold))
    return op_node(combine, violating_cond, balanced_cond)


def _simple_candidate(
    entry=None,
    *,
    stop=None,
    target=None,
    sizing=None,
    regime_gates=(),
    exit_tree=None,
    creation_method: CreationMethod = CreationMethod.RANDOM,
    seed: int = 1,
):
    z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
    if entry is None:
        entry = op_node(
            OperatorId.ENTRY_LONG, op_node(OperatorId.LESS_THAN, z, constant_node(-2.0))
        )
    return build_candidate(
        entry_tree=entry,
        exit_tree=exit_tree,
        stop=stop,
        target=target,
        sizing=sizing,
        regime_gates=regime_gates,
        strategy_family="test_phase2",
        creation_method=creation_method,
        grammar_version="v",
        feature_set_version="v",
        cost_model_version="v",
        random_seed=seed,
    )


# ---------------------------------------------------------------------------
# Task 1: feature-capability resolution must fail closed
# ---------------------------------------------------------------------------


class TestFeatureCapabilityResolutionFailClosed:
    def test_research_eligible_stops_before_wfo_with_structured_artifact(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "features.generator.FeatureGenerator.generate",
            side_effect=RuntimeError("synthetic feature generation failure"),
        ):
            backend = EventDrivenDiscoveryBackend(bars=_bars(seed=1), require_real_bars=False)
        assert backend.feature_resolution_status == "failed"
        assert backend.available_feature_ids is None

        calls = {"n": 0}
        orig_evaluate = backend.evaluate

        def wrapped(cand):
            calls["n"] += 1
            return orig_evaluate(cand)

        backend.evaluate = wrapped  # type: ignore[method-assign]

        reg = ExperimentRegistry(tmp_path / "reg")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        ev = CandidateEvaluator(
            registry=reg,
            budget=budget,
            counters=counters,
            backend=backend,
            research_eligible=True,
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=1)
        rec = ev.evaluate(cand)

        assert rec.outcome is EvalOutcome.FEATURE_CAPABILITY_RESOLUTION_FAILED
        assert rec.rejection_reason == FEATURE_CAPABILITY_RESOLUTION_FAILED

        # Required artifact fields.
        meta = rec.meta
        assert meta["research_eligible"] is True
        assert meta["feature_resolution_status"] == FEATURE_CAPABILITY_RESOLUTION_FAILED
        assert "synthetic feature generation failure" in meta["resolution_error"]
        assert meta["available_features"] is None
        assert meta["dataset_capabilities"] == []
        assert meta["candidate_id"] == cand.candidate_id
        assert meta["full_wfo_consumed"] is False

        # Zero Full WFO budget consumed; real backend evaluation never called.
        assert counters.full_wfo == 0
        assert calls["n"] == 0

    def test_full_wfo_counter_unchanged_across_multiple_candidates(self, tmp_path: Path) -> None:
        with patch(
            "features.generator.FeatureGenerator.generate",
            side_effect=RuntimeError("boom"),
        ):
            backend = EventDrivenDiscoveryBackend(bars=_bars(seed=2), require_real_bars=False)
        reg = ExperimentRegistry(tmp_path / "reg2")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        ev = CandidateEvaluator(
            registry=reg, budget=budget, counters=counters, backend=backend, research_eligible=True
        )
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        for s in range(3):
            # Distinct entry constants so each candidate gets a distinct
            # content-derived candidate_id (never a duplicate re-submission).
            entry = op_node(
                OperatorId.ENTRY_LONG,
                op_node(OperatorId.LESS_THAN, z, constant_node(-2.0 - s * 0.1)),
            )
            cand = _simple_candidate(entry, seed=s)
            rec = ev.evaluate(cand)
            assert rec.outcome is EvalOutcome.FEATURE_CAPABILITY_RESOLUTION_FAILED
        assert counters.full_wfo == 0

    def test_original_exception_preserved_on_backend(self) -> None:
        boom = RuntimeError("original exception object")
        with patch("features.generator.FeatureGenerator.generate", side_effect=boom):
            backend = EventDrivenDiscoveryBackend(bars=_bars(seed=3), require_real_bars=False)
        assert backend.feature_resolution_exception is boom
        assert "original exception object" in (backend.feature_resolution_error or "")

    def test_non_research_fallback_is_labeled_and_not_confused_with_research(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "features.generator.FeatureGenerator.generate",
            side_effect=RuntimeError("boom"),
        ):
            backend = EventDrivenDiscoveryBackend(bars=_bars(seed=4), require_real_bars=False)
        reg = ExperimentRegistry(tmp_path / "reg3")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        ev = CandidateEvaluator(
            registry=reg,
            budget=budget,
            counters=counters,
            backend=backend,
            research_eligible=False,
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=4)
        rec = ev.evaluate(cand)

        # Never confused with a real research rejection.
        assert rec.outcome is not EvalOutcome.FEATURE_CAPABILITY_RESOLUTION_FAILED
        assert rec.outcome in {EvalOutcome.REGISTERED, EvalOutcome.REJECTED, EvalOutcome.RISK_FAILED}
        assert rec.meta.get("feature_resolution_status") == PERMISSIVE_NON_RESEARCH_FALLBACK
        assert rec.meta.get("resolution_error")


# ---------------------------------------------------------------------------
# Task 2: semantic threshold-domain validation inside CandidateEvaluator
# ---------------------------------------------------------------------------


class _BoomBackend:
    """Backend that must never be called once a candidate is rejected pre-WFO."""

    backend_kind = "boom_backend_should_never_run"
    is_full_event_wfo = True

    def evaluate(self, candidate):  # pragma: no cover - should never execute
        raise AssertionError("backend.evaluate() must not be called for a rejected candidate")


def _evaluator(tmp_path: Path, name: str) -> CandidateEvaluator:
    reg = ExperimentRegistry(tmp_path / name)
    budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=10)
    counters = BudgetCounters()
    return CandidateEvaluator(registry=reg, budget=budget, counters=counters, backend=_BoomBackend())


class TestSemanticThresholdValidation:
    def test_rank_below_valid_range_rejected(self, tmp_path: Path) -> None:
        rank_feat = feature_node("liq.volume_pct_20", ValueType.RANK)
        violating = op_node(OperatorId.GREATER_THAN, rank_feat, constant_node(-0.2))
        cond = _and_with_balanced_gate(
            violating,
            balance_feature_id="price.rolling_z_20",
            balance_value_type=ValueType.ZSCORE,
            threshold=0.0,
        )
        entry = op_node(OperatorId.ENTRY_LONG, cond)
        cand = _simple_candidate(entry)
        ev = _evaluator(tmp_path, "rank")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert rec.rejection_reason == INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_minutes_since_open_negative_rejected(self, tmp_path: Path) -> None:
        minutes_feat = feature_node("temp.minutes_since_open", ValueType.TIME)
        violating = op_node(OperatorId.LESS_THAN, minutes_feat, constant_node(-1.0))
        cond = _and_with_balanced_gate(
            violating,
            balance_feature_id="price.rolling_z_20",
            balance_value_type=ValueType.ZSCORE,
            threshold=0.0,
            combine=OperatorId.OR,
        )
        entry = op_node(OperatorId.ENTRY_LONG, cond)
        cand = _simple_candidate(entry)
        ev = _evaluator(tmp_path, "minutes")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_one_bar_return_above_valid_range_rejected(self, tmp_path: Path) -> None:
        ret_feat = feature_node("price.simple_return_1", ValueType.RETURN)
        violating = op_node(OperatorId.GREATER_THAN, ret_feat, constant_node(1.2))
        cond = _and_with_balanced_gate(
            violating,
            balance_feature_id="price.rolling_z_20",
            balance_value_type=ValueType.ZSCORE,
            threshold=0.0,
            combine=OperatorId.OR,
        )
        entry = op_node(OperatorId.ENTRY_LONG, cond)
        cand = _simple_candidate(entry)
        ev = _evaluator(tmp_path, "return")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_invalid_atr_stop_multiplier_rejected(self, tmp_path: Path) -> None:
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, constant_node(25.0))
        cand = _simple_candidate(stop=stop)
        ev = _evaluator(tmp_path, "atr_stop")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_invalid_atr_target_multiplier_rejected(self, tmp_path: Path) -> None:
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        target = op_node(OperatorId.ATR_TARGET, atr, constant_node(35.0))
        cand = _simple_candidate(target=target)
        ev = _evaluator(tmp_path, "atr_target")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_invalid_threshold_under_abs_rejected(self, tmp_path: Path) -> None:
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        violating = op_node(OperatorId.GREATER_THAN, op_node(OperatorId.ABS, z), constant_node(50.0))
        cond = _and_with_balanced_gate(
            violating,
            balance_feature_id="liq.volume_pct_20",
            balance_value_type=ValueType.RANK,
            threshold=0.5,
            combine=OperatorId.OR,
        )
        entry = op_node(OperatorId.ENTRY_LONG, cond)
        cand = _simple_candidate(entry)
        ev = _evaluator(tmp_path, "abs")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_invalid_between_bound_rejected(self, tmp_path: Path) -> None:
        rank_feat = feature_node("liq.volume_pct_20", ValueType.RANK)
        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(OperatorId.BETWEEN, rank_feat, constant_node(-0.5), constant_node(0.5)),
        )
        cand = _simple_candidate(entry)
        ev = _evaluator(tmp_path, "between")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_invalid_threshold_inside_regime_gate_rejected(self, tmp_path: Path) -> None:
        regime_feat = feature_node("regime.trend_state", ValueType.REGIME)
        gate_cond = op_node(OperatorId.GREATER_THAN, regime_feat, constant_node(5.0))
        regime_gate = op_node(OperatorId.REGIME_GATE, gate_cond, regime_feat)
        cand = _simple_candidate(regime_gates=(regime_gate,))
        ev = _evaluator(tmp_path, "regime_gate")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        assert ev.counters.full_wfo == 0

    def test_manually_constructed_candidate_not_from_generator_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Generation-time checks are never invoked for a hand-built candidate —
        only CandidateEvaluator can catch this, proving the gate is not
        creation-path dependent."""
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        violating = op_node(OperatorId.LESS_THAN, z, constant_node(-99.0))
        cond = _and_with_balanced_gate(
            violating,
            balance_feature_id="liq.volume_pct_20",
            balance_value_type=ValueType.RANK,
            threshold=0.5,
            combine=OperatorId.OR,
        )
        entry = op_node(OperatorId.ENTRY_LONG, cond)
        cand = _simple_candidate(entry, creation_method=CreationMethod.RANDOM)
        ev = _evaluator(tmp_path, "manual")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN

    def test_hypothetical_invalid_mutation_output_is_rejected_not_clamped(
        self, tmp_path: Path
    ) -> None:
        """Even if a mutation somehow produced an out-of-domain constant, the
        evaluator must reject it outright — never silently clamp it in."""
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, constant_node(-2.0))
        cand = _simple_candidate(stop=stop, creation_method=CreationMethod.MUTATION)
        ev = _evaluator(tmp_path, "mutation_output")
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN
        # Never silently repaired to a valid value — reported as-is.
        violation = rec.meta["violations"][0]
        assert violation["tested_value"] == pytest.approx(-2.0)

    def test_violation_artifact_has_required_fields(self, tmp_path: Path) -> None:
        rank_feat = feature_node("liq.volume_pct_20", ValueType.RANK)
        violating = op_node(OperatorId.GREATER_THAN, rank_feat, constant_node(-0.2))
        cond = _and_with_balanced_gate(
            violating,
            balance_feature_id="price.rolling_z_20",
            balance_value_type=ValueType.ZSCORE,
            threshold=0.0,
        )
        entry = op_node(OperatorId.ENTRY_LONG, cond)
        cand = _simple_candidate(entry)
        ev = _evaluator(tmp_path, "fields")
        rec = ev.evaluate(cand)
        violation = rec.meta["violations"][0]
        for key in (
            "tree_path",
            "node_path",
            "feature_id",
            "parameter_or_constant",
            "tested_value",
            "valid_range",
            "units",
            "reason",
        ):
            assert key in violation

    def test_valid_candidate_not_rejected_by_threshold_check(self, tmp_path: Path) -> None:
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        entry = op_node(OperatorId.ENTRY_LONG, op_node(OperatorId.LESS_THAN, z, constant_node(-2.0)))
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, constant_node(1.5))
        cand = _simple_candidate(entry, stop=stop)
        reg = ExperimentRegistry(tmp_path / "valid")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        ev = CandidateEvaluator(
            registry=reg,
            budget=budget,
            counters=counters,
            backend=SyntheticOOSBackend(),
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
        )
        rec = ev.evaluate(cand)
        assert rec.outcome is not EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN


class TestMutationAndCrossoverDomainSampling:
    def test_mutation_samples_atr_stop_mult_within_domain(self) -> None:
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, parameter_node("atr_stop_mult", 1.5))
        mutator = Mutator()
        for seed in range(200):
            rng = np.random.default_rng(seed)
            mutated = mutator.mutate_tree(stop, rng)
            for n in mutated.walk():
                if n.kind.value == "PARAMETER" and n.name == "atr_stop_mult":
                    val = float(n.meta["default"])
                    assert 0.05 <= val <= 20.0, f"seed={seed}: atr_stop_mult={val} out of domain"

    def test_crossover_cross_domain_value_still_caught_by_evaluator(self, tmp_path: Path) -> None:
        """Crossover swaps whole typed subtrees (not raw constant resampling),
        so a numeric value can legally cross from one feature's domain into a
        structurally-compatible but semantically different slot. This proves
        the evaluator-level gate (Task 2) is the safety net that guarantees no
        invalid candidate ever proceeds, regardless of crossover's mechanism.
        """
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        tree_a = op_node(OperatorId.ATR_STOP, atr, constant_node(1.5))
        tree_b = op_node(OperatorId.LESS_THAN, z, constant_node(-2.0))
        rng = np.random.default_rng(0)
        child_a, _ = Crossover().crossover_trees(tree_a, tree_b, rng)
        # child_a must be the ATR_STOP tree with a swapped-in constant.
        assert child_a.name == OperatorId.ATR_STOP.value
        cand = _simple_candidate(stop=child_a, creation_method=CreationMethod.CROSSOVER)
        ev = _evaluator(tmp_path, "crossover_domain")
        rec = ev.evaluate(cand)
        # Either the swap produced an out-of-domain multiplier (rejected) or it
        # happened to land in-domain — either way the backend boom-guard would
        # have fired if evaluation proceeded incorrectly past a violation.
        if rec.outcome is EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN:
            assert ev.counters.full_wfo == 0
        else:
            assert child_a.children[1].meta.get("value", 0.0) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Task 3: real Stress scenario semantics
# ---------------------------------------------------------------------------


def _event_backend_with_bars(seed: int = 5) -> EventDrivenDiscoveryBackend:
    return EventDrivenDiscoveryBackend(bars=_bars(seed=seed), require_real_bars=False)


def _fake_trade(trade_id: str, exit_time: str, net_pnl: float) -> dict:
    return {"trade_id": trade_id, "exit_time": exit_time, "net_pnl": net_pnl, "fold_id": 0}


class TestRemovedBestDaySemantics:
    def test_uses_real_baseline_closed_trade_pnl_not_inference(self) -> None:
        base = _event_backend_with_bars(seed=10)
        trades = [
            _fake_trade("T1", "2024-01-02T09:00:00+00:00", 50.0),
            _fake_trade("T2", "2024-01-02T09:05:00+00:00", 25.0),
            _fake_trade("T3", "2024-01-03T09:00:00+00:00", 10.0),
            _fake_trade("T4", "2024-01-04T09:00:00+00:00", -5.0),
        ]
        arts = {"closed_trades": trades}
        built = build_stress_backend_for_scenario(base, "removed_best_day", baseline_artifacts=arts)
        assert not isinstance(built, StressScenarioStatus)
        changes = built.execution_changes
        # Day 2024-01-02 has 75.0 aggregate net PnL — the real highest day.
        assert changes["removed_day"] == "2024-01-02 00:00:00+00:00"
        assert changes["baseline_day_net_pnl"] == pytest.approx(75.0)
        assert set(changes["removed_trade_ids"]) == {"T1", "T2"}
        assert len(changes["removed_trade_timestamps"]) == 2
        assert "rerun_methodology" in changes
        assert "arbitrary" not in changes["rerun_methodology"].lower()

    def test_baseline_trades_unavailable_returns_status_not_arbitrary_fallback(self) -> None:
        base = _event_backend_with_bars(seed=11)
        built = build_stress_backend_for_scenario(
            base, "removed_best_day", baseline_artifacts={"closed_trades": []}
        )
        assert isinstance(built, StressScenarioStatus)
        assert built.stress_status == STRESS_BASELINE_TRADES_UNAVAILABLE

    def test_baseline_artifacts_none_returns_status(self) -> None:
        base = _event_backend_with_bars(seed=12)
        built = build_stress_backend_for_scenario(base, "removed_best_day", baseline_artifacts=None)
        assert isinstance(built, StressScenarioStatus)
        assert built.stress_status == STRESS_BASELINE_TRADES_UNAVAILABLE


class TestRemovedBestTradesSemantics:
    def test_not_an_alias_for_removed_best_day(self) -> None:
        base = _event_backend_with_bars(seed=13)
        # Best AGGREGATE day (01-02, sum=100) is distinct from the single
        # highest-PnL individual trade (B3=90 on 01-05, which never sums past
        # 100 on its own day) — proves the two scenarios genuinely diverge.
        trades = [
            _fake_trade("A1", "2024-01-02T09:00:00+00:00", 20.0),
            _fake_trade("A2", "2024-01-02T09:05:00+00:00", 20.0),
            _fake_trade("A3", "2024-01-02T09:10:00+00:00", 20.0),
            _fake_trade("A4", "2024-01-02T09:15:00+00:00", 20.0),
            _fake_trade("A5", "2024-01-02T09:20:00+00:00", 20.0),
            _fake_trade("B1", "2024-01-03T09:00:00+00:00", 1.0),
            _fake_trade("B2", "2024-01-04T09:00:00+00:00", 1.0),
            _fake_trade("B3", "2024-01-05T09:00:00+00:00", 90.0),
            _fake_trade("B4", "2024-01-06T09:00:00+00:00", 1.0),
            _fake_trade("B5", "2024-01-07T09:00:00+00:00", 1.0),
        ]
        arts = {"closed_trades": trades}
        day_built = build_stress_backend_for_scenario(base, "removed_best_day", baseline_artifacts=arts)
        trades_built = build_stress_backend_for_scenario(
            base, "removed_best_trades", baseline_artifacts=arts
        )
        assert not isinstance(day_built, StressScenarioStatus)
        assert not isinstance(trades_built, StressScenarioStatus)

        day_changes = day_built.execution_changes
        trades_changes = trades_built.execution_changes
        assert day_changes["removed_day"] == "2024-01-02 00:00:00+00:00"
        # removed_best_trades must target the single highest-PnL trade (B3 on
        # 01-05), not the highest-aggregate-PnL day (01-02).
        assert "B3" in trades_changes["removed_trade_ids"]
        assert "A1" not in trades_changes["removed_trade_ids"]
        assert set(trades_changes["removed_trade_ids"]) != set(day_changes["removed_trade_ids"])
        for key in (
            "removed_trade_ids",
            "removed_trade_timestamps",
            "removed_fraction",
            "baseline_removed_net_pnl",
            "rerun_methodology",
        ):
            assert key in trades_changes
        assert "NOT_an_alias_for_removed_best_day" in trades_changes["rerun_methodology"]

    def test_baseline_unavailable_when_no_trades(self) -> None:
        base = _event_backend_with_bars(seed=14)
        built = build_stress_backend_for_scenario(
            base, "removed_best_trades", baseline_artifacts={"closed_trades": []}
        )
        assert isinstance(built, StressScenarioStatus)
        assert built.stress_status == STRESS_BASELINE_TRADES_UNAVAILABLE


class TestSymbolExclusionAndUnsupported:
    def test_symbol_exclusion_single_symbol_not_applicable(self) -> None:
        base = _event_backend_with_bars(seed=15)
        built = build_stress_backend_for_scenario(base, "symbol_exclusion")
        assert isinstance(built, StressScenarioStatus)
        assert built.stress_status == NOT_APPLICABLE_SINGLE_SYMBOL

    def test_unknown_scenario_returns_unsupported_not_baseline_rerun(self) -> None:
        base = _event_backend_with_bars(seed=16)
        built = build_stress_backend_for_scenario(base, "made_up_scenario_xyz")
        assert isinstance(built, StressScenarioStatus)
        assert built.stress_status == UNSUPPORTED_STRESS_SCENARIO


class TestStressTesterScenarioStatusHandling:
    def _fake_folds(self) -> list[FoldOOSMetrics]:
        return [
            FoldOOSMetrics(
                fold_id=0,
                expectancy=0.05,
                sharpe=1.2,
                profit_factor=1.3,
                calmar=0.4,
                max_drawdown=-0.05,
                n_trades=10,
            )
        ]

    def test_symbol_exclusion_never_executed_never_touches_backend(self) -> None:
        calls = {"n": 0}

        def _factory(scenario: str):
            calls["n"] += 1
            raise AssertionError("backend_factory must not be called for symbol_exclusion")

        budget = SearchBudget(max_stress_evaluations=5)
        counters = BudgetCounters()
        tester = StressTester(budget=budget, counters=counters, backend_factory=_factory)
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("symbol_exclusion",))
        assert len(results) == 1
        r = results[0]
        assert r.status == "not_applicable"
        assert r.passed is False
        assert r.failure_reason == NOT_APPLICABLE_SINGLE_SYMBOL
        assert calls["n"] == 0
        assert counters.stress == 0

    def test_unknown_scenario_never_executed(self) -> None:
        budget = SearchBudget(max_stress_evaluations=5)
        counters = BudgetCounters()
        tester = StressTester(
            budget=budget,
            counters=counters,
            backend_factory=lambda s: (_ for _ in ()).throw(
                AssertionError("must not build backend for unknown scenario")
            ),
        )
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("nonexistent_scenario",))
        assert results[0].status == "unsupported"
        assert results[0].passed is False
        assert counters.stress == 0

    def test_stress_status_backend_excluded_from_pass_rate(self) -> None:
        class _FakeStatusBackend:
            stress_status = STRESS_BASELINE_TRADES_UNAVAILABLE
            backend_kind = "stress_status_no_rerun"

            def evaluate(self, candidate):
                raise AssertionError("must not evaluate when stress_status is set")

        budget = SearchBudget(max_stress_evaluations=5)
        counters = BudgetCounters()
        tester = StressTester(
            budget=budget, counters=counters, backend_factory=lambda s: _FakeStatusBackend()
        )
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("removed_best_day",))
        assert results[0].status == "baseline_unavailable"
        assert results[0].passed is False
        assert counters.stress == 0

    def test_promotion_gate_excludes_non_executed_from_denominator(self) -> None:
        executed_pass = StressResult(
            scenario="base_costs",
            candidate_id="c1",
            fitness=1.0,
            median_expectancy=0.1,
            max_drawdown=-0.05,
            passed=True,
            status="executed",
        )
        not_applicable = StressResult(
            scenario="symbol_exclusion",
            candidate_id="c1",
            fitness=0.0,
            median_expectancy=0.0,
            max_drawdown=0.0,
            passed=False,
            status="not_applicable",
        )
        gate = PromotionGate(min_oos_fitness=0.0, min_stress_pass_rate=1.0)

        class _FrozenStub:
            ranking_source = "validation_oos"
            frozen_id = "f1"
            candidate_id = "c1"
            oos_fitness = 1.0

        decision = gate.decide(
            _FrozenStub(), train_score=None,
            stress_results=[executed_pass, not_applicable],
        )
        # If the not-applicable scenario counted in the denominator with
        # passed=False, pass_rate would be 0.5 < 1.0 and this would be HELD.
        assert decision.stress_pass_rate == pytest.approx(1.0)


class TestStressSignalSourceIntegrity:
    def _make_tester(self, fake_backend, *, research_eligible: bool) -> StressTester:
        budget = SearchBudget(max_stress_evaluations=5)
        counters = BudgetCounters()
        return StressTester(
            budget=budget,
            counters=counters,
            research_eligible=research_eligible,
            synthetic_stress_forbidden=research_eligible,
            backend_factory=lambda s: fake_backend,
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
        )

    def _folds(self) -> list[FoldOOSMetrics]:
        return [
            FoldOOSMetrics(
                fold_id=0, expectancy=0.05, sharpe=1.2, profit_factor=1.3,
                calmar=0.4, max_drawdown=-0.05, n_trades=10,
            )
        ]

    class _FakeBackend:
        backend_kind = STRESS_BACKEND_KIND
        is_full_event_wfo = True

        def __init__(self, folds, artifacts):
            self._folds = folds
            self.last_run_artifacts = artifacts

        def evaluate(self, candidate):
            return self._folds, {}

    def test_valid_research_scenario_reads_signal_source_from_artifact(self) -> None:
        arts = {
            "signal_source": "candidate_dsl_trees",
            "wfo_completed_folds": 3,
        }
        backend = self._FakeBackend(self._folds(), arts)
        tester = self._make_tester(backend, research_eligible=True)
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("base_costs",))
        r = results[0]
        assert r.signal_source == "candidate_dsl_trees"
        assert r.completed_fold_count == 3
        assert r.integrity_ok is True

    def test_hardcoded_signal_source_never_used_wrong_value_marks_integrity_failure(self) -> None:
        arts = {"signal_source": "not_the_real_source", "wfo_completed_folds": 3}
        backend = self._FakeBackend(self._folds(), arts)
        tester = self._make_tester(backend, research_eligible=True)
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("base_costs",))
        r = results[0]
        assert r.signal_source == "not_the_real_source"
        assert r.integrity_ok is False
        assert r.passed is False

    def test_zero_completed_folds_marks_integrity_failure(self) -> None:
        arts = {"signal_source": "candidate_dsl_trees", "wfo_completed_folds": 0}
        backend = self._FakeBackend(self._folds(), arts)
        tester = self._make_tester(backend, research_eligible=True)
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("base_costs",))
        r = results[0]
        assert r.completed_fold_count == 0
        assert r.integrity_ok is False
        assert r.passed is False

    def test_non_research_run_permissive_on_integrity(self) -> None:
        arts = {"signal_source": "missing_or_wrong", "wfo_completed_folds": 0}
        backend = self._FakeBackend(self._folds(), arts)
        tester = self._make_tester(backend, research_eligible=False)
        cand = _simple_candidate()
        results = tester.run(cand, base_fitness=1.0, scenarios=("base_costs",))
        r = results[0]
        assert r.integrity_ok is False
        # Non-research: integrity issue is recorded but does not force failure.
        assert r.passed is True
