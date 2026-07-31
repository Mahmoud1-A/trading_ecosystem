"""Focused tests for the Phase 1 cost-controlled repair pass on fix/alpha-budget-drain.

Covers exactly the three Phase 1 blockers:
  1. Direction-aware family entries (condition sign must determine direction).
  2. ParameterRobustness tree coverage (entry/exit/stop/target/sizing/regime_gates).
  3. Full WFO accounting (budget only advances on proven real WFO completion).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from discovery.candidate import build_candidate
from discovery.evaluator import CandidateEvaluator, EvalOutcome, SyntheticOOSBackend
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.expression_tree import ExprNode, constant_node, feature_node, op_node, parameter_node
from discovery.family_generator import materialize_family_spec
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import FamilyCampaignConfig, MultiFamilyCampaign
from discovery.operators import OperatorId
from discovery.parameter_robustness import (
    ROBUSTNESS_PARAMETER_NOT_BOUND,
    ParameterNotBoundError,
    ParameterRobustness,
)
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.types import CreationMethod, ValueType
from registry.experiment_registry import ExperimentRegistry


TZ = "America/Chicago"

# Feature ids that actually carry the directional signal for each pattern
# (excludes incidental features like temp.minutes_since_open used only for
# optional AND-window confirmation).
_DIRECTION_FEATURES: dict[str, tuple[str, ...]] = {
    "zscore_extreme": ("price.rolling_z_20", "liq.dist_session_vwap"),
    "vwap_zscore": ("price.rolling_z_20", "liq.dist_session_vwap"),
    "dist_mean_threshold": (
        "price.dist_rolling_mean_20",
        "liq.dist_session_vwap",
        "price.rolling_z_20",
    ),
    "vwap_deviation": (
        "price.dist_rolling_mean_20",
        "liq.dist_session_vwap",
        "price.rolling_z_20",
    ),
    "return_persistence": ("price.return_5", "price.simple_return_1"),
    "cross_momentum": ("price.return_5", "price.simple_return_1"),
    "breakout_distance": (
        "price.breakout_distance_20",
        "price.breakdown_distance_20",
    ),
    "compression_release": (
        "price.breakout_distance_20",
        "price.breakdown_distance_20",
    ),
    "gap_fade_entry": ("price.close_to_open",),
    "vwap_gap_reversion": ("price.close_to_open", "liq.dist_session_vwap"),
}

# Group A: LONG -> negative/below comparison, SHORT -> positive/above comparison.
# Group B: LONG -> positive/above comparison, SHORT -> negative/below comparison.
_GROUP_A_PATTERNS = {
    "zscore_extreme",
    "vwap_zscore",
    "dist_mean_threshold",
    "vwap_deviation",
    "gap_fade_entry",
    "vwap_gap_reversion",
}
_GROUP_B_PATTERNS = {
    "return_persistence",
    "cross_momentum",
    "breakout_distance",
    "compression_release",
}

_UP_OPS = {OperatorId.GREATER_THAN.value, OperatorId.CROSS_ABOVE.value}
_DOWN_OPS = {OperatorId.LESS_THAN.value, OperatorId.CROSS_BELOW.value}

_FAMILY_PATTERNS = {
    "mean_reversion": ("zscore_extreme", "dist_mean_threshold"),
    "VWAP_reversion": ("vwap_deviation", "vwap_zscore"),
    "momentum": ("return_persistence", "cross_momentum"),
    "breakout": ("breakout_distance", "compression_release"),
    "gap_fade": ("gap_fade_entry", "vwap_gap_reversion"),
}


def _direction_bearing_ops(node: ExprNode, feature_ids: tuple[str, ...]) -> list[str]:
    """Collect comparison/cross operator names whose left child is a directional feature."""
    found: list[str] = []
    for n in node.walk():
        if n.kind.value != "OPERATOR" or n.name not in (_UP_OPS | _DOWN_OPS):
            continue
        if not n.children:
            continue
        left = n.children[0]
        if left.kind.value == "FEATURE" and left.name in feature_ids:
            found.append(n.name)
    return found


class TestDirectionLinkedConditionUnit:
    """Deterministic, non-random proof that condition sign always matches direction."""

    @pytest.mark.parametrize("family_id", list(_FAMILY_PATTERNS))
    def test_condition_sign_matches_direction_for_both_sides(self, family_id: str) -> None:
        spec = materialize_family_spec(family_id, seed=3)
        gen = CandidateGenerator.from_family_spec(spec)
        for pattern in _FAMILY_PATTERNS[family_id]:
            assert pattern in _GROUP_A_PATTERNS or pattern in _GROUP_B_PATTERNS
            group_a = pattern in _GROUP_A_PATTERNS
            feats = _DIRECTION_FEATURES[pattern]
            for long_side in (True, False):
                # Multiple rng draws so both the plain and AND/OR-augmented
                # branches of the pattern get exercised deterministically.
                for rng_seed in (1, 2, 3, 4, 5, 6, 7, 8):
                    rng = np.random.default_rng(rng_seed)
                    cond = gen._direction_linked_cond(pattern, rng, long_side=long_side)
                    ops = _direction_bearing_ops(cond, feats)
                    assert ops, f"{family_id}/{pattern}: no directional comparison found"
                    for op_name in ops:
                        is_up = op_name in _UP_OPS
                        expected_up = (not long_side) if group_a else long_side
                        assert is_up == expected_up, (
                            f"{family_id}/{pattern} long_side={long_side}: "
                            f"got {op_name} (up={is_up}), expected up={expected_up}"
                        )

    def test_gap_fade_never_uses_abs_on_gap(self) -> None:
        """Explicit regression guard: no ABS(gap) followed by direction choice."""
        spec = materialize_family_spec("gap_fade", seed=4)
        gen = CandidateGenerator.from_family_spec(spec)
        for pattern in ("gap_fade_entry", "vwap_gap_reversion"):
            for long_side in (True, False):
                for rng_seed in range(1, 12):
                    rng = np.random.default_rng(rng_seed)
                    cond = gen._direction_linked_cond(pattern, rng, long_side=long_side)
                    for n in cond.walk():
                        if n.kind.value == "OPERATOR" and n.name == OperatorId.ABS.value:
                            child = n.children[0] if n.children else None
                            assert not (
                                child is not None
                                and child.kind.value == "FEATURE"
                                and child.name == "price.close_to_open"
                            ), "gap_fade must not wrap the gap feature in ABS()"


class TestDirectionAwareFamilyEntriesIntegration:
    """End-to-end proof via the public generate() pipeline + provenance."""

    @pytest.mark.parametrize("family_id", list(_FAMILY_PATTERNS))
    def test_generated_candidates_direction_matches_condition_and_provenance(
        self, family_id: str
    ) -> None:
        spec = materialize_family_spec(family_id, seed=17)
        gen = CandidateGenerator.from_family_spec(spec)
        seen_long = False
        seen_short = False
        group_a = family_id not in {"momentum", "breakout"}
        checked = 0
        for s in range(0, 250):
            try:
                cand = gen.generate(seed=s, max_attempts=12)
            except RuntimeError:
                continue
            checked += 1
            direction = cand.family_provenance.get("direction")
            assert direction in {"ENTRY_LONG", "ENTRY_SHORT"}
            # Provenance direction must match the actual wrapping operator.
            assert cand.entry_tree.name == direction

            pattern_feats = set()
            for feats in (_DIRECTION_FEATURES[p] for p in _FAMILY_PATTERNS[family_id]):
                pattern_feats.update(feats)
            ops = _direction_bearing_ops(cand.entry_tree, tuple(pattern_feats))
            long_side = direction == "ENTRY_LONG"
            for op_name in ops:
                is_up = op_name in _UP_OPS
                expected_up = (not long_side) if group_a else long_side
                assert is_up == expected_up, (family_id, direction, op_name)

            seen_long = seen_long or long_side
            seen_short = seen_short or not long_side
            if seen_long and seen_short and checked >= 6:
                break
        assert seen_long, f"{family_id}: never generated an ENTRY_LONG candidate"
        assert seen_short, f"{family_id}: never generated an ENTRY_SHORT candidate"


class TestParameterRobustnessTreeCoverage:
    def _atr_candidate(self, *, stop_mult: float = 1.5, target_mult: float = 2.5):
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        entry = op_node(OperatorId.ENTRY_LONG, op_node(OperatorId.LESS_THAN, z, parameter_node("z_entry", -2.0)))
        exit_tree = op_node(
            OperatorId.EXIT_SIGNAL, op_node(OperatorId.GREATER_THAN, z, parameter_node("z_exit", -0.25))
        )
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, parameter_node("atr_stop_mult", stop_mult))
        target = op_node(OperatorId.ATR_TARGET, atr, parameter_node("atr_target_mult", target_mult))
        return build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            stop=stop,
            target=target,
            strategy_family="test_robustness",
            creation_method=CreationMethod.RANDOM,
            grammar_version="v",
            feature_set_version="v",
            cost_model_version="v",
            random_seed=1,
        )

    def test_perturbing_atr_stop_mult_changes_stop_tree_and_candidate_id(self) -> None:
        cand = self._atr_candidate()
        rob = ParameterRobustness()
        perturbed, changed_paths = rob._with_param(cand, "atr_stop_mult", 3.3)
        assert any(p.startswith("stop") for p in changed_paths)
        assert not any(p.startswith("target") for p in changed_paths)
        assert perturbed.stop.children[1].meta["default"] == pytest.approx(3.3)
        # Target must be untouched.
        assert perturbed.target.children[1].meta["default"] == pytest.approx(2.5)
        assert perturbed.candidate_id != cand.candidate_id

    def test_perturbing_atr_target_mult_changes_target_tree_and_candidate_id(self) -> None:
        cand = self._atr_candidate()
        rob = ParameterRobustness()
        perturbed, changed_paths = rob._with_param(cand, "atr_target_mult", 5.1)
        assert any(p.startswith("target") for p in changed_paths)
        assert not any(p.startswith("stop") for p in changed_paths)
        assert perturbed.target.children[1].meta["default"] == pytest.approx(5.1)
        assert perturbed.stop.children[1].meta["default"] == pytest.approx(1.5)
        assert perturbed.candidate_id != cand.candidate_id

    def test_regime_gate_and_sizing_are_covered(self) -> None:
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        entry = op_node(OperatorId.ENTRY_LONG, op_node(OperatorId.LESS_THAN, z, parameter_node("z_entry", -2.0)))
        gate_feat = feature_node("regime.trend_state", ValueType.REGIME)
        gate_cond = op_node(OperatorId.GREATER_THAN, gate_feat, parameter_node("regime_thr", 0.1))
        regime_gate = op_node(OperatorId.REGIME_GATE, gate_cond, gate_feat)
        sizing = op_node(OperatorId.ABS, parameter_node("risk_frac", 0.01))
        cand = build_candidate(
            entry_tree=entry,
            sizing=sizing,
            regime_gates=(regime_gate,),
            strategy_family="test_robustness_gates",
            creation_method=CreationMethod.RANDOM,
            grammar_version="v",
            feature_set_version="v",
            cost_model_version="v",
            random_seed=2,
        )
        rob = ParameterRobustness()
        perturbed_gate, gate_paths = rob._with_param(cand, "regime_thr", 0.9)
        assert any(p.startswith("regime_gate[0]") for p in gate_paths)
        assert perturbed_gate.regime_gates[0].children[0].children[1].meta["default"] == pytest.approx(0.9)

        perturbed_sizing, sizing_paths = rob._with_param(cand, "risk_frac", 0.05)
        assert any(p.startswith("sizing") for p in sizing_paths)
        assert perturbed_sizing.sizing.children[0].meta["default"] == pytest.approx(0.05)

    def test_unbound_parameter_rejected(self) -> None:
        cand = self._atr_candidate()
        rob = ParameterRobustness()
        with pytest.raises(ParameterNotBoundError, match=ROBUSTNESS_PARAMETER_NOT_BOUND):
            rob._with_param(cand, "definitely_not_a_real_param", 1.0)

    def test_probe_reports_not_bound_result_without_crashing(self) -> None:
        # Build a candidate whose `parameters` dict includes a name that is not
        # actually present in any tree (defensive: simulates a stale/foreign
        # parameter map) to prove `probe()` degrades to a rejection, not a crash.
        cand = self._atr_candidate()
        object.__setattr__(cand, "parameters", {**cand.parameters, "ghost_param": 0.5})
        rob = ParameterRobustness(relative_steps=(-0.1, 0.0, 0.1))
        results = rob.probe(cand)
        ghost = next(r for r in results if r.parameter == "ghost_param")
        assert ghost.accepted is False
        assert ghost.reason == ROBUSTNESS_PARAMETER_NOT_BOUND
        # Real, bound params must still be probed successfully alongside it.
        bound_names = {r.parameter for r in results if r.reason != ROBUSTNESS_PARAMETER_NOT_BOUND}
        assert {"atr_stop_mult", "atr_target_mult", "z_entry", "z_exit"}.issubset(bound_names)


def _bars(n: int = 900, seed: int = 0) -> pd.DataFrame:
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


class TestFullWfoAccounting:
    def test_synthetic_backend_never_advances_full_wfo(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg_synth")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        ev = CandidateEvaluator(
            registry=reg, budget=budget, counters=counters, backend=SyntheticOOSBackend()
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=1)
        rec = ev.evaluate(cand)
        # A completed evaluation (not duplicate/precheck/invalid/feature-unavailable).
        assert rec.outcome in {EvalOutcome.REGISTERED, EvalOutcome.REJECTED, EvalOutcome.RISK_FAILED}
        # Not is_full_event_wfo -> must never consume full_wfo budget.
        assert counters.full_wfo == 0
        assert rec.meta.get("is_full_wfo_completion") is False

    def test_real_backend_completion_advances_full_wfo_exactly_once(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg_real")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        backend = EventDrivenDiscoveryBackend(bars=_bars(600, seed=2), require_real_bars=False)
        ev = CandidateEvaluator(registry=reg, budget=budget, counters=counters, backend=backend)
        cand = CandidateGenerator().seed_template_mean_reversion(seed=2)
        rec = ev.evaluate(cand)
        assert counters.full_wfo == 1
        assert rec.meta.get("is_full_wfo_completion") is True
        assert rec.train_metrics.get("signal_source") == "candidate_dsl_trees"
        assert rec.train_metrics.get("is_full_event_wfo") is True
        assert int(rec.train_metrics.get("wfo_completed_folds", 0)) > 0

    def test_duplicate_precheck_and_invalid_dsl_never_advance_full_wfo(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg_precheck")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=10)
        counters = BudgetCounters()
        backend = EventDrivenDiscoveryBackend(bars=_bars(600, seed=3), require_real_bars=False)
        ev = CandidateEvaluator(registry=reg, budget=budget, counters=counters, backend=backend)
        cand = CandidateGenerator().seed_template_mean_reversion(seed=3)

        rec1 = ev.evaluate(cand)
        # A real WFO evaluation completed (regardless of downstream economic
        # accept/reject) — this alone must move the Full WFO counter.
        assert rec1.outcome in {EvalOutcome.REGISTERED, EvalOutcome.REJECTED, EvalOutcome.RISK_FAILED}
        assert rec1.meta.get("is_full_wfo_completion") is True
        wfo_after_first = counters.full_wfo
        assert wfo_after_first == 1

        # Duplicate re-submission must not touch the backend or the counter.
        rec2 = ev.evaluate(cand)
        assert rec2.outcome is EvalOutcome.DUPLICATE_SKIPPED
        assert counters.full_wfo == wfo_after_first

        # INVALID_DSL_TYPE path (registered directly, backend never called) must
        # not touch the counter either.
        ev.register_invalid_dsl(ValueError("bad dsl"), seed=999, operation="test")
        assert counters.full_wfo == wfo_after_first

    def test_eval_failure_before_completion_never_advances_full_wfo(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg_fail")
        budget = SearchBudget(max_full_wfo_evaluations=5, max_evaluated_candidates=5)
        counters = BudgetCounters()
        backend = EventDrivenDiscoveryBackend(bars=_bars(600, seed=4), require_real_bars=False)

        def _boom(_candidate):
            raise RuntimeError("simulated mid-evaluation failure")

        backend.evaluate = _boom  # type: ignore[method-assign]
        ev = CandidateEvaluator(registry=reg, budget=budget, counters=counters, backend=backend)
        cand = CandidateGenerator().seed_template_mean_reversion(seed=4)
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.EVAL_FAILED
        assert counters.full_wfo == 0

    def test_multi_family_campaign_full_wfo_uses_authoritative_delta(self, tmp_path: Path) -> None:
        """SyntheticOOSBackend campaign: family_stats.full_wfo must stay 0 —
        previously it was incremented for almost every non-duplicate outcome."""
        registry = ExperimentRegistry(tmp_path / "reg_campaign")
        cfg = FamilyCampaignConfig(
            requested_family_count=2,
            min_candidates_per_family=4,
            total_candidate_budget=8,
            max_full_wfo=6,
            adaptive_reallocation=False,
            seed=5,
            min_oos_trades=1,
            min_oos_trades_per_fold=1,
            max_runtime_seconds=30,
        )
        campaign = MultiFamilyCampaign(
            config=cfg,
            registry=registry,
            backend=SyntheticOOSBackend(),
            discovery_run_id="test_phase1_full_wfo",
        )
        result = campaign.run()
        for st in result.family_stats:
            assert st.full_wfo == 0, f"{st.family_id}: full_wfo should be 0 with a non-event-driven backend"
