"""Regression: Alpha Miner seed-template must not loop on a fixed candidate_id."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from discovery.evaluator import EvalOutcome, EvaluationRecord, SyntheticOOSBackend
from discovery.generator import CandidateGenerator
from discovery.search_budget import (
    DUPLICATE_CANDIDATE_ID,
    GENERATION_DIVERSITY_EXHAUSTED,
    BudgetCounters,
    SearchBudget,
)
from discovery.search_controller import SearchController
from registry.experiment_registry import ExperimentRegistry


def _budget(**kwargs: object) -> SearchBudget:
    defaults = dict(
        max_generated_candidates=40,
        max_evaluated_candidates=30,
        max_full_wfo_evaluations=8,
        max_runtime_seconds=60.0,
        max_candidates_per_family=40,
        max_candidates_per_complexity_tier=40,
        max_candidates_per_feature_family=40,
        population_size=4,
        elite_count=1,
        stagnation_limit=8,
        stagnation_generations=8,
        minimum_generations_before_stagnation=4,
    )
    defaults.update(kwargs)
    return SearchBudget(**defaults)


class TestSeedTemplateOnce:
    def test_seed_template_called_once(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "registry")
        budget = _budget(max_generated_candidates=8, max_evaluated_candidates=8, population_size=3)
        ctrl = SearchController(registry=reg, budget=budget, seed=11)
        gen = ctrl.generator
        real_seed = gen.seed_template_mean_reversion
        real_generate = gen.generate
        seed_calls = {"n": 0}
        generate_calls = {"n": 0}

        def counting_seed(*, seed: int = 0):
            seed_calls["n"] += 1
            return real_seed(seed=seed)

        def counting_generate(*, seed: int = 0):
            generate_calls["n"] += 1
            return real_generate(seed=seed)

        with (
            patch.object(gen, "seed_template_mean_reversion", side_effect=counting_seed),
            patch.object(gen, "generate", side_effect=counting_generate),
        ):
            ctrl.run()

        assert seed_calls["n"] == 1
        assert generate_calls["n"] >= 1

    def test_failed_seed_does_not_loop(self, tmp_path: Path) -> None:
        """When seed never enters population, controller must not re-call seed template."""
        reg = ExperimentRegistry(tmp_path / "registry")
        budget = _budget(
            max_generated_candidates=6,
            max_evaluated_candidates=6,
            population_size=3,
            max_runtime_seconds=15.0,
        )
        ctrl = SearchController(registry=reg, budget=budget, seed=3)
        seed_ids: list[str] = []
        real_seed = ctrl.generator.seed_template_mean_reversion
        real_generate = ctrl.generator.generate

        def tracking_seed(*, seed: int = 0):
            cand = real_seed(seed=seed)
            seed_ids.append(cand.candidate_id)
            return cand

        def tracking_generate(*, seed: int = 0):
            return real_generate(seed=seed)

        # Force seed evaluation to never enter population (reject all fitness).
        original_evaluate = ctrl.evaluator.evaluate

        def reject_seed_keep_others(cand):
            rec = original_evaluate(cand)
            if cand.creation_method.value == "SEED_TEMPLATE":
                return EvaluationRecord(
                    outcome=EvalOutcome.REJECTED,
                    candidate_id=cand.candidate_id,
                    lineage_id=cand.lineage_id,
                    trial_id=rec.trial_id,
                    fitness=None,
                    rejection_reason="forced_seed_reject",
                    runtime_seconds=0.0,
                )
            return rec

        with (
            patch.object(ctrl.generator, "seed_template_mean_reversion", side_effect=tracking_seed),
            patch.object(ctrl.generator, "generate", side_effect=tracking_generate),
            patch.object(ctrl.evaluator, "evaluate", side_effect=reject_seed_keep_others),
        ):
            result = ctrl.run()

        assert len(seed_ids) == 1
        assert result.stop_reason != "empty_population" or ctrl.counters.unique_generated >= 1
        # After seed, random generation must produce different IDs
        unique_ids = {t.candidate_id for t in reg.all_trials()}
        assert len(unique_ids) >= 1


class TestUniqueBudgetAccounting:
    def test_duplicate_seed_does_not_consume_unique_budget(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=10,
            max_candidates_per_family=10,
            max_candidates_per_complexity_tier=10,
            max_candidates_per_feature_family=10,
        )
        counters = BudgetCounters()
        gen = CandidateGenerator()
        # Fixed AST → same candidate_id regardless of seed argument
        a = gen.seed_template_mean_reversion(seed=1)
        b = gen.seed_template_mean_reversion(seed=999)
        assert a.candidate_id == b.candidate_id

        assert (
            counters.record_generated(
                budget,
                family=a.strategy_family,
                complexity=a.complexity_score,
                feature_ids=a.feature_ids,
                candidate_id=a.candidate_id,
            )
            is None
        )
        assert counters.generated == 1
        assert counters.unique_generated == 1
        assert counters.generated_attempts == 1

        reason = counters.record_generated(
            budget,
            family=b.strategy_family,
            complexity=b.complexity_score,
            feature_ids=b.feature_ids,
            candidate_id=b.candidate_id,
        )
        assert reason == DUPLICATE_CANDIDATE_ID
        assert counters.generated == 1  # unique budget unchanged
        assert counters.unique_generated == 1
        assert counters.duplicate_attempts == 1
        assert counters.generated_attempts == 2
        assert counters.stop_reason(budget) is None

    def test_random_replacements_have_different_candidate_ids(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "registry")
        budget = _budget(max_generated_candidates=12, max_evaluated_candidates=12, population_size=4)
        ctrl = SearchController(registry=reg, budget=budget, seed=42)
        result = ctrl.run()
        ids = [t.candidate_id for t in reg.all_trials() if t.rejection_reason != "duplicate_not_reevaluated"]
        # Unique non-duplicate registry rows should include multiple IDs after seed+random
        unique = set(ids)
        assert len(unique) >= 2 or ctrl.counters.unique_generated >= 2
        assert ctrl.counters.unique_generated == ctrl.counters.generated
        assert ctrl.counters.generated_attempts >= ctrl.counters.unique_generated
        assert result.stop_reason is not None


class TestDiversityExhaustion:
    def test_generation_diversity_exhausted_terminal(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=5,
            max_evaluated_candidates=50,
            max_runtime_seconds=30.0,
            population_size=2,
            max_candidates_per_family=50,
            max_candidates_per_complexity_tier=50,
            max_candidates_per_feature_family=50,
        )
        counters = BudgetCounters()
        # One unique accept, then flood duplicates past attempt safety cap
        counters.record_generated(
            budget,
            family="f",
            complexity=3.0,
            feature_ids=("ohlcv.close",),
            candidate_id="cand_only",
        )
        for _ in range(counters.max_generation_attempts(budget)):
            counters.record_generated(
                budget,
                family="f",
                complexity=3.0,
                feature_ids=("ohlcv.close",),
                candidate_id="cand_only",
            )
        assert counters.diversity_exhausted(budget) is True
        assert counters.stop_reason(budget) == GENERATION_DIVERSITY_EXHAUSTED
        assert counters.generated == 1
        assert counters.duplicate_attempts >= 1


@dataclass
class _FakeFullWfoBackend(SyntheticOOSBackend):
    """Test double that HONESTLY proves full-WFO completion.

    Merely flipping ``is_full_event_wfo`` on a synthetic backend is no longer
    sufficient to advance the Full WFO budget (Phase 1 fix) — the evaluator now
    requires the backend's own returned artifacts to prove completion
    (``signal_source == candidate_dsl_trees``, ``is_full_event_wfo`` in the
    train-metrics payload, and ``wfo_completed_folds > 0``). This double
    supplies those artifacts so budget/plumbing tests stay fast without a real
    event-driven backend.
    """

    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"

    def evaluate(self, candidate):
        folds, train = super().evaluate(candidate)
        train = {
            **train,
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
        }
        return folds, train


class TestFortyCandidateWfoReach:
    def test_forty_candidate_run_reaches_real_wfo(self, tmp_path: Path) -> None:
        """A 40-candidate synthetic run must evaluate via backend (WFO path active)."""
        reg = ExperimentRegistry(tmp_path / "registry")
        budget = _budget(
            max_generated_candidates=40,
            max_evaluated_candidates=30,
            max_full_wfo_evaluations=8,
            population_size=4,
            max_runtime_seconds=45.0,
        )
        ctrl = SearchController(registry=reg, budget=budget, seed=21)
        # A backend that truthfully reports full-WFO completion artifacts so
        # the Full WFO counter advances (see _FakeFullWfoBackend docstring).
        backend = _FakeFullWfoBackend(seed_salt=7)
        ctrl.evaluator.backend = backend

        eval_ids: list[str] = []
        original = ctrl.evaluator.evaluate

        def tracking_evaluate(cand):
            eval_ids.append(cand.candidate_id)
            return original(cand)

        with patch.object(ctrl.evaluator, "evaluate", side_effect=tracking_evaluate):
            result = ctrl.run()

        assert ctrl.counters.unique_generated >= 4
        assert ctrl.counters.generated == ctrl.counters.unique_generated
        # Unique IDs across generation — not a single seed looping
        assert len(set(eval_ids)) >= 4
        assert ctrl.counters.evaluated >= 1
        assert ctrl.counters.full_wfo >= 1
        assert result.stop_reason in {
            "max_generated_candidates",
            "max_evaluated_candidates",
            "max_full_wfo_evaluations",
            "stagnation_limit",
            "empty_population",
            "completed",
            GENERATION_DIVERSITY_EXHAUSTED,
            "max_runtime_seconds",
        }


class TestRichestTrialReporting:
    def test_duplicate_does_not_overwrite_evaluated_record(self) -> None:
        from control_plane.alpha_results import _latest_by_candidate
        from registry.experiment_registry import TrialRecord, TrialStatus

        rich = TrialRecord(
            trial_id="t1",
            candidate_id="cand_same",
            lineage_id="lin",
            strategy_family="mean_reversion_template",
            parameters={},
            config_snapshot={"is_full_event_wfo": True},
            system_version="t",
            git_commit_hash="g",
            data_hash="d",
            random_seed=1,
            train_window=None,
            validation_window=None,
            vault_version=None,
            gross_metrics={},
            net_metrics={"fitness": 1.0},
            ranking_score=1.25,
            rejection_reason=None,
            trade_log_path=None,
            equity_curve_path=None,
            execution_assumptions={},
            cost_model_version="c",
            code_hash="h",
            trial_status=TrialStatus.COMPLETED.value,
            failure_reason=None,
            fold_records=[{"fold_id": 0}, {"fold_id": 1}],
            ranking_source="validation_oos",
        )
        dup = TrialRecord(
            trial_id="t2",
            candidate_id="cand_same",
            lineage_id="lin",
            strategy_family="mean_reversion_template",
            parameters={},
            config_snapshot={},
            system_version="t",
            git_commit_hash="g",
            data_hash="d",
            random_seed=1,
            train_window=None,
            validation_window=None,
            vault_version=None,
            gross_metrics={},
            net_metrics={"duplicate": True},
            ranking_score=None,
            rejection_reason="duplicate_not_reevaluated",
            trade_log_path=None,
            equity_curve_path=None,
            execution_assumptions={},
            cost_model_version="c",
            code_hash="h",
            trial_status=TrialStatus.COMPLETED.value,
            failure_reason=None,
            fold_records=[],
            ranking_source="validation_oos",
        )
        kept = _latest_by_candidate([rich, dup])
        assert kept["cand_same"].trial_id == "t1"
        assert kept["cand_same"].ranking_score == 1.25
        assert len(kept["cand_same"].fold_records) == 2
