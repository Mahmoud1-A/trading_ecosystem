"""
Phase 6D acceptance tests — Autonomous Alpha Miner.

Proves: registry visibility, duplicate skip, OOS-only ranking, budget freeze,
reproducible rankings, DSR/PBO population, behavioral clustering, stress/robustness
persistence, Vault isolation, freeze-before-Vault, one-shot Vault access.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from discovery import (
    RANKING_SOURCE_OOS,
    BehavioralDeduper,
    CandidateEvaluator,
    CandidateGenerator,
    EvalOutcome,
    FitnessError,
    FoldOOSMetrics,
    MinerVaultError,
    MinerVaultGateway,
    PromotionGate,
    PromotionStatus,
    RobustFitness,
    SearchBudget,
    SearchController,
    VaultSubmissionRequest,
    assert_oos_only_promotion,
    freeze_candidate,
)
from metrics.dsr import compute_deflated_sharpe
from metrics.pbo import compute_pbo
from registry.experiment_registry import ExperimentRegistry
from validation.vault import ValidationVault, VaultAccessError


TZ = ZoneInfo("America/Chicago")


def _registry(tmp_path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "registry")


def _budget(**kwargs) -> SearchBudget:
    defaults = dict(
        max_generated_candidates=20,
        max_evaluated_candidates=16,
        max_runtime_seconds=30.0,
        max_full_wfo_evaluations=16,
        max_stress_evaluations=8,
        max_vault_submissions=2,
        stagnation_limit=3,
        population_size=4,
        elite_count=1,
        max_candidates_per_family=20,
        max_candidates_per_complexity_tier=20,
        max_candidates_per_feature_family=20,
    )
    defaults.update(kwargs)
    return SearchBudget(**defaults)


def _vault(tmp_path) -> ValidationVault:
    idx = pd.date_range("2024-06-01 08:30", periods=40, freq="5min", tz=TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )
    return ValidationVault(
        vault_version="vault_v1",
        data=df,
        data_hash="vaulthash",
        store_path=tmp_path / "vault",
    )


class TestSearchBudget:
    def test_budget_change_creates_new_run_id(self) -> None:
        a = SearchBudget(max_generated_candidates=10)
        b = a.with_overrides(max_generated_candidates=11)
        assert a.budget_id != b.budget_id


class TestRegistryAndDuplicates:
    def test_every_candidate_reaches_registry(self, tmp_path) -> None:
        reg = _registry(tmp_path)
        budget = _budget()
        from discovery.search_budget import BudgetCounters

        ev = CandidateEvaluator(
            registry=reg,
            budget=budget,
            counters=BudgetCounters(),
            discovery_run_id="run_test",
        )
        gen = CandidateGenerator()
        c1 = gen.seed_template_mean_reversion(seed=1)
        c2 = gen.generate(seed=3)
        r1 = ev.evaluate(c1)
        r2 = ev.evaluate(c2)
        assert r1.trial_id is not None
        assert r2.trial_id is not None
        assert len(reg.all_trials()) == 2

    def test_rejected_trials_remain_visible(self, tmp_path) -> None:
        reg = _registry(tmp_path)
        from discovery.search_budget import BudgetCounters

        ev = CandidateEvaluator(
            registry=reg,
            budget=_budget(),
            counters=BudgetCounters(),
        )
        gen = CandidateGenerator()
        c = gen.seed_template_mean_reversion(seed=1)
        ev.evaluate(c)
        # Force a rejected duplicate registration path
        again = gen.seed_template_mean_reversion(seed=2)
        assert again.candidate_id == c.candidate_id
        rec = ev.evaluate(again)
        assert rec.outcome is EvalOutcome.DUPLICATE_SKIPPED
        rejected = reg.rejected_trials()
        assert any(t.rejection_reason == "duplicate_not_reevaluated" for t in rejected)

    def test_duplicates_not_reevaluated(self, tmp_path) -> None:
        reg = _registry(tmp_path)
        from discovery.search_budget import BudgetCounters

        counters = BudgetCounters()
        ev = CandidateEvaluator(registry=reg, budget=_budget(), counters=counters)
        gen = CandidateGenerator()
        c = gen.seed_template_mean_reversion(seed=1)
        ev.evaluate(c)
        evaluated_after_first = counters.evaluated
        ev.evaluate(gen.seed_template_mean_reversion(seed=99))
        assert counters.evaluated == evaluated_after_first


class TestOOSRanking:
    def test_fitness_rejects_non_oos_ranking_source(self) -> None:
        folds = [
            FoldOOSMetrics(0, 0.1, 0.5, 1.2, 0.3, -0.05),
            FoldOOSMetrics(1, 0.05, 0.4, 1.1, 0.2, -0.08),
        ]
        with pytest.raises(FitnessError, match="validation_oos"):
            RobustFitness().score(folds, ranking_source="training_is")

    def test_training_metrics_cannot_promote(self) -> None:
        assert assert_oos_only_promotion(train_score=999.0, oos_fitness=-1.0, promote_threshold=0.0) is False
        assert assert_oos_only_promotion(train_score=999.0, oos_fitness=1.0, promote_threshold=0.0) is True

        gen = CandidateGenerator()
        cand = gen.seed_template_mean_reversion(seed=1)
        frozen = freeze_candidate(
            cand,
            oos_fitness=-5.0,
            ranking_source=RANKING_SOURCE_OOS,
            discovery_run_id="run_x",
            code_hash="c",
            data_hash="d",
        )
        decision = PromotionGate(min_oos_fitness=0.0).decide(frozen, train_score=1e9)
        assert decision.status is PromotionStatus.REJECTED
        assert decision.train_score_ignored == 1e9


class TestBudgetStopAndRepro:
    def test_search_stops_at_frozen_budget(self, tmp_path) -> None:
        budget = _budget(
            max_generated_candidates=6,
            max_evaluated_candidates=5,
            max_full_wfo_evaluations=5,
            population_size=3,
            stagnation_limit=10,
        )
        ctrl = SearchController(
            registry=_registry(tmp_path),
            budget=budget,
            seed=7,
            discovery_run_id="run_budget",
        )
        result = ctrl.run()
        assert result.stop_reason in {
            "max_generated_candidates",
            "max_evaluated_candidates",
            "max_full_wfo_evaluations",
            "stagnation_limit",
            "empty_population",
            "completed",
        }
        assert ctrl.counters.generated <= budget.max_generated_candidates
        assert ctrl.counters.evaluated <= budget.max_evaluated_candidates
        assert result.registered_trials >= 1

    def test_identical_runs_reproduce_rankings(self, tmp_path) -> None:
        budget = _budget(
            max_generated_candidates=8,
            max_evaluated_candidates=6,
            max_full_wfo_evaluations=6,
            population_size=3,
            stagnation_limit=2,
            max_stress_evaluations=4,
        )
        r1 = SearchController(
            registry=_registry(tmp_path / "a"),
            budget=budget,
            seed=123,
            discovery_run_id="run_a",
        ).run()
        r2 = SearchController(
            registry=_registry(tmp_path / "b"),
            budget=budget,
            seed=123,
            discovery_run_id="run_b",
        ).run()
        assert r1.reproducible_fingerprint == r2.reproducible_fingerprint
        assert [x["candidate_id"] for x in r1.rankings] == [x["candidate_id"] for x in r2.rankings]
        assert [x["fitness"] for x in r1.rankings] == [x["fitness"] for x in r2.rankings]


class TestDSRPBOAndBehavior:
    def test_dsr_pbo_use_full_population(self, tmp_path) -> None:
        reg = _registry(tmp_path)
        from discovery.search_budget import BudgetCounters

        ev = CandidateEvaluator(registry=reg, budget=_budget(), counters=BudgetCounters())
        gen = CandidateGenerator()
        for seed in range(5):
            ev.evaluate(gen.generate(seed=10 + seed))
        # Duplicate reject still in ledger
        ev.evaluate(gen.seed_template_mean_reversion(seed=1))
        ev.evaluate(gen.seed_template_mean_reversion(seed=2))
        pop = reg.trial_population()
        assert pop.total_trials == len(reg.all_trials())
        assert pop.total_trials >= 5
        # DSR/PBO consume full population (may be insufficient_data but must accept pop)
        dsr = compute_deflated_sharpe(0.5, pop, n_observations=50)
        assert dsr.total_trials == pop.total_trials
        n_trials = max(pop.total_trials, 2)
        matrix = np.random.default_rng(0).normal(0, 0.01, size=(64, n_trials))
        pbo = compute_pbo(matrix, population=pop, n_splits=4)
        assert pbo.total_trials == pop.total_trials

    def test_behavioral_duplicates_clustered(self, tmp_path) -> None:
        reg = _registry(tmp_path)
        from discovery.search_budget import BudgetCounters

        ev = CandidateEvaluator(registry=reg, budget=_budget(), counters=BudgetCounters())
        gen = CandidateGenerator()
        c1 = gen.seed_template_mean_reversion(seed=1)
        c2 = gen.seed_template_mean_reversion(seed=2)  # same identity
        # Two different candidates with correlated synthetic behavior
        a = gen.generate(seed=20)
        b = gen.generate(seed=21)
        ra = ev.evaluate(a)
        rb = ev.evaluate(b)
        # Force near-identical signatures
        from discovery.behavioral_dedup import BehaviorSignature

        sig_a = BehaviorSignature(
            candidate_id=a.candidate_id,
            signal_vector=(0.1, 0.2, 0.3, 0.4),
            daily_pnl=(1.0, 2.0, 1.5, 2.5),
            feature_ids=a.feature_ids,
            complexity=a.complexity_score,
            fitness=1.0,
        )
        sig_b = BehaviorSignature(
            candidate_id=b.candidate_id,
            signal_vector=(0.11, 0.21, 0.29, 0.41),
            daily_pnl=(1.05, 1.95, 1.55, 2.4),
            feature_ids=a.feature_ids,  # same features
            complexity=b.complexity_score + 5,
            fitness=0.8,
        )
        clusters = BehavioralDeduper(similarity_threshold=0.8).cluster([sig_a, sig_b])
        assert len(clusters) == 1
        assert clusters[0].representative_id == a.candidate_id  # higher fitness, lower complexity tie-break via fitness
        _ = (c1, c2, ra, rb)


class TestStressRobustnessPersistence:
    def test_stress_and_parameter_results_persist(self, tmp_path) -> None:
        budget = _budget(
            max_generated_candidates=5,
            max_evaluated_candidates=4,
            max_full_wfo_evaluations=4,
            population_size=2,
            stagnation_limit=1,
            max_stress_evaluations=8,
        )
        ctrl = SearchController(
            registry=_registry(tmp_path),
            budget=budget,
            seed=3,
            discovery_run_id="run_stress",
        )
        result = ctrl.run()
        # Find a finalist record with stress attached
        stressed = [
            r
            for r in ctrl.evaluator.records()
            if r.stress_results or r.robustness_results
        ]
        assert result.finalists
        assert stressed, "expected stress/robustness persistence on finalist records"
        assert any(r.stress_results for r in stressed)
        # Parameter robustness may be empty if candidate has no free params; template has params
        assert any(r.robustness_results for r in stressed)


class TestVaultIsolation:
    def test_miner_cannot_access_vault_data(self, tmp_path) -> None:
        gw = MinerVaultGateway(vault=_vault(tmp_path))
        with pytest.raises(MinerVaultError, match="must not access"):
            gw.peek_vault_forbidden()

        ctrl = SearchController(
            registry=_registry(tmp_path / "reg"),
            budget=_budget(max_generated_candidates=4, max_evaluated_candidates=3, population_size=2, stagnation_limit=1),
            seed=1,
            discovery_run_id="run_novault",
        )
        # Search must complete without vault
        result = ctrl.run()
        ctrl.assert_no_vault_access()
        assert result.discovery_run_id == "run_novault"

    def test_candidate_must_be_frozen_before_vault(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        gw = MinerVaultGateway(vault=vault)
        gen = CandidateGenerator()
        cand = gen.seed_template_mean_reversion(seed=1)
        from discovery.freezing import FreezeError

        # Unfrozen object must not be accepted
        with pytest.raises((TypeError, AttributeError, MinerVaultError, FreezeError)):
            gw.submit(
                VaultSubmissionRequest(frozen=cand),  # type: ignore[arg-type]
                evaluate_fn=lambda df: {"n": len(df)},
            )

        frozen = freeze_candidate(
            cand,
            oos_fitness=0.5,
            ranking_source=RANKING_SOURCE_OOS,
            discovery_run_id="run_f",
            code_hash="code",
            data_hash="data",
        )
        out = gw.submit(
            VaultSubmissionRequest(frozen=frozen),
            evaluate_fn=lambda df: {"n": len(df), "sharpe": 0.1},
        )
        assert out["vault_result"]["access_count"] == 1

    def test_vault_access_is_one_shot(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        gw = MinerVaultGateway(vault=vault)
        gen = CandidateGenerator()
        cand = gen.seed_template_mean_reversion(seed=1)
        frozen = freeze_candidate(
            cand,
            oos_fitness=0.5,
            ranking_source=RANKING_SOURCE_OOS,
            discovery_run_id="run_f",
            code_hash="code",
            data_hash="data",
        )
        gw.submit(VaultSubmissionRequest(frozen=frozen), evaluate_fn=lambda df: {"ok": True})
        with pytest.raises(MinerVaultError):
            gw.submit(VaultSubmissionRequest(frozen=frozen), evaluate_fn=lambda df: {"ok": True})

        # Underlying vault also enforces lineage one-shot
        with pytest.raises(VaultAccessError):
            vault.evaluate_frozen_candidate(
                frozen.to_lineage(),
                evaluate_fn=lambda df: {"ok": True},
            )


class TestRankingSourceOnTrials:
    def test_ranking_uses_oos_only(self, tmp_path) -> None:
        reg = _registry(tmp_path)
        from discovery.search_budget import BudgetCounters

        ev = CandidateEvaluator(registry=reg, budget=_budget(), counters=BudgetCounters())
        c = CandidateGenerator().seed_template_mean_reversion(seed=1)
        rec = ev.evaluate(c)
        assert rec.fitness is not None
        assert rec.fitness.ranking_source == RANKING_SOURCE_OOS
        trials = reg.all_trials()
        assert trials[-1].ranking_source == RANKING_SOURCE_OOS
        # Train diagnostic present but not used as ranking_score source
        assert "sharpe" in rec.train_metrics
        assert rec.train_metrics["sharpe"] != rec.fitness.fitness
