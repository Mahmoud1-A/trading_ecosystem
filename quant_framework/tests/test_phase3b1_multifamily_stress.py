"""Phase 3B.1: MultiFamily Score Qualified → real Stress orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from discovery.evaluator import EvalOutcome, EvaluationRecord, SyntheticOOSBackend
from discovery.fitness import FitnessResult, FoldOOSMetrics
from discovery.multi_family_campaign import (
    PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_ROBUSTNESS_SCREENING,
    REAL_STRESS_BACKEND_REQUIRED,
    SCORE_QUALIFIED,
    STATISTICS_NOT_RUN,
    STRESS_BUDGET_EXHAUSTED,
    STRESS_FAILED,
    STRESS_NOT_ENTERED,
    STRESS_PASSED,
    STRESS_TESTED,
    SYNTHETIC_STRESS_FORBIDDEN,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
)
from discovery.stress import StressResult
from discovery.stress_backend import (
    NOT_APPLICABLE_SINGLE_SYMBOL,
    STRESS_BACKEND_KIND,
    STRESS_BASELINE_TRADES_UNAVAILABLE,
    UNSUPPORTED_STRESS_SCENARIO,
)
from registry.experiment_registry import ExperimentRegistry


def _good_folds(n: int = 3) -> list[FoldOOSMetrics]:
    return [
        FoldOOSMetrics(
            fold_id=i,
            expectancy=0.08,
            sharpe=1.4,
            profit_factor=1.5,
            calmar=0.6,
            max_drawdown=-0.04,
            n_trades=12,
        )
        for i in range(n)
    ]


def _bad_folds(n: int = 3) -> list[FoldOOSMetrics]:
    return [
        FoldOOSMetrics(
            fold_id=i,
            expectancy=-0.05,
            sharpe=-0.4,
            profit_factor=0.7,
            calmar=-0.2,
            max_drawdown=-0.25,
            n_trades=8,
        )
        for i in range(n)
    ]


@dataclass
class HonestFullWfoBackend(SyntheticOOSBackend):
    """Proves Full WFO completion via returned artifacts (not flags alone)."""

    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"
    force_reject: bool = False

    def evaluate(self, candidate):
        folds = _bad_folds() if self.force_reject else _good_folds()
        # Keep SyntheticOOSBackend identity hashing path available for variety.
        if not self.force_reject:
            folds, _ = super().evaluate(candidate)
            # Ensure economic gates clear for Score Qualified path.
            folds = [
                FoldOOSMetrics(
                    fold_id=f.fold_id,
                    expectancy=max(0.05, abs(float(f.expectancy)) + 0.05),
                    sharpe=max(0.8, float(f.sharpe)),
                    profit_factor=max(1.2, float(f.profit_factor)),
                    calmar=max(0.3, float(f.calmar)),
                    max_drawdown=-min(0.08, abs(float(f.max_drawdown))),
                    n_trades=max(8, int(f.n_trades)),
                )
                for f in folds
            ]
        train = {
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
        }
        return folds, train


@dataclass
class _RealStressScenarioBackend:
    scenario: str
    folds: list[FoldOOSMetrics]
    pass_economic: bool = True
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)
    backend_kind: str = STRESS_BACKEND_KIND
    cost_model_changes: dict[str, Any] = field(default_factory=dict)
    execution_changes: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, candidate):
        folds = self.folds if self.pass_economic else _bad_folds()
        self.last_run_artifacts = {
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
            "trades_count": sum(f.n_trades for f in folds),
            "orders_count": sum(f.n_trades for f in folds),
            "fills_count": sum(f.n_trades for f in folds),
            "scenario": self.scenario,
            "candidate_id": candidate.candidate_id,
        }
        return folds, dict(self.last_run_artifacts)


@dataclass
class _StatusBackend:
    stress_status: str
    backend_kind: str = "stress_status_no_rerun"

    def evaluate(self, candidate):
        raise AssertionError(f"must not evaluate when stress_status={self.stress_status}")


def _make_real_stress_factory(
    *,
    fail_scenarios: frozenset[str] | None = None,
    status_map: dict[str, str] | None = None,
    call_log: list[str] | None = None,
) -> Callable[[str], Any]:
    fail_scenarios = fail_scenarios or frozenset()
    status_map = status_map or {}
    call_log = call_log if call_log is not None else []

    def _factory(scenario: str):
        call_log.append(scenario)
        if scenario in status_map:
            return _StatusBackend(stress_status=status_map[scenario])
        return _RealStressScenarioBackend(
            scenario=scenario,
            folds=_good_folds(),
            pass_economic=scenario not in fail_scenarios,
        )

    _factory.backend_kind = STRESS_BACKEND_KIND  # type: ignore[attr-defined]
    _factory.research_eligible = True  # type: ignore[attr-defined]
    _factory.synthetic_stress_forbidden = True  # type: ignore[attr-defined]
    return _factory


def _evo_cfg(**overrides) -> FamilyCampaignConfig:
    base = dict(
        requested_family_count=1,
        min_candidates_per_family=3,
        total_candidate_budget=10,
        max_full_wfo=8,
        adaptive_reallocation=False,
        seed=21,
        min_oos_trades=1,
        min_oos_trades_per_fold=1,
        max_runtime_seconds=60,
        family_local_evolution=True,
        evolution_generations=2,
        population_size=2,
        stagnation_generations=99,
        allow_cross_family_crossover=False,
        minimum_improvement=1e-4,
        max_stress_evaluations=20,
        max_stress_scenarios_per_candidate=6,
        min_stress_pass_rate=0.5,
        stress_scenarios=(
            "base_costs",
            "costs_2x",
            "wider_spread",
            "symbol_exclusion",
        ),
        fail_closed_unsupported_stress=True,
    )
    base.update(overrides)
    return FamilyCampaignConfig(**base)


def _run(
    tmp_path: Path,
    *,
    stress_factory=None,
    research_eligible: bool = True,
    backend=None,
    run_id: str = "test_phase3b1",
    **cfg_overrides,
):
    registry = ExperimentRegistry(tmp_path / f"reg_{run_id}")
    campaign = MultiFamilyCampaign(
        config=_evo_cfg(**cfg_overrides),
        registry=registry,
        backend=backend or HonestFullWfoBackend(seed_salt=7),
        discovery_run_id=run_id,
        research_eligible=research_eligible,
        synthetic_stress_forbidden=research_eligible,
        stress_backend_factory=stress_factory,
    )
    return campaign.run(), campaign


class TestScoreQualifiedStressEntry:
    def test_only_score_qualified_enter_stress(self, tmp_path: Path) -> None:
        call_log: list[str] = []
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(call_log=call_log),
            run_id="sq_only",
        )
        entered_ids = {
            s.candidate_id
            for s in result.candidate_stress_summaries
            if s.final_decision in {STRESS_PASSED, STRESS_FAILED}
        }
        sq_from_gen = {
            cid
            for g in result.generation_records
            for cid in g.score_qualified_candidate_ids
        }
        assert entered_ids
        assert entered_ids <= sq_from_gen
        # Every entered candidate has SCORE_QUALIFIED then STRESS_TESTED in history.
        for cid in entered_ids:
            events = [e for e in result.candidate_status_history if e.candidate_id == cid]
            statuses = [e.new_status for e in events]
            assert SCORE_QUALIFIED in statuses
            assert STRESS_TESTED in statuses
            assert statuses.index(SCORE_QUALIFIED) < statuses.index(STRESS_TESTED)

    def test_structural_parent_failed_score_never_enters_stress(
        self, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        result, campaign = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(call_log=call_log),
            backend=HonestFullWfoBackend(seed_salt=9, force_reject=True),
            run_id="struct_no_stress",
            max_full_wfo=4,
            total_candidate_budget=6,
            evolution_generations=1,
        )
        # Force-reject backend yields no Score Qualified.
        assert sum(s.score_qualified for s in result.family_stats) == 0
        assert result.candidate_stress_summaries == []
        not_entered = [
            e for e in result.candidate_status_history if e.new_status == STRESS_NOT_ENTERED
        ]
        assert not_entered
        assert call_log == []
        assert all(
            STRESS_PASSED not in [e.new_status for e in result.candidate_status_history if e.candidate_id == ev.candidate_id]
            for ev in not_entered
        )


class TestRealStressBackend:
    def test_real_stress_backend_used(self, tmp_path: Path) -> None:
        call_log: list[str] = []
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(call_log=call_log),
            run_id="real_backend",
        )
        summaries = [
            s
            for s in result.candidate_stress_summaries
            if s.final_decision in {STRESS_PASSED, STRESS_FAILED}
        ]
        assert summaries
        assert call_log
        for s in summaries:
            assert s.backend_kind == STRESS_BACKEND_KIND
            assert s.research_eligible is True

    def test_synthetic_stress_rejected(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=None,
            backend=HonestFullWfoBackend(seed_salt=3),
            research_eligible=True,
            run_id="synth_forbidden",
            max_full_wfo=6,
        )
        # Score Qualified may exist, but synthetic binding must fail closed.
        failed = [
            s
            for s in result.candidate_stress_summaries
            if s.final_decision == STRESS_FAILED
        ]
        assert failed
        assert all(
            s.final_reason in {SYNTHETIC_STRESS_FORBIDDEN, REAL_STRESS_BACKEND_REQUIRED}
            for s in failed
        )
        assert all(s.stress_budget_consumed == 0 for s in failed)


class TestScenarioSemantics:
    def test_signal_source_from_scenario_artifacts(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            run_id="signal_src",
            stress_scenarios=("base_costs", "costs_2x"),
            max_stress_scenarios_per_candidate=2,
        )
        executed = [
            s for s in result.candidate_stress_summaries if s.total_scenarios_executed > 0
        ]
        assert executed
        for s in executed:
            assert s.signal_source_proof
            assert all(v == "candidate_dsl_trees" for v in s.signal_source_proof.values())
            assert all(n > 0 for n in s.completed_fold_proof.values())

    def test_non_applicable_excluded_from_denominator(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            run_id="na_denom",
            stress_scenarios=("base_costs", "symbol_exclusion"),
            max_stress_scenarios_per_candidate=2,
            min_stress_pass_rate=1.0,
        )
        executed = [
            s for s in result.candidate_stress_summaries if s.total_scenarios_executed > 0
        ]
        assert executed
        for s in executed:
            assert "symbol_exclusion" in s.scenario_statuses
            assert s.scenario_statuses["symbol_exclusion"] == "not_applicable"
            assert s.not_applicable_count >= 1
            assert s.denominator == s.total_scenarios_executed
            assert "symbol_exclusion" not in [
                name
                for name, st in s.scenario_statuses.items()
                if st == "executed"
            ]
            # Pass rate uses executed only — N/A must not dilute.
            assert s.denominator == 1
            assert s.pass_rate == pytest.approx(1.0)

    def test_unsupported_scenarios_do_not_pass(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(
                status_map={"made_up_xyz": UNSUPPORTED_STRESS_SCENARIO}
            ),
            run_id="unsupported",
            stress_scenarios=("base_costs", "made_up_xyz"),
            max_stress_scenarios_per_candidate=2,
            fail_closed_unsupported_stress=True,
        )
        # StressTester treats unknown names as unsupported before factory;
        # also cover status_map path via a known scenario renamed in factory.
        # Use costs_2x mapped to unsupported status:
        result2, _ = _run(
            tmp_path / "b",
            stress_factory=_make_real_stress_factory(
                status_map={"costs_2x": UNSUPPORTED_STRESS_SCENARIO}
            ),
            run_id="unsupported2",
            stress_scenarios=("base_costs", "costs_2x"),
            max_stress_scenarios_per_candidate=2,
        )
        failed = [
            s for s in result2.candidate_stress_summaries if s.final_decision == STRESS_FAILED
        ]
        assert failed
        assert any(s.unsupported_count >= 1 for s in failed)
        assert all(s.final_decision != STRESS_PASSED or s.unsupported_count == 0 for s in result2.candidate_stress_summaries)

    def test_missing_baseline_trades_no_fake_fallback(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(
                status_map={"removed_best_day": STRESS_BASELINE_TRADES_UNAVAILABLE}
            ),
            run_id="baseline_missing",
            stress_scenarios=("base_costs", "removed_best_day"),
            max_stress_scenarios_per_candidate=2,
        )
        failed = [
            s for s in result.candidate_stress_summaries if s.final_decision == STRESS_FAILED
        ]
        assert failed
        for s in failed:
            assert s.stress_budget_consumed == 1  # only base_costs executed
            assert s.scenario_statuses.get("removed_best_day") == "baseline_unavailable"


class TestStatusHistoryAndGates:
    def test_stress_passed_requires_score_qualified_then_tested(
        self, tmp_path: Path
    ) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            run_id="status_order",
            min_stress_pass_rate=0.5,
        )
        passed = [
            s for s in result.candidate_stress_summaries if s.final_decision == STRESS_PASSED
        ]
        assert passed
        for s in passed:
            events = [
                e
                for e in result.candidate_status_history
                if e.candidate_id == s.candidate_id
            ]
            statuses = [e.new_status for e in events]
            assert statuses.count(SCORE_QUALIFIED) >= 1
            assert statuses.count(STRESS_TESTED) >= 1
            assert statuses.count(STRESS_PASSED) >= 1
            assert statuses.index(SCORE_QUALIFIED) < statuses.index(STRESS_TESTED)
            assert statuses.index(STRESS_TESTED) < statuses.index(STRESS_PASSED)
            # Sequences are deterministic and increasing.
            seqs = [e.sequence for e in events]
            assert seqs == sorted(seqs)

    def test_stress_failed_does_not_enter_later_stages(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(
                fail_scenarios=frozenset({"base_costs", "costs_2x", "wider_spread"})
            ),
            run_id="failed_no_later",
            stress_scenarios=("base_costs", "costs_2x", "wider_spread"),
            max_stress_scenarios_per_candidate=3,
            min_stress_pass_rate=1.0,
        )
        assert result.as_dict()["finalists"] == []
        assert result.as_dict()["promoted"] == []
        assert result.research_shortlist == []
        assert result.vault_candidates == []
        assert result.paper_candidates == []
        assert result.robustness_pipeline_complete is True
        # Statistics/clustering may run but enter zero robustness-passed candidates.
        assert result.statistics_pipeline_complete is True
        assert result.clustering_pipeline_complete is True
        assert all(
            s.final_decision != "STATISTICALLY_PASSED"
            for s in result.candidate_statistics_summaries
        )

    def test_stress_passed_does_not_populate_downstream(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            run_id="passed_not_finalist",
        )
        assert any(s.final_decision == STRESS_PASSED for s in result.candidate_stress_summaries)
        payload = result.as_dict()
        assert payload["finalists"] == []
        assert payload["promoted"] == []
        assert payload["vault_candidates"] == []
        assert payload["paper_candidates"] == []
        assert result.pipeline_level == (
            "MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST"
        )
        assert result.stress_pipeline_complete is True
        assert result.robustness_pipeline_complete is True
        assert result.statistics_pipeline_complete is True
        assert result.post_wfo_pipeline_complete is False
        assert "VAULT_NOT_RUN" in result.post_wfo_blocked_reasons
        assert "STATISTICS_NOT_RUN" not in result.post_wfo_blocked_reasons
        assert "ROBUSTNESS_NOT_RUN" not in result.post_wfo_blocked_reasons
        for claim in (
            "finalist",
            "promoted",
            "research shortlisted",
            "vault eligible",
            "paper eligible",
        ):
            assert claim in payload["stress_passed_does_not_mean"]
        assert result.pipeline_level == "MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST"


class TestStressBudget:
    def test_stress_budgets_cannot_be_exceeded(self, tmp_path: Path) -> None:
        call_log: list[str] = []
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(call_log=call_log),
            run_id="budget_cap",
            max_stress_evaluations=2,
            stress_scenarios=("base_costs", "costs_2x", "wider_spread"),
            max_stress_scenarios_per_candidate=3,
            max_full_wfo=8,
            total_candidate_budget=12,
            evolution_generations=2,
        )
        acct = result.stress_accounting
        assert acct is not None
        assert acct.stress_evaluations_consumed <= 2
        assert acct.stress_evaluations_consumed == len(call_log)
        # Non-applicable / not-entered must not invent extra consumption.
        for s in result.candidate_stress_summaries:
            assert s.stress_budget_consumed >= 0
        # If more SQ than budget allows, some remain NOT_ENTERED for budget.
        if acct.candidates_score_qualified > 1:
            assert acct.stop_reason == STRESS_BUDGET_EXHAUSTED or any(
                e.new_status == STRESS_NOT_ENTERED and e.reason == STRESS_BUDGET_EXHAUSTED
                for e in result.candidate_status_history
            )

    def test_not_applicable_and_bind_failure_consume_zero_extra(
        self, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(call_log=call_log),
            run_id="zero_budget_na",
            stress_scenarios=("symbol_exclusion",),
            max_stress_scenarios_per_candidate=1,
            max_stress_evaluations=10,
        )
        # symbol_exclusion alone is not_applicable — zero stress budget.
        assert call_log == []
        for s in result.candidate_stress_summaries:
            if s.total_scenarios_configured == 1 and "symbol_exclusion" in s.scenario_names:
                assert s.stress_budget_consumed == 0
                assert s.denominator == 0
                assert s.final_decision == STRESS_FAILED
