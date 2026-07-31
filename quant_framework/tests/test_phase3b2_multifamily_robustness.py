"""Phase 3B.2: MultiFamily STRESS_PASSED → real Parameter Robustness orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from discovery.candidate import build_candidate
from discovery.evaluator import SyntheticOOSBackend
from discovery.expression_tree import feature_node, op_node, parameter_node
from discovery.fitness import FoldOOSMetrics
from discovery.multi_family_campaign import (
    PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_ROBUSTNESS_SCREENING,
    REAL_ROBUSTNESS_BACKEND_REQUIRED,
    ROBUSTNESS_BUDGET_EXHAUSTED,
    ROBUSTNESS_FAILED,
    ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD,
    ROBUSTNESS_INTEGRITY_FAILED,
    ROBUSTNESS_NOT_ENTERED,
    ROBUSTNESS_PASSED,
    ROBUSTNESS_TESTED,
    SCORE_QUALIFIED,
    STATISTICS_NOT_RUN,
    STRESS_FAILED,
    STRESS_PASSED,
    STRESS_TESTED,
    SYNTHETIC_ROBUSTNESS_FORBIDDEN,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
)
from discovery.operators import OperatorId
from discovery.parameter_robustness import (
    PARAMETER_ROBUST,
    ROBUSTNESS_BACKEND_KIND_INVALID,
    ROBUSTNESS_PARAMETER_NOT_BOUND,
    ROBUSTNESS_POINT_NOT_APPLICABLE,
    ROBUSTNESS_SIGNAL_SOURCE_INVALID,
    ROBUSTNESS_WFO_INCOMPLETE,
    ParameterRobustness,
)
from discovery.stress_backend import STRESS_BACKEND_KIND
from discovery.types import CreationMethod, ValueType
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


@dataclass
class HonestEventDrivenBackend:
    """Controlled real event-driven backend (not SyntheticOOSBackend)."""

    seed_salt: int = 0
    backend_kind: str = "event_driven_wfo"
    is_full_event_wfo: bool = True
    call_log: list[tuple[str, str]] = field(default_factory=list)
    force_signal_source: str | None = None
    force_backend_kind: str | None = None
    force_completed_folds: int | None = None
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, candidate):
        self.call_log.append(("evaluate", candidate.candidate_id))
        folds = _good_folds()
        is_robustness = bool((candidate.family_provenance or {}).get("robustness_param"))
        completed = len(folds)
        if is_robustness and self.force_completed_folds is not None:
            completed = int(self.force_completed_folds)
        signal = "candidate_dsl_trees"
        if is_robustness and self.force_signal_source is not None:
            signal = self.force_signal_source
        kind = self.backend_kind
        if is_robustness and self.force_backend_kind is not None:
            kind = self.force_backend_kind
        train = {
            "signal_source": signal,
            "is_full_event_wfo": bool(self.is_full_event_wfo and completed > 0),
            "wfo_completed_folds": completed,
            "backend_kind": kind,
            "candidate_id": candidate.candidate_id,
            "orders_count": 10,
            "fills_count": 10,
            "trades_count": sum(f.n_trades for f in folds),
        }
        self.last_run_artifacts = dict(train)
        return folds, train


@dataclass
class HonestFullWfoBackend(SyntheticOOSBackend):
    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"

    def evaluate(self, candidate):
        folds, _ = super().evaluate(candidate)
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
            "backend_kind": self.backend_kind,
            "candidate_id": candidate.candidate_id,
        }
        return folds, train


@dataclass
class _RealStressScenarioBackend:
    scenario: str
    folds: list[FoldOOSMetrics]
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)
    backend_kind: str = STRESS_BACKEND_KIND

    def evaluate(self, candidate):
        folds = self.folds
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


def _make_real_stress_factory(
    *,
    fail_scenarios: frozenset[str] | None = None,
    call_log: list[str] | None = None,
) -> Callable[[str], Any]:
    fail_scenarios = fail_scenarios or frozenset()
    call_log = call_log if call_log is not None else []

    def _factory(scenario: str):
        call_log.append(scenario)
        folds = _good_folds()
        if scenario in fail_scenarios:
            folds = [
                FoldOOSMetrics(
                    fold_id=i,
                    expectancy=-0.05,
                    sharpe=-0.4,
                    profit_factor=0.7,
                    calmar=-0.2,
                    max_drawdown=-0.25,
                    n_trades=8,
                )
                for i in range(3)
            ]
        return _RealStressScenarioBackend(scenario=scenario, folds=folds)

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
        max_stress_scenarios_per_candidate=4,
        min_stress_pass_rate=0.5,
        stress_scenarios=("base_costs", "costs_2x", "wider_spread"),
        fail_closed_unsupported_stress=True,
        max_robustness_candidates=4,
        max_robustness_evaluations=40,
        max_parameters_per_candidate=2,
        max_points_per_parameter=5,
        allow_one_sided_neighborhood=False,
        min_valid_neighborhood_points=3,
    )
    base.update(overrides)
    return FamilyCampaignConfig(**base)


def _run(
    tmp_path: Path,
    *,
    stress_factory=None,
    research_eligible: bool = True,
    backend=None,
    run_id: str = "test_phase3b2",
    **cfg_overrides,
):
    registry = ExperimentRegistry(tmp_path / f"reg_{run_id}")
    campaign = MultiFamilyCampaign(
        config=_evo_cfg(**cfg_overrides),
        registry=registry,
        backend=backend or HonestEventDrivenBackend(seed_salt=7),
        discovery_run_id=run_id,
        research_eligible=research_eligible,
        synthetic_stress_forbidden=research_eligible,
        synthetic_robustness_forbidden=research_eligible,
        stress_backend_factory=stress_factory,
    )
    return campaign.run(), campaign


def _atr_candidate(*, stop_mult: float = 1.5, target_mult: float = 2.5):
    z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
    entry = op_node(
        OperatorId.ENTRY_LONG,
        op_node(OperatorId.LESS_THAN, z, parameter_node("z_entry", -2.0)),
    )
    exit_tree = op_node(
        OperatorId.EXIT_SIGNAL,
        op_node(OperatorId.GREATER_THAN, z, parameter_node("z_exit", -0.25)),
    )
    atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
    stop = op_node(OperatorId.ATR_STOP, atr, parameter_node("atr_stop_mult", stop_mult))
    target = op_node(
        OperatorId.ATR_TARGET, atr, parameter_node("atr_target_mult", target_mult)
    )
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


class TestStressPassedRobustnessEntry:
    def test_only_stress_passed_enter_robustness(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            run_id="rob_entry",
        )
        entered = {
            s.candidate_id
            for s in result.candidate_robustness_summaries
            if s.final_decision in {ROBUSTNESS_PASSED, ROBUSTNESS_FAILED}
        }
        stress_passed = {
            s.candidate_id
            for s in result.candidate_stress_summaries
            if s.final_decision == STRESS_PASSED
        }
        assert entered
        assert entered <= stress_passed
        for cid in entered:
            events = [e for e in result.candidate_status_history if e.candidate_id == cid]
            statuses = [e.new_status for e in events]
            assert STRESS_PASSED in statuses
            assert ROBUSTNESS_TESTED in statuses
            assert statuses.index(STRESS_PASSED) < statuses.index(ROBUSTNESS_TESTED)

    def test_stress_failed_consume_zero_robustness_budget(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(
                fail_scenarios=frozenset({"base_costs", "costs_2x", "wider_spread"})
            ),
            run_id="failed_zero_rob",
            min_stress_pass_rate=1.0,
            max_robustness_evaluations=40,
        )
        failed = [
            s
            for s in result.candidate_stress_summaries
            if s.final_decision == STRESS_FAILED
        ]
        assert failed
        rob_entered = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision in {ROBUSTNESS_PASSED, ROBUSTNESS_FAILED}
        ]
        assert rob_entered == []
        acct = result.robustness_accounting
        assert acct is not None
        assert acct.robustness_backend_calls == 0
        assert acct.robustness_counter_delta == 0
        assert acct.candidates_robustness_entered == 0


class TestRealRobustnessBackend:
    def test_real_event_driven_backend_used(self, tmp_path: Path) -> None:
        backend = HonestEventDrivenBackend(seed_salt=11)
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=backend,
            run_id="real_rob_backend",
        )
        summaries = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision in {ROBUSTNESS_PASSED, ROBUSTNESS_FAILED}
        ]
        assert summaries
        assert any(cid for _, cid in backend.call_log)
        for s in summaries:
            assert s.backend_kind == "event_driven_wfo"
            assert s.research_eligible is True

    def test_synthetic_robustness_rejected(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=HonestFullWfoBackend(seed_salt=3),
            research_eligible=True,
            run_id="synth_rob_forbidden",
        )
        passed_stress = [
            s for s in result.candidate_stress_summaries if s.final_decision == STRESS_PASSED
        ]
        assert passed_stress
        failed = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_FAILED
        ]
        assert failed
        assert all(
            s.final_reason
            in {SYNTHETIC_ROBUSTNESS_FORBIDDEN, REAL_ROBUSTNESS_BACKEND_REQUIRED}
            for s in failed
        )
        assert all(s.budget_consumed == 0 for s in failed)


class TestRobustnessBudgetAccounting:
    def test_each_backend_call_consumes_one_unit(self, tmp_path: Path) -> None:
        backend = HonestEventDrivenBackend(seed_salt=5)
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=backend,
            run_id="rob_budget_unit",
            max_parameters_per_candidate=1,
            max_points_per_parameter=3,
            max_robustness_candidates=1,
        )
        entered = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision in {ROBUSTNESS_PASSED, ROBUSTNESS_FAILED}
            and s.robustness_backend_calls > 0
        ]
        assert entered
        for s in entered:
            assert s.robustness_backend_calls == s.robustness_counter_delta
        acct = result.robustness_accounting
        assert acct is not None
        assert acct.robustness_backend_calls == acct.robustness_counter_delta
        assert acct.robustness_backend_calls == acct.robustness_evaluations_consumed


class TestDomainAndBinding:
    def test_unbound_parameter_consumes_zero_budget(self) -> None:
        cand = _atr_candidate()
        object.__setattr__(cand, "parameters", {**cand.parameters, "ghost_param": 0.5})
        backend = HonestEventDrivenBackend()
        counters = type("C", (), {"robustness": 0})()
        rob = ParameterRobustness(
            backend=backend,
            research_eligible=True,
            synthetic_robustness_forbidden=True,
            relative_steps=(-0.1, 0.0, 0.1),
        )
        results = rob.probe(
            cand,
            parameter_names=("ghost_param",),
            counters=counters,
            max_evaluations=10,
            require_integrity=True,
        )
        assert results[0].reason == ROBUSTNESS_PARAMETER_NOT_BOUND
        assert results[0].backend_call_count == 0
        assert counters.robustness == 0
        assert backend.call_log == []

    def test_out_of_domain_consumes_zero_and_not_zero_fitness(self) -> None:
        # Rank-like threshold: pushing far above 1.0 should be N/A, not fitness 0.
        feat = feature_node("liq.volume_pct_20", ValueType.RATIO)
        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(OperatorId.GREATER_THAN, feat, parameter_node("vol_pct_thr", 0.9)),
        )
        cand = build_candidate(
            entry_tree=entry,
            strategy_family="test_domain",
            creation_method=CreationMethod.RANDOM,
            grammar_version="v",
            feature_set_version="v",
            cost_model_version="v",
            random_seed=9,
        )
        backend = HonestEventDrivenBackend()
        counters = type("C", (), {"robustness": 0})()
        rob = ParameterRobustness(
            backend=backend,
            research_eligible=True,
            synthetic_robustness_forbidden=True,
            relative_steps=(0.0, 0.5),  # 0.9 * 1.5 = 1.35 likely out of [0,1]
            min_valid_neighborhood_points=1,
            allow_one_sided_neighborhood=True,
        )
        results = rob.probe(
            cand,
            parameter_names=("vol_pct_thr",),
            counters=counters,
            max_evaluations=10,
            require_integrity=True,
        )
        r = results[0]
        na = [p for p in r.points if p["point_status"] == ROBUSTNESS_POINT_NOT_APPLICABLE]
        assert na
        for p in na:
            assert p["fitness"] is None
            assert 0.0 not in r.fitness_curve or p["relative_step"] != 0.0
        # N/A points must not be in the fitness curve as artificial zeros.
        assert all(
            p["relative_step"] not in r.valid_evaluated_steps
            for p in na
        )

    def test_direction_incoherent_rejected_before_wfo(self) -> None:
        # Fade feature with wrong signed threshold for ENTRY_LONG.
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(OperatorId.GREATER_THAN, z, parameter_node("z_thr", 1.0)),
        )
        cand = build_candidate(
            entry_tree=entry,
            strategy_family="mean_reversion",
            creation_method=CreationMethod.RANDOM,
            grammar_version="v",
            feature_set_version="v",
            cost_model_version="v",
            random_seed=3,
            family_provenance={"family_id": "mean_reversion"},
        )
        from discovery.family_generator import StrategyFamilyGenerator

        gen = StrategyFamilyGenerator(seed=1)
        families = gen.generate(count=1, family_ids=["mean_reversion"])
        spec = families[0]
        backend = HonestEventDrivenBackend()
        counters = type("C", (), {"robustness": 0})()
        rob = ParameterRobustness(
            backend=backend,
            research_eligible=True,
            synthetic_robustness_forbidden=True,
            relative_steps=(0.0,),
            grammar=spec.to_grammar(),
            min_valid_neighborhood_points=1,
        )
        before = len(backend.call_log)
        rob.probe(
            cand,
            parameter_names=("z_thr",),
            family_spec=spec,
            counters=counters,
            max_evaluations=10,
            require_integrity=True,
        )
        # Center may already be incoherent → N/A, zero backend calls.
        assert counters.robustness == 0
        assert len(backend.call_log) == before

    def test_stop_target_paths_and_perturbed_ids(self) -> None:
        cand = _atr_candidate()
        rob = ParameterRobustness(backend=HonestEventDrivenBackend())
        stop_p, stop_paths = rob._with_param(cand, "atr_stop_mult", 3.3)
        tgt_p, tgt_paths = rob._with_param(cand, "atr_target_mult", 5.1)
        assert any(p.startswith("stop") for p in stop_paths)
        assert any(p.startswith("target") for p in tgt_paths)
        assert stop_p.candidate_id != cand.candidate_id
        assert tgt_p.candidate_id != cand.candidate_id
        assert stop_p.candidate_id != tgt_p.candidate_id


class TestIntegrityHardFail:
    def test_wrong_signal_source_hard_fails(self, tmp_path: Path) -> None:
        backend = HonestEventDrivenBackend(force_signal_source="hand_crafted")
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=backend,
            run_id="bad_signal",
            max_robustness_candidates=1,
            max_parameters_per_candidate=1,
        )
        failed = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_FAILED
        ]
        assert failed
        assert any(
            s.final_reason in {ROBUSTNESS_INTEGRITY_FAILED, ROBUSTNESS_SIGNAL_SOURCE_INVALID}
            or any(
                p.get("rejection_reason") == ROBUSTNESS_SIGNAL_SOURCE_INVALID
                for ps in s.parameter_summaries
                for p in ps.get("points", [])
            )
            for s in failed
        )

    def test_wrong_backend_kind_hard_fails(self, tmp_path: Path) -> None:
        backend = HonestEventDrivenBackend(force_backend_kind="synthetic_oos_probe")
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=backend,
            run_id="bad_kind",
            max_robustness_candidates=1,
            max_parameters_per_candidate=1,
        )
        failed = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_FAILED
        ]
        assert failed
        assert any(
            any(
                p.get("rejection_reason") == ROBUSTNESS_BACKEND_KIND_INVALID
                for ps in s.parameter_summaries
                for p in ps.get("points", [])
            )
            or s.final_reason
            in {ROBUSTNESS_INTEGRITY_FAILED, ROBUSTNESS_BACKEND_KIND_INVALID}
            for s in failed
        )

    def test_zero_folds_hard_fails(self, tmp_path: Path) -> None:
        backend = HonestEventDrivenBackend(force_completed_folds=0)
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=backend,
            run_id="zero_folds",
            max_robustness_candidates=1,
            max_parameters_per_candidate=1,
        )
        failed = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_FAILED
        ]
        assert failed
        assert any(
            any(
                p.get("rejection_reason") == ROBUSTNESS_WFO_INCOMPLETE
                for ps in s.parameter_summaries
                for p in ps.get("points", [])
            )
            or s.final_reason in {ROBUSTNESS_INTEGRITY_FAILED, ROBUSTNESS_WFO_INCOMPLETE}
            for s in failed
        )


class TestCandidateLevelDecision:
    def test_pass_requires_all_selected_parameters(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=HonestEventDrivenBackend(seed_salt=2),
            run_id="all_params",
            max_parameters_per_candidate=2,
            max_points_per_parameter=5,
            max_robustness_candidates=2,
        )
        passed = [
            s
            for s in result.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_PASSED
        ]
        for s in passed:
            assert s.accepted_parameter_count == len(s.selected_parameter_names)
            assert s.failed_parameter_count == 0
            assert all(
                ps["final_parameter_decision"] == PARAMETER_ROBUST
                for ps in s.parameter_summaries
            )

    def test_budget_truncation_cannot_pass(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=HonestEventDrivenBackend(seed_salt=4),
            run_id="budget_trunc",
            max_robustness_candidates=2,
            max_robustness_evaluations=2,
            max_parameters_per_candidate=2,
            max_points_per_parameter=5,
        )
        for s in result.candidate_robustness_summaries:
            if s.final_decision == ROBUSTNESS_PASSED:
                pytest.fail("budget truncation must not yield ROBUSTNESS_PASSED")
            if s.robustness_backend_calls > 0 or "BUDGET" in (s.final_reason or ""):
                assert s.final_decision == ROBUSTNESS_FAILED
                assert s.final_reason in {
                    ROBUSTNESS_BUDGET_EXHAUSTED,
                    "ROBUSTNESS_INCOMPLETE",
                    ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD,
                }

    def test_insufficient_neighborhood_cannot_pass(self) -> None:
        cand = _atr_candidate()
        backend = HonestEventDrivenBackend()
        rob = ParameterRobustness(
            backend=backend,
            research_eligible=True,
            synthetic_robustness_forbidden=True,
            relative_steps=(0.0,),  # only center — insufficient
            min_valid_neighborhood_points=3,
        )
        results = rob.probe(
            cand,
            parameter_names=("atr_stop_mult",),
            counters=type("C", (), {"robustness": 0})(),
            max_evaluations=10,
            require_integrity=True,
        )
        assert results[0].accepted is False
        assert results[0].final_parameter_decision != PARAMETER_ROBUST
        assert ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD in (
            results[0].reason,
            results[0].final_parameter_decision,
        ) or results[0].reason == ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD


class TestStatusAndCapability:
    def test_status_history_cannot_skip_stress_passed(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=HonestEventDrivenBackend(seed_salt=8),
            run_id="status_order",
        )
        for s in result.candidate_robustness_summaries:
            if s.final_decision != ROBUSTNESS_PASSED:
                continue
            events = [
                e for e in result.candidate_status_history if e.candidate_id == s.candidate_id
            ]
            statuses = [e.new_status for e in events]
            assert SCORE_QUALIFIED in statuses
            assert STRESS_TESTED in statuses
            assert STRESS_PASSED in statuses
            assert ROBUSTNESS_TESTED in statuses
            assert ROBUSTNESS_PASSED in statuses
            assert statuses.index(STRESS_PASSED) < statuses.index(ROBUSTNESS_PASSED)

    def test_robustness_passed_does_not_populate_downstream(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(),
            backend=HonestEventDrivenBackend(seed_salt=6),
            run_id="no_downstream",
        )
        payload = result.as_dict()
        # Phase 3C now runs after robustness; Vault/Paper/Live stay blocked.
        assert result.pipeline_level == (
            "MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST"
        )
        assert result.stress_pipeline_complete is True
        assert result.robustness_pipeline_complete is True
        assert result.statistics_pipeline_complete is True
        assert result.clustering_pipeline_complete is True
        assert result.research_shortlist_pipeline_complete is True
        assert result.post_wfo_pipeline_complete is False
        assert result.vault_pipeline_complete is False
        assert result.paper_pipeline_complete is False
        assert result.live_pipeline_complete is False
        assert payload["finalists"] == []
        assert payload["promoted"] == []
        assert payload["vault_candidates"] == []
        assert payload["paper_candidates"] == []
        assert "VAULT_NOT_RUN" in result.post_wfo_blocked_reasons
        assert "PAPER_NOT_RUN" in result.post_wfo_blocked_reasons
        assert "LIVE_NOT_RUN" in result.post_wfo_blocked_reasons
        assert "STATISTICS_NOT_RUN" not in result.post_wfo_blocked_reasons
        assert "ROBUSTNESS_NOT_RUN" not in result.post_wfo_blocked_reasons
        for claim in (
            "finalist",
            "promoted",
            "vault eligible",
            "paper eligible",
            "live eligible",
        ):
            assert claim in payload["research_shortlisted_does_not_mean"]
        for claim in (
            "finalist",
            "promoted",
            "research shortlisted",
            "vault eligible",
            "paper eligible",
            "live eligible",
        ):
            assert claim in payload["robustness_passed_does_not_mean"]

    def test_not_entered_recorded_for_non_stress_passed(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            stress_factory=_make_real_stress_factory(
                fail_scenarios=frozenset({"base_costs", "costs_2x", "wider_spread"})
            ),
            run_id="not_entered",
            min_stress_pass_rate=1.0,
        )
        not_entered = [
            e
            for e in result.candidate_status_history
            if e.new_status == ROBUSTNESS_NOT_ENTERED
        ]
        assert not_entered
        assert all(str(e.reason).startswith(ROBUSTNESS_NOT_ENTERED) for e in not_entered)
