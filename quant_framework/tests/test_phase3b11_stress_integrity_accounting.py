"""Phase 3B.1.1: Stress integrity hard-fail + original-WFO baseline accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.evaluator import (
    CandidateEvaluator,
    EvalOutcome,
    SyntheticOOSBackend,
    snapshot_baseline_wfo_artifacts,
)
from discovery.expression_tree import constant_node, feature_node, op_node
from discovery.fitness import FoldOOSMetrics, RobustFitness
from discovery.multi_family_campaign import (
    BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO,
    STRESS_BACKEND_KIND_INVALID,
    STRESS_FAILED,
    STRESS_INTEGRITY_FAILED,
    STRESS_PASSED,
    STRESS_SIGNAL_SOURCE_INVALID,
    STRESS_WFO_INCOMPLETE,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
)
from discovery.operators import OperatorId
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.stress import StressResult, StressTester
from discovery.stress_backend import (
    STRESS_BACKEND_KIND,
    STRESS_BASELINE_TRADES_UNAVAILABLE,
)
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


def _closed_trades_for(candidate_id: str) -> list[dict[str, Any]]:
    return [
        {
            "trade_id": f"{candidate_id}_t1",
            "exit_time": "2024-01-02T15:00:00+00:00",
            "net_pnl": 100.0,
        },
        {
            "trade_id": f"{candidate_id}_t2",
            "exit_time": "2024-01-03T15:00:00+00:00",
            "net_pnl": 40.0,
        },
        {
            "trade_id": f"{candidate_id}_t3",
            "exit_time": "2024-01-04T15:00:00+00:00",
            "net_pnl": 10.0,
        },
    ]


@dataclass
class ArtifactFullWfoBackend(SyntheticOOSBackend):
    """Full WFO backend that publishes candidate-specific last_run_artifacts."""

    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)
    evaluate_calls: list[str] = field(default_factory=list)

    def evaluate(self, candidate):
        self.evaluate_calls.append(candidate.candidate_id)
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
        closed = _closed_trades_for(candidate.candidate_id)
        self.last_run_artifacts = {
            "candidate_id": candidate.candidate_id,
            "signal_source": "candidate_dsl_trees",
            "backend_kind": self.backend_kind,
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
            "closed_trades": closed,
            "orders_count": len(closed),
            "fills_count": len(closed),
            "trades_count": len(closed),
            "oos_ranges": [
                {"fold_id": str(f.fold_id), "start": "2024-01-01", "end": "2024-01-10"}
                for f in folds
            ],
        }
        train = {
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
            "backend_kind": self.backend_kind,
            "orders_count": len(closed),
            "fills_count": len(closed),
            "trades_count": len(closed),
            "oos_ranges": list(self.last_run_artifacts["oos_ranges"]),
        }
        return folds, train


@dataclass
class _RealStressScenarioBackend:
    scenario: str
    folds: list[FoldOOSMetrics]
    pass_economic: bool = True
    signal_source: str = "candidate_dsl_trees"
    backend_kind: str = STRESS_BACKEND_KIND
    completed_fold_count: int | None = None
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)
    cost_model_changes: dict[str, Any] = field(default_factory=dict)
    execution_changes: dict[str, Any] = field(default_factory=dict)
    evaluate_calls: list[str] = field(default_factory=list)

    def evaluate(self, candidate):
        self.evaluate_calls.append(candidate.candidate_id)
        folds = self.folds if self.pass_economic else [
            FoldOOSMetrics(
                fold_id=0,
                expectancy=-0.05,
                sharpe=-0.4,
                profit_factor=0.7,
                calmar=-0.2,
                max_drawdown=-0.25,
                n_trades=8,
            )
        ]
        n_folds = (
            self.completed_fold_count
            if self.completed_fold_count is not None
            else len(folds)
        )
        self.last_run_artifacts = {
            "signal_source": self.signal_source,
            "is_full_event_wfo": True,
            "wfo_completed_folds": n_folds,
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


def _make_stress_factory(
    *,
    fail_scenarios: frozenset[str] | None = None,
    status_map: dict[str, str] | None = None,
    call_log: list[str] | None = None,
    integrity_overrides: dict[str, dict[str, Any]] | None = None,
    capture_baseline: list[dict[str, Any] | None] | None = None,
) -> Callable[[str], Any]:
    fail_scenarios = fail_scenarios or frozenset()
    status_map = status_map or {}
    call_log = call_log if call_log is not None else []
    integrity_overrides = integrity_overrides or {}
    baseline_cache: dict[str, Any] = {"arts": None}

    def _factory(scenario: str):
        call_log.append(scenario)
        if capture_baseline is not None:
            capture_baseline.append(
                dict(baseline_cache["arts"]) if baseline_cache["arts"] else None
            )
        if scenario in status_map:
            return _StatusBackend(stress_status=status_map[scenario])
        ov = integrity_overrides.get(scenario, {})
        return _RealStressScenarioBackend(
            scenario=scenario,
            folds=_good_folds(),
            pass_economic=scenario not in fail_scenarios,
            signal_source=str(ov.get("signal_source", "candidate_dsl_trees")),
            backend_kind=str(ov.get("backend_kind", STRESS_BACKEND_KIND)),
            completed_fold_count=ov.get("completed_fold_count"),
        )

    _factory.backend_kind = STRESS_BACKEND_KIND  # type: ignore[attr-defined]
    _factory.research_eligible = True  # type: ignore[attr-defined]
    _factory.synthetic_stress_forbidden = True  # type: ignore[attr-defined]
    _factory.baseline_cache = baseline_cache  # type: ignore[attr-defined]
    return _factory


def _simple_candidate(*, seed: int = 1) -> StrategyCandidate:
    z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
    entry = op_node(
        OperatorId.ENTRY_LONG,
        op_node(OperatorId.LESS_THAN, z, constant_node(-2.0 - 0.01 * seed)),
    )
    return build_candidate(
        entry_tree=entry,
        strategy_family="test_phase3b11",
        creation_method=CreationMethod.RANDOM,
        grammar_version="v",
        feature_set_version="v",
        cost_model_version="v",
        random_seed=seed,
    )


def _stress_result(
    scenario: str,
    *,
    passed: bool = True,
    integrity_ok: bool = True,
    signal_source: str = "candidate_dsl_trees",
    backend_kind: str = STRESS_BACKEND_KIND,
    completed_fold_count: int = 3,
    status: str = "executed",
) -> StressResult:
    return StressResult(
        scenario=scenario,
        candidate_id="c1",
        fitness=1.0 if passed else 0.0,
        median_expectancy=0.08 if passed else -0.05,
        max_drawdown=-0.04,
        passed=passed,
        failure_reason=None if passed else "economic",
        signal_source=signal_source,
        backend_kind=backend_kind,
        status=status,
        completed_fold_count=completed_fold_count,
        integrity_ok=integrity_ok,
    )


def _campaign() -> MultiFamilyCampaign:
    registry = ExperimentRegistry(Path("_unused_phase3b11"))
    return MultiFamilyCampaign(
        config=FamilyCampaignConfig(
            requested_family_count=1,
            min_candidates_per_family=2,
            total_candidate_budget=4,
            max_full_wfo=4,
            seed=1,
        ),
        registry=registry,
        backend=SyntheticOOSBackend(seed_salt=1),
        discovery_run_id="decide_only",
        research_eligible=True,
    )


class TestStressIntegrityHardFail:
    def test_nine_passed_one_integrity_false_cannot_pass(self) -> None:
        campaign = _campaign()
        results = [
            _stress_result(f"s{i}", passed=True, integrity_ok=True) for i in range(9)
        ]
        bad = _stress_result(
            "s_bad",
            passed=False,
            integrity_ok=False,
            signal_source="wrong_source",
            completed_fold_count=3,
        )
        results.append(bad)
        decision, reason, stats = campaign._decide_stress_outcome(
            results, required_pass_rate=0.5, fail_closed_unsupported=True
        )
        assert decision == STRESS_FAILED
        assert reason == STRESS_SIGNAL_SOURCE_INVALID
        assert len(stats["integrity_failures"]) == 1
        assert stats["integrity_failures"][0]["scenario"] == "s_bad"
        assert stats["pass_rate"] == pytest.approx(0.9)

    def test_wrong_backend_kind_fails_with_valid_signal_and_folds(self) -> None:
        campaign = _campaign()
        results = [
            _stress_result("ok", passed=True),
            _stress_result(
                "bad_kind",
                passed=True,
                integrity_ok=False,
                signal_source="candidate_dsl_trees",
                backend_kind="synthetic_oos_probe",
                completed_fold_count=3,
            ),
        ]
        decision, reason, stats = campaign._decide_stress_outcome(
            results, required_pass_rate=0.0, fail_closed_unsupported=True
        )
        assert decision == STRESS_FAILED
        assert reason == STRESS_BACKEND_KIND_INVALID
        assert stats["integrity_failures"][0]["backend_kind"] == "synthetic_oos_probe"

    def test_wrong_signal_source_fails(self) -> None:
        campaign = _campaign()
        results = [
            _stress_result(
                "bad_src",
                passed=True,
                integrity_ok=False,
                signal_source="not_dsl",
                completed_fold_count=2,
            )
        ]
        decision, reason, _ = campaign._decide_stress_outcome(
            results, required_pass_rate=0.0, fail_closed_unsupported=True
        )
        assert decision == STRESS_FAILED
        assert reason == STRESS_SIGNAL_SOURCE_INVALID

    def test_zero_completed_folds_fails(self) -> None:
        campaign = _campaign()
        results = [
            _stress_result(
                "no_folds",
                passed=True,
                integrity_ok=False,
                completed_fold_count=0,
            )
        ]
        decision, reason, _ = campaign._decide_stress_outcome(
            results, required_pass_rate=0.0, fail_closed_unsupported=True
        )
        assert decision == STRESS_FAILED
        assert reason == STRESS_WFO_INCOMPLETE

    def test_pass_rate_cannot_override_integrity_failure(self) -> None:
        campaign = _campaign()
        results = [
            _stress_result(f"ok{i}") for i in range(9)
        ] + [
            _stress_result(
                "integrity_only",
                passed=False,
                integrity_ok=False,
                signal_source="candidate_dsl_trees",
                backend_kind="not_real_event",
                completed_fold_count=5,
            )
        ]
        decision, reason, stats = campaign._decide_stress_outcome(
            results, required_pass_rate=0.1, fail_closed_unsupported=True
        )
        assert stats["pass_rate"] >= 0.1
        assert decision == STRESS_FAILED
        assert reason == STRESS_BACKEND_KIND_INVALID
        assert reason != "stress_pass_rate_ok"

    def test_integrity_ok_false_catch_all(self) -> None:
        campaign = _campaign()
        # All individual fields look valid but integrity_ok is false.
        results = [
            _stress_result(
                "flag_false",
                passed=True,
                integrity_ok=False,
                signal_source="candidate_dsl_trees",
                backend_kind=STRESS_BACKEND_KIND,
                completed_fold_count=3,
            )
        ]
        decision, reason, stats = campaign._decide_stress_outcome(
            results, required_pass_rate=0.0, fail_closed_unsupported=True
        )
        assert decision == STRESS_FAILED
        assert reason == STRESS_INTEGRITY_FAILED
        assert stats["integrity_failures"][0]["integrity_ok"] is False


class TestBaselineWfoArtifacts:
    def test_original_wfo_closed_trades_copied_to_record(self, tmp_path: Path) -> None:
        backend = ArtifactFullWfoBackend(seed_salt=11)
        registry = ExperimentRegistry(tmp_path / "reg_base")
        ev = CandidateEvaluator(
            backend=backend,
            registry=registry,
            budget=SearchBudget(
                max_generated_candidates=5,
                max_evaluated_candidates=5,
                max_full_wfo_evaluations=5,
                max_stress_evaluations=0,
                min_oos_trades=1,
                min_oos_trades_per_fold=1,
            ),
            counters=BudgetCounters(),
            fitness_model=RobustFitness(min_total_oos_trades=1, min_oos_trades_per_fold=1),
            discovery_run_id="baseline_copy",
        )
        cand = _simple_candidate(seed=11)
        rec = ev.evaluate(cand)
        assert rec.outcome is EvalOutcome.REGISTERED
        snap = rec.meta["baseline_wfo_artifacts"]
        assert snap["candidate_id"] == cand.candidate_id
        assert snap["signal_source"] == "candidate_dsl_trees"
        assert snap["closed_trades"] == _closed_trades_for(cand.candidate_id)
        assert snap["wfo_completed_folds"] > 0
        # Snapshot is a deep copy — mutating backend artifacts must not affect it.
        backend.last_run_artifacts["closed_trades"].append({"trade_id": "x"})
        assert len(rec.meta["baseline_wfo_artifacts"]["closed_trades"]) == 3

    def test_later_candidate_cannot_overwrite_prior_snapshot(self, tmp_path: Path) -> None:
        backend = ArtifactFullWfoBackend(seed_salt=12)
        registry = ExperimentRegistry(tmp_path / "reg_iso")
        ev = CandidateEvaluator(
            backend=backend,
            registry=registry,
            budget=SearchBudget(
                max_generated_candidates=5,
                max_evaluated_candidates=5,
                max_full_wfo_evaluations=5,
                max_stress_evaluations=0,
                min_oos_trades=1,
                min_oos_trades_per_fold=1,
            ),
            counters=BudgetCounters(),
            fitness_model=RobustFitness(min_total_oos_trades=1, min_oos_trades_per_fold=1),
            discovery_run_id="baseline_iso",
        )
        cand_a = _simple_candidate(seed=21)
        cand_b = _simple_candidate(seed=22)
        rec_a = ev.evaluate(cand_a)
        rec_b = ev.evaluate(cand_b)
        assert rec_a.meta["baseline_wfo_artifacts"]["candidate_id"] == cand_a.candidate_id
        assert rec_b.meta["baseline_wfo_artifacts"]["candidate_id"] == cand_b.candidate_id
        assert (
            rec_a.meta["baseline_wfo_artifacts"]["closed_trades"][0]["trade_id"]
            == f"{cand_a.candidate_id}_t1"
        )
        assert (
            rec_b.meta["baseline_wfo_artifacts"]["closed_trades"][0]["trade_id"]
            == f"{cand_b.candidate_id}_t1"
        )
        # Backend now points at B; A's snapshot remains A.
        assert backend.last_run_artifacts["candidate_id"] == cand_b.candidate_id
        assert rec_a.meta["baseline_wfo_artifacts"]["candidate_id"] == cand_a.candidate_id

    def test_snapshot_helper_never_aliases_mutable_backend_arts(self) -> None:
        backend = ArtifactFullWfoBackend(seed_salt=1)
        cand = _simple_candidate(seed=31)
        backend.evaluate(cand)
        snap = snapshot_baseline_wfo_artifacts(
            candidate_id=cand.candidate_id,
            backend=backend,
            train_metrics={},
        )
        backend.last_run_artifacts["closed_trades"].clear()
        assert len(snap["closed_trades"]) == 3


class TestStressBaselineWiring:
    def test_stress_receives_candidate_specific_baseline(self) -> None:
        captured: list[dict[str, Any] | None] = []
        call_log: list[str] = []
        factory = _make_stress_factory(call_log=call_log, capture_baseline=captured)
        budget = SearchBudget(max_stress_evaluations=5)
        counters = BudgetCounters()
        tester = StressTester(
            budget=budget,
            counters=counters,
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
            backend_factory=factory,
            research_eligible=True,
            synthetic_stress_forbidden=True,
        )
        cand = _simple_candidate(seed=41)
        arts = {
            "candidate_id": cand.candidate_id,
            "signal_source": "candidate_dsl_trees",
            "backend_kind": "event_driven_wfo",
            "is_full_event_wfo": True,
            "wfo_completed_folds": 3,
            "closed_trades": _closed_trades_for(cand.candidate_id),
            "orders_count": 3,
            "fills_count": 3,
            "trades_count": 3,
            "oos_ranges": [],
        }
        tester.run(
            cand,
            base_fitness=1.0,
            scenarios=("base_costs",),
            baseline_artifacts=arts,
        )
        assert captured
        assert captured[0] is not None
        assert captured[0]["candidate_id"] == cand.candidate_id
        assert captured[0]["closed_trades"][0]["trade_id"] == f"{cand.candidate_id}_t1"
        assert tester.last_run_accounting["baseline_artifact_source"] == (
            BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO
        )
        assert tester.last_run_accounting["hidden_baseline_rerun"] is False

    def test_stress_tester_never_calls_base_backend_to_seed(self) -> None:
        class _BaseSpy:
            backend_kind = "event_driven_wfo"
            calls = 0

            def evaluate(self, candidate):
                self.calls += 1
                raise AssertionError("hidden baseline reevaluation must not occur")

        baseline_cache: dict[str, Any] = {"arts": None}
        base = _BaseSpy()

        def _factory(scenario: str):
            return _RealStressScenarioBackend(scenario=scenario, folds=_good_folds())

        _factory.baseline_cache = baseline_cache  # type: ignore[attr-defined]
        _factory.base_event_backend = base  # type: ignore[attr-defined]

        tester = StressTester(
            budget=SearchBudget(max_stress_evaluations=3),
            counters=BudgetCounters(),
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
            backend_factory=_factory,
            research_eligible=True,
        )
        cand = _simple_candidate(seed=42)
        arts = {
            "candidate_id": cand.candidate_id,
            "closed_trades": _closed_trades_for(cand.candidate_id),
            "signal_source": "candidate_dsl_trees",
            "wfo_completed_folds": 2,
        }
        tester.run(
            cand,
            base_fitness=1.0,
            scenarios=("base_costs",),
            baseline_artifacts=arts,
        )
        assert base.calls == 0
        assert baseline_cache["arts"]["candidate_id"] == cand.candidate_id
        assert tester.last_run_accounting["hidden_baseline_rerun"] is False

    def test_removed_best_day_uses_original_wfo_trades(self) -> None:
        from discovery.stress_backend import _extract_closed_trades

        arts = {
            "candidate_id": "c_day",
            "closed_trades": _closed_trades_for("c_day"),
        }
        extracted = _extract_closed_trades(arts)
        assert extracted[0]["trade_id"] == "c_day_t1"
        assert max(t["net_pnl"] for t in extracted) == 100.0
        # StressTester seeds factory cache from these arts — no base reeval.
        baseline_cache: dict[str, Any] = {"arts": None}
        call_log: list[str] = []

        def _factory(scenario: str):
            call_log.append(scenario)
            arts_seen = baseline_cache.get("arts")
            assert arts_seen is not None
            assert arts_seen["closed_trades"][0]["trade_id"] == "c_day_t1"
            return _StatusBackend(stress_status=STRESS_BASELINE_TRADES_UNAVAILABLE)

        _factory.baseline_cache = baseline_cache  # type: ignore[attr-defined]
        tester = StressTester(
            budget=SearchBudget(max_stress_evaluations=3),
            counters=BudgetCounters(),
            backend_factory=_factory,
            research_eligible=True,
        )
        results = tester.run(
            _simple_candidate(seed=51),
            base_fitness=1.0,
            scenarios=("removed_best_day",),
            baseline_artifacts=arts,
        )
        assert results[0].status == "baseline_unavailable"
        assert tester.last_run_accounting["scenario_backend_calls"] == 0
        assert tester.last_run_accounting["hidden_baseline_rerun"] is False

    def test_removed_best_trades_uses_original_qualifying_trades(self) -> None:
        from discovery.stress_backend import REMOVED_BEST_TRADES_FRACTION, _extract_closed_trades

        arts = {"closed_trades": _closed_trades_for("c_tr")}
        trades = _extract_closed_trades(arts)
        ranked = sorted(trades, key=lambda t: float(t["net_pnl"]), reverse=True)
        n_remove = max(1, int(round(len(ranked) * REMOVED_BEST_TRADES_FRACTION)))
        removed = ranked[:n_remove]
        assert removed[0]["trade_id"] == "c_tr_t1"
        assert removed[0]["net_pnl"] == 100.0

        baseline_cache: dict[str, Any] = {"arts": None}

        def _factory(scenario: str):
            assert baseline_cache["arts"]["closed_trades"][0]["trade_id"] == "c_tr_t1"
            return _StatusBackend(stress_status=STRESS_BASELINE_TRADES_UNAVAILABLE)

        _factory.baseline_cache = baseline_cache  # type: ignore[attr-defined]
        tester = StressTester(
            budget=SearchBudget(max_stress_evaluations=3),
            counters=BudgetCounters(),
            backend_factory=_factory,
            research_eligible=True,
        )
        results = tester.run(
            _simple_candidate(seed=52),
            base_fitness=1.0,
            scenarios=("removed_best_trades",),
            baseline_artifacts=arts,
        )
        assert results[0].failure_reason == STRESS_BASELINE_TRADES_UNAVAILABLE
        assert tester.last_run_accounting["hidden_baseline_rerun"] is False

    def test_missing_original_trades_zero_backend_calls(self) -> None:
        call_log: list[str] = []
        factory = _make_stress_factory(
            call_log=call_log,
            status_map={
                "removed_best_day": STRESS_BASELINE_TRADES_UNAVAILABLE,
                "removed_best_trades": STRESS_BASELINE_TRADES_UNAVAILABLE,
            },
        )
        counters = BudgetCounters()
        tester = StressTester(
            budget=SearchBudget(max_stress_evaluations=10),
            counters=counters,
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
            backend_factory=factory,
            research_eligible=True,
        )
        results = tester.run(
            _simple_candidate(seed=53),
            base_fitness=1.0,
            scenarios=("removed_best_day", "removed_best_trades"),
            baseline_artifacts=None,
        )
        assert all(r.status == "baseline_unavailable" for r in results)
        assert all(r.failure_reason == STRESS_BASELINE_TRADES_UNAVAILABLE for r in results)
        assert counters.stress == 0
        assert tester.last_run_accounting["scenario_backend_calls"] == 0
        assert tester.last_run_accounting["stress_counter_delta"] == 0
        # Factory was consulted for status, but evaluate was never called.
        assert call_log == ["removed_best_day", "removed_best_trades"]

    def test_scenario_backend_calls_equal_stress_counter_delta(self) -> None:
        call_log: list[str] = []
        factory = _make_stress_factory(call_log=call_log)
        counters = BudgetCounters()
        tester = StressTester(
            budget=SearchBudget(max_stress_evaluations=10),
            counters=counters,
            fitness_model=RobustFitness(min_total_oos_trades=0, min_oos_trades_per_fold=0),
            backend_factory=factory,
            research_eligible=True,
        )
        tester.run(
            _simple_candidate(seed=54),
            base_fitness=1.0,
            scenarios=("base_costs", "costs_2x", "symbol_exclusion"),
            baseline_artifacts={"candidate_id": "c_acct", "closed_trades": []},
        )
        acct = tester.last_run_accounting
        assert acct["scenario_backend_calls"] == acct["stress_counter_delta"]
        assert acct["scenario_backend_calls"] == 2  # symbol_exclusion N/A
        assert counters.stress == 2
        assert acct["hidden_baseline_rerun"] is False

    def test_no_hidden_baseline_rerun_when_max_stress_zero(self, tmp_path: Path) -> None:
        backend = ArtifactFullWfoBackend(seed_salt=3)

        class _CountingFactory:
            backend_kind = STRESS_BACKEND_KIND
            baseline_cache = {"arts": None}

            def __call__(self, scenario: str):
                raise AssertionError("Stress factory must not be called when budget is 0")

        factory = _CountingFactory()
        registry = ExperimentRegistry(tmp_path / "reg_zero")
        campaign = MultiFamilyCampaign(
            config=FamilyCampaignConfig(
                requested_family_count=1,
                min_candidates_per_family=3,
                total_candidate_budget=8,
                max_full_wfo=6,
                adaptive_reallocation=False,
                seed=21,
                min_oos_trades=1,
                min_oos_trades_per_fold=1,
                max_runtime_seconds=60,
                family_local_evolution=True,
                evolution_generations=1,
                population_size=2,
                stagnation_generations=99,
                allow_cross_family_crossover=False,
                minimum_improvement=1e-4,
                max_stress_evaluations=0,
                max_stress_scenarios_per_candidate=4,
                min_stress_pass_rate=0.5,
                stress_scenarios=("base_costs", "costs_2x"),
            ),
            registry=registry,
            backend=backend,
            discovery_run_id="zero_stress_budget",
            research_eligible=True,
            synthetic_stress_forbidden=True,
            stress_backend_factory=factory,
        )
        result = campaign.run()
        assert result.stress_accounting is not None
        assert result.stress_accounting.stress_evaluations_consumed == 0
        for s in result.candidate_stress_summaries:
            assert s.scenario_backend_calls == 0
            assert s.stress_counter_delta == 0
            assert s.hidden_baseline_rerun is False

    def test_multifamily_integrity_override_fails_candidate(self, tmp_path: Path) -> None:
        call_log: list[str] = []
        factory = _make_stress_factory(
            call_log=call_log,
            integrity_overrides={
                "costs_2x": {
                    "backend_kind": "wrong_backend",
                    "signal_source": "candidate_dsl_trees",
                    "completed_fold_count": 3,
                }
            },
        )
        registry = ExperimentRegistry(tmp_path / "reg_int")
        campaign = MultiFamilyCampaign(
            config=FamilyCampaignConfig(
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
                max_stress_scenarios_per_candidate=3,
                min_stress_pass_rate=0.0,  # would pass economically
                stress_scenarios=("base_costs", "costs_2x", "wider_spread"),
            ),
            registry=registry,
            backend=ArtifactFullWfoBackend(seed_salt=7),
            discovery_run_id="integrity_mf",
            research_eligible=True,
            synthetic_stress_forbidden=True,
            stress_backend_factory=factory,
        )
        result = campaign.run()
        entered = [
            s
            for s in result.candidate_stress_summaries
            if s.final_decision in {STRESS_PASSED, STRESS_FAILED}
        ]
        assert entered
        assert all(s.final_decision == STRESS_FAILED for s in entered)
        assert all(s.final_reason == STRESS_BACKEND_KIND_INVALID for s in entered)
        assert all(s.integrity_failures for s in entered)
        assert all(
            s.scenario_backend_calls == s.stress_counter_delta for s in entered
        )
        assert all(s.hidden_baseline_rerun is False for s in entered)
        assert all(
            s.baseline_artifact_source == BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO
            for s in entered
        )
