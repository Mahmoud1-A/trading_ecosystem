"""Phase 3C: DSR/PBO → Behavioral Clustering → Research Shortlist."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from discovery.candidate import build_candidate
from discovery.expression_tree import feature_node, op_node, parameter_node
from discovery.fitness import FoldOOSMetrics
from discovery.multi_family_campaign import (
    BEHAVIORALLY_CLUSTERED,
    DSR_INSUFFICIENT_DATA,
    DSR_PASSED,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
    PBO_PASSED,
    PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST,
    RESEARCH_SHORTLISTED,
    ROBUSTNESS_PASSED,
    SCORE_QUALIFIED,
    STATISTICS_TESTED,
    STATISTICALLY_PASSED,
    STRESS_PASSED,
    STRESS_TESTED,
    VAULT_NOT_RUN,
)
from discovery.operators import OperatorId
from discovery.research_shortlist_pipeline import (
    RESEARCH_SHORTLISTED_DOES_NOT_MEAN,
    ResearchShortlistConfig,
    AlignedPerformanceMatrix,
    align_performance_matrix,
    build_behavioral_signature,
    build_full_wfo_trial_population,
    evaluate_candidate_dsr_pbo,
    extract_oos_return_series,
    extract_oos_statistics_series,
    extract_trade_timing_vector,
    extract_exposure_vector,
    hard_gate_check,
    _full_wfo_completed,
    STATISTICS_INSUFFICIENT_DATA,
    OBS_INSUFFICIENT,
    PARTITION_TRADING_DAY,
    PARTITION_WFO_VALIDATION_PERIOD,
)
from discovery.stress_backend import STRESS_BACKEND_KIND
from discovery.types import CreationMethod, ValueType
from registry.experiment_registry import ExperimentRegistry


def _folds_for_candidate(candidate_id: str, n: int = 24) -> list[FoldOOSMetrics]:
    """Diverse per-candidate fold artifacts (enough obs for DSR/PBO)."""
    digest = hashlib.sha256(candidate_id.encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16) % (2**31 - 1))
    base = 0.06 + 0.04 * (int(digest[8:10], 16) / 255.0)
    folds: list[FoldOOSMetrics] = []
    for i in range(n):
        exp = float(base + rng.normal(0.0, 0.02))
        folds.append(
            FoldOOSMetrics(
                fold_id=i,
                expectancy=max(0.02, exp),
                sharpe=max(0.5, exp * 8.0 + float(rng.normal(0, 0.15))),
                profit_factor=max(1.15, 1.2 + exp),
                calmar=max(0.3, exp * 4.0),
                max_drawdown=-min(0.08, abs(float(rng.uniform(0.02, 0.06)))),
                turnover=float(0.1 + (int(digest[10 + (i % 4)], 16) % 8) * 0.05),
                n_trades=max(6, 8 + (i % 5) + (int(digest[12], 16) % 4)),
            )
        )
    return folds


def _closed_trades(candidate_id: str, folds: list[FoldOOSMetrics]) -> list[dict[str, Any]]:
    digest = hashlib.sha256(candidate_id.encode()).hexdigest()
    trades: list[dict[str, Any]] = []
    hour_bias = int(digest[0:2], 16) % 24
    for f in folds:
        for t_i in range(max(1, int(f.n_trades) // 4)):
            hour = (hour_bias + t_i * 3 + f.fold_id) % 24
            trades.append(
                {
                    "fold_id": f.fold_id,
                    "phase": "validation_oos",
                    "net_pnl": float(f.expectancy) * (0.8 + 0.1 * t_i),
                    "qty": float(1.0 + (hour % 5) * 0.25),
                    "exit_time": f"2024-01-{(f.fold_id % 28) + 1:02d}T{hour:02d}:15:00",
                    "entry_time": f"2024-01-{(f.fold_id % 28) + 1:02d}T{hour:02d}:00:00",
                }
            )
    return trades


@dataclass
class RichEventDrivenBackend:
    """Full-WFO backend with enough OOS artifacts for DSR/PBO + clustering."""

    seed_salt: int = 0
    backend_kind: str = "event_driven_wfo"
    is_full_event_wfo: bool = True
    n_folds: int = 24
    call_log: list[tuple[str, str]] = field(default_factory=list)
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, candidate):
        self.call_log.append(("evaluate", candidate.candidate_id))
        folds = _folds_for_candidate(candidate.candidate_id, n=self.n_folds)
        is_rob = bool((candidate.family_provenance or {}).get("robustness_param"))
        if is_rob:
            folds = [
                FoldOOSMetrics(
                    fold_id=f.fold_id,
                    expectancy=max(0.04, float(f.expectancy) * 0.95),
                    sharpe=max(0.6, float(f.sharpe) * 0.95),
                    profit_factor=max(1.1, float(f.profit_factor) * 0.98),
                    calmar=max(0.25, float(f.calmar)),
                    max_drawdown=float(f.max_drawdown),
                    turnover=float(f.turnover),
                    n_trades=max(6, int(f.n_trades)),
                )
                for f in folds
            ]
        closed = _closed_trades(candidate.candidate_id, folds)
        train = {
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
            "backend_kind": self.backend_kind,
            "candidate_id": candidate.candidate_id,
            "orders_count": len(closed),
            "fills_count": len(closed),
            "trades_count": sum(f.n_trades for f in folds),
            "closed_trades": closed,
            "oos_ranges": [
                {
                    "start": f"2024-01-{i + 1:02d}",
                    "end": f"2024-01-{i + 2:02d}",
                    "fold_id": str(i),
                }
                for i in range(len(folds))
            ],
        }
        self.last_run_artifacts = dict(train)
        return folds, train


@dataclass
class _RealStressScenarioBackend:
    scenario: str
    folds: list[FoldOOSMetrics]
    last_run_artifacts: dict[str, Any] = field(default_factory=dict)
    backend_kind: str = STRESS_BACKEND_KIND

    def evaluate(self, candidate):
        self.last_run_artifacts = {
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(self.folds),
            "trades_count": sum(f.n_trades for f in self.folds),
            "orders_count": sum(f.n_trades for f in self.folds),
            "fills_count": sum(f.n_trades for f in self.folds),
            "scenario": self.scenario,
            "candidate_id": candidate.candidate_id,
        }
        return self.folds, dict(self.last_run_artifacts)


def _make_real_stress_factory() -> Callable[[str], Any]:
    def _factory(scenario: str):
        folds = [
            FoldOOSMetrics(
                fold_id=i,
                expectancy=0.07,
                sharpe=1.2,
                profit_factor=1.4,
                calmar=0.5,
                max_drawdown=-0.04,
                n_trades=10,
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
        requested_family_count=3,
        min_candidates_per_family=3,
        total_candidate_budget=18,
        max_full_wfo=12,
        adaptive_reallocation=False,
        seed=77,
        min_oos_trades=1,
        min_oos_trades_per_fold=1,
        max_runtime_seconds=120,
        family_local_evolution=True,
        evolution_generations=2,
        population_size=2,
        stagnation_generations=99,
        allow_cross_family_crossover=False,
        minimum_improvement=1e-4,
        max_stress_evaluations=40,
        max_stress_scenarios_per_candidate=3,
        min_stress_pass_rate=0.5,
        stress_scenarios=("base_costs", "costs_2x", "wider_spread"),
        fail_closed_unsupported_stress=True,
        max_robustness_candidates=6,
        max_robustness_evaluations=80,
        max_parameters_per_candidate=1,
        max_points_per_parameter=3,
        allow_one_sided_neighborhood=False,
        min_valid_neighborhood_points=3,
        min_dsr=0.05,
        max_pbo=0.95,
        pbo_n_splits=4,
        behavioral_similarity_threshold=0.55,
        min_oos_observations_for_dsr=20,
    )
    base.update(overrides)
    return FamilyCampaignConfig(**base)


def _run(tmp_path: Path, *, run_id: str = "phase3c", **cfg_overrides):
    registry = ExperimentRegistry(tmp_path / f"reg_{run_id}")
    campaign = MultiFamilyCampaign(
        config=_evo_cfg(**cfg_overrides),
        registry=registry,
        backend=RichEventDrivenBackend(seed_salt=3),
        discovery_run_id=run_id,
        research_eligible=True,
        stress_backend_factory=_make_real_stress_factory(),
    )
    return campaign.run(), campaign


class TestStatisticsHonesty:
    def test_insufficient_data_never_fabricates_ok(self) -> None:
        from discovery.evaluator import EvaluationRecord, EvalOutcome
        from discovery.fitness import FitnessResult

        folds = [
            FoldOOSMetrics(
                fold_id=0,
                expectancy=0.1,
                sharpe=1.0,
                profit_factor=1.5,
                calmar=0.5,
                max_drawdown=-0.02,
                n_trades=5,
            )
        ]
        rec = EvaluationRecord(
            outcome=EvalOutcome.REGISTERED,
            candidate_id="c1",
            lineage_id="l1",
            trial_id="t1",
            fitness=FitnessResult(
                fitness=1.0,
                ranking_source="validation_oos",
                components={},
                fold_scores=(1.0,),
            ),
            rejection_reason=None,
            oos_folds=folds,
            train_metrics={
                "signal_source": "candidate_dsl_trees",
                "is_full_event_wfo": True,
                "wfo_completed_folds": 1,
            },
            meta={
                "is_full_wfo_completion": True,
                "baseline_wfo_artifacts": {
                    "candidate_id": "c1",
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": 1,
                    "closed_trades": [],
                },
            },
        )
        pop, _ids, series = build_full_wfo_trial_population([rec, rec])
        cfg = ResearchShortlistConfig(min_oos_observations_for_dsr=20, min_dsr=0.95)
        aligned = align_performance_matrix([rec, rec], candidate_ids=["c1", "c1"])
        matrix = aligned.matrix if isinstance(aligned, AlignedPerformanceMatrix) else None
        dsr, pbo, _meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=pop,
            performance_matrix=aligned if isinstance(aligned, AlignedPerformanceMatrix) else matrix,
            trial_column_index=0,
            config=cfg,
        )
        assert dsr.status.value == "INSUFFICIENT_DATA"
        assert dsr.deflated_sharpe is None
        assert pbo is not None
        assert pbo.status.value == "INSUFFICIENT_DATA"
        assert pbo.pbo is None

    def test_hard_gates_require_all_stages(self) -> None:
        ok, gates, reason = hard_gate_check(
            score_qualified=True,
            stress_passed=True,
            robustness_passed=True,
            dsr_decision=DSR_PASSED,
            pbo_decision=PBO_PASSED,
        )
        assert ok is True
        assert gates == [
            "SCORE_QUALIFIED",
            "STRESS_PASSED",
            "ROBUSTNESS_PASSED",
            "DSR_PASSED",
            "PBO_PASSED",
        ]
        ok2, _, reason2 = hard_gate_check(
            score_qualified=True,
            stress_passed=True,
            robustness_passed=True,
            dsr_decision=DSR_INSUFFICIENT_DATA,
            pbo_decision=PBO_PASSED,
        )
        assert ok2 is False
        assert reason2 == DSR_INSUFFICIENT_DATA


class TestPhase3CCampaign:
    def test_pipeline_level_and_vault_blocked(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, run_id="pipe_level")
        assert result.pipeline_level == PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST
        assert result.statistics_pipeline_complete is True
        assert result.clustering_pipeline_complete is True
        assert result.research_shortlist_pipeline_complete is True
        assert result.vault_pipeline_complete is False
        assert result.paper_pipeline_complete is False
        assert result.live_pipeline_complete is False
        assert result.post_wfo_pipeline_complete is False
        assert VAULT_NOT_RUN in result.post_wfo_blocked_reasons
        payload = result.as_dict()
        assert payload["finalists"] == []
        assert payload["promoted"] == []
        assert payload["vault_candidates"] == []
        assert payload["paper_candidates"] == []
        for claim in RESEARCH_SHORTLISTED_DOES_NOT_MEAN:
            assert claim in payload["research_shortlisted_does_not_mean"]

    def test_status_history_includes_statistics_and_clustering(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, run_id="status_hist")
        statuses = {e.new_status for e in result.candidate_status_history}
        assert STATISTICS_TESTED in statuses
        assert BEHAVIORALLY_CLUSTERED in statuses or any(
            e.new_status == "CLUSTERING_NOT_ENTERED" for e in result.candidate_status_history
        )
        for s in result.candidate_robustness_summaries:
            if s.final_decision != ROBUSTNESS_PASSED:
                continue
            events = [
                e for e in result.candidate_status_history if e.candidate_id == s.candidate_id
            ]
            names = [e.new_status for e in events]
            assert SCORE_QUALIFIED in names or STRESS_TESTED in names
            assert STRESS_PASSED in names
            assert ROBUSTNESS_PASSED in names
            assert STATISTICS_TESTED in names

    def test_statistics_artifacts_present(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, run_id="stats_arts")
        assert result.statistics_accounting is not None
        assert result.population_stats
        payload = result.as_dict()
        assert "candidate_statistics_summaries" in payload
        assert "statistics_accounting" in payload
        for s in result.candidate_statistics_summaries:
            assert s.dsr_payload.get("status") in {"OK", "INSUFFICIENT_DATA", "INVALID_INPUT"}
            if s.dsr_status == DSR_INSUFFICIENT_DATA:
                assert s.dsr_value is None
                assert s.dsr_reason

    def test_at_least_three_clusters_and_reproducible_shortlist(self, tmp_path: Path) -> None:
        result_a, _ = _run(tmp_path, run_id="val_a")
        result_b, _ = _run(tmp_path, run_id="val_b")
        assert len(result_a.clusters) >= 3, (
            f"expected >=3 clusters, got {len(result_a.clusters)}: {result_a.clusters}"
        )
        assert result_a.reproducible_fingerprint == result_b.reproducible_fingerprint
        assert result_a.as_dict()["clusters"] == result_b.as_dict()["clusters"]
        assert result_a.as_dict()["research_shortlist"] == result_b.as_dict()["research_shortlist"]
        # Validation requires a reproducible shortlist artifact (may be empty with reasons).
        assert "research_shortlist" in result_a.as_dict()
        assert result_a.research_shortlist_pipeline_complete is True
        for entry in result_a.research_shortlist:
            d = entry if isinstance(entry, dict) else entry.as_dict()
            assert d["vault_eligible"] is False
            assert d["paper_eligible"] is False
            assert d["live_eligible"] is False
            assert d["finalist"] is False
            assert "SCORE_QUALIFIED" in d["gates_passed"]
            assert "STRESS_PASSED" in d["gates_passed"]
            assert "ROBUSTNESS_PASSED" in d["gates_passed"]
            assert "DSR_PASSED" in d["gates_passed"]
            assert "PBO_PASSED" in d["gates_passed"]
            assert d.get("provenance")
        # Prefer non-empty shortlist when gates pass; assert funnel fields always.
        funnel = result_a.as_dict()["family_funnel"]
        assert all("research_shortlisted" in f for f in funnel)
        assert all("statistically_passed" in f for f in funnel)
        assert all("robustness_passed" in f for f in funnel)
        assert len(result_a.research_shortlist) >= 1, (
            f"expected reproducible shortlist entries, rejects={result_a.shortlist_rejects}"
        )

    def test_shortlist_rejects_carry_exact_reasons(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, run_id="reject_reasons", min_dsr=0.999999, max_pbo=0.0)
        assert result.research_shortlist_pipeline_complete is True
        if not result.research_shortlist:
            reasons = result.empty_collections_reasons.get("research_shortlist") or []
            assert reasons or result.shortlist_rejects
        for rej in result.shortlist_rejects:
            d = rej.as_dict() if hasattr(rej, "as_dict") else rej
            assert d["reason"]


class TestBehavioralSignatureArtifacts:
    def test_signature_uses_oos_returns_timing_exposure(self) -> None:
        from discovery.evaluator import EvaluationRecord, EvalOutcome
        from discovery.fitness import FitnessResult

        cand = build_candidate(
            entry_tree=op_node(
                OperatorId.GREATER_THAN,
                feature_node("price.return_5", ValueType.RETURN),
                parameter_node("thr", 0.0, ValueType.SCALAR),
            ),
            exit_tree=feature_node("price.simple_return_1", ValueType.BOOLEAN),
            stop=None,
            target=None,
            sizing=None,
            regime_gates=(),
            strategy_family="momentum",
            creation_method=CreationMethod.RANDOM,
            generation=0,
            parent_ids=(),
            grammar_version="g1",
            feature_set_version="f1",
            cost_model_version="c1",
            asset_universe=("ES",),
            random_seed=1,
        )
        folds = _folds_for_candidate(cand.candidate_id, n=24)
        closed = _closed_trades(cand.candidate_id, folds)
        rec = EvaluationRecord(
            outcome=EvalOutcome.REGISTERED,
            candidate_id=cand.candidate_id,
            lineage_id=cand.lineage_id,
            trial_id="t",
            fitness=FitnessResult(
                fitness=1.2,
                ranking_source="validation_oos",
                components={},
                fold_scores=(1.2,),
            ),
            rejection_reason=None,
            oos_folds=folds,
            train_metrics={
                "signal_source": "candidate_dsl_trees",
                "is_full_event_wfo": True,
                "wfo_completed_folds": len(folds),
            },
            meta={
                "is_full_wfo_completion": True,
                "baseline_wfo_artifacts": {
                    "candidate_id": cand.candidate_id,
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": len(folds),
                    "closed_trades": closed,
                },
            },
        )
        rets = extract_oos_return_series(rec)
        assert len(rets) >= 20
        series = extract_oos_statistics_series(rec)
        assert series.is_sufficient
        assert series.observation_type != OBS_INSUFFICIENT
        sig = build_behavioral_signature(cand, rec)
        assert sig is not None
        assert len(sig.signal_vector) >= 8
        assert len(sig.daily_pnl) >= 8
        assert sig.component_availability
        # daily_pnl must be time-aligned daily values, not timing∥exposure blend.
        timing, tmeta = extract_trade_timing_vector(rec)
        exposure, emeta = extract_exposure_vector(rec)
        if tmeta.get("available") and emeta.get("available"):
            blend = list(timing[:8]) + list(exposure[:8])
            assert not np.allclose(sig.daily_pnl[: len(blend)], blend)


class TestAdversarialIntegrity:
    """Adversarial proofs required by the Phase 3C final audit."""

    def test_single_candidate_cannot_produce_valid_pbo(self) -> None:
        from discovery.evaluator import EvaluationRecord, EvalOutcome
        from discovery.fitness import FitnessResult
        from metrics.pbo import PBOStatus

        folds = _folds_for_candidate("solo_cand", n=24)
        closed = _closed_trades("solo_cand", folds)
        rec = EvaluationRecord(
            outcome=EvalOutcome.REGISTERED,
            candidate_id="solo_cand",
            lineage_id="l_solo",
            trial_id="t_solo",
            fitness=FitnessResult(
                fitness=1.0,
                ranking_source="validation_oos",
                components={},
                fold_scores=(1.0,),
            ),
            rejection_reason=None,
            oos_folds=folds,
            train_metrics={
                "signal_source": "candidate_dsl_trees",
                "is_full_event_wfo": True,
                "wfo_completed_folds": len(folds),
            },
            meta={
                "is_full_wfo_completion": True,
                "baseline_wfo_artifacts": {
                    "candidate_id": "solo_cand",
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": len(folds),
                    "closed_trades": closed,
                },
            },
        )
        pop, _ids, series = build_full_wfo_trial_population([rec])
        assert len(series) == 1
        aligned = align_performance_matrix([rec], candidate_ids=["solo_cand"])
        assert isinstance(aligned, AlignedPerformanceMatrix)
        assert aligned.matrix is None
        dsr, pbo, _meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=pop,
            performance_matrix=aligned,
            trial_column_index=0,
            config=ResearchShortlistConfig(min_oos_observations_for_dsr=20),
        )
        assert pbo is not None
        assert pbo.status == PBOStatus.INSUFFICIENT_DATA
        assert pbo.pbo is None
        # Even if DSR were somehow passed, PBO insufficiency must still block shortlist.
        ok, _, reason = hard_gate_check(
            score_qualified=True,
            stress_passed=True,
            robustness_passed=True,
            dsr_decision=DSR_PASSED,
            pbo_decision="PBO_INSUFFICIENT_DATA",
        )
        assert ok is False
        assert reason == "PBO_INSUFFICIENT_DATA"

    def test_aggregate_only_pf_cannot_fabricate_statistics(self) -> None:
        from discovery.evaluator import EvaluationRecord, EvalOutcome
        from discovery.fitness import FitnessResult

        # No OOS folds, no closed trades — only a train-side PF aggregate claim.
        rec = EvaluationRecord(
            outcome=EvalOutcome.REGISTERED,
            candidate_id="agg_only",
            lineage_id="l_agg",
            trial_id="t_agg",
            fitness=FitnessResult(
                fitness=9.9,
                ranking_source="validation_oos",
                components={"profit_factor": 5.0, "expectancy": 1.0},
                fold_scores=(9.9,),
            ),
            rejection_reason=None,
            oos_folds=[],
            train_metrics={
                "signal_source": "candidate_dsl_trees",
                "is_full_event_wfo": True,
                "wfo_completed_folds": 3,
                "profit_factor": 5.0,
            },
            meta={
                "is_full_wfo_completion": True,
                "baseline_wfo_artifacts": {
                    "candidate_id": "agg_only",
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": 3,
                    "closed_trades": [],
                },
            },
        )
        pop, ids, series = build_full_wfo_trial_population([rec])
        assert ids == []
        assert series == []
        aligned = align_performance_matrix([rec], candidate_ids=["agg_only"])
        assert isinstance(aligned, AlignedPerformanceMatrix)
        assert aligned.matrix is None
        # Population with no scored trials still must not fabricate OK DSR/PBO.
        from metrics.trial_population import TrialSelection, build_trial_population

        empty_pop = build_trial_population(
            [],
            total_trials=1,
            rejected_trials=0,
            failed_trials=1,
            selection=TrialSelection.ALL_TRIALS,
        )
        dsr, pbo, _meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=empty_pop,
            performance_matrix=aligned,
            trial_column_index=None,
            config=ResearchShortlistConfig(min_oos_observations_for_dsr=20),
        )
        assert dsr.status.value == "INSUFFICIENT_DATA"
        assert dsr.deflated_sharpe is None
        assert pbo is None or pbo.status.value == "INSUFFICIENT_DATA"

    def test_clustering_uses_behavior_not_candidate_or_family_hash(self) -> None:
        from discovery.behavioral_dedup import behavioral_similarity
        from discovery.evaluator import EvaluationRecord, EvalOutcome
        from discovery.fitness import FitnessResult

        def _cand(cid: str, family: str, seed: int):
            return build_candidate(
                entry_tree=op_node(
                    OperatorId.GREATER_THAN,
                    feature_node("price.return_5", ValueType.RETURN),
                    parameter_node("thr", 0.0, ValueType.SCALAR),
                ),
                exit_tree=feature_node("price.simple_return_1", ValueType.BOOLEAN),
                stop=None,
                target=None,
                sizing=None,
                regime_gates=(),
                strategy_family=family,
                creation_method=CreationMethod.RANDOM,
                generation=0,
                parent_ids=(),
                grammar_version="g1",
                feature_set_version="f1",
                cost_model_version="c1",
                asset_universe=("ES",),
                random_seed=seed,
            )

        def _rec(cid: str, folds, closed):
            return EvaluationRecord(
                outcome=EvalOutcome.REGISTERED,
                candidate_id=cid,
                lineage_id=f"l_{cid}",
                trial_id=f"t_{cid}",
                fitness=FitnessResult(
                    fitness=1.0,
                    ranking_source="validation_oos",
                    components={},
                    fold_scores=(1.0,),
                ),
                rejection_reason=None,
                oos_folds=folds,
                train_metrics={
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": len(folds),
                },
                meta={
                    "is_full_wfo_completion": True,
                    "baseline_wfo_artifacts": {
                        "candidate_id": cid,
                        "signal_source": "candidate_dsl_trees",
                        "is_full_event_wfo": True,
                        "wfo_completed_folds": len(folds),
                        "closed_trades": closed,
                    },
                },
            )

        # Identical OOS behavior artifacts, different IDs / families.
        folds = _folds_for_candidate("behavior_twin_a", n=24)
        closed = _closed_trades("behavior_twin_a", folds)
        a = _cand("id_aaaa_family_x", "momentum", 1)
        b = _cand("id_bbbb_family_y", "mean_reversion", 2)
        # Force shared candidate_id on records for artifact extraction, then rename on signature via cand.
        rec_a = _rec(a.candidate_id, folds, closed)
        rec_b = _rec(b.candidate_id, folds, closed)
        # Overwrite baseline trades to the *same* behavioral series (not ID-derived).
        same_trades = list(closed)
        rec_a.meta["baseline_wfo_artifacts"]["closed_trades"] = same_trades
        rec_b.meta["baseline_wfo_artifacts"]["closed_trades"] = same_trades
        rec_a.oos_folds = folds
        rec_b.oos_folds = folds
        sig_a = build_behavioral_signature(a, rec_a)
        sig_b = build_behavioral_signature(b, rec_b)
        assert sig_a is not None and sig_b is not None
        assert sig_a.candidate_id != sig_b.candidate_id
        # Primary weights are OOS signal/PnL (0.8); feature Jaccard is only 0.2.
        assert np.allclose(sig_a.signal_vector, sig_b.signal_vector)
        assert np.allclose(sig_a.daily_pnl, sig_b.daily_pnl)
        twin_sim = behavioral_similarity(sig_a, sig_b)
        assert twin_sim >= 0.80

        # Different OOS behavior, similar ID prefixes / same family → must not force-cluster.
        folds_c = _folds_for_candidate("behavior_opposite", n=24)
        closed_c = _closed_trades("behavior_opposite", folds_c)
        c = _cand("id_aaaa_family_z", "momentum", 3)
        rec_c = _rec(c.candidate_id, folds_c, closed_c)
        sig_c = build_behavioral_signature(c, rec_c)
        assert sig_c is not None
        # Opposite path should be materially less similar than identical twins.
        assert behavioral_similarity(sig_a, sig_c) < behavioral_similarity(sig_a, sig_b)

    def test_cannot_skip_upstream_status_into_shortlist(self) -> None:
        for missing in (
            {"score_qualified": False, "stress_passed": True, "robustness_passed": True},
            {"score_qualified": True, "stress_passed": False, "robustness_passed": True},
            {"score_qualified": True, "stress_passed": True, "robustness_passed": False},
        ):
            ok, gates, reason = hard_gate_check(
                dsr_decision=DSR_PASSED,
                pbo_decision=PBO_PASSED,
                **missing,
            )
            assert ok is False
            assert reason is not None
            assert "PBO_PASSED" not in gates

    def test_dashboard_renders_report_fields_not_hardcoded_success(self) -> None:
        from pathlib import Path

        dash = (
            Path(__file__).resolve().parents[1]
            / "control_plane"
            / "static"
            / "assets"
            / "dashboard.js"
        )
        text = dash.read_text(encoding="utf-8")
        assert "report.research_shortlist" in text
        assert "report.clusters" in text
        assert "report.candidate_statistics_summaries" in text
        assert "report.candidate_status_history" in text
        assert "Vault / Paper / Live remain blocked" in text
        assert "vault_pipeline_complete" in text or "report.vault_pipeline_complete" in text or "VAULT" in text
        # Must not hardcode research shortlist / vault approval as always true.
        assert "vault_eligible: true" not in text.lower()
        assert 'vault_eligible: true' not in text
        assert "live_pipeline_complete = true" not in text


# ---------------------------------------------------------------------------
# Phase 3C.1 adversarial acceptance: statistical + behavioral integrity
# ---------------------------------------------------------------------------


def _rec_with_trades(
    cid: str,
    trades: list[dict[str, Any]],
    *,
    folds: list[FoldOOSMetrics] | None = None,
    spoof_full_wfo_flag: bool = False,
    signal_source: str = "candidate_dsl_trees",
    is_full_event_wfo: bool = True,
    baseline_candidate_id: str | None = None,
    extra_baseline: dict[str, Any] | None = None,
):
    from discovery.evaluator import EvaluationRecord, EvalOutcome
    from discovery.fitness import FitnessResult

    folds = folds or [
        FoldOOSMetrics(
            fold_id=i,
            expectancy=0.05,
            sharpe=1.0,
            profit_factor=1.3,
            calmar=0.4,
            max_drawdown=-0.03,
            n_trades=4,
        )
        for i in range(max(1, len({t.get("fold_id") for t in trades}) or 1))
    ]
    baseline = {
        "candidate_id": baseline_candidate_id if baseline_candidate_id is not None else cid,
        "signal_source": signal_source,
        "is_full_event_wfo": is_full_event_wfo,
        "wfo_completed_folds": len(folds),
        "closed_trades": trades,
    }
    if extra_baseline:
        baseline.update(extra_baseline)
    return EvaluationRecord(
        outcome=EvalOutcome.REGISTERED,
        candidate_id=cid,
        lineage_id=f"l_{cid}",
        trial_id=f"t_{cid}",
        fitness=FitnessResult(
            fitness=1.0,
            ranking_source="validation_oos",
            components={},
            fold_scores=(1.0,),
        ),
        rejection_reason=None,
        oos_folds=folds,
        train_metrics={
            "signal_source": signal_source,
            "is_full_event_wfo": is_full_event_wfo,
            "wfo_completed_folds": len(folds),
        },
        meta={
            "is_full_wfo_completion": True if spoof_full_wfo_flag else True,
            "baseline_wfo_artifacts": baseline,
        },
    )


class TestPhase3C1StatisticalBehavioralIntegrity:
    """Adversarial proofs for PHASE 3C.1 alignment + signature repairs."""

    def test_spoofed_full_wfo_flag_alone_cannot_pass_integrity(self) -> None:
        from discovery.evaluator import EvaluationRecord, EvalOutcome
        from discovery.fitness import FitnessResult

        # Flag set, but missing independent Full-WFO evidence.
        rec = EvaluationRecord(
            outcome=EvalOutcome.REGISTERED,
            candidate_id="spoof",
            lineage_id="l_spoof",
            trial_id="t_spoof",
            fitness=FitnessResult(
                fitness=9.0,
                ranking_source="validation_oos",
                components={},
                fold_scores=(9.0,),
            ),
            rejection_reason=None,
            oos_folds=[],
            train_metrics={},
            meta={"is_full_wfo_completion": True, "baseline_wfo_artifacts": {}},
        )
        assert _full_wfo_completed(rec) is False
        pop, ids, series = build_full_wfo_trial_population([rec])
        assert ids == []
        assert series == []
        cand = build_candidate(
            entry_tree=op_node(
                OperatorId.GREATER_THAN,
                feature_node("price.return_5", ValueType.RETURN),
                parameter_node("thr", 0.0, ValueType.SCALAR),
            ),
            exit_tree=feature_node("price.simple_return_1", ValueType.BOOLEAN),
            stop=None,
            target=None,
            sizing=None,
            regime_gates=(),
            strategy_family="momentum",
            creation_method=CreationMethod.RANDOM,
            generation=0,
            parent_ids=(),
            grammar_version="g1",
            feature_set_version="f1",
            cost_model_version="c1",
            asset_universe=("ES",),
            random_seed=1,
        )
        assert build_behavioral_signature(cand, rec) is None

        # Mismatched baseline candidate_id also fails even with trades + flag.
        folds = _folds_for_candidate("real", n=8)
        closed = _closed_trades("real", folds)
        bad = _rec_with_trades(
            "real",
            closed,
            folds=folds,
            spoof_full_wfo_flag=True,
            baseline_candidate_id="someone_else",
        )
        bad.meta["is_full_wfo_completion"] = True
        assert _full_wfo_completed(bad) is False
        pop2, ids2, _ = build_full_wfo_trial_population([bad])
        assert ids2 == []

    def test_pbo_rows_are_common_calendar_periods_not_trade_ordinals(self) -> None:
        # Candidate A: 2 trades on day1 + day3. Candidate B: 3 trades on day1,2,3.
        trades_a = [
            {
                "fold_id": 0,
                "net_pnl": 10.0,
                "qty": 1.0,
                "net_return": 0.01,
                "exit_time": "2024-03-01T10:00:00",
            },
            {
                "fold_id": 2,
                "net_pnl": 20.0,
                "qty": 1.0,
                "net_return": 0.02,
                "exit_time": "2024-03-03T11:00:00",
            },
        ]
        trades_b = [
            {
                "fold_id": 0,
                "net_pnl": 5.0,
                "qty": 1.0,
                "net_return": 0.005,
                "exit_time": "2024-03-01T09:00:00",
            },
            {
                "fold_id": 1,
                "net_pnl": 7.0,
                "qty": 1.0,
                "net_return": 0.007,
                "exit_time": "2024-03-02T09:00:00",
            },
            {
                "fold_id": 2,
                "net_pnl": 9.0,
                "qty": 1.0,
                "net_return": 0.009,
                "exit_time": "2024-03-03T09:00:00",
            },
        ]
        rec_a = _rec_with_trades("cand_a", trades_a)
        rec_b = _rec_with_trades("cand_b", trades_b)
        aligned = align_performance_matrix(
            [rec_a, rec_b], candidate_ids=["cand_a", "cand_b"]
        )
        assert isinstance(aligned, AlignedPerformanceMatrix)
        assert aligned.is_usable
        assert aligned.matrix is not None
        assert aligned.partition_frequency == PARTITION_WFO_VALIDATION_PERIOD
        assert aligned.common_time_index == ("fold:0", "fold:1", "fold:2")
        # Row 0 = fold 0 for BOTH candidates (not trade#0 vs trade#0 across unequal lengths).
        assert aligned.matrix.shape == (3, 2)
        assert aligned.missing_counts["cand_a"] == 1  # no fold:1
        assert aligned.fill_policy
        assert aligned.alignment_fingerprint
        # cand_a fold1 filled with 0 under zero-when-no-trade policy.
        assert float(aligned.matrix[1, 0]) == 0.0
        assert float(aligned.matrix[0, 0]) != 0.0
        assert float(aligned.matrix[0, 1]) != 0.0

    def test_different_trade_counts_do_not_ordinal_align(self) -> None:
        # Different trade counts + different timestamps: alignment must be by day.
        trades_a = [
            {
                "net_pnl": 1.0,
                "qty": 1.0,
                "net_return": 0.01,
                "exit_time": "2024-06-01T10:00:00",
            },
            {
                "net_pnl": 2.0,
                "qty": 1.0,
                "net_return": 0.02,
                "exit_time": "2024-06-05T10:00:00",
            },
        ]
        trades_b = [
            {
                "net_pnl": 9.0,
                "qty": 1.0,
                "net_return": 0.09,
                "exit_time": "2024-06-01T12:00:00",
            },
            {
                "net_pnl": 8.0,
                "qty": 1.0,
                "net_return": 0.08,
                "exit_time": "2024-06-02T12:00:00",
            },
            {
                "net_pnl": 7.0,
                "qty": 1.0,
                "net_return": 0.07,
                "exit_time": "2024-06-05T12:00:00",
            },
            {
                "net_pnl": 6.0,
                "qty": 1.0,
                "net_return": 0.06,
                "exit_time": "2024-06-06T12:00:00",
            },
        ]
        rec_a = _rec_with_trades("a", trades_a, folds=_folds_for_candidate("a", n=2))
        rec_b = _rec_with_trades("b", trades_b, folds=_folds_for_candidate("b", n=2))
        aligned = align_performance_matrix([rec_a, rec_b], candidate_ids=["a", "b"])
        assert aligned.is_usable and aligned.matrix is not None
        assert aligned.partition_frequency == PARTITION_TRADING_DAY
        assert "2024-06-01" in aligned.common_time_index
        assert "2024-06-05" in aligned.common_time_index
        # Prove NOT min-length ordinal truncate (that would yield shape (2, 2)).
        assert aligned.matrix.shape[0] == len(aligned.common_time_index)
        assert aligned.matrix.shape[0] >= 3
        # Legacy float-series API must refuse (trade-sequence only).
        assert align_performance_matrix([[0.1, 0.2], [0.3, 0.4, 0.5]]) is None

    def test_unalignable_data_produces_pbo_insufficient(self) -> None:
        # Trades with no timestamps and no fold_id → cannot time-align.
        trades = [{"net_pnl": 1.0, "qty": 1.0, "net_return": 0.01} for _ in range(5)]
        rec_a = _rec_with_trades("u1", trades)
        rec_b = _rec_with_trades("u2", list(trades))
        aligned = align_performance_matrix(
            [rec_a, rec_b], candidate_ids=["u1", "u2"]
        )
        assert isinstance(aligned, AlignedPerformanceMatrix)
        assert aligned.matrix is None
        assert aligned.reason is not None
        pop, ids, _ = build_full_wfo_trial_population([rec_a, rec_b])
        dsr, pbo, meta = evaluate_candidate_dsr_pbo(
            rec=rec_a,
            population=pop,
            performance_matrix=aligned,
            trial_column_index=0,
            config=ResearchShortlistConfig(min_oos_observations_for_dsr=2),
        )
        assert pbo is not None
        assert pbo.status.value == "INSUFFICIENT_DATA"
        assert pbo.pbo is None

    def test_dsr_never_uses_fold_expectancy_as_returns(self) -> None:
        folds = [
            FoldOOSMetrics(
                fold_id=i,
                expectancy=0.5,
                sharpe=2.0,
                profit_factor=2.0,
                calmar=1.0,
                max_drawdown=-0.01,
                n_trades=20,
            )
            for i in range(30)
        ]
        rec = _rec_with_trades("fold_only", [], folds=folds)
        series = extract_oos_statistics_series(rec)
        assert series.observation_type == OBS_INSUFFICIENT
        assert "fold_expectancy" in (series.reason or "")
        assert extract_oos_return_series(rec) == []
        pop, ids, _ = build_full_wfo_trial_population([rec])
        # No normalized series → not scored into population as a valid return trial.
        assert rec.candidate_id not in ids or series.observation_count == 0
        empty_pop = pop
        dsr, _pbo, meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=empty_pop,
            performance_matrix=None,
            trial_column_index=None,
            config=ResearchShortlistConfig(min_oos_observations_for_dsr=20, min_dsr=0.01),
        )
        assert dsr.status.value == "INSUFFICIENT_DATA"
        assert dsr.deflated_sharpe is None
        assert meta.get("observation_type") == OBS_INSUFFICIENT

    def test_raw_unnormalized_pnl_cannot_pass_as_returns(self) -> None:
        # Changing position sizes; only raw PnL — must NOT become a return series.
        trades = [
            {"net_pnl": 10.0, "exit_time": f"2024-04-{(i % 28) + 1:02d}T10:00:00", "fold_id": i}
            for i in range(30)
        ]
        # Deliberately omit qty/notional/normalized return fields.
        rec = _rec_with_trades("raw_pnl", trades)
        series = extract_oos_statistics_series(rec)
        assert series.observation_type == OBS_INSUFFICIENT
        assert extract_oos_return_series(rec) == []
        pop, ids, _ = build_full_wfo_trial_population([rec])
        assert "raw_pnl" not in ids
        dsr, _pbo, meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=pop,
            performance_matrix=None,
            trial_column_index=None,
            config=ResearchShortlistConfig(min_oos_observations_for_dsr=20),
        )
        assert dsr.status.value == "INSUFFICIENT_DATA"
        assert "unnormalized" in (meta.get("reason") or series.reason or "").lower() or (
            series.reason and "STATISTICS_INSUFFICIENT_DATA" in series.reason
        )

    def test_builtin_hash_absent_from_research_shortlist_production_paths(self) -> None:
        from pathlib import Path
        import ast

        root = Path(__file__).resolve().parents[1] / "discovery"
        targets = [
            root / "research_shortlist_pipeline.py",
            root / "behavioral_dedup.py",
            root / "multi_family_campaign.py",
        ]
        for path in targets:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "hash", f"builtin hash() in {path}"
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    # reject builtins.hash(...)
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "builtins"
                        and node.func.attr == "hash"
                    ):
                        raise AssertionError(f"builtins.hash in {path}")

    def test_invalid_timestamps_do_not_create_artificial_timing_evidence(self) -> None:
        trades = [
            {"net_pnl": 1.0, "qty": 1.0, "net_return": 0.01, "exit_time": "not-a-time"},
            {"net_pnl": 2.0, "qty": 1.0, "net_return": 0.02, "exit_time": "???bad???"},
            {"net_pnl": 3.0, "qty": 1.0, "net_return": 0.03},  # missing
        ]
        rec = _rec_with_trades("bad_ts", trades)
        timing, meta = extract_trade_timing_vector(rec, n_bins=16)
        assert meta.get("available") is False
        assert meta.get("status") == "timing_data_unavailable"
        assert timing == ()
        assert meta.get("skipped_invalid_or_missing") == 3

    def test_pnl_is_not_used_as_exposure(self) -> None:
        trades = [
            {
                "net_pnl": 1000.0,
                "realized_pnl": 1000.0,
                "exit_time": "2024-05-01T10:00:00",
                "net_return": 0.01,
            },
            {
                "net_pnl": 50.0,
                "exit_time": "2024-05-02T10:00:00",
                "net_return": 0.02,
            },
        ]
        rec = _rec_with_trades("no_exp", trades)
        exposure, meta = extract_exposure_vector(rec, n_bins=8)
        assert meta.get("available") is False
        assert exposure == ()
        assert meta.get("skipped_pnl_used_as_exposure") == 2

    def test_cross_process_behavioral_signature_identical(self, tmp_path: Path) -> None:
        import subprocess
        import sys
        import json as _json

        script = tmp_path / "sig_probe.py"
        script.write_text(
            """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(r'''%s''').resolve()))
from discovery.candidate import build_candidate
from discovery.expression_tree import feature_node, op_node, parameter_node
from discovery.operators import OperatorId
from discovery.types import CreationMethod, ValueType
from discovery.fitness import FoldOOSMetrics
from discovery.evaluator import EvaluationRecord, EvalOutcome
from discovery.fitness import FitnessResult
from discovery.research_shortlist_pipeline import build_behavioral_signature

folds = [
    FoldOOSMetrics(fold_id=i, expectancy=0.05, sharpe=1.1, profit_factor=1.3,
                   calmar=0.4, max_drawdown=-0.03, n_trades=4)
    for i in range(6)
]
trades = []
for i, f in enumerate(folds):
    trades.append({
        "fold_id": f.fold_id,
        "net_pnl": 1.0 + i * 0.1,
        "qty": 2.0,
        "net_return": 0.01 + i * 0.001,
        "exit_time": f"2024-07-{(i%%28)+1:02d}T{(10+i)%%24:02d}:00:00",
    })
cand = build_candidate(
    entry_tree=op_node(OperatorId.GREATER_THAN,
        feature_node("price.return_5", ValueType.RETURN),
        parameter_node("thr", 0.0, ValueType.SCALAR)),
    exit_tree=feature_node("price.simple_return_1", ValueType.BOOLEAN),
    stop=None, target=None, sizing=None, regime_gates=(),
    strategy_family="momentum", creation_method=CreationMethod.RANDOM,
    generation=0, parent_ids=(), grammar_version="g1",
    feature_set_version="f1", cost_model_version="c1",
    asset_universe=("ES",), random_seed=42,
)
rec = EvaluationRecord(
    outcome=EvalOutcome.REGISTERED,
    candidate_id=cand.candidate_id,
    lineage_id=cand.lineage_id,
    trial_id="t",
    fitness=FitnessResult(fitness=1.2, ranking_source="validation_oos",
                          components={}, fold_scores=(1.2,)),
    rejection_reason=None,
    oos_folds=folds,
    train_metrics={"signal_source":"candidate_dsl_trees","is_full_event_wfo":True,
                   "wfo_completed_folds":len(folds)},
    meta={"is_full_wfo_completion": True,
          "baseline_wfo_artifacts": {
              "candidate_id": cand.candidate_id,
              "signal_source": "candidate_dsl_trees",
              "is_full_event_wfo": True,
              "wfo_completed_folds": len(folds),
              "closed_trades": trades,
          }},
)
sig = build_behavioral_signature(cand, rec)
assert sig is not None
print(json.dumps(sig.as_dict(), sort_keys=True))
"""
            % str(Path(__file__).resolve().parents[1]).replace("\\", "\\\\"),
            encoding="utf-8",
        )
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        # Force different hash seeds across processes.
        results = []
        for seed in ("0", "random", "1"):
            e = dict(env)
            e["PYTHONHASHSEED"] = seed
            e["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                env=e,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr
            results.append(_json.loads(proc.stdout.strip()))
        assert results[0] == results[1] == results[2]
        assert results[0]["daily_pnl"]
        assert results[0]["signature_kind"]

