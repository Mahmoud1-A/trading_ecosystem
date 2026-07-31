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
    align_performance_matrix,
    build_behavioral_signature,
    build_full_wfo_trial_population,
    evaluate_candidate_dsr_pbo,
    extract_oos_return_series,
    hard_gate_check,
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
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": 1,
                    "closed_trades": [],
                },
            },
        )
        pop, _ids, series = build_full_wfo_trial_population([rec, rec])
        cfg = ResearchShortlistConfig(min_oos_observations_for_dsr=20, min_dsr=0.95)
        matrix = align_performance_matrix(series)
        dsr, pbo, _meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=pop,
            performance_matrix=matrix,
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
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": len(folds),
                    "closed_trades": closed,
                },
            },
        )
        rets = extract_oos_return_series(rec)
        assert len(rets) >= 20
        sig = build_behavioral_signature(cand, rec)
        assert sig is not None
        assert len(sig.signal_vector) >= 8
        assert len(sig.daily_pnl) >= 8


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
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": len(folds),
                    "closed_trades": closed,
                },
            },
        )
        pop, _ids, series = build_full_wfo_trial_population([rec])
        assert len(series) == 1
        matrix = align_performance_matrix(series)
        assert matrix is None
        dsr, pbo, _meta = evaluate_candidate_dsr_pbo(
            rec=rec,
            population=pop,
            performance_matrix=matrix,
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
        matrix = align_performance_matrix(series)
        assert matrix is None
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
            performance_matrix=None,
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
