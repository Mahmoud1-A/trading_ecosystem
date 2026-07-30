"""SCORE_QUALIFIED hard gates: expectancy, PF, trades, MaxDD — no Stress on reject."""

from __future__ import annotations

from pathlib import Path

import pytest
from control_plane.alpha_results import STATUS_NOT_EVALUATED, build_candidate_row
from control_plane.stage_machine import CandidateStage, classify_candidate
from discovery.fitness import (
    INSUFFICIENT_OOS_TRADES,
    MAX_DRAWDOWN_EXCEEDED,
    NEGATIVE_EXPECTANCY,
    NO_OOS_TRADES,
    PF_BELOW_ONE,
    SCORE_QUALIFIED_REJECT_REASONS,
    FoldOOSMetrics,
    RobustFitness,
)
from discovery.search_budget import search_budget_from_config
from discovery.search_controller import SearchController
from discovery.evaluator import EvalOutcome, EvaluationRecord
from registry.experiment_registry import ExperimentRegistry, TrialRecord, TrialStatus


def _fold(
    *,
    fold_id: int = 0,
    n_trades: int = 10,
    expectancy: float = 0.1,
    profit_factor: float = 1.5,
    max_drawdown: float = -0.05,
) -> FoldOOSMetrics:
    return FoldOOSMetrics(
        fold_id=fold_id,
        expectancy=expectancy,
        sharpe=0.5,
        profit_factor=profit_factor,
        calmar=0.4,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
    )


def _passing_folds() -> list[FoldOOSMetrics]:
    return [
        _fold(fold_id=0, expectancy=0.2, profit_factor=1.4, max_drawdown=-0.08, n_trades=10),
        _fold(fold_id=1, expectancy=0.15, profit_factor=1.3, max_drawdown=-0.06, n_trades=10),
    ]


class TestScoreQualifiedHardGates:
    def test_negative_expectancy_exact_reason(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.25)
        result = fit.score(
            [
                _fold(expectancy=-0.1, profit_factor=1.5, n_trades=10),
                _fold(fold_id=1, expectancy=-0.05, profit_factor=1.4, n_trades=10),
            ]
        )
        assert result.rejected is True
        assert result.rejection_reason == NEGATIVE_EXPECTANCY

    def test_pf_below_one_exact_reason(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.25)
        result = fit.score(
            [
                _fold(expectancy=0.1, profit_factor=0.9, n_trades=10),
                _fold(fold_id=1, expectancy=0.2, profit_factor=0.95, n_trades=10),
            ]
        )
        assert result.rejected is True
        assert result.rejection_reason == PF_BELOW_ONE

    def test_pf_exactly_one_rejected(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.25)
        result = fit.score(
            [
                _fold(expectancy=0.1, profit_factor=1.0, n_trades=10),
                _fold(fold_id=1, expectancy=0.2, profit_factor=1.0, n_trades=10),
            ]
        )
        assert result.rejected is True
        assert result.rejection_reason == PF_BELOW_ONE

    def test_insufficient_oos_trades_exact_reason(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.25)
        result = fit.score(
            [
                _fold(expectancy=0.2, profit_factor=1.5, n_trades=2),
                _fold(fold_id=1, expectancy=0.2, profit_factor=1.5, n_trades=3),
            ]
        )
        assert result.rejected is True
        assert result.rejection_reason == INSUFFICIENT_OOS_TRADES

    def test_max_drawdown_exceeded_exact_reason(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.10)
        result = fit.score(
            [
                _fold(expectancy=0.2, profit_factor=1.5, max_drawdown=-0.05, n_trades=10),
                _fold(fold_id=1, expectancy=0.2, profit_factor=1.5, max_drawdown=-0.15, n_trades=10),
            ]
        )
        assert result.rejected is True
        assert result.rejection_reason == MAX_DRAWDOWN_EXCEEDED

    def test_all_gates_pass(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.20)
        result = fit.score(_passing_folds())
        assert result.rejected is False
        assert result.rejection_reason is None
        assert result.fitness > float("-inf")

    def test_gate_order_trades_before_expectancy(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.20)
        result = fit.score(
            [
                _fold(expectancy=-1.0, profit_factor=0.1, n_trades=0),
                _fold(fold_id=1, expectancy=-1.0, profit_factor=0.1, n_trades=0),
            ]
        )
        assert result.rejection_reason == NO_OOS_TRADES

    def test_budget_reads_max_oos_drawdown(self) -> None:
        budget = search_budget_from_config(
            {"max_generated_candidates": 10, "max_oos_drawdown": 0.12}
        )
        assert budget.max_oos_drawdown == pytest.approx(0.12)

    def test_controller_syncs_drawdown_limit(self, tmp_path: Path) -> None:
        budget = search_budget_from_config(
            {
                "max_generated_candidates": 4,
                "max_evaluated_candidates": 4,
                "max_full_wfo_evaluations": 2,
                "min_oos_trades": 5,
                "max_oos_drawdown": 0.07,
                "stagnation_generations": 99,
                "population_size": 1,
            }
        )
        ctrl = SearchController(
            registry=ExperimentRegistry(tmp_path / "reg"),
            budget=budget,
            seed=1,
            skip_seed_template=True,
        )
        assert ctrl.evaluator.fitness_model.max_oos_drawdown == pytest.approx(0.07)
        assert ctrl.stress_tester.fitness_model.max_oos_drawdown == pytest.approx(0.07)


class TestScoreRejectBlocksStressAndStage:
    def test_stage_machine_maps_all_exact_reasons(self) -> None:
        for reason in SCORE_QUALIFIED_REJECT_REASONS:
            out = classify_candidate(
                rejection_reason=reason,
                ranking_score=None,
                is_full_event_wfo=True,
                fold_count=2,
                completed_fold_count=2,
                stress_status="NOT_EVALUATED",
                dsr_status="OK",
                pbo_status="OK",
                behavioral_cluster=None,
                is_cluster_representative=False,
                controller_shortlist=False,
                total_oos_trades=20,
            )
            assert out["evaluation_stage"] == CandidateStage.SCORE_REJECTED.value
            assert out["promotion_label"] == "NONE"

    def test_dashboard_row_marks_score_reject_not_evaluated_stats(self) -> None:
        trial = TrialRecord(
            trial_id="t1",
            candidate_id="cand_neg",
            lineage_id="lin",
            strategy_family="momentum",
            parameters={},
            config_snapshot={
                "is_full_event_wfo": True,
                "wfo_fold_count": 2,
                "wfo_completed_folds": 2,
                "backend_kind": "event_driven_wfo",
            },
            system_version="t",
            git_commit_hash="g",
            data_hash="d",
            random_seed=1,
            train_window=None,
            validation_window=None,
            vault_version=None,
            gross_metrics={},
            net_metrics={
                "total_oos_trades": 20,
                "fold_trade_counts": [10, 10],
                "min_oos_trades": 8,
            },
            ranking_score=None,
            rejection_reason=NEGATIVE_EXPECTANCY,
            trade_log_path=None,
            equity_curve_path=None,
            execution_assumptions={},
            cost_model_version="c",
            code_hash="h",
            trial_status=TrialStatus.COMPLETED.value,
            fold_records=[f.as_dict() for f in _passing_folds()],
        )
        row = build_candidate_row(
            trial,
            controller_shortlist_ids=set(),
            cluster_by_id={},
            representative_ids=set(),
            enrichment={},
            pop_stats={"dsr_status": "OK", "pbo_status": "OK"},
        )
        assert row["evaluation_stage"] == CandidateStage.SCORE_REJECTED.value
        assert row["rejection_reason"] == NEGATIVE_EXPECTANCY
        assert row["dsr"] == STATUS_NOT_EVALUATED
        assert row["pbo"] == STATUS_NOT_EVALUATED

    def test_rejected_record_is_not_registered_outcome(self) -> None:
        """Stress loop only iterates REGISTERED finalists — rejected must not qualify."""
        fit = RobustFitness(min_total_oos_trades=8, max_oos_drawdown=0.20)
        result = fit.score(
            [
                _fold(expectancy=-0.2, profit_factor=1.5, n_trades=10),
                _fold(fold_id=1, expectancy=-0.1, profit_factor=1.4, n_trades=10),
            ]
        )
        assert result.rejected
        assert result.rejection_reason == NEGATIVE_EXPECTANCY
        # Mimic evaluator: rejected fitness → EvalOutcome.REJECTED (not REGISTERED).
        rec = EvaluationRecord(
            outcome=EvalOutcome.REJECTED,
            candidate_id="c1",
            lineage_id="l1",
            trial_id="t1",
            fitness=result,
            rejection_reason=result.rejection_reason,
        )
        assert rec.outcome is not EvalOutcome.REGISTERED
        assert rec.rejection_reason in SCORE_QUALIFIED_REJECT_REASONS
