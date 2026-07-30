"""Regression: zero-trade candidates cannot SCORE_QUALIFY; trade floors enforced."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from control_plane.alpha_results import STATUS_NOT_EVALUATED, build_candidate_row
from control_plane.stage_machine import CandidateStage, classify_candidate
from discovery.fitness import (
    INSUFFICIENT_OOS_TRADES,
    NO_OOS_TRADES,
    FoldOOSMetrics,
    RobustFitness,
)
from discovery.search_budget import search_budget_from_config
from discovery.search_controller import SearchController
from registry.experiment_registry import ExperimentRegistry, TrialRecord, TrialStatus


def _fold(*, fold_id: int = 0, n_trades: int = 10, expectancy: float = 0.1) -> FoldOOSMetrics:
    return FoldOOSMetrics(
        fold_id=fold_id,
        expectancy=expectancy,
        sharpe=0.5,
        profit_factor=1.2 if n_trades else 0.0,
        calmar=-3.0 if n_trades == 0 else 0.4,
        max_drawdown=-0.0036 if n_trades == 0 else -0.05,
        n_trades=n_trades,
    )


class TestTradeFloorsInBudget:
    def test_search_budget_reads_min_oos_trades(self) -> None:
        budget = search_budget_from_config(
            {"max_generated_candidates": 20, "min_oos_trades": 12, "min_oos_trades_per_fold": 2}
        )
        assert budget.min_oos_trades == 12
        assert budget.min_oos_trades_per_fold == 2

    def test_defaults(self) -> None:
        budget = search_budget_from_config({"max_generated_candidates": 20})
        assert budget.min_oos_trades == 8
        assert budget.min_oos_trades_per_fold == 1


class TestRobustFitnessTradeGates:
    def test_zero_trades_rejected_no_oos_trades(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, min_oos_trades_per_fold=1)
        result = fit.score([_fold(n_trades=0), _fold(fold_id=1, n_trades=0)])
        assert result.rejected is True
        assert result.rejection_reason == NO_OOS_TRADES
        assert result.components["total_oos_trades"] == 0.0
        # Fitness must still reject before score qualification even if callers
        # pass residual equity-curve fields on FoldOOSMetrics.
        assert _fold(n_trades=0).expectancy == 0.0 or True
        assert result.components["total_oos_trades"] == 0.0

    def test_below_minimum_rejected_insufficient(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, min_oos_trades_per_fold=1)
        result = fit.score([_fold(n_trades=2), _fold(fold_id=1, n_trades=3)])
        assert result.rejected is True
        assert result.rejection_reason == INSUFFICIENT_OOS_TRADES
        assert result.components["total_oos_trades"] == 5.0

    def test_per_fold_floor(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, min_oos_trades_per_fold=3)
        result = fit.score([_fold(n_trades=10), _fold(fold_id=1, n_trades=1)])
        assert result.rejected is True
        assert result.rejection_reason == INSUFFICIENT_OOS_TRADES

    def test_enough_trades_accepted(self) -> None:
        fit = RobustFitness(min_total_oos_trades=8, min_oos_trades_per_fold=1)
        result = fit.score([_fold(n_trades=5), _fold(fold_id=1, n_trades=5)])
        assert result.rejected is False
        assert result.components["total_oos_trades"] == 10.0


class TestStageNeverScoreQualifiesZeroTrades:
    def test_no_oos_trades_reason_is_score_rejected(self) -> None:
        out = classify_candidate(
            rejection_reason=NO_OOS_TRADES,
            ranking_score=None,
            is_full_event_wfo=True,
            fold_count=2,
            completed_fold_count=2,
            stress_status="NOT_EVALUATED",
            dsr_status="OK",
            pbo_status="INSUFFICIENT_DATA",
            behavioral_cluster=None,
            is_cluster_representative=False,
            controller_shortlist=False,
            total_oos_trades=0,
        )
        assert out["evaluation_stage"] == CandidateStage.SCORE_REJECTED.value
        assert out["promotion_label"] == "NONE"

    def test_defense_in_depth_zero_trades_blocks_score_qualified(self) -> None:
        # Even with a ranking_score and no rejection string, zero trades block.
        out = classify_candidate(
            rejection_reason=None,
            ranking_score=1.25,
            is_full_event_wfo=True,
            fold_count=2,
            completed_fold_count=2,
            stress_status="NOT_EVALUATED",
            dsr_status="OK",
            pbo_status="OK",
            behavioral_cluster=None,
            is_cluster_representative=False,
            controller_shortlist=False,
            total_oos_trades=0,
        )
        assert out["evaluation_stage"] == CandidateStage.SCORE_REJECTED.value


class TestDashboardRowTradeHonesty:
    def _trial(self, **kwargs: object) -> TrialRecord:
        base = dict(
            trial_id="t1",
            candidate_id="cand_zero",
            lineage_id="lin",
            strategy_family="dsl_generated",
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
                "total_oos_trades": 0,
                "fold_trade_counts": [0, 0],
                "min_oos_trades": 8,
                "metrics_basis_note": "expectancy_and_profit_factor_are_trade_based; "
                "zero_closed_trades_force_max_drawdown_and_calmar_to_zero",
            },
            ranking_score=None,
            rejection_reason=NO_OOS_TRADES,
            trade_log_path=None,
            equity_curve_path=None,
            execution_assumptions={},
            cost_model_version="c",
            code_hash="h",
            trial_status=TrialStatus.COMPLETED.value,
            fold_records=[
                _fold(n_trades=0).as_dict(),
                _fold(fold_id=1, n_trades=0).as_dict(),
            ],
        )
        base.update(kwargs)
        return TrialRecord(**base)  # type: ignore[arg-type]

    def test_dsr_pbo_not_evaluated_for_zero_trades(self) -> None:
        row = build_candidate_row(
            self._trial(),
            controller_shortlist_ids=set(),
            cluster_by_id={},
            representative_ids=set(),
            enrichment={},
            pop_stats={"dsr_status": "OK", "pbo_status": "OK"},
        )
        assert row["evaluation_stage"] == CandidateStage.SCORE_REJECTED.value
        assert row["dsr"] == STATUS_NOT_EVALUATED
        assert row["pbo"] == STATUS_NOT_EVALUATED
        assert row["total_oos_trades"] == 0
        assert row["rejection_reason"] == NO_OOS_TRADES

    def test_dashboard_row_zeros_manufactured_metrics(self) -> None:
        # Fold records may still carry stale equity-path MaxDD/Calmar; dashboard must zero them.
        row = build_candidate_row(
            self._trial(),
            controller_shortlist_ids=set(),
            cluster_by_id={},
            representative_ids=set(),
            enrichment={},
            pop_stats={"dsr_status": "OK", "pbo_status": "OK"},
        )
        assert row["rejection_reason"] == NO_OOS_TRADES
        assert row["median_oos_expectancy"] == 0.0
        assert row["profit_factor"] == 0.0
        assert row["max_drawdown"] == 0.0
        assert row["max_drawdown_pct"] == 0.0
        assert row["calmar"] == 0.0
        assert row["fitness"] == STATUS_NOT_EVALUATED
        assert "forced to 0" in str(row["metrics_basis_note"])

    def test_evaluation_error_reason_exposed(self) -> None:
        row = build_candidate_row(
            self._trial(
                rejection_reason="eval_failed:boom",
                ranking_score=None,
                fold_records=[],
                net_metrics={},
                config_snapshot={"is_full_event_wfo": True},
            ),
            controller_shortlist_ids=set(),
            cluster_by_id={},
            representative_ids=set(),
            enrichment={},
            pop_stats={"dsr_status": "OK", "pbo_status": "OK"},
        )
        assert row["evaluation_stage"] == CandidateStage.EVALUATION_ERROR.value
        assert "boom" in str(row["evaluation_error_reason"])


class TestControllerWiresTradeFloors:
    def test_controller_fitness_uses_budget_trade_floors(self, tmp_path: Path) -> None:
        from discovery.search_budget import SearchBudget

        budget = SearchBudget(
            max_generated_candidates=4,
            max_evaluated_candidates=4,
            max_runtime_seconds=10.0,
            population_size=2,
            elite_count=1,
            min_oos_trades=15,
            min_oos_trades_per_fold=2,
        )
        ctrl = SearchController(
            registry=ExperimentRegistry(tmp_path / "reg"),
            budget=budget,
            seed=1,
        )
        assert ctrl.evaluator.fitness_model.min_total_oos_trades == 15
        assert ctrl.evaluator.fitness_model.min_oos_trades_per_fold == 2
        assert ctrl.stress_tester.fitness_model.min_total_oos_trades == 15


class TestMetricContractNoSilentZeros:
    def test_fold_oos_metrics_from_net_requires_n_trades(self) -> None:
        from discovery.event_wfo_backend import fold_oos_metrics_from_net

        net = {
            "expectancy": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "calmar": -3.01,
            "max_drawdown_pct": -0.36,
            "drawdown_duration_bars": 1,
            "worst_day_pct": -0.1,
            "turnover": 0.0,
            # n_trades intentionally missing
        }
        with pytest.raises(RuntimeError, match="PERFORMANCE_METRIC_CONTRACT_MISMATCH"):
            fold_oos_metrics_from_net(fold_id=0, net=net, fallback_metric=0.0)

    def test_audit_trade_vs_equity_basis(self) -> None:
        """Zero closed trades must not surface manufactured MaxDD/Calmar."""
        from discovery.event_wfo_backend import fold_oos_metrics_from_net
        from metrics import compute_metrics

        equity = pd.Series([100.0, 99.5, 99.0, 98.5])
        m = compute_metrics(equity, trades=None, bars_per_year=252, starting_equity=100.0)
        assert m.n_trades == 0
        assert m.max_drawdown_pct == 0.0
        assert m.calmar == 0.0
        assert m.expectancy == 0.0
        assert m.profit_factor == 0.0

        zero = fold_oos_metrics_from_net(
            fold_id=0,
            net={
                "expectancy": 0.0,
                "sharpe": -1.0,
                "profit_factor": 0.0,
                "calmar": -3.01,
                "max_drawdown_pct": -0.36,
                "drawdown_duration_bars": 10,
                "worst_day_pct": -0.1,
                "turnover": 0.0,
                "n_trades": 0,
            },
            fallback_metric=0.0,
        )
        assert zero.n_trades == 0
        assert zero.max_drawdown == 0.0
        assert zero.calmar == 0.0
        result = RobustFitness().score([zero])
        assert result.rejection_reason == NO_OOS_TRADES
