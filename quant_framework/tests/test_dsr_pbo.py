"""
Phase 5.5-I acceptance — DSR / PBO over the complete Experiment Registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metrics.dsr import DSRStatus, compute_deflated_sharpe
from metrics.pbo import PBOStatus, compute_pbo
from metrics.trial_population import TopNOnlyInputError, TrialSelection, build_trial_population
from registry import ExperimentRegistry
from registry.experiment_registry import TrialStatus


def _seed_registry(
    path: Path,
    *,
    scores: list[float],
    rejected: list[bool] | None = None,
    failed: int = 0,
) -> ExperimentRegistry:
    reg = ExperimentRegistry(path)
    rejected = rejected or [False] * len(scores)
    for i, (score, is_rej) in enumerate(zip(scores, rejected, strict=True)):
        reg.create_trial(
            candidate_id=f"c{i}",
            lineage_id=f"l{i}",
            strategy_family="mean_reversion_vwap_bb",
            parameters={"lookback": 10 + i},
            config_snapshot={"v": 1},
            system_version="0.5.5-phase5.5",
            data_hash="d",
            random_seed=42,
            cost_model_version="futures_cost_v1",
            code_hash="code",
            ranking_score=score,
            rejection_reason="failed_stability" if is_rej else None,
            trial_id=f"trial_{i}",
        )
    for j in range(failed):
        reg.create_trial(
            candidate_id=f"fail{j}",
            lineage_id=f"lf{j}",
            strategy_family="mean_reversion_vwap_bb",
            parameters={"lookback": 99},
            config_snapshot={"v": 1},
            system_version="0.5.5-phase5.5",
            data_hash="d",
            random_seed=42,
            cost_model_version="futures_cost_v1",
            code_hash="code",
            ranking_score=None,
            trial_status=TrialStatus.FAILED,
            failure_reason="engine_error",
            trial_id=f"failed_{j}",
        )
    return reg


class TestDSRRegistryIntegration:
    def test_insufficient_observations_returns_insufficient_data(self, tmp_path: Path) -> None:
        reg = _seed_registry(tmp_path / "r1", scores=[1.0, 0.5, -0.2])
        result = compute_deflated_sharpe(
            1.0, reg.trial_population(), n_observations=5, min_observations=20
        )
        assert result.status == DSRStatus.INSUFFICIENT_DATA
        assert result.deflated_sharpe is None
        assert result.reason
        assert result.total_trials == 3
        assert result.calculation_version

    def test_failed_trials_change_the_multiple_testing_adjustment(self, tmp_path: Path) -> None:
        base = _seed_registry(tmp_path / "base", scores=[1.2, 0.8, 0.3, -0.1, 0.5])
        with_failed = _seed_registry(
            tmp_path / "failed",
            scores=[1.2, 0.8, 0.3, -0.1, 0.5],
            failed=4,
        )
        dsr_base = compute_deflated_sharpe(1.2, base.trial_population(), n_observations=100)
        dsr_failed = compute_deflated_sharpe(
            1.2, with_failed.trial_population(), n_observations=100
        )
        assert dsr_base.status == DSRStatus.OK
        assert dsr_failed.status == DSRStatus.OK
        assert dsr_failed.total_trials > dsr_base.total_trials
        assert dsr_failed.failed_trials == 4
        # More trials raise expected max Sharpe → lower deflated Sharpe
        assert dsr_failed.expected_max_sharpe is not None
        assert dsr_base.expected_max_sharpe is not None
        assert dsr_failed.expected_max_sharpe > dsr_base.expected_max_sharpe
        assert dsr_failed.deflated_sharpe is not None
        assert dsr_base.deflated_sharpe is not None
        assert dsr_failed.deflated_sharpe <= dsr_base.deflated_sharpe

    def test_top_n_only_input_is_rejected(self) -> None:
        pop = build_trial_population(
            [2.0, 1.5, 1.0],
            total_trials=3,
            selection=TrialSelection.TOP_N,
        )
        with pytest.raises(TopNOnlyInputError):
            compute_deflated_sharpe(2.0, pop, n_observations=100)

    def test_identical_registry_data_produces_identical_results(self, tmp_path: Path) -> None:
        scores = [0.9, 0.4, -0.2, 1.1, 0.05, 0.7]
        a = _seed_registry(tmp_path / "a", scores=scores, rejected=[False, True, False, False, True, False])
        b = _seed_registry(tmp_path / "b", scores=scores, rejected=[False, True, False, False, True, False])
        out_a = a.evaluate_overfitting(observed_sharpe=1.1, n_observations=80)
        out_b = b.evaluate_overfitting(observed_sharpe=1.1, n_observations=80)
        assert out_a["dsr"] == out_b["dsr"]
        assert out_a["trial_population"] == out_b["trial_population"]
        assert out_a["dsr"]["status"] == DSRStatus.OK.value
        assert out_a["dsr"]["total_trials"] == 6
        assert out_a["dsr"]["rejected_trials"] == 2


class TestPBORegistryIntegration:
    def test_insufficient_splits_returns_insufficient_data(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.normal(0, 0.01, size=(12, 3))
        pop = build_trial_population([0.5, 0.2, -0.1], selection=TrialSelection.ALL_TRIALS)
        result = compute_pbo(matrix, population=pop, n_splits=2, min_splits=4)
        assert result.status == PBOStatus.INSUFFICIENT_DATA
        assert result.pbo is None

    def test_top_n_only_rejected_for_pbo(self) -> None:
        matrix = np.random.default_rng(1).normal(0, 0.01, size=(40, 3))
        pop = build_trial_population([1.0, 0.5, 0.2], selection=TrialSelection.TOP_N)
        with pytest.raises(TopNOnlyInputError):
            compute_pbo(matrix, population=pop, n_splits=4)

    def test_failed_trials_inflate_total_trial_count(self, tmp_path: Path) -> None:
        reg = _seed_registry(tmp_path / "pbo", scores=[0.8, 0.3, -0.1], failed=5)
        pop = reg.trial_population()
        assert pop.total_trials == 8
        assert pop.failed_trials == 5
        assert pop.scored_trials == 3
