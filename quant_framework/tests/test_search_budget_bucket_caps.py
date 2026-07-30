"""Regression: family/tier/feature-family caps come from Search budget JSON."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from control_plane.app import create_app
from control_plane.run_manager import RunManager
from discovery.search_budget import (
    BUDGET_BUCKETS_EXHAUSTED,
    COMPLEXITY_TIER_CAP,
    FAMILY_CAP,
    FEATURE_FAMILY_CAP,
    BudgetCounters,
    SearchBudget,
    resolve_bucket_caps,
    search_budget_from_config,
)
from discovery.search_controller import SearchController
from fastapi.testclient import TestClient
from registry.experiment_registry import ExperimentRegistry


class TestResolveBucketCaps:
    def test_omitted_caps_default_to_max_generated(self) -> None:
        caps = resolve_bucket_caps(
            {"max_generated_candidates": 200},
            max_generated_candidates=200,
        )
        assert caps["requested"] == {
            "max_candidates_per_family": None,
            "max_candidates_per_complexity_tier": None,
            "max_candidates_per_feature_family": None,
        }
        assert caps["effective"] == {
            "max_candidates_per_family": 200,
            "max_candidates_per_complexity_tier": 200,
            "max_candidates_per_feature_family": 200,
        }

    def test_explicit_caps_are_honored(self) -> None:
        caps = resolve_bucket_caps(
            {
                "max_generated_candidates": 200,
                "max_candidates_per_family": 50,
                "max_candidates_per_complexity_tier": 75,
                "max_candidates_per_feature_family": 100,
            },
            max_generated_candidates=200,
        )
        assert caps["requested"]["max_candidates_per_family"] == 50
        assert caps["effective"]["max_candidates_per_family"] == 50
        assert caps["effective"]["max_candidates_per_complexity_tier"] == 75
        assert caps["effective"]["max_candidates_per_feature_family"] == 100

    def test_search_budget_from_config_campaign_200(self) -> None:
        budget = search_budget_from_config(
            {
                "max_generated_candidates": 200,
                "max_evaluated_candidates": 200,
                "max_full_wfo_evaluations": 200,
                "population_size": 4,
                "max_candidates_per_family": 200,
                "max_candidates_per_complexity_tier": 200,
                "max_candidates_per_feature_family": 200,
            }
        )
        assert budget.max_generated_candidates == 200
        assert budget.max_candidates_per_family == 200
        assert budget.max_candidates_per_complexity_tier == 200
        assert budget.max_candidates_per_feature_family == 200

    def test_from_config_defaults_caps_to_max_generated_not_20(self) -> None:
        budget = search_budget_from_config({"max_generated_candidates": 200})
        assert budget.max_candidates_per_family == 200
        assert budget.max_candidates_per_complexity_tier == 200
        assert budget.max_candidates_per_feature_family == 200
        assert budget.max_candidates_per_family != 20


class TestExactCapRejectionReasons:
    def test_family_cap_reason(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=100,
            max_candidates_per_family=1,
            max_candidates_per_complexity_tier=100,
            max_candidates_per_feature_family=100,
        )
        counters = BudgetCounters()
        assert (
            counters.record_generated(
                budget, family="mean_reversion", complexity=3.0, feature_ids=("ohlcv.close",)
            )
            is None
        )
        reason = counters.record_generated(
            budget, family="mean_reversion", complexity=3.0, feature_ids=("ohlcv.close",)
        )
        assert reason == FAMILY_CAP

    def test_complexity_tier_cap_reason(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=100,
            max_candidates_per_family=100,
            max_candidates_per_complexity_tier=1,
            max_candidates_per_feature_family=100,
        )
        counters = BudgetCounters()
        assert (
            counters.record_generated(
                budget, family="a", complexity=3.0, feature_ids=("ohlcv.close",)
            )
            is None
        )
        reason = counters.record_generated(
            budget, family="b", complexity=4.0, feature_ids=("ohlcv.open",)
        )
        assert reason == COMPLEXITY_TIER_CAP

    def test_feature_family_cap_reason(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=100,
            max_candidates_per_family=100,
            max_candidates_per_complexity_tier=100,
            max_candidates_per_feature_family=1,
        )
        counters = BudgetCounters()
        assert (
            counters.record_generated(
                budget, family="a", complexity=3.0, feature_ids=("ohlcv.close",)
            )
            is None
        )
        reason = counters.record_generated(
            budget, family="b", complexity=20.0, feature_ids=("ohlcv.open",)
        )
        assert reason == FEATURE_FAMILY_CAP

    def test_generic_budget_family_or_tier_cap_never_returned(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=50,
            max_candidates_per_family=1,
            max_candidates_per_complexity_tier=1,
            max_candidates_per_feature_family=1,
        )
        counters = BudgetCounters()
        counters.record_generated(
            budget, family="x", complexity=3.0, feature_ids=("ohlcv.close",)
        )
        for fam in ("x", "y", "z"):
            reason = counters.record_generated(
                budget, family=fam, complexity=3.0, feature_ids=("ohlcv.close",)
            )
            assert reason in {FAMILY_CAP, COMPLEXITY_TIER_CAP, FEATURE_FAMILY_CAP}
            assert reason != "budget_family_or_tier_cap"


class TestBucketExhaustionStopsGeneration:
    def test_stop_reason_fires_when_buckets_exhausted(self) -> None:
        budget = SearchBudget(
            max_generated_candidates=200,
            max_evaluated_candidates=200,
            max_runtime_seconds=30.0,
            max_candidates_per_family=1,
            max_candidates_per_complexity_tier=1,
            max_candidates_per_feature_family=1,
            population_size=2,
            stagnation_limit=100,
        )
        counters = BudgetCounters()
        counters.record_generated(
            budget, family="a", complexity=3.0, feature_ids=("ohlcv.close",)
        )
        # Fill rejection streak without accepting further candidates
        for i in range(20):
            counters.record_generated(
                budget, family="a", complexity=3.0, feature_ids=("ohlcv.close",)
            )
        assert counters.buckets_exhausted(budget) is True
        assert counters.stop_reason(budget) == BUDGET_BUCKETS_EXHAUSTED

    def test_controller_stops_without_endless_loop(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "registry")
        budget = SearchBudget(
            max_generated_candidates=200,
            max_evaluated_candidates=200,
            max_full_wfo_evaluations=200,
            max_runtime_seconds=20.0,
            max_candidates_per_family=2,
            max_candidates_per_complexity_tier=2,
            max_candidates_per_feature_family=2,
            population_size=4,
            elite_count=1,
            stagnation_limit=50,
            stagnation_generations=50,
            minimum_generations_before_stagnation=1,
        )
        ctrl = SearchController(registry=reg, budget=budget, seed=7)
        t0 = time.perf_counter()
        result = ctrl.run()
        elapsed = time.perf_counter() - t0
        assert elapsed < 15.0
        assert result.stop_reason in {
            BUDGET_BUCKETS_EXHAUSTED,
            "max_generated_candidates",
            "max_evaluated_candidates",
            "max_runtime_seconds",
            "stagnation_limit",
            "empty_population",
            "completed",
        }
        # Must not keep generating forever past bucket caps
        assert ctrl.counters.generated <= 6
        rejected = [
            t
            for t in reg.all_trials()
            if t.rejection_reason
            in {FAMILY_CAP, COMPLEXITY_TIER_CAP, FEATURE_FAMILY_CAP}
        ]
        if rejected:
            assert "budget_family_or_tier_cap" not in {
                t.rejection_reason for t in reg.all_trials()
            }


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    mgr = RunManager(tmp_path / "cp", max_workers=2)
    return TestClient(create_app(mgr, root=tmp_path / "cp"))


def _wait(client: TestClient, run_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/api/runs/{run_id}").json()
        if last["state"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"run {run_id} did not terminate: {last}")


class TestControlPlaneEffectiveCaps:
    def test_report_exposes_requested_and_effective_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "smoke_test": True,
                "dataset": "synthetic_demo",
                "random_seed": 13,
                "search_budget": {
                    "max_generated_candidates": 200,
                    "max_evaluated_candidates": 8,
                    "max_full_wfo_evaluations": 8,
                    "population_size": 2,
                    "max_runtime_seconds": 20,
                    "max_candidates_per_family": 200,
                    "max_candidates_per_complexity_tier": 200,
                    "max_candidates_per_feature_family": 200,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        finished = _wait(client, run_id)
        assert finished["state"] == "COMPLETED"
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        consumed = report["search_budget_consumed"]
        assert consumed["max_candidates_per_family_effective"] == 200
        assert consumed["max_candidates_per_complexity_tier_effective"] == 200
        assert consumed["max_candidates_per_feature_family_effective"] == 200
        assert consumed["max_candidates_per_family_requested"] == 200
        assert consumed["bucket_caps_effective"] == {
            "max_candidates_per_family": 200,
            "max_candidates_per_complexity_tier": 200,
            "max_candidates_per_feature_family": 200,
        }
        frozen = report.get("frozen_search_budget") or {}
        assert frozen["effective"]["max_candidates_per_family"] == 200
        dist = report.get("rejection_reason_distribution") or {}
        assert "budget_family_or_tier_cap" not in dist

    def test_omitted_caps_effective_equal_max_generated(self, client: TestClient) -> None:
        resp = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "smoke_test": True,
                "dataset": "synthetic_demo",
                "random_seed": 17,
                "search_budget": {
                    "max_generated_candidates": 48,
                    "max_evaluated_candidates": 6,
                    "max_full_wfo_evaluations": 6,
                    "population_size": 2,
                    "max_runtime_seconds": 15,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        _wait(client, run_id)
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        consumed = report["search_budget_consumed"]
        assert consumed["max_candidates_per_family_requested"] is None
        assert consumed["max_candidates_per_family_effective"] == 48
        assert consumed["max_candidates_per_complexity_tier_effective"] == 48
        assert consumed["max_candidates_per_feature_family_effective"] == 48
