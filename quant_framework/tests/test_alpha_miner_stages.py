"""Alpha Miner stage semantics — FINALIST requires institutional gates."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from control_plane.app import create_app
from control_plane.alpha_results import build_alpha_miner_report
from control_plane.run_manager import RunManager
from control_plane.stage_machine import CandidateStage, classify_candidate, map_terminal_reason
from discovery.evaluator import SyntheticOOSBackend
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.generator import CandidateGenerator
from discovery.search_budget import SearchBudget
from discovery.search_controller import SearchController
from fastapi.testclient import TestClient
from registry.experiment_registry import ExperimentRegistry
from validation.event_driven_wfo import proxy_metric_evaluate_fn


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "stage_cp"


@pytest.fixture()
def client(root: Path) -> TestClient:
    return TestClient(create_app(RunManager(root, max_workers=1), root=root))


def _wait(client: TestClient, run_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/api/runs/{run_id}").json()
        if last["state"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return last
        time.sleep(0.05)
    raise TimeoutError(last)


class TestStageMachineUnit:
    def test_no_finalist_without_full_wfo(self) -> None:
        out = classify_candidate(
            rejection_reason=None,
            ranking_score=1.2,
            is_full_event_wfo=False,
            fold_count=0,
            completed_fold_count=0,
            stress_status="PASSED",
            dsr_status="OK",
            pbo_status="OK",
            behavioral_cluster="beh_0",
            is_cluster_representative=True,
            controller_shortlist=True,
        )
        assert out["promotion_label"] != "FINALIST"
        assert out["evaluation_stage"] == CandidateStage.TOP_RANKED_UNVALIDATED.value
        assert out["vault_eligible"] is False

    def test_no_finalist_before_stress(self) -> None:
        out = classify_candidate(
            rejection_reason=None,
            ranking_score=1.2,
            is_full_event_wfo=True,
            fold_count=3,
            completed_fold_count=3,
            stress_status="NOT_EVALUATED",
            dsr_status="OK",
            pbo_status="OK",
            behavioral_cluster="beh_0",
            is_cluster_representative=True,
            controller_shortlist=True,
        )
        assert out["promotion_label"] != "FINALIST"
        assert out["vault_eligible"] is False

    def test_no_finalist_before_cluster(self) -> None:
        out = classify_candidate(
            rejection_reason=None,
            ranking_score=1.2,
            is_full_event_wfo=True,
            fold_count=3,
            completed_fold_count=3,
            stress_status="PASSED",
            dsr_status="OK",
            pbo_status="OK",
            behavioral_cluster=None,
            is_cluster_representative=True,
            controller_shortlist=True,
        )
        assert out["promotion_label"] != "FINALIST"

    def test_insufficient_stats_blocks_vault(self) -> None:
        out = classify_candidate(
            rejection_reason=None,
            ranking_score=1.2,
            is_full_event_wfo=True,
            fold_count=3,
            completed_fold_count=3,
            stress_status="PASSED",
            dsr_status="INSUFFICIENT_DATA",
            pbo_status="INSUFFICIENT_DATA",
            behavioral_cluster="beh_0",
            is_cluster_representative=True,
            controller_shortlist=True,
        )
        assert out["promotion_label"] == "RESEARCH_SHORTLISTED"
        assert out["vault_eligible"] is False
        assert out["paper_eligible"] is False

    def test_terminal_reason_mapping(self) -> None:
        assert map_terminal_reason("max_generated_candidates") == "GENERATED_BUDGET_EXHAUSTED"
        assert map_terminal_reason("max_evaluated_candidates") == "EVALUATED_BUDGET_EXHAUSTED"
        assert map_terminal_reason("max_full_wfo_evaluations") == "FULL_WFO_BUDGET_EXHAUSTED"
        assert map_terminal_reason("max_runtime_seconds") == "RUNTIME_EXHAUSTED"
        assert map_terminal_reason("stagnation_limit") == "STAGNATION_LIMIT"
        assert map_terminal_reason("completed") == "CONVERGENCE"
        assert map_terminal_reason(None, cancelled=True) == "CANCELLED"


class TestProxyCannotPromote:
    def test_proxy_metric_not_used_as_backend(self) -> None:
        assert "proxy_metric" in proxy_metric_evaluate_fn.__name__
        backend = SyntheticOOSBackend()
        assert backend.is_full_event_wfo is False
        assert backend.backend_kind == "synthetic_oos_probe"
        # proxy helper is not a BacktestBackend and must not be assigned
        assert not hasattr(proxy_metric_evaluate_fn, "is_full_event_wfo")

    def test_synthetic_smoke_zero_finalists(self, client: TestClient) -> None:
        resp = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "smoke_test": True,
                "dataset": "synthetic_demo",
                "random_seed": 3,
                "search_budget": {
                    "max_generated_candidates": 12,
                    "max_evaluated_candidates": 10,
                    "max_full_wfo_evaluations": 10,
                    "population_size": 4,
                    "max_runtime_seconds": 45,
                    "evaluation_backend": "synthetic_oos_probe",
                },
            },
        )
        run_id = resp.json()["run_id"]
        finished = _wait(client, run_id)
        assert finished["state"] == "COMPLETED"
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        assert report["full_wfo_evaluations"] == 0
        assert int(report["finalist_count"]) == 0
        assert isinstance(report["finalists"], int)
        assert report["discovery_result"] == "NO_QUALIFIED_CANDIDATE"
        assert "mandatory Alpha Miner gates" in (report.get("message_no_finalists") or "")
        for c in report["candidates"]:
            assert c["promotion_label"] != "FINALIST"
            if c.get("controller_shortlist") or c["candidate_id"] in set(
                report.get("controller_shortlist_ids") or []
            ):
                assert c["promotion_label"] in {
                    "TOP_RANKED_UNVALIDATED",
                    "SHORTLISTED",
                    "RESEARCH_SHORTLISTED",
                    "NONE",
                } or c.get("rejected")


class TestCountersAndDuplicates:
    def test_duplicate_counters_match_distinct_ids(self, client: TestClient) -> None:
        run_id = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "smoke_test": True,
                "dataset": "synthetic_demo",
                "random_seed": 9,
                "search_budget": {
                    "max_generated_candidates": 12,
                    "max_evaluated_candidates": 10,
                    "population_size": 4,
                    "max_runtime_seconds": 45,
                },
            },
        ).json()["run_id"]
        _wait(client, run_id)
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        cands = client.get(f"/api/runs/{run_id}/candidates").json()
        dup_rows = [
            c for c in cands["candidates"] if c.get("evaluation_stage") == "DUPLICATE"
        ]
        assert report["duplicate_candidates"] == len({c["candidate_id"] for c in dup_rows})
        assert cands.get("duplicate_count", report["duplicate_candidates"]) == report[
            "duplicate_candidates"
        ]

    def test_dashboard_counters_match_registry(self, client: TestClient) -> None:
        run_id = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "smoke_test": True,
                "dataset": "synthetic_demo",
                "random_seed": 5,
                "search_budget": {
                    "max_generated_candidates": 12,
                    "max_evaluated_candidates": 10,
                    "population_size": 4,
                    "max_runtime_seconds": 45,
                },
            },
        ).json()["run_id"]
        finished = _wait(client, run_id)
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        reg = ExperimentRegistry(Path(finished["artifact_dir"]) / "registry")
        distinct = {t.candidate_id: t for t in reg.all_trials()}
        assert report["generated_candidates"] == len(distinct)
        assert report["finalist_count"] == 0  # synthetic path


class TestInstitutionalPath:
    def test_event_driven_backend_marks_full_wfo(self, tmp_path: Path) -> None:
        backend = EventDrivenDiscoveryBackend()
        assert backend.is_full_event_wfo is True
        assert backend.backend_kind == "event_driven_wfo"
        gen = CandidateGenerator()
        cand = gen.seed_template_mean_reversion(seed=1)
        folds, train = backend.evaluate(cand)
        assert len(folds) >= 1
        assert train.get("is_full_event_wfo") is True
        assert train.get("proxy_metric_used") is False
        assert train.get("wfo_completed_folds", 0) >= 1

    def test_institutional_lifecycle_ordered(self, tmp_path: Path) -> None:
        """Small institutional-path SearchController run with event-driven backend."""
        reg = ExperimentRegistry(tmp_path / "reg")
        budget = SearchBudget(
            max_generated_candidates=4,
            max_evaluated_candidates=3,
            max_full_wfo_evaluations=3,
            max_stress_evaluations=2,
            population_size=2,
            elite_count=1,
            stagnation_limit=2,
            max_runtime_seconds=120,
            max_candidates_per_family=20,
            max_candidates_per_complexity_tier=20,
            max_candidates_per_feature_family=20,
        )
        ctrl = SearchController(
            registry=reg,
            budget=budget,
            seed=2,
            discovery_run_id="inst_1",
        )
        ctrl.evaluator.backend = EventDrivenDiscoveryBackend()
        result = ctrl.run()
        report = build_alpha_miner_report(
            registry=reg,
            counters=ctrl.counters,
            budget=budget,
            result=result,
            elapsed_seconds=1.0,
            evaluation_records=ctrl.evaluator.records(),
            evaluation_backend="event_driven_wfo",
        )
        assert report["full_wfo_evaluations"] >= 1
        assert isinstance(report["finalists"], int)
        # Lifecycle evidence on at least one evaluated candidate
        rows = [r for r in report["candidates"] if r.get("is_full_event_wfo")]
        assert rows
        row = rows[0]
        assert row["wfo"]["completed_fold_count"] >= 1
        assert row["backend_kind"] == "event_driven_wfo"
        # FINALIST only if all gates passed; otherwise shortlist/research labels
        for r in report["candidates"]:
            if r["promotion_label"] == "FINALIST":
                assert r["stress_status"] == "PASSED"
                assert r["behavioral_cluster"] not in {"NOT_EVALUATED", None, ""}
                assert r["is_full_event_wfo"] is True
                assert r["vault_eligible"] is True
            if r["dsr"] == "INSUFFICIENT_DATA" or r["pbo"] == "INSUFFICIENT_DATA":
                assert r["vault_eligible"] is False
                assert r["promotion_label"] != "FINALIST"
