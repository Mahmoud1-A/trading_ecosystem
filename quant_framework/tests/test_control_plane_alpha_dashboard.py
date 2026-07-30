"""Phase 12 dashboard integration — Alpha Miner run lifecycle hardening."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from control_plane.app import create_app
from control_plane.run_manager import RunManager
from fastapi.testclient import TestClient


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "cp_alpha"


@pytest.fixture()
def client(root: Path) -> TestClient:
    mgr = RunManager(root, max_workers=2)
    return TestClient(create_app(mgr, root=root))


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/api/runs/{run_id}").json()
        if last["state"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"run {run_id} did not terminate: {last}")


def _start_miner(client: TestClient, **extra: object) -> str:
    payload = {
        "run_type": "ALPHA_MINER",
        "confirm_alpha_miner": True,
        "smoke_test": True,
        "dataset": "synthetic_demo",
        "random_seed": 11,
        "search_budget": {
            "max_generated_candidates": 12,
            "max_evaluated_candidates": 10,
            "max_full_wfo_evaluations": 10,
            "population_size": 4,
            "max_runtime_seconds": 45,
            "stagnation_limit": 3,
        },
        **extra,
    }
    resp = client.post("/api/runs", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


class TestRunListSemantics:
    def test_completed_visible_in_all_and_default(self, client: TestClient) -> None:
        run_id = client.post("/api/runs", json={"run_type": "RESEARCH_DEMO"}).json()["run_id"]
        _wait_terminal(client, run_id)
        all_runs = client.get("/api/runs", params={"filter": "ALL"}).json()["runs"]
        assert any(r["run_id"] == run_id for r in all_runs)
        completed = client.get("/api/runs", params={"filter": "COMPLETED"}).json()["runs"]
        assert any(r["run_id"] == run_id for r in completed)
        default = client.get("/api/runs", params={"filter": "DEFAULT"}).json()["runs"]
        assert any(r["run_id"] == run_id for r in default)

    def test_active_filter_excludes_completed(self, client: TestClient) -> None:
        run_id = client.post("/api/runs", json={"run_type": "RESEARCH_DEMO"}).json()["run_id"]
        _wait_terminal(client, run_id)
        active = client.get("/api/runs", params={"filter": "ACTIVE"}).json()["runs"]
        assert all(
            r["state"] not in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"} for r in active
        )
        assert not any(r["run_id"] == run_id for r in active)


class TestTerminalCancel:
    def test_cancel_unavailable_for_terminal(self, client: TestClient) -> None:
        run_id = client.post("/api/runs", json={"run_type": "RESEARCH_DEMO"}).json()["run_id"]
        finished = _wait_terminal(client, run_id)
        assert finished["state"] == "COMPLETED"
        assert finished.get("cancellable") is False
        resp = client.post(f"/api/runs/{run_id}/cancel")
        assert resp.status_code == 409


class TestAlphaMinerDashboard:
    def test_counters_persist_after_completion(self, client: TestClient) -> None:
        run_id = _start_miner(client)
        finished = _wait_terminal(client, run_id)
        assert finished["state"] == "COMPLETED"
        assert finished["generated_count"] > 0
        assert finished["evaluated_count"] > 0
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        assert report["generated_candidates"] == finished["generated_count"]
        assert report["evaluated_candidates"] == finished["evaluated_count"]
        assert "search_budget_consumed" in report
        assert report["elapsed_time"] >= 0
        assert "throughput_per_second" in report
        assert report["terminal_reason"]
        summary = client.get(f"/api/runs/{run_id}/summary").json()
        assert summary["software_execution_status"] == "SUCCESS"
        assert summary["discovery_result"] in {
            "FINALISTS_FOUND",
            "NO_FINALISTS",
            "NO_QUALIFIED_CANDIDATE",
            "BUDGET_EXHAUSTED",
        }
        assert summary["profitability_result"] in {
            "POSITIVE",
            "NEGATIVE",
            "FLAT",
            "NOT_AVAILABLE",
        }
        assert summary["statistical_result"] in {
            "PASSED",
            "FAILED",
            "INSUFFICIENT_DATA",
            "NOT_EVALUATED",
        }
        assert summary["vault_result"] in {
            "NOT_SUBMITTED",
            "NOT_ELIGIBLE",
            "PASSED",
            "FAILED",
        }

    def test_candidates_survive_restart(self, root: Path, client: TestClient) -> None:
        run_id = _start_miner(client)
        _wait_terminal(client, run_id)
        before = client.get(f"/api/runs/{run_id}/candidates").json()
        assert before["registry_total"] >= 1
        mgr2 = RunManager(root, max_workers=1)
        client2 = TestClient(create_app(mgr2, root=root))
        after = client2.get(f"/api/runs/{run_id}/candidates").json()
        assert after["registry_total"] == before["registry_total"]
        assert after["rejected_visible"] is True
        report = client2.get(f"/api/runs/{run_id}/alpha_miner").json()
        assert report["generated_candidates"] >= 1

    def test_no_finalist_status(self, client: TestClient) -> None:
        run_id = _start_miner(client)
        finished = _wait_terminal(client, run_id)
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        summary = client.get(f"/api/runs/{run_id}/summary").json()
        # Default dashboard backend is synthetic — true FINALIST count must be 0
        assert int(report["finalist_count"]) == 0
        assert report["qualified_candidate_status"] == "NO_QUALIFIED_CANDIDATE"
        assert "mandatory Alpha Miner gates" in (report.get("message_no_finalists") or "")
        assert summary["discovery_result"] in {
            "NO_QUALIFIED_CANDIDATE",
            "NO_FINALISTS",
            "BUDGET_EXHAUSTED",
        }
        assert isinstance(report["finalists"], int)

    def test_missing_stats_not_dash(self, client: TestClient) -> None:
        run_id = _start_miner(client)
        _wait_terminal(client, run_id)
        cands = client.get(f"/api/runs/{run_id}/candidates").json()["candidates"]
        assert cands
        for c in cands:
            assert c.get("dsr") != "-"
            assert c.get("pbo") != "-"
            assert str(c.get("dsr")) in {
                "NOT_EVALUATED",
                "NOT_APPLICABLE",
                "INSUFFICIENT_DATA",
                "OK",
                "FAILED",
            } or isinstance(c.get("dsr"), (int, float))
        summary = client.get(f"/api/runs/{run_id}/summary").json()
        assert summary["statistical_result"] != "-"
        assert summary["statistical_result"] in {
            "PASSED",
            "FAILED",
            "INSUFFICIENT_DATA",
            "NOT_EVALUATED",
        }

    def test_sse_reconnect_missed_miner_events(self, client: TestClient) -> None:
        run_id = _start_miner(client)
        _wait_terminal(client, run_id)
        events = client.get(f"/api/runs/{run_id}/events").json()["events"]
        types = {e["event_type"] for e in events}
        assert "RUN_STARTED" in types
        assert "RUN_COMPLETED" in types
        assert "GENERATION_STARTED" in types
        assert "CANDIDATE_GENERATED" in types or "CANDIDATE_EVALUATED" in types
        assert "FINALIST_SELECTED" in types
        mid = events[len(events) // 3]["seq"]
        missed = client.get(
            f"/api/runs/{run_id}/events", params={"after_seq": mid}
        ).json()["events"]
        assert missed
        assert all(e["seq"] > mid for e in missed)

    def test_counters_match_registry(self, client: TestClient) -> None:
        run_id = _start_miner(client)
        finished = _wait_terminal(client, run_id)
        cands = client.get(f"/api/runs/{run_id}/candidates").json()
        from registry.experiment_registry import ExperimentRegistry

        reg = ExperimentRegistry(Path(finished["artifact_dir"]) / "registry")
        distinct = {t.candidate_id: t for t in reg.all_trials()}
        assert cands["registry_total"] == len(distinct)
        report = client.get(f"/api/runs/{run_id}/alpha_miner").json()
        assert report["generated_candidates"] == len(distinct)
        assert report["duplicate_candidates"] == sum(
            1
            for t in distinct.values()
            if t.rejection_reason and "duplicate" in t.rejection_reason.lower()
        )
        assert report["finalist_count"] == sum(
            1 for c in cands["candidates"] if c.get("promotion_label") == "FINALIST"
        )

    def test_small_run_remains_visible(self, client: TestClient) -> None:
        run_id = _start_miner(client)
        _wait_terminal(client, run_id)
        active = client.get("/api/runs", params={"filter": "ACTIVE"}).json()["runs"]
        assert not any(r["run_id"] == run_id for r in active)
        recent = client.get("/api/runs", params={"filter": "DEFAULT"}).json()["runs"]
        assert any(r["run_id"] == run_id and r["state"] == "COMPLETED" for r in recent)
