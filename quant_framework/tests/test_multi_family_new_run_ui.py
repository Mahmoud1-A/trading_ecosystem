"""New Run UI + control-plane wiring for Strategy Family Generator (Multi-Family)."""

from __future__ import annotations

from pathlib import Path

import pytest
from control_plane.app import create_app
from control_plane.multi_family_ui import (
    MULTI_FAMILY_STRATEGY_FAMILY,
    preview_multi_family_specs,
    validate_multi_family_launch,
)
from control_plane.run_manager import RunManager
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    mgr = RunManager(tmp_path / "cp_mf", max_workers=1)
    return TestClient(create_app(mgr, root=tmp_path / "cp_mf"))


def _js(client: TestClient) -> str:
    return client.get("/assets/dashboard.js").text


class TestNewRunDashboardControls:
    def test_run_mode_and_multi_family_controls_present(self, client: TestClient) -> None:
        js = _js(client)
        assert 'value="single">Single Family' in js
        assert 'value="multi">Multi-Family Generated' in js
        assert 'id="mf_family_count"' in js
        assert 'id="mf_per_family"' in js
        assert 'id="mf_blueprints"' in js
        assert 'id="mf_adaptive"' in js
        assert "multi_family_generated" in js
        assert "Generated family specs" in js
        assert "collectRunBody" in js
        assert "strategy_family: multi ? \"multi_family_generated\"" in js
        assert "MULTI_FAMILY_TOO_FEW" in js

    def test_frozen_preview_includes_family_specs_path(self, client: TestClient) -> None:
        js = _js(client)
        assert "preview_family_specs" in js
        assert "/api/strategy_families/preview" in js
        assert "/api/strategy_families/blueprints" in js


class TestBlueprintsAndPreviewApi:
    def test_list_blueprints(self, client: TestClient) -> None:
        resp = client.get("/api/strategy_families/blueprints", params={"seed": 42})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["count"] >= 2
        assert data["multi_family_strategy_family"] == MULTI_FAMILY_STRATEGY_FAMILY
        ids = {b["family_id"] for b in data["blueprints"]}
        assert "mean_reversion" in ids
        assert "momentum" in ids
        fps = {b["effective_grammar_fingerprint"] for b in data["blueprints"]}
        assert len(fps) >= 2

    def test_preview_ok(self, client: TestClient) -> None:
        resp = client.post(
            "/api/strategy_families/preview",
            json={
                "seed": 7,
                "requested_family_count": 4,
                "family_ids": [
                    "mean_reversion",
                    "momentum",
                    "breakout",
                    "volatility_expansion",
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["strategy_family"] == MULTI_FAMILY_STRATEGY_FAMILY
        assert data["family_count"] == 4
        assert data["distinct_grammar_fingerprints"] >= 2
        assert len(data["families"]) == 4

    def test_preview_rejects_fewer_than_two_blueprints(self, client: TestClient) -> None:
        resp = client.post(
            "/api/strategy_families/preview",
            json={
                "seed": 7,
                "requested_family_count": 2,
                "family_ids": ["mean_reversion"],
            },
        )
        assert resp.status_code == 400
        assert "MULTI_FAMILY_TOO_FEW" in resp.json()["detail"]

    def test_preview_rejects_family_count_one(self, client: TestClient) -> None:
        resp = client.post(
            "/api/strategy_families/preview",
            json={"seed": 7, "requested_family_count": 1},
        )
        assert resp.status_code == 400
        assert "MULTI_FAMILY_TOO_FEW" in resp.json()["detail"]


class TestCreateRunMultiFamilyContract:
    def _mf_payload(self, **extra: object) -> dict:
        return {
            "run_type": "ALPHA_MINER",
            "confirm_alpha_miner": True,
            "smoke_test": True,
            "dataset": "synthetic_demo",
            "random_seed": 11,
            "strategy_family": MULTI_FAMILY_STRATEGY_FAMILY,
            "search_budget": {
                "max_generated_candidates": 24,
                "max_evaluated_candidates": 24,
                "max_full_wfo_evaluations": 8,
                "population_size": 4,
                "max_runtime_seconds": 20,
            },
            "multi_family": {
                "enabled": True,
                "requested_family_count": 4,
                "min_candidates_per_family": 6,
                "total_candidate_budget": 24,
                "max_full_wfo": 8,
                "adaptive_reallocation": True,
                "family_ids": [
                    "mean_reversion",
                    "momentum",
                    "breakout",
                    "volatility_expansion",
                ],
                "seed": 11,
            },
            **extra,
        }

    def test_launch_rejects_stale_strategy_family_labels(self, client: TestClient) -> None:
        for stale in ("mean_reversion_vwap_bb", "dsl_generated"):
            payload = self._mf_payload(strategy_family=stale)
            resp = client.post("/api/runs", json=payload)
            assert resp.status_code == 400, resp.text
            detail = str(resp.json().get("detail") or resp.text)
            assert "multi_family_generated" in detail or "must not remain" in detail

    def test_launch_rejects_single_blueprint(self, client: TestClient) -> None:
        payload = self._mf_payload()
        payload["multi_family"]["family_ids"] = ["mean_reversion"]
        payload["multi_family"]["requested_family_count"] = 2
        resp = client.post("/api/runs", json=payload)
        assert resp.status_code == 400, resp.text
        assert "MULTI_FAMILY_TOO_FEW" in str(resp.json().get("detail") or resp.text)

    def test_launch_accepts_and_stores_preview_specs(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json=self._mf_payload())
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        run = client.get(f"/api/runs/{run_id}").json()
        snap = run.get("config_snapshot") or {}
        assert snap.get("strategy_family") == MULTI_FAMILY_STRATEGY_FAMILY
        mf = snap.get("multi_family") or {}
        assert mf.get("enabled") is True
        assert int(mf.get("distinct_grammar_fingerprints") or 0) >= 2
        assert len(mf.get("preview_family_specs") or []) >= 2
        client.post(f"/api/runs/{run_id}/cancel")


class TestValidateHelpers:
    def test_validate_requires_multi_family_generated(self) -> None:
        with pytest.raises(ValueError, match="multi_family_generated"):
            validate_multi_family_launch(
                strategy_family="mean_reversion_vwap_bb",
                multi_family={"enabled": True, "requested_family_count": 4},
                random_seed=1,
            )

    def test_preview_helpers_fail_fast(self) -> None:
        with pytest.raises(ValueError, match="MULTI_FAMILY_TOO_FEW"):
            preview_multi_family_specs(seed=1, family_count=1, family_ids=None)
        with pytest.raises(ValueError, match="MULTI_FAMILY_TOO_FEW"):
            preview_multi_family_specs(
                seed=1, family_count=2, family_ids=["mean_reversion"]
            )
