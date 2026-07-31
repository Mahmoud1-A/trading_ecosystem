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

_PIPELINE_FIELDS = (
    "family_local_evolution",
    "evolution_generations",
    "stagnation_generations",
    "minimum_improvement",
    "allow_cross_family_crossover",
    "stress_scenarios",
    "max_stress_evaluations",
    "max_stress_scenarios_per_candidate",
    "min_stress_pass_rate",
    "fail_closed_unsupported_stress",
    "max_robustness_candidates",
    "max_robustness_evaluations",
    "max_parameters_per_candidate",
    "max_points_per_parameter",
    "min_valid_neighborhood_points",
    "allow_one_sided_neighborhood",
    "min_dsr",
    "max_pbo",
    "pbo_n_splits",
    "min_oos_observations_for_dsr",
    "behavioral_similarity_threshold",
)

_FROZEN_PREVIEW_MARKERS = (
    '"requested_family_count": 8',
    '"min_candidates_per_family": 6',
    '"total_candidate_budget": 96',
    '"max_evaluated_candidates": 96',
    '"max_full_wfo": 32',
    '"family_local_evolution": true',
    '"evolution_generations": 3',
    '"max_stress_evaluations": 24',
    '"max_robustness_candidates": 3',
    '"min_dsr": 0.95',
    '"max_pbo": 0.5',
)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    mgr = RunManager(tmp_path / "cp_mf", max_workers=1)
    return TestClient(create_app(mgr, root=tmp_path / "cp_mf"))


def _js(client: TestClient) -> str:
    return client.get("/assets/dashboard.js").text


def _collect_run_body_fn(js: str) -> str:
    start = js.index("function collectRunBody()")
    end = js.index("\nconst RUN_TABS", start)
    return js[start:end]


class TestNewRunDashboardControls:
    def test_run_mode_and_multi_family_controls_present(self, client: TestClient) -> None:
        js = _js(client)
        assert 'value="single">Single Family' in js
        assert 'value="multi">Multi-Family Generated' in js
        assert 'id="mf_family_count"' in js
        assert 'id="mf_per_family"' in js
        assert 'id="mf_total_budget"' in js
        assert "Initial candidates per family" in js
        assert "Total candidate budget" in js
        assert "<label>Candidates per family</label>" not in js
        assert 'id="mf_blueprints"' in js
        assert 'id="mf_adaptive"' in js
        assert 'id="mf_family_local_evolution"' in js
        assert 'id="mf_evolution_generations"' in js
        assert 'id="mf_stagnation_generations"' in js
        assert 'id="mf_minimum_improvement"' in js
        assert 'id="mf_stress_scenarios"' in js
        assert 'id="mf_max_stress_evaluations"' in js
        assert 'id="mf_max_stress_scenarios_per_candidate"' in js
        assert 'id="mf_min_stress_pass_rate"' in js
        assert 'id="mf_max_robustness_candidates"' in js
        assert 'id="mf_max_robustness_evaluations"' in js
        assert 'id="mf_max_parameters_per_candidate"' in js
        assert 'id="mf_max_points_per_parameter"' in js
        assert 'id="mf_min_valid_neighborhood_points"' in js
        assert 'id="mf_min_dsr"' in js
        assert 'id="mf_max_pbo"' in js
        assert 'id="mf_pbo_n_splits"' in js
        assert 'id="mf_min_oos_observations_for_dsr"' in js
        assert 'id="mf_behavioral_similarity_threshold"' in js
        assert "multi_family_generated" in js
        assert "Generated family specs" in js
        assert "collectRunBody" in js
        assert "strategy_family: multi ? \"multi_family_generated\"" in js
        assert "MULTI_FAMILY_TOO_FEW" in js
        assert "MULTI_FAMILY_INVALID" in js

    def test_frozen_preview_includes_family_specs_path(self, client: TestClient) -> None:
        js = _js(client)
        assert "preview_family_specs" in js
        assert "/api/strategy_families/preview" in js
        assert "/api/strategy_families/blueprints" in js

    def test_frozen_preview_includes_pipeline_defaults(self, client: TestClient) -> None:
        js = _js(client)
        assert "Frozen config preview" in js
        assert "JSON.stringify(body, null, 2)" in js
        body_fn = _collect_run_body_fn(js)
        # Defaults wired into multi_family so frozen preview shows them.
        assert "family_local_evolution:" in body_fn
        assert 'mfNum("mf_evolution_generations", 3)' in body_fn
        assert 'mfNum("mf_max_stress_evaluations", 24)' in body_fn
        assert 'mfNum("mf_max_robustness_candidates", 3)' in body_fn
        assert 'mfNum("mf_min_dsr", 0.95)' in body_fn
        assert 'mfNum("mf_max_pbo", 0.5)' in body_fn
        assert 'id="mf_family_local_evolution" checked' in js
        assert 'id="mf_family_count" type="number" min="2" max="8" value="8"' in js
        assert 'id="mf_per_family" type="number" min="1" max="100" value="6"' in js
        assert 'id="mf_total_budget" type="number" min="1" max="10000" value="96"' in js
        assert 'id="mf_evolution_generations" type="number" min="1" max="50" value="3"' in js
        assert 'id="mf_max_stress_evaluations" type="number" min="0" max="500" value="24"' in js
        assert 'id="mf_max_robustness_candidates" type="number" min="0" max="50" value="3"' in js
        assert 'id="mf_min_dsr" type="number" min="0" max="5" step="0.01" value="0.95"' in js
        assert 'id="mf_max_pbo" type="number" min="0" max="1" step="0.01" value="0.5"' in js
        # Campaign 0.1: 8×6 initial = 48, total budget 96 → 48 evolutionary slots;
        # max_full_wfo = floor(96/3) = 32; max_evaluated_candidates = 96.
        assert "max_evaluated_candidates: totalBudget" in body_fn
        assert "Math.floor(totalBudget / 3)" in body_fn
        for marker in _FROZEN_PREVIEW_MARKERS:
            # Proof markers: UI defaults + collectRunBody wiring produce these keys.
            key = marker.split(":")[0].strip().strip('"')
            assert key in body_fn or key in js

    def test_collect_run_body_sends_complete_multifamily_pipeline(self, client: TestClient) -> None:
        body_fn = _collect_run_body_fn(_js(client))
        assert "body.multi_family = {" in body_fn
        for field in _PIPELINE_FIELDS:
            assert f"{field}:" in body_fn or f'"{field}"' in body_fn
        assert "allow_cross_family_crossover: false" in body_fn
        assert "fail_closed_unsupported_stress: true" in body_fn
        assert "allow_one_sided_neighborhood: false" in body_fn
        # Must live inside multi_family, not only search_budget.
        mf_block = body_fn[body_fn.index("body.multi_family = {") :]
        for field in (
            "family_local_evolution",
            "max_stress_evaluations",
            "max_robustness_candidates",
            "min_dsr",
            "max_pbo",
            "stress_scenarios",
        ):
            assert field in mf_block
        budget_assign = body_fn[
            body_fn.index("body.search_budget = {") : body_fn.index("body.multi_family = {")
        ]
        assert "family_local_evolution" not in budget_assign
        assert "min_dsr" not in budget_assign
        assert "max_stress_evaluations" not in budget_assign

    def test_initial_population_and_total_budget_are_independent(self, client: TestClient) -> None:
        js = _js(client)
        body_fn = _collect_run_body_fn(js)
        assert 'id="mf_total_budget"' in js
        assert "mf_total_budget" in body_fn
        assert "total_candidate_budget: totalBudget" in body_fn
        assert "min_candidates_per_family: perFamily" in body_fn
        # Must NOT auto-derive total as family_count * initial candidates.
        assert "totalBudget = perFamily * familyCount" not in body_fn
        assert "totalBudget = familyCount * perFamily" not in body_fn
        assert "Independent of initial population" in body_fn
        assert "|| 96)" in body_fn

    def test_campaign_01_budget_defaults_accepted(self, client: TestClient) -> None:
        js = _js(client)
        body_fn = _collect_run_body_fn(js)
        # 8 families × 6 initial with total budget 96 is the accepted Campaign 0.1 shape.
        assert 'id="mf_family_count" type="number" min="2" max="8" value="8"' in js
        assert 'id="mf_per_family" type="number" min="1" max="100" value="6"' in js
        assert 'id="mf_total_budget" type="number" min="1" max="10000" value="96"' in js
        assert "total_candidate_budget: totalBudget" in body_fn
        assert "min_candidates_per_family: perFamily" in body_fn
        assert "requested_family_count: familyCount" in body_fn
        # Validation accepts when total >= initial population (8*6=48 <= 96).
        assert "totalBudget >= initialPop" in js
        assert "total candidate budget is smaller than the initial population" in js

    def test_total_budget_below_initial_population_rejected(self, client: TestClient) -> None:
        js = _js(client)
        assert "function multiFamilyBudgetError" in js
        assert (
            "MULTI_FAMILY_INVALID: total candidate budget is smaller than the initial population"
            in js
        )
        assert "multiFamilyBudgetError(body)" in js
        # Start is blocked when validation fails.
        start_idx = js.index('$("#start_btn").onclick')
        start_chunk = js[start_idx : start_idx + 900]
        assert "mfBudgetErr" in start_chunk
        assert "return;" in start_chunk

    def test_total_budget_above_max_generated_syncs_upward(self, client: TestClient) -> None:
        body_fn = _collect_run_body_fn(_js(client))
        # Explicit synchronization: raise max_generated_candidates to at least totalBudget.
        assert "Math.max(rawMaxGen, totalBudget)" in body_fn
        assert "max_generated_candidates: maxGenerated" in body_fn
        assert "synchronize generated cap upward" in body_fn

    def test_frozen_preview_contains_independent_budget_values(self, client: TestClient) -> None:
        js = _js(client)
        body_fn = _collect_run_body_fn(js)
        for key in (
            "requested_family_count",
            "min_candidates_per_family",
            "total_candidate_budget",
            "max_evaluated_candidates",
            "max_full_wfo",
            "family_local_evolution",
            "evolution_generations",
        ):
            assert key in body_fn
        assert 'id="mf_family_count" type="number" min="2" max="8" value="8"' in js
        assert 'id="mf_per_family" type="number" min="1" max="100" value="6"' in js
        assert 'id="mf_total_budget" type="number" min="1" max="10000" value="96"' in js
        assert 'id="mf_evolution_generations" type="number" min="1" max="50" value="3"' in js
        assert "max_evaluated_candidates: totalBudget" in body_fn
        assert "Math.floor(totalBudget / 3)" in body_fn

    def test_single_family_requests_omit_multi_family(self, client: TestClient) -> None:
        body_fn = _collect_run_body_fn(_js(client))
        assert "const multi = isMultiFamilyMode();" in body_fn
        assert "if (multi) {" in body_fn
        # multi_family is only assigned inside the multi branch.
        before_multi = body_fn[: body_fn.index("if (multi) {")]
        assert "body.multi_family" not in before_multi
        assert 'strategy_family: multi ? "multi_family_generated" : $("#family").value' in body_fn
        # search_budget overrides for campaign caps happen only inside the multi branch.
        assert "max_evaluated_candidates: totalBudget" not in before_multi
        assert "max_evaluated_candidates: totalBudget" in body_fn[body_fn.index("if (multi) {") :]
        # Runtime propagation is Multi-Family only; single-family keeps search_budget alone.
        assert "max_runtime_seconds: Number(budget.max_runtime_seconds ?? 7200)" not in before_multi

    def test_search_budget_runtime_propagated_to_multi_family(self, client: TestClient) -> None:
        js = _js(client)
        body_fn = _collect_run_body_fn(js)
        mf_block = body_fn[body_fn.index("body.multi_family = {") :]
        # Same Search Budget JSON drives both search_budget and multi_family runtime.
        assert "max_runtime_seconds: Number(budget.max_runtime_seconds ?? 7200)" in mf_block
        assert "max_runtime_seconds" in body_fn
        # Frozen preview serializes the full body, so both keys appear together.
        assert "JSON.stringify(body, null, 2)" in js
        # Default fallback is the campaign-facing 7200, not the backend FamilyCampaignConfig 600.
        assert "?? 7200)" in mf_block
        assert "?? 600)" not in mf_block
        assert "max_runtime_seconds: 600" not in mf_block

    def test_invalid_multi_family_runtime_rejected(self, client: TestClient) -> None:
        js = _js(client)
        assert "function multiFamilyBudgetError" in js
        assert (
            "MULTI_FAMILY_INVALID: max_runtime_seconds must be a positive finite number"
            in js
        )
        err_fn = js[js.index("function multiFamilyBudgetError") : js.index("function collectRunBody")]
        assert "Number.isFinite(runtime)" in err_fn
        assert "runtime > 0" in err_fn
        assert "mf.max_runtime_seconds" in err_fn

    def test_generated_card_prefers_campaign_total(self, client: TestClient) -> None:
        js = _js(client)
        assert "generation_accounting?.generated_total" in js
        assert "budget_allocation?.generated_total" in js
        assert "budget_allocation?.campaign_generated" in js

    def test_cross_family_crossover_forced_false(self, client: TestClient) -> None:
        body_fn = _collect_run_body_fn(_js(client))
        assert "allow_cross_family_crossover: false" in body_fn
        assert "allow_cross_family_crossover: true" not in body_fn
        assert 'id="mf_allow_cross_family_crossover"' not in _js(client)

    def test_vault_paper_live_confirmations_default_false(self, client: TestClient) -> None:
        js = _js(client)
        assert 'id="c_vault"' in js
        assert 'id="c_paper"' in js
        assert 'id="c_vault" checked' not in js
        assert 'id="c_paper" checked' not in js
        assert "confirm_vault: $("#c_vault").checked" in js
        assert "confirm_paper: $("#c_paper").checked" in js
        assert "confirm_live" not in js
        assert "Live trading" in js and "OFF" in js
        from control_plane.security import CreateRunRequest

        req = CreateRunRequest(run_type="ALPHA_MINER", confirm_alpha_miner=True, smoke_test=True)
        assert req.confirm_vault is False
        assert req.confirm_paper is False
        assert getattr(req, "confirm_live", False) is False


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
                "max_runtime_seconds": 7200,
            },
            "multi_family": {
                "enabled": True,
                "requested_family_count": 4,
                "min_candidates_per_family": 6,
                "total_candidate_budget": 24,
                "max_full_wfo": 8,
                "max_runtime_seconds": 7200,
                "adaptive_reallocation": True,
                "family_ids": [
                    "mean_reversion",
                    "momentum",
                    "breakout",
                    "volatility_expansion",
                ],
                "seed": 11,
                "family_local_evolution": True,
                "evolution_generations": 2,
                "stagnation_generations": 1,
                "minimum_improvement": 0.0001,
                "allow_cross_family_crossover": False,
                "stress_scenarios": [
                    "base_costs",
                    "costs_2x",
                    "wider_spread",
                    "worse_slippage",
                    "delayed_execution",
                    "removed_best_day",
                ],
                "max_stress_evaluations": 24,
                "max_stress_scenarios_per_candidate": 6,
                "min_stress_pass_rate": 0.5,
                "fail_closed_unsupported_stress": True,
                "max_robustness_candidates": 3,
                "max_robustness_evaluations": 18,
                "max_parameters_per_candidate": 2,
                "max_points_per_parameter": 3,
                "min_valid_neighborhood_points": 3,
                "allow_one_sided_neighborhood": False,
                "min_dsr": 0.95,
                "max_pbo": 0.5,
                "pbo_n_splits": 4,
                "min_oos_observations_for_dsr": 20,
                "behavioral_similarity_threshold": 0.85,
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
        assert mf.get("family_local_evolution") is True
        assert mf.get("evolution_generations") == 2
        assert mf.get("max_stress_evaluations") == 24
        assert mf.get("max_robustness_candidates") == 3
        assert mf.get("min_dsr") == 0.95
        assert mf.get("max_pbo") == 0.5
        assert mf.get("allow_cross_family_crossover") is False
        assert snap.get("live_trading_enabled") is False
        # Search-budget runtime must land on multi_family (not backend 600 default).
        assert (snap.get("search_budget") or {}).get("max_runtime_seconds") == 7200
        assert mf.get("max_runtime_seconds") == 7200
        client.post(f"/api/runs/{run_id}/cancel")

    def test_launch_stores_runtime_7200_not_backend_default(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json=self._mf_payload())
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        snap = client.get(f"/api/runs/{run_id}").json().get("config_snapshot") or {}
        assert (snap.get("search_budget") or {}).get("max_runtime_seconds") == 7200
        assert (snap.get("multi_family") or {}).get("max_runtime_seconds") == 7200
        # Explicit campaign value must not collapse to FamilyCampaignConfig default.
        assert (snap.get("multi_family") or {}).get("max_runtime_seconds") != 600
        client.post(f"/api/runs/{run_id}/cancel")

    def test_launch_stores_complete_pipeline_in_multi_family(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json=self._mf_payload())
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        snap = client.get(f"/api/runs/{run_id}").json().get("config_snapshot") or {}
        mf = snap.get("multi_family") or {}
        budget = snap.get("search_budget") or {}
        for field in _PIPELINE_FIELDS:
            assert field in mf, field
        assert mf["stress_scenarios"] == [
            "base_costs",
            "costs_2x",
            "wider_spread",
            "worse_slippage",
            "delayed_execution",
            "removed_best_day",
        ]
        assert "family_local_evolution" not in budget
        assert "min_dsr" not in budget
        client.post(f"/api/runs/{run_id}/cancel")

    def test_single_family_launch_unchanged(self, client: TestClient) -> None:
        payload = {
            "run_type": "ALPHA_MINER",
            "confirm_alpha_miner": True,
            "smoke_test": True,
            "dataset": "synthetic_demo",
            "random_seed": 11,
            "strategy_family": "mean_reversion_vwap_bb",
            "search_budget": {
                "max_generated_candidates": 12,
                "max_evaluated_candidates": 12,
                "max_full_wfo_evaluations": 4,
                "population_size": 4,
                "max_runtime_seconds": 20,
            },
        }
        resp = client.post("/api/runs", json=payload)
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        snap = client.get(f"/api/runs/{run_id}").json().get("config_snapshot") or {}
        assert snap.get("strategy_family") == "mean_reversion_vwap_bb"
        mf = snap.get("multi_family") or {}
        assert not mf.get("enabled")
        assert snap.get("live_trading_enabled") is False
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
