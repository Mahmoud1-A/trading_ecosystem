"""
Phase 8 acceptance tests — Dashboard cutover and control plane.

Prove: dashboard counts match Registry, monotonic event counts, snapshots ≠ funnel,
legacy promotion blocked, new backend authoritative, legacy read-only, safe rollback.
"""

from __future__ import annotations

import pytest

from api import DiscoveryAPI, HealthAPI, PaperAPI, PortfolioAPI, VaultAPI
from api.types import BackendAuthority, CounterKind
from dashboard import ControlPlane, CutoverError, CutoverStage
from portfolio import (
    ConstructionMethod,
    PortfolioConstraints,
    PortfolioPipelineStage,
    build_portfolio_version,
    freeze_portfolio,
)
from registry.experiment_registry import ExperimentRegistry


def _seed_registry(path) -> ExperimentRegistry:
    reg = ExperimentRegistry(path)
    for i in range(5):
        reg.create_trial(
            candidate_id=f"c{i}",
            lineage_id=f"l{i}",
            strategy_family="dsl",
            parameters={"p": float(i)},
            config_snapshot={"v": 1},
            system_version="0.8.0-phase8",
            data_hash="d",
            random_seed=i,
            cost_model_version="cost_v1",
            code_hash="code",
            ranking_score=0.1 * i if i < 4 else None,
            rejection_reason="precheck:always-false" if i == 4 else None,
            trial_id=f"t{i}",
        )
    return reg


def _plane(tmp_path) -> ControlPlane:
    reg = _seed_registry(tmp_path / "reg")
    discovery = DiscoveryAPI(registry=reg)
    discovery.sync_from_registry(finalist_ids={"c3"})
    return ControlPlane(
        discovery=discovery,
        portfolio=PortfolioAPI(),
        vault=VaultAPI(),
        paper=PaperAPI(),
        health=HealthAPI(),
    )


class TestRegistryAlignedCounts:
    def test_dashboard_counts_match_registry_queries(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        reg = plane.discovery.registry
        counters = {c.name: c for c in plane.discovery.funnel_counters()}
        assert counters["registry_total_trials"].value == len(reg.all_trials())
        assert counters["registry_total_trials"].kind is CounterKind.CUMULATIVE_EVENT
        rejected = reg.rejected_trials()
        assert counters["precheck_rejected"].value == len(
            [t for t in rejected if t.rejection_reason and "precheck" in t.rejection_reason]
        )
        view = plane.dashboard_payload()["candidate_funnel"]
        event_names = {c["name"] for c in view["event_counts"]}
        snap_names = {c["name"] for c in view["snapshots"]}
        assert "current_finalist_set_size" in snap_names
        assert "current_finalist_set_size" not in event_names
        assert "current_finalist_set_size" not in view["funnel_stage_names"]

    def test_event_counts_are_monotonic(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        before = plane.discovery.session.generated
        # Ingest another trial event
        trial = plane.discovery.registry.all_trials()[0]
        plane.discovery.ingest_trial(trial, is_finalist=False)
        assert plane.discovery.session.generated == before + 1
        plane.discovery.ingest_trial(trial, is_finalist=False)
        assert plane.discovery.session.generated == before + 2

    def test_snapshot_values_not_treated_as_funnel_counts(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        # Book-like snapshot must not appear as funnel stage
        view = plane.dashboard_payload()["candidate_funnel"]
        assert "legacy_book_size" not in view["funnel_stage_names"]
        for c in view["snapshots"]:
            assert c["kind"] == CounterKind.SNAPSHOT.value
            assert c["name"] not in view["funnel_stage_names"]


class TestCutoverPromotionAndAuthority:
    def test_old_promotion_logic_cannot_promote(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        # Allowed only in PARALLEL with promotion still on
        plane.attempt_legacy_promote("legacy_cand")
        assert "legacy_cand" in plane.legacy.archive.get("promoted", [])

        plane.advance(CutoverStage.LEGACY_READONLY)
        with pytest.raises(CutoverError, match="disabled"):
            plane.attempt_legacy_promote("another")
        assert plane.paper.promotions_blocked_legacy >= 1

    def test_new_backend_is_authoritative_after_switch(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        plane.advance(CutoverStage.LEGACY_READONLY)
        plane.advance(CutoverStage.COMPARE)
        plane.advance(CutoverStage.SWITCHED)
        assert plane.authority is BackendAuthority.NEW
        payload = plane.dashboard_payload()
        assert payload["authority"] == BackendAuthority.NEW.value
        assert "candidate_funnel" in payload
        assert "legacy_readonly_view" not in payload  # switched off parallel dual view

    def test_legacy_dashboard_read_only_during_migration(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        plane.advance(CutoverStage.LEGACY_READONLY)
        assert plane.legacy.read_only is True
        assert plane.legacy.promotion_enabled is False
        snap = plane.legacy.snapshot()
        assert snap["read_only"] is True
        # Parallel payload still exposes legacy read-only view
        payload = plane.dashboard_payload()
        assert "legacy_readonly_view" in payload

    def test_cutover_can_be_rolled_back_safely(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        plane.advance(CutoverStage.LEGACY_READONLY)
        plane.advance(CutoverStage.COMPARE)
        plane.advance(CutoverStage.SWITCHED)
        assert plane.authority is BackendAuthority.NEW
        prev = plane.rollback()
        assert prev is CutoverStage.COMPARE
        assert plane.legacy.read_only is True
        prev2 = plane.rollback()
        assert prev2 is CutoverStage.LEGACY_READONLY
        plane.rollback()
        assert plane.stage is CutoverStage.PARALLEL
        assert plane.authority is BackendAuthority.PARALLEL
        # After full rollback, legacy promotion can work again
        plane.attempt_legacy_promote("restored")


class TestPortfolioAndHealthPanels:
    def test_portfolio_and_execution_panels(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        version = build_portfolio_version(
            weights={"c0": 0.6, "c1": 0.4},
            lineage_ids={"c0": "l0", "c1": "l1"},
            construction_method=ConstructionMethod.GREEDY_MARGINAL.value,
            constraints=PortfolioConstraints().as_dict(),
            feature_set_version="fs1",
            cost_model_version="cost_v1",
            stage=PortfolioPipelineStage.STABILITY_PASSED,
        )
        frozen = freeze_portfolio(version)
        plane.portfolio.record_version(version)
        plane.portfolio.record_version(frozen)
        plane.health.execution.record_order()
        plane.health.execution.record_fill(partial=True, latency_ms=12.0)
        plane.health.execution.record_costs(spread=0.1, commission=0.2)
        plane.health.data_quality.missing_bars = 3
        plane.health.data_quality.dataset_versions["gold"] = "v1"
        plane.vault.record_access(lineage_id="l0", vault_version="vault_v1", success=True)

        payload = plane.dashboard_payload()
        assert payload["portfolio_pipeline"]["current_members"] == list(frozen.member_candidate_ids)
        assert any(c["name"] == "orders" for c in payload["execution_quality"]["counters"])
        assert payload["data_quality"]["provider_healthy"] is True
        assert payload["system_health"]["vault"]["counters"]

    def test_compare_outputs_logs_diffs(self, tmp_path) -> None:
        plane = _plane(tmp_path)
        plane.legacy.funnel_counts = {"generated": 99}
        plane.advance(CutoverStage.COMPARE)
        report = plane.compare_outputs()
        assert report["match"] is False
        assert "generated" in report["diffs"]
        assert plane.preserve_legacy_archive()["deleted"] is False
