"""Minimal MultiFamily research-claim safety hardening tests.

Proves the incomplete MultiFamily path is declared as WFO screening only and
cannot emit Finalist / Promoted / Shortlist / Vault / Paper claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from discovery.evaluator import SyntheticOOSBackend
from discovery.multi_family_campaign import (
    CROSS_FAMILY_CROSSOVER_UNSUPPORTED,
    EMPTY_COLLECTIONS_REASONS,
    FAMILY_LOCAL_EVOLUTION_NOT_READY,
    NOT_RESEARCH_SHORTLISTED,
    PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING,
    POST_WFO_BLOCKED_REASONS,
    POST_WFO_PIPELINE_NOT_RUN,
    SCORE_QUALIFIED_MEANING,
    STRESS_NOT_RUN,
    ROBUSTNESS_NOT_RUN,
    CLUSTERING_NOT_RUN,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
)
from discovery.promotion import POST_WFO_PIPELINE_NOT_RUN as PROMO_POST_WFO_NOT_RUN
from discovery.promotion import PromotionGate, PromotionStatus
from discovery.freezing import FrozenCandidate
from registry.experiment_registry import ExperimentRegistry


def _cfg(**overrides) -> FamilyCampaignConfig:
    base = dict(
        requested_family_count=2,
        min_candidates_per_family=4,
        total_candidate_budget=8,
        max_full_wfo=4,
        adaptive_reallocation=False,
        seed=11,
        min_oos_trades=1,
        min_oos_trades_per_fold=1,
        max_runtime_seconds=30,
        family_local_evolution=False,
        allow_cross_family_crossover=False,
    )
    base.update(overrides)
    return FamilyCampaignConfig(**base)


def _run_campaign(tmp_path: Path, **cfg_overrides):
    registry = ExperimentRegistry(tmp_path / "reg_mfc_safety")
    campaign = MultiFamilyCampaign(
        config=_cfg(**cfg_overrides),
        registry=registry,
        backend=SyntheticOOSBackend(),
        discovery_run_id="test_mfc_safety",
    )
    return campaign.run()


class TestMultiFamilyPipelineHonesty:
    def test_pipeline_level_is_wfo_screening(self, tmp_path: Path) -> None:
        result = _run_campaign(tmp_path)
        assert result.pipeline_level == PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING
        assert result.as_dict()["pipeline_level"] == PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING
        dr = result.discovery_results[0]
        assert dr.pipeline_level == PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING

    def test_post_wfo_pipeline_complete_is_false(self, tmp_path: Path) -> None:
        result = _run_campaign(tmp_path)
        assert result.post_wfo_pipeline_complete is False
        assert result.as_dict()["post_wfo_pipeline_complete"] is False
        assert result.discovery_results[0].post_wfo_pipeline_complete is False

    def test_score_qualified_meaning_is_economic_wfo_only(self, tmp_path: Path) -> None:
        result = _run_campaign(tmp_path)
        assert result.score_qualified_meaning == SCORE_QUALIFIED_MEANING
        payload = result.as_dict()
        assert payload["score_qualified_meaning"] == SCORE_QUALIFIED_MEANING
        for claim in (
            "finalist",
            "promoted",
            "stress passed",
            "research shortlisted",
            "vault eligible",
            "paper eligible",
        ):
            assert claim in payload["score_qualified_does_not_mean"]


class TestNoFalseFinalistsOrPromotions:
    def test_score_qualified_does_not_become_finalist_or_promoted(self, tmp_path: Path) -> None:
        result = _run_campaign(tmp_path)
        total_sq = sum(s.score_qualified for s in result.family_stats)
        # Even when Score Qualified candidates exist, post-WFO surfaces stay empty.
        dr = result.discovery_results[0]
        assert dr.finalists == []
        assert dr.promoted == []
        assert dr.clusters == []
        assert dr.research_shortlist == []
        assert dr.vault_candidates == []
        assert dr.paper_candidates == []
        assert result.research_shortlist == []
        assert result.vault_candidates == []
        assert result.paper_candidates == []
        payload = result.as_dict()
        assert payload["finalists"] == []
        assert payload["promoted"] == []
        assert payload["clusters"] == []
        # Rankings may list Score Qualified candidates; that is not a finalist claim.
        if total_sq > 0:
            assert any(r.get("candidate_id") for r in dr.rankings)

    def test_empty_collections_include_explicit_reasons(self, tmp_path: Path) -> None:
        result = _run_campaign(tmp_path)
        reasons = result.empty_collections_reasons
        for key in (
            "finalists",
            "promoted",
            "clusters",
            "research_shortlist",
            "vault_candidates",
            "paper_candidates",
        ):
            assert key in reasons
            assert reasons[key] == list(EMPTY_COLLECTIONS_REASONS[key])
            assert reasons[key], f"{key} must not be silently empty without reasons"

        assert POST_WFO_PIPELINE_NOT_RUN in result.post_wfo_blocked_reasons
        assert STRESS_NOT_RUN in result.post_wfo_blocked_reasons
        assert ROBUSTNESS_NOT_RUN in result.post_wfo_blocked_reasons
        assert CLUSTERING_NOT_RUN in result.post_wfo_blocked_reasons
        assert NOT_RESEARCH_SHORTLISTED in result.post_wfo_blocked_reasons
        assert list(result.post_wfo_blocked_reasons) == list(POST_WFO_BLOCKED_REASONS)

        dr = result.discovery_results[0]
        assert dr.empty_collections_reasons["finalists"]
        assert POST_WFO_PIPELINE_NOT_RUN in dr.empty_collections_reasons["finalists"]
        assert CLUSTERING_NOT_RUN in dr.empty_collections_reasons["clusters"]
        assert NOT_RESEARCH_SHORTLISTED in dr.empty_collections_reasons["vault_candidates"]

    def test_promotion_gate_refuses_incomplete_post_wfo(self) -> None:
        frozen = FrozenCandidate(
            frozen_id="fz1",
            candidate_id="c1",
            lineage_id="lin1",
            strategy_family="mean_reversion",
            parameters={},
            config_hash="cfg",
            feature_set_version="fs1",
            cost_model_version="cm1",
            grammar_version="g1",
            code_hash="code",
            data_hash="data",
            execution_assumptions={},
            expression_snapshot={},
            oos_fitness=1.0,
            ranking_source="validation_oos",
            discovery_run_id="d1",
        )
        decision = PromotionGate().refuse_incomplete_post_wfo_pipeline(frozen)
        assert decision.status is PromotionStatus.REJECTED
        assert decision.reason == PROMO_POST_WFO_NOT_RUN
        assert decision.reason == POST_WFO_PIPELINE_NOT_RUN


class TestIncompleteEvolutionGuards:
    def test_family_local_evolution_fails_fast(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "reg_evo")
        campaign = MultiFamilyCampaign(
            config=_cfg(family_local_evolution=True),
            registry=registry,
            backend=SyntheticOOSBackend(),
            discovery_run_id="test_evo_not_ready",
        )
        with pytest.raises(RuntimeError, match=FAMILY_LOCAL_EVOLUTION_NOT_READY):
            campaign.run()

    def test_cross_family_crossover_fails_fast(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "reg_xover")
        campaign = MultiFamilyCampaign(
            config=_cfg(allow_cross_family_crossover=True),
            registry=registry,
            backend=SyntheticOOSBackend(),
            discovery_run_id="test_xover_unsupported",
        )
        with pytest.raises(RuntimeError, match=CROSS_FAMILY_CROSSOVER_UNSUPPORTED):
            campaign.run()

    def test_evolution_defaults_remain_disabled(self) -> None:
        cfg = FamilyCampaignConfig()
        assert cfg.family_local_evolution is False
        assert cfg.allow_cross_family_crossover is False
