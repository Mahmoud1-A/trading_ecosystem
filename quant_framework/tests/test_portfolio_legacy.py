"""
Phase 7 acceptance tests — Portfolio construction and legacy migration.

Prove: no inherited legacy status, forced reevaluation, Book size is snapshot
not funnel, constraints enforced, correlation penalty, marginal improvement,
unstable allocations rejected, frozen immutability, Vault only when frozen.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from discovery.types import CreationMethod
from legacy import (
    FORBIDDEN_INHERITED_FIELDS,
    FUNNEL_STAGES,
    LEGACY_BOOK_SIZE,
    LegacyImporter,
    LegacyStrategyRecord,
    adapt_legacy_record,
    make_legacy_book,
)
from portfolio import (
    CandidateFunnelStage,
    CandidatePool,
    ConstructionMethod,
    ConstraintViolation,
    MemberMeta,
    PoolMember,
    PortfolioConstraints,
    PortfolioOptimizer,
    PortfolioPipelineStage,
    PortfolioPromoter,
    PortfolioVaultError,
    PortfolioVersionError,
    RiskBudget,
    build_correlation_matrix,
    build_portfolio_version,
    correlation_penalty,
    freeze_portfolio,
    marginal_contribution,
    mutate_frozen_weights,
    portfolio_quality,
    validate_weights,
)
from registry.experiment_registry import ExperimentRegistry
from validation.vault import ValidationVault


TZ = ZoneInfo("America/Chicago")


def _member(
    cid: str,
    *,
    family: str = "mr",
    expected: float = 0.5,
    symbols: tuple[str, ...] = ("ES",),
    turnover: float = 1.0,
    prop: float = 0.05,
) -> MemberMeta:
    return MemberMeta(
        candidate_id=cid,
        lineage_id=f"lin_{cid}",
        family=family,
        symbols=symbols,
        asset_class="futures",
        regime_bucket="range",
        session="rth",
        turnover=turnover,
        margin_usage=0.2,
        prop_breach_prob=prop,
        expected_oos=expected,
    )


def _vault(tmp_path) -> ValidationVault:
    idx = pd.date_range("2024-06-01 08:30", periods=30, freq="5min", tz=TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )
    return ValidationVault(
        vault_version="vault_v1",
        data=df,
        data_hash="vh",
        store_path=tmp_path / "vault",
    )


class TestLegacyMigration:
    def test_legacy_members_receive_no_inherited_status(self) -> None:
        record = LegacyStrategyRecord(
            legacy_id="L1",
            name="OldBookStar",
            strategy_family="legacy_mr",
            parameters={"z_entry": -2.0, "z_exit": -0.2},
            book_member=True,
            vault_eligible=True,
            paper_status="PAPER",
            legacy_fitness=9.9,
            promotion_status="BOOK",
        )
        cand = adapt_legacy_record(record, random_seed=1)
        assert cand.creation_method is CreationMethod.LEGACY_IMPORT
        payload = cand.as_dict()
        for field in FORBIDDEN_INHERITED_FIELDS:
            assert field not in payload
        assert payload.get("book_member") is None or "book_member" not in payload

    def test_all_imported_strategies_are_reevaluated(self, tmp_path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg")
        book = make_legacy_book(5)
        report = LegacyImporter(registry=reg).import_and_reevaluate(book)
        assert report.imported == 5
        assert report.reevaluated == 5
        assert report.as_dict()["all_reevaluated"] is True
        assert len(reg.all_trials()) == 5
        # Legacy fitness must not appear as ranking
        for t in reg.all_trials():
            assert t.parameters  # reevaluated params present
            assert "legacy_fitness" not in t.net_metrics

    def test_book_size_is_not_a_funnel_stage(self) -> None:
        assert LEGACY_BOOK_SIZE == 29
        assert "BOOK_SIZE" not in FUNNEL_STAGES
        assert CandidateFunnelStage.FINALIST.value in FUNNEL_STAGES
        report = make_legacy_book()
        assert len(report) == 29
        # Migration report separates snapshot from events
        from legacy.migration_report import MigrationReport

        mr = MigrationReport(legacy_book_size_snapshot=29)
        d = mr.as_dict()
        assert d["snapshots"]["legacy_book_size"] == 29
        assert "legacy_book_size" not in d["funnel_event_counts"]
        assert "BOOK_SIZE" not in d["funnel_event_counts"]


class TestPortfolioConstraintsAndCorr:
    def test_portfolio_constraints_cannot_be_bypassed(self) -> None:
        members = [_member("a"), _member("b", family="x"), _member("c", family="y")]
        constraints = PortfolioConstraints(max_members=2, max_strategy_weight=0.6)
        with pytest.raises(ConstraintViolation, match="max_members"):
            validate_weights({"a": 0.4, "b": 0.3, "c": 0.3}, members, constraints)
        with pytest.raises(ConstraintViolation, match="max_strategy_weight"):
            validate_weights(
                {"a": 0.9, "b": 0.1},
                members[:2],
                PortfolioConstraints(max_strategy_weight=0.5),
            )

    def test_correlated_strategies_are_penalized(self) -> None:
        pnl = {
            "a": (1.0, 2.0, 3.0, 4.0),
            "b": (1.1, 2.1, 2.9, 4.1),  # highly correlated
            "c": (-1.0, 0.5, -0.2, 1.0),
        }
        pairwise = build_correlation_matrix(pnl)
        w_corr = {"a": 0.5, "b": 0.5}
        w_div = {"a": 0.5, "c": 0.5}
        assert correlation_penalty(w_corr, pairwise) > correlation_penalty(w_div, pairwise)
        members = [
            _member("a", expected=0.4),
            _member("b", expected=0.4),
            _member("c", expected=0.4),
        ]
        q_corr = portfolio_quality(w_corr, members[:2], pairwise)
        q_div = portfolio_quality(w_div, [members[0], members[2]], pairwise)
        assert q_div > q_corr


class TestMarginalAndStability:
    def test_adding_strategy_must_improve_marginal_quality(self) -> None:
        # Solo A is risky (high prop) so a diversified add can raise quality
        members = [
            _member("a", expected=0.55, prop=0.35, symbols=("ES",)),
            _member("b", expected=0.50, family="other", prop=0.05, symbols=("NQ",)),
            _member("clone", expected=0.52, prop=0.30, symbols=("ES",)),  # correlated with a
        ]
        pairwise = {
            ("a", "b"): 0.05,
            ("a", "clone"): 0.95,
            ("b", "clone"): 0.15,
        }
        current = {"a": 1.0}
        delta_b = marginal_contribution(
            "b", current_weights=current, members=members, pairwise=pairwise, new_weight=0.45
        )
        delta_clone = marginal_contribution(
            "clone",
            current_weights=current,
            members=members,
            pairwise=pairwise,
            new_weight=0.45,
        )
        assert delta_b > 0
        assert delta_b > delta_clone

        opt = PortfolioOptimizer(
            constraints=PortfolioConstraints(
                max_members=3,
                max_strategy_weight=0.7,
                max_symbol_exposure=0.75,
                max_asset_class_exposure=1.0,
                max_regime_concentration=1.0,
                max_session_concentration=1.0,
            ),
            risk_budget=RiskBudget(max_member_risk_share=0.6),
        )
        result = opt.optimize(members, pairwise=pairwise, method=ConstructionMethod.GREEDY_MARGINAL)
        assert result.accepted
        assert "b" in result.weights
        assert "clone" not in result.weights or result.weights.get("clone", 0) <= result.weights.get(
            "b", 0
        )

    def test_unstable_allocations_are_rejected(self) -> None:
        members = [_member("a", expected=0.1), _member("b", expected=0.1, family="x")]
        pairwise = {("a", "b"): 0.99}
        opt = PortfolioOptimizer(
            constraints=PortfolioConstraints(max_pairwise_correlation=0.5, max_strategy_weight=0.6),
            max_mean_corr=0.5,
        )
        result = opt.optimize(
            members,
            pairwise=pairwise,
            method=ConstructionMethod.EQUAL_RISK_CONTRIBUTION,
            vols={"a": 0.1, "b": 0.1},
        )
        # Either constraint violation or unstable_high_mean_correlation
        assert result.accepted is False


class TestFreezeAndVault:
    def test_portfolio_versions_immutable_after_freezing(self) -> None:
        version = build_portfolio_version(
            weights={"a": 0.6, "b": 0.4},
            lineage_ids={"a": "la", "b": "lb"},
            construction_method=ConstructionMethod.GREEDY_MARGINAL.value,
            constraints=PortfolioConstraints().as_dict(),
            feature_set_version="fs1",
            cost_model_version="cost_v1",
        )
        frozen = freeze_portfolio(version)
        assert frozen.frozen is True
        assert frozen.stage is PortfolioPipelineStage.FROZEN
        with pytest.raises(PortfolioVersionError, match="immutable"):
            mutate_frozen_weights(frozen, {"a": 1.0})
        with pytest.raises(PortfolioVersionError, match="already frozen"):
            freeze_portfolio(frozen)

    def test_only_frozen_portfolios_can_access_vault(self, tmp_path) -> None:
        version = build_portfolio_version(
            weights={"a": 1.0},
            lineage_ids={"a": "la"},
            construction_method="equal_risk_contribution",
            constraints={},
            feature_set_version="fs1",
            cost_model_version="cost_v1",
        )
        promoter = PortfolioPromoter(vault=_vault(tmp_path))
        with pytest.raises(PortfolioVaultError, match="frozen"):
            promoter.mark_vault_eligible(version)
        with pytest.raises(PortfolioVaultError, match="frozen"):
            promoter.submit_to_vault(version, evaluate_fn=lambda df: {"n": len(df)})

        frozen = freeze_portfolio(version)
        eligible = promoter.mark_vault_eligible(frozen)
        assert eligible.stage is PortfolioPipelineStage.VAULT_ELIGIBLE
        passed, result = promoter.submit_to_vault(
            eligible, evaluate_fn=lambda df: {"n": len(df), "ok": True}
        )
        assert passed.stage is PortfolioPipelineStage.VAULT_PASSED
        assert result["access_count"] == 1


class TestPoolSnapshotVsEvents:
    def test_pool_separates_snapshot_from_events(self) -> None:
        pool = CandidatePool()
        for i in range(3):
            pool.add(
                PoolMember(
                    meta=_member(f"c{i}", expected=0.2 * i),
                    stage=CandidateFunnelStage.FINALIST,
                    pnl_series=(0.1, 0.2, -0.05),
                )
            )
        pool.event_counts["GENERATED"] = 100
        d = pool.as_dict()
        assert d["finalist_count_snapshot"] == 3
        assert d["event_counts"]["GENERATED"] == 100
        assert d["finalist_count_snapshot"] != d["event_counts"]["GENERATED"]
