"""Phase 1 acceptance tests — configuration models."""

from __future__ import annotations

from datetime import date, time

import pytest
from pydantic import ValidationError
from zoneinfo import ZoneInfo

from config import (
    CFDAssetSpec,
    CostModel,
    DrawdownType,
    FuturesAssetSpec,
    IntrabarAmbiguityPolicy,
    PropProfile,
    SessionHours,
    SystemConfig,
    default_cfd_index,
    default_es_futures,
    default_prop_profile,
    default_system_config,
)
from config.asset_spec import ContinuousSeriesConfig, RolloverRules
from config.cost_model import default_cfd_cost_model, default_futures_cost_model
from config.models import AssetClass, ContinuousAdjustment, SettlementBehavior
from config.system_config import DataConfig


class TestPropProfile:
    def test_default_challenge_profile(self) -> None:
        p = default_prop_profile("challenge")
        assert p.prop_hard_daily_loss_limit == -0.015
        assert p.prop_total_drawdown_limit == -0.05
        assert p.internal_soft_daily_limit == -0.010
        assert p.drawdown_type == DrawdownType.STATIC
        assert p.close_positions_on_hard_breach is True
        assert p.cancel_pending_orders_on_hard_breach is True

    def test_soft_must_be_above_hard(self) -> None:
        with pytest.raises(ValidationError):
            PropProfile(
                prop_hard_daily_loss_limit=-0.015,
                prop_total_drawdown_limit=-0.05,
                internal_soft_daily_limit=-0.02,  # more severe than hard — invalid
            )

    def test_hard_limits_must_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            PropProfile(
                prop_hard_daily_loss_limit=0.015,
                prop_total_drawdown_limit=-0.05,
                internal_soft_daily_limit=-0.01,
            )

    def test_timezone_must_be_valid_iana(self) -> None:
        with pytest.raises(ValidationError):
            PropProfile(
                prop_hard_daily_loss_limit=-0.015,
                prop_total_drawdown_limit=-0.05,
                internal_soft_daily_limit=-0.01,
                daily_reset_timezone="Not/AZone",
            )

    def test_zoneinfo_resolves(self) -> None:
        p = default_prop_profile("funded")
        assert p.zoneinfo() == ZoneInfo("America/New_York")
        assert p.daily_reset_time == time(17, 0)
        assert p.drawdown_type == DrawdownType.TRAILING

    def test_frozen(self) -> None:
        p = default_prop_profile()
        with pytest.raises(ValidationError):
            p.name = "mutated"  # type: ignore[misc]


class TestFuturesAssetSpec:
    def test_es_template(self) -> None:
        es = default_es_futures(active_contract="ESH24")
        assert es.asset_class == AssetClass.FUTURES
        assert es.symbol == "ES"
        assert es.exchange == "CME"
        assert es.multiplier == 50.0
        assert es.tick_size == 0.25
        assert es.tick_value == 12.50
        assert es.active_contract == "ESH24"
        assert es.expiration == date(2024, 3, 15)
        assert es.settlement_behavior == SettlementBehavior.MARK_TO_MARKET
        assert es.continuous_series.enabled is True
        assert es.continuous_series.exclude_roll_gap_from_pnl is True
        assert es.rollover_rules.persist_decision is True

    def test_maintenance_cannot_exceed_initial(self) -> None:
        with pytest.raises(ValidationError):
            FuturesAssetSpec(
                symbol="ES",
                exchange="CME",
                multiplier=50.0,
                tick_size=0.25,
                tick_value=12.50,
                active_contract="ESH24",
                initial_margin=10_000.0,
                maintenance_margin=12_000.0,
            )

    def test_requires_active_tradable_contract(self) -> None:
        with pytest.raises(ValidationError):
            FuturesAssetSpec(
                symbol="ES",
                exchange="CME",
                multiplier=50.0,
                tick_size=0.25,
                tick_value=12.50,
                active_contract="",
                initial_margin=12_000.0,
                maintenance_margin=10_800.0,
            )

    def test_session_hours_timezone(self) -> None:
        hours = SessionHours(timezone="America/Chicago", open_time=time(8, 30), close_time=time(15, 0))
        assert hours.zoneinfo().key == "America/Chicago"
        with pytest.raises(ValidationError):
            SessionHours(timezone="Fake/Zone")

    def test_continuous_series_config(self) -> None:
        cfg = ContinuousSeriesConfig(
            adjustment=ContinuousAdjustment.BACKWARD_RATIO,
            exclude_roll_gap_from_pnl=True,
        )
        assert cfg.exclude_roll_gap_from_pnl is True
        rules = RolloverRules(days_before_expiry=8)
        assert rules.days_before_expiry == 8


class TestCFDAssetSpec:
    def test_cfd_template(self) -> None:
        cfd = default_cfd_index("US500")
        assert cfd.asset_class == AssetClass.CFD
        assert cfd.broker_symbol == "US500"
        assert cfd.leverage == 20.0
        assert cfd.variable_spread_by_time is True
        assert cfd.weekend_gaps is True
        assert cfd.stop_out_level == 0.5

    def test_leverage_margin_extreme_inconsistency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CFDAssetSpec(
                broker_symbol="US500",
                leverage=20.0,
                margin_rate=0.90,  # wildly inconsistent with 1/20
            )

    def test_futures_and_cfd_are_distinct_types(self) -> None:
        assert not isinstance(default_es_futures(), CFDAssetSpec)
        assert not isinstance(default_cfd_index(), FuturesAssetSpec)


class TestCostModel:
    def test_futures_and_cfd_defaults_differ(self) -> None:
        fut = default_futures_cost_model()
        cfd = default_cfd_cost_model()
        assert fut.version != cfd.version
        assert fut.overnight_swap_enabled is False
        assert cfd.overnight_swap_enabled is True
        assert fut.commission_per_contract > 0
        assert "participation_rate_cap" in CostModel.model_fields


class TestSystemConfig:
    def test_default_system_config_futures(self) -> None:
        cfg = default_system_config("futures")
        assert cfg.system_version.startswith("0.12.0")
        assert isinstance(cfg.asset, FuturesAssetSpec)
        assert cfg.intrabar_policy == IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE
        assert cfg.data.require_timezone is True

    def test_default_system_config_cfd(self) -> None:
        cfg = default_system_config("cfd")
        assert isinstance(cfg.asset, CFDAssetSpec)

    def test_snapshot_is_json_serializable(self) -> None:
        cfg = default_system_config()
        snap = cfg.to_snapshot()
        assert snap["prop_profile"]["prop_hard_daily_loss_limit"] == -0.015
        assert snap["cost_model"]["version"] == "futures_cost_v1"
        assert "asset" in snap

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(
                asset=default_es_futures(),
                prop_profile=default_prop_profile(),
                cost_model=default_futures_cost_model(),
                unexpected=True,  # type: ignore[call-arg]
            )

    def test_data_config_path_coercion(self) -> None:
        d = DataConfig(path="sample.csv")
        assert d.path is not None
        assert d.path.name == "sample.csv"
