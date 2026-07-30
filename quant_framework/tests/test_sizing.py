"""
Phase 5.5-G acceptance — volatility-target sizing under margin and risk budgets.

Hand-calculated Futures and CFD cases. Size is always the minimum of the binding
constraints and is never increased to meet the minimum lot.
"""

from __future__ import annotations

import pytest

from config.asset_spec import default_cfd_index, default_es_futures
from config.models import RiskState
from config.prop_profile import default_prop_profile
from config.strategy_config import SizingParams
from engine.portfolio import Portfolio
from risk.position_sizing import volatility_target_size


class TestVolatilityTargeting:
    def test_size_scales_with_equity_and_atr(self) -> None:
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        sizing = SizingParams(risk_per_trade_pct=0.005, atr_stop_mult=1.5, min_lot=0.01, lot_step=0.01)
        asset = default_es_futures()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=4.0,
        )
        assert not sd.rejected
        assert sd.allowed_quantity >= sizing.min_lot

    def test_never_increases_above_cap(self) -> None:
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        sizing = SizingParams(max_lots=2.0, min_lot=0.01, lot_step=0.01, risk_per_trade_pct=0.05)
        asset = default_es_futures()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=0.5,
        )
        assert sd.allowed_quantity <= sizing.max_lots + 1e-9

    def test_rejects_below_min_lot(self) -> None:
        portfolio = Portfolio(1_000.0)
        portfolio.cash = 1_000.0
        sizing = SizingParams(min_lot=1.0, lot_step=1.0, risk_per_trade_pct=0.001)
        asset = default_es_futures()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=50.0,
        )
        assert sd.rejected or sd.allowed_quantity == 0.0


class TestFuturesHandCalculated:
    def test_vol_target_and_margin_min(self) -> None:
        """
        ES: atr=4, stop_mult=1.5 => stop=6 points; point_value=50 => risk/contract=300
        risk_pct=0.005 on 100k => risk$=500 => vol_target=500/300=1.666...
        free_margin=24_000 / initial_margin=12_000 => margin_cap=2.0
        max_lots=10, exposures open. Floor to lot_step=1.0 => 1.0
        """
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        sizing = SizingParams(
            risk_per_trade_pct=0.005,
            atr_stop_mult=1.5,
            min_lot=1.0,
            lot_step=1.0,
            max_lots=10.0,
        )
        asset = default_es_futures()
        profile = default_prop_profile()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=4.0,
            prop_profile=profile,
            free_margin=24_000.0,
            used_margin=0.0,
            max_symbol_exposure=10.0,
            max_portfolio_exposure=10.0,
        )
        assert not sd.rejected
        assert sd.allowed_quantity == pytest.approx(1.0)
        assert sd.constraints["vol_target"] == pytest.approx(500.0 / 300.0)
        assert sd.constraints["free_margin"] == pytest.approx(2.0)

    def test_insufficient_free_margin_rejects(self) -> None:
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        sizing = SizingParams(min_lot=1.0, lot_step=1.0, risk_per_trade_pct=0.01)
        asset = default_es_futures()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=2.0,
            free_margin=5_000.0,  # < initial_margin 12_000
            used_margin=0.0,
        )
        assert sd.rejected
        assert sd.reason == "insufficient_free_margin"

    def test_hard_daily_budget_exhausted(self) -> None:
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        sizing = SizingParams(min_lot=1.0, lot_step=1.0, risk_per_trade_pct=0.01)
        asset = default_es_futures()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=2.0,
            remaining_hard_daily_budget=0.0,
        )
        assert sd.rejected
        assert sd.reason == "hard_daily_budget_exhausted"

    def test_reduce_only_cannot_increase_exposure(self) -> None:
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        pos = portfolio.get_position("ES")
        pos.quantity = 2.0
        pos.avg_price = 100.0
        sizing = SizingParams(min_lot=1.0, lot_step=1.0, risk_per_trade_pct=0.05)
        asset = default_es_futures()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=2.0,
            risk_state=RiskState.REDUCE_ONLY,
            reduce_only=False,
            symbol="ES",
        )
        assert sd.rejected
        assert sd.reason == "reduce_only"

    def test_caution_applies_configured_size_reduction(self) -> None:
        portfolio = Portfolio(100_000.0)
        portfolio.cash = 100_000.0
        sizing = SizingParams(
            risk_per_trade_pct=0.01,
            atr_stop_mult=1.0,
            min_lot=0.01,
            lot_step=0.01,
            max_lots=100.0,
        )
        asset = default_es_futures()
        # stop=2 pts * 50 = 100$/contract; risk$=1000 => qty=10 without caution
        from config.prop_profile import PropProfile

        data = default_prop_profile().model_dump()
        data["internal_risk_budget"] = 0.5
        data["max_symbol_exposure"] = 100.0
        data["max_portfolio_exposure"] = 100.0
        profile = PropProfile(**data)
        # Large free margin so margin does not bind before the caution scalar
        normal = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=2.0,
            risk_state=RiskState.NORMAL,
            prop_profile=profile,
            free_margin=1_000_000.0,
        )
        caution = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=2.0,
            risk_state=RiskState.CAUTION,
            prop_profile=profile,
            free_margin=1_000_000.0,
        )
        assert not normal.rejected and not caution.rejected
        assert normal.allowed_quantity == pytest.approx(10.0)
        assert caution.allowed_quantity == pytest.approx(5.0)


class TestCFDHandCalculated:
    def test_cfd_margin_and_leverage_cap(self) -> None:
        """
        US500 CFD at price 5000, contract_size=1, leverage=20 => margin/unit = 250
        equity=50_000, risk_pct=0.01, atr=20, stop_mult=1.5 => stop=30 pts
        risk$/unit=30 => vol_target=500/30≈16.666
        free_margin=5_000 => margin_cap=5000/250=20
        max_lots=10 binds => 10 after floor to lot_step 0.01
        """
        portfolio = Portfolio(50_000.0)
        portfolio.cash = 50_000.0
        sizing = SizingParams(
            risk_per_trade_pct=0.01,
            atr_stop_mult=1.5,
            min_lot=0.01,
            lot_step=0.01,
            max_lots=10.0,
        )
        asset = default_cfd_index()
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=20.0,
            price=5_000.0,
            leverage=20.0,
            free_margin=5_000.0,
            used_margin=0.0,
            stop_out_level=0.5,
            symbol="US500",
        )
        assert not sd.rejected
        assert sd.allowed_quantity == pytest.approx(10.0)
        assert sd.binding_constraint == "max_lots"
        assert sd.constraints["free_margin"] == pytest.approx(20.0)
        assert sd.constraints["vol_target"] == pytest.approx(500.0 / 30.0)

    def test_never_rounds_up_to_meet_min_lot(self) -> None:
        portfolio = Portfolio(10_000.0)
        portfolio.cash = 10_000.0
        sizing = SizingParams(
            risk_per_trade_pct=0.001,
            atr_stop_mult=2.0,
            min_lot=1.0,
            lot_step=1.0,
            max_lots=10.0,
        )
        # CFD min_lot comes from the asset spec — set it to 1.0 so a 0.25
        # vol-target size cannot be rounded up into a trade.
        asset = default_cfd_index().model_copy(update={"min_lot": 1.0, "lot_step": 1.0})
        # stop=40, risk$=10 => qty=0.25 < min_lot 1.0 => reject, never bump to 1
        sd = volatility_target_size(
            portfolio=portfolio,
            sizing=sizing,
            asset=asset,
            atr_points=20.0,
            price=5_000.0,
            symbol="US500",
            free_margin=1_000_000.0,
        )
        assert sd.rejected
        assert sd.allowed_quantity == 0.0
        assert sd.reason == "below_minimum_tradable"
