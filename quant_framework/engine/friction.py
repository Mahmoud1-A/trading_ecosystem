"""Configurable friction / cost application at fill time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.asset_spec import CFDAssetSpec, FuturesAssetSpec
from config.cost_model import CostModel
from config.models import AssetClass, Side
from engine.fills import CostBreakdown


@dataclass(frozen=True)
class FrictionContext:
    """Market context required to price friction for a fill."""

    mid_price: float
    atr: float | None
    bar_volume: float
    is_near_session_open: bool = False
    is_near_session_close: bool = False
    hour_utc: int | None = None
    # When both set, half-spread is taken from observed BID/ASK (Dukascopy).
    bid: float | None = None
    ask: float | None = None


class FrictionEngine:
    """
    Apply CostModel to produce a CostBreakdown and an executable fill price.

    Deterministic given the same CostModel, context, side, quantity, and RNG seed.
    """

    def __init__(
        self,
        cost_model: CostModel,
        asset: FuturesAssetSpec | CFDAssetSpec,
        *,
        seed: int = 42,
    ) -> None:
        self.cost_model = cost_model
        self.asset = asset
        self._rng = np.random.default_rng(seed)

    @property
    def tick_size(self) -> float:
        return float(self.asset.tick_size)

    @property
    def point_value(self) -> float:
        if isinstance(self.asset, FuturesAssetSpec):
            return float(self.asset.multiplier)
        return float(self.asset.contract_size)

    def _spread_ticks(self, ctx: FrictionContext) -> float:
        ticks = float(self.cost_model.fixed_spread_ticks)
        if self.cost_model.dynamic_spread_enabled and ctx.atr is not None and self.tick_size > 0:
            ticks += float(self.cost_model.dynamic_spread_atr_mult) * (ctx.atr / self.tick_size)
        if isinstance(self.asset, CFDAssetSpec):
            markup = float(self.asset.broker_spread_markup)
            ticks += markup / self.tick_size if self.tick_size > 0 else 0.0
        return max(0.0, ticks)

    def _slippage_ticks(self, ctx: FrictionContext) -> float:
        # Half-normal style non-negative slip around configured mean (deterministic RNG)
        base = float(self.cost_model.slippage_ticks_mean)
        noise = abs(self._rng.normal(0.0, float(self.cost_model.slippage_ticks_std)))
        ticks = base + noise
        if self.cost_model.volatility_dependent_slippage and ctx.atr is not None and self.tick_size > 0:
            ticks += float(self.cost_model.volatility_slippage_mult) * (ctx.atr / self.tick_size) * 0.1
        if self.cost_model.time_of_day_slippage and (
            ctx.is_near_session_open or ctx.is_near_session_close
        ):
            ticks *= float(self.cost_model.open_close_slippage_mult)
        if self.cost_model.liquidity_dependent_slippage and ctx.bar_volume > 0:
            ratio = float(self.cost_model.liquidity_volume_ref) / max(ctx.bar_volume, 1.0)
            ticks *= max(1.0, min(ratio, 3.0))
        return max(0.0, ticks)

    def quote_prices(self, ctx: FrictionContext) -> tuple[float, float, float]:
        """Return (bid, ask, half_spread_price)."""
        if (
            ctx.bid is not None
            and ctx.ask is not None
            and float(ctx.ask) >= float(ctx.bid)
        ):
            bid = float(ctx.bid)
            ask = float(ctx.ask)
            half = max(0.0, (ask - bid) / 2.0)
            return bid, ask, half
        half = self._spread_ticks(ctx) * self.tick_size / 2.0
        bid = ctx.mid_price - half
        ask = ctx.mid_price + half
        return bid, ask, half

    def executable_price(
        self,
        *,
        side: Side,
        reference_price: float,
        ctx: FrictionContext,
    ) -> tuple[float, float, float]:
        """
        Compute fill price after spread + slippage.

        Returns (fill_price, spread_cost_per_unit_price, slippage_cost_per_unit_price).
        """
        _, _, half_spread = self.quote_prices(ctx)
        slip_ticks = self._slippage_ticks(ctx)
        slip_price = slip_ticks * self.tick_size
        if side == Side.BUY:
            fill = reference_price + half_spread + slip_price
        else:
            fill = reference_price - half_spread - slip_price
        return fill, half_spread, slip_price

    def commission(self, quantity: float) -> float:
        raw = float(self.cost_model.commission_per_contract) * abs(quantity)
        return max(raw, float(self.cost_model.minimum_commission)) if quantity else 0.0

    def build_costs(
        self,
        *,
        side: Side,
        quantity: float,
        reference_price: float,
        ctx: FrictionContext,
        include_rollover: bool = False,
    ) -> tuple[float, CostBreakdown]:
        fill_px, half_spread, slip_price = self.executable_price(
            side=side, reference_price=reference_price, ctx=ctx
        )
        pv = self.point_value
        spread_cost = half_spread * abs(quantity) * pv
        slippage_cost = slip_price * abs(quantity) * pv
        commission = self.commission(quantity)

        financing = 0.0
        if (
            self.cost_model.overnight_swap_enabled
            and isinstance(self.asset, CFDAssetSpec)
            and self.asset.asset_class == AssetClass.CFD
        ):
            # Financing applied elsewhere on overnight; zero at intrabar fill by default.
            financing = 0.0

        rollover = 0.0
        if include_rollover and isinstance(self.asset, FuturesAssetSpec):
            rollover = (
                float(self.cost_model.futures_rollover_cost_ticks)
                * self.tick_size
                * abs(quantity)
                * pv
            )

        impact = 0.0
        if self.cost_model.market_impact_enabled and ctx.bar_volume > 0:
            participation = abs(quantity) / max(ctx.bar_volume, 1.0)
            impact = (
                float(self.cost_model.market_impact_coeff)
                * participation
                * reference_price
                * abs(quantity)
                * pv
            )

        costs = CostBreakdown(
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            financing_cost=financing,
            rollover_cost=rollover,
            market_impact_cost=impact,
        )
        return fill_px, costs

    def max_fill_quantity(self, bar_volume: float, requested: float) -> float:
        cap = float(self.cost_model.participation_rate_cap) * max(bar_volume, 0.0)
        if cap <= 0:
            return 0.0
        return float(min(requested, cap))
