"""Configurable execution cost / friction model."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from config.models import NonNegativeFloat, PositiveFloat, UnitInterval


class CostModel(BaseModel):
    """
    Friction assumptions applied at fill time (Phase 2+).

    Every trade report must attribute gross PnL into commission, spread, slippage,
    financing, rollover, and net PnL components.
    """

    model_config = {"extra": "forbid", "frozen": True}

    version: str = Field("cost_v1", min_length=1)
    commission_per_contract: NonNegativeFloat = 2.50
    minimum_commission: NonNegativeFloat = 0.0
    fixed_spread_ticks: NonNegativeFloat = 1.0
    dynamic_spread_enabled: bool = True
    dynamic_spread_atr_mult: NonNegativeFloat = 0.10
    slippage_ticks_mean: NonNegativeFloat = 0.5
    slippage_ticks_std: NonNegativeFloat = 0.25
    volatility_dependent_slippage: bool = True
    volatility_slippage_mult: NonNegativeFloat = 0.5
    time_of_day_slippage: bool = True
    open_close_slippage_mult: PositiveFloat = 1.5
    liquidity_dependent_slippage: bool = True
    liquidity_volume_ref: PositiveFloat = 1_000.0
    overnight_swap_enabled: bool = True
    futures_rollover_cost_ticks: NonNegativeFloat = 0.0
    market_impact_enabled: bool = False
    market_impact_coeff: NonNegativeFloat = 0.0
    participation_rate_cap: UnitInterval = Field(
        0.10,
        description="Max fraction of bar volume a fill may consume.",
    )

    @model_validator(mode="after")
    def _nonneg_std(self) -> CostModel:
        if self.slippage_ticks_std < 0:
            raise ValueError("slippage_ticks_std must be >= 0")
        return self


def default_futures_cost_model() -> CostModel:
    return CostModel(
        version="futures_cost_v1",
        commission_per_contract=2.50,
        minimum_commission=2.50,
        fixed_spread_ticks=1.0,
        overnight_swap_enabled=False,
        futures_rollover_cost_ticks=1.0,
    )


def default_cfd_cost_model() -> CostModel:
    return CostModel(
        version="cfd_cost_v1",
        commission_per_contract=0.0,
        minimum_commission=0.0,
        fixed_spread_ticks=2.0,
        overnight_swap_enabled=True,
        futures_rollover_cost_ticks=0.0,
    )
