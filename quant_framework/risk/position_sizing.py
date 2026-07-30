"""Volatility-targeted position sizing — expanded Phase 4."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config.asset_spec import CFDAssetSpec, FuturesAssetSpec
from config.models import RiskState
from config.prop_profile import PropProfile
from config.strategy_config import SizingParams
from engine.portfolio import Portfolio
from risk.exposure import snapshot_exposure


@dataclass(frozen=True)
class SizingDecision:
    allowed_quantity: float
    rejected: bool
    reason: str = ""
    risk_dollars: float = 0.0
    vol_scalar: float = 1.0
    binding_constraint: str = ""
    constraints: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed_quantity": self.allowed_quantity,
            "rejected": self.rejected,
            "reason": self.reason,
            "risk_dollars": self.risk_dollars,
            "vol_scalar": self.vol_scalar,
            "binding_constraint": self.binding_constraint,
            "constraints": dict(self.constraints),
        }


def _round_lot(qty: float, step: float, minimum: float) -> float:
    if qty < minimum:
        return 0.0
    steps = math.floor(qty / step + 1e-12)
    return max(minimum, steps * step) if steps >= 1 else 0.0


def margin_per_unit(
    asset: FuturesAssetSpec | CFDAssetSpec,
    *,
    price: float | None = None,
    initial_margin_per_unit: float | None = None,
    leverage: float | None = None,
) -> float | None:
    """
    Initial margin required for one contract / lot.

    Futures use the exchange initial margin. CFDs derive it from notional and either
    the explicit leverage override or the spec's ``margin_rate``. Returns ``None``
    when the requirement cannot be determined (e.g. CFD without a price).
    """
    if initial_margin_per_unit is not None:
        return float(initial_margin_per_unit)
    if isinstance(asset, FuturesAssetSpec):
        return float(asset.initial_margin)
    if price is None or price <= 0:
        return None
    notional = float(asset.contract_size) * float(price)
    if leverage is not None and leverage > 0:
        return notional / float(leverage)
    return notional * float(asset.margin_rate)


def size_by_risk_budget(
    *,
    requested_qty: float,
    risk_state: RiskState,
    prop_profile: PropProfile,
    portfolio: Portfolio,
    symbol: str,
    stop_distance_points: float | None = None,
    tick_value: float = 1.0,
) -> SizingDecision:
    """
    Apply FSM risk budget and exposure caps to a requested quantity.

    Never increases size to force a trade.
    """
    if requested_qty <= 0:
        return SizingDecision(0.0, True, "non_positive_quantity")

    qty = float(requested_qty)
    if risk_state == RiskState.CAUTION:
        qty *= float(prop_profile.internal_risk_budget)

    exp = snapshot_exposure(portfolio, symbol)
    remaining_symbol = float(prop_profile.max_symbol_exposure) - exp.symbol_exposure
    remaining_book = float(prop_profile.max_portfolio_exposure) - exp.portfolio_exposure
    qty = min(qty, max(0.0, remaining_symbol), max(0.0, remaining_book))

    if qty <= 0:
        return SizingDecision(0.0, True, "exposure_cap")

    if stop_distance_points is not None and stop_distance_points > 0:
        equity = portfolio.equity()
        risk_cash = equity * float(prop_profile.internal_risk_budget) * 0.01
        per_contract_risk = stop_distance_points * tick_value
        if per_contract_risk > 0:
            atr_cap = risk_cash / per_contract_risk
            qty = min(qty, atr_cap)

    if qty < requested_qty * 0.01:
        return SizingDecision(0.0, True, "below_minimum_tradable")

    return SizingDecision(qty, False)


def volatility_target_size(
    *,
    portfolio: Portfolio,
    sizing: SizingParams,
    asset: FuturesAssetSpec | CFDAssetSpec,
    atr_points: float,
    realized_vol: float | None = None,
    risk_state: RiskState = RiskState.NORMAL,
    risk_budget_multiplier: float = 1.0,
    symbol: str = "ES",
    prop_profile: PropProfile | None = None,
    price: float | None = None,
    initial_margin_per_unit: float | None = None,
    maintenance_margin_per_unit: float | None = None,
    leverage: float | None = None,
    stop_out_level: float | None = None,
    used_margin: float = 0.0,
    free_margin: float | None = None,
    remaining_soft_daily_budget: float | None = None,
    remaining_hard_daily_budget: float | None = None,
    reduce_only: bool = False,
    max_symbol_exposure: float | None = None,
    max_portfolio_exposure: float | None = None,
) -> SizingDecision:
    """
    ATR / volatility-targeted sizing under every simultaneously binding constraint.

    The allowed quantity is the minimum of the vol-target size and every applicable
    cap: configured max lots, symbol and portfolio exposure, free margin, the
    maintenance-margin stop-out buffer, and the remaining soft/hard daily risk
    budgets. Sizing is only ever reduced — it is never increased to reach the
    minimum tradable lot, and a size below the minimum lot is rejected instead.

    Margin/budget arguments left as ``None`` are treated as not applicable rather
    than as zero, so callers opt in to each constraint explicitly.
    """
    constraints: dict[str, float] = {}
    if atr_points <= 0 or not math.isfinite(atr_points):
        return SizingDecision(0.0, True, "invalid_atr", binding_constraint="atr")

    equity = portfolio.equity()
    stop_dist = atr_points * float(sizing.atr_stop_mult)
    if isinstance(asset, FuturesAssetSpec):
        point_value = float(asset.multiplier)
    else:
        point_value = float(asset.contract_size)

    per_contract_risk = stop_dist * point_value
    risk_pct = float(sizing.risk_per_trade_pct) * float(risk_budget_multiplier)
    if risk_state in {RiskState.HALTED, RiskState.MANUAL_LOCK}:
        return SizingDecision(0.0, True, f"risk_state_{risk_state.value.lower()}",
                              binding_constraint="risk_state")

    exp = snapshot_exposure(portfolio, symbol)

    if risk_state == RiskState.REDUCE_ONLY:
        # REDUCE_ONLY may never add exposure: only an explicit reduce-only order
        # bounded by the currently open quantity is permitted.
        if not reduce_only or exp.symbol_exposure <= 0:
            return SizingDecision(0.0, True, "reduce_only", binding_constraint="reduce_only")

    risk_dollars = equity * risk_pct
    qty = risk_dollars / max(per_contract_risk, 1e-12)
    constraints["vol_target"] = qty

    # CAUTION trades a reduced fraction of the per-trade risk budget
    if risk_state == RiskState.CAUTION and prop_profile is not None:
        qty *= float(prop_profile.internal_risk_budget)
        constraints["caution_budget"] = qty

    vol_scalar = 1.0
    if realized_vol is not None and realized_vol > 0:
        target_bar = float(sizing.target_vol_annual) / math.sqrt(float(sizing.bars_per_year))
        vol_scalar = min(1.0, target_bar / realized_vol)
        qty *= vol_scalar
        constraints["vol_scalar"] = qty

    binding = "vol_target"

    def _apply(name: str, cap: float) -> None:
        nonlocal qty, binding
        capped = max(0.0, float(cap))
        constraints[name] = capped
        if capped + 1e-12 < qty:
            binding = name
            qty = capped

    _apply("max_lots", float(sizing.max_lots))

    symbol_cap = float(max_symbol_exposure) if max_symbol_exposure is not None else None
    book_cap = float(max_portfolio_exposure) if max_portfolio_exposure is not None else None
    if prop_profile is not None:
        symbol_cap = min(
            symbol_cap if symbol_cap is not None else float("inf"),
            float(prop_profile.max_symbol_exposure),
        )
        book_cap = min(
            book_cap if book_cap is not None else float("inf"),
            float(prop_profile.max_portfolio_exposure),
        )
    if symbol_cap is None:
        symbol_cap = float(sizing.max_lots)
    if book_cap is None:
        book_cap = float(sizing.max_lots)

    if reduce_only:
        # A reducing order can never exceed what is currently open
        _apply("reduce_only_open_quantity", exp.symbol_exposure)
    else:
        _apply("symbol_exposure", symbol_cap - exp.symbol_exposure)
        _apply("portfolio_exposure", book_cap - exp.portfolio_exposure)

    unit_margin = margin_per_unit(
        asset,
        price=price,
        initial_margin_per_unit=initial_margin_per_unit,
        leverage=leverage,
    )
    if unit_margin is not None and unit_margin > 0 and not reduce_only:
        available = float(free_margin) if free_margin is not None else max(0.0, equity - used_margin)
        if available <= 0:
            return SizingDecision(
                0.0,
                True,
                "insufficient_free_margin",
                risk_dollars=risk_dollars,
                vol_scalar=vol_scalar,
                binding_constraint="free_margin",
                constraints=constraints,
            )
        margin_cap = available / unit_margin
        # Cannot afford even one minimum lot under current free margin
        min_lot_peek = float(asset.min_lot if isinstance(asset, CFDAssetSpec) else sizing.min_lot)
        if margin_cap + 1e-12 < min_lot_peek:
            return SizingDecision(
                0.0,
                True,
                "insufficient_free_margin",
                risk_dollars=risk_dollars,
                vol_scalar=vol_scalar,
                binding_constraint="free_margin",
                constraints={**constraints, "free_margin": margin_cap, "min_lot": min_lot_peek},
            )
        _apply("free_margin", margin_cap)

        # Stop-out buffer: equity must stay above stop_out_level * used margin
        level = (
            float(stop_out_level)
            if stop_out_level is not None
            else (float(asset.stop_out_level) if isinstance(asset, CFDAssetSpec) else None)
        )
        maint = (
            float(maintenance_margin_per_unit)
            if maintenance_margin_per_unit is not None
            else (float(asset.maintenance_margin) if isinstance(asset, FuturesAssetSpec) else unit_margin)
        )
        if level is not None and level > 0 and maint > 0:
            allowed_maint = equity / level - float(used_margin)
            _apply("stop_out_buffer", allowed_maint / maint)

    for name, budget in (
        ("hard_daily_budget", remaining_hard_daily_budget),
        ("soft_daily_budget", remaining_soft_daily_budget),
    ):
        if budget is None:
            continue
        if float(budget) <= 0:
            return SizingDecision(
                0.0,
                True,
                f"{name}_exhausted",
                risk_dollars=risk_dollars,
                vol_scalar=vol_scalar,
                binding_constraint=name,
                constraints=constraints,
            )
        _apply(name, float(budget) / max(per_contract_risk, 1e-12))

    lot_step = float(asset.lot_step if isinstance(asset, CFDAssetSpec) else sizing.lot_step)
    min_lot = float(asset.min_lot if isinstance(asset, CFDAssetSpec) else sizing.min_lot)
    # _round_lot floors to the lot step and never rounds up past the requested size
    qty = _round_lot(qty, lot_step, min_lot)

    if qty <= 0:
        return SizingDecision(
            0.0,
            True,
            "below_minimum_tradable",
            risk_dollars=risk_dollars,
            vol_scalar=vol_scalar,
            binding_constraint=binding,
            constraints=constraints,
        )

    return SizingDecision(
        qty,
        False,
        risk_dollars=risk_dollars,
        vol_scalar=vol_scalar,
        binding_constraint=binding,
        constraints=constraints,
    )
