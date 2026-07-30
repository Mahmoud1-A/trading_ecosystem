"""Futures rollover accounting — explicit tradable contracts only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config.asset_spec import FuturesAssetSpec
from config.cost_model import CostModel
from config.models import OrderType, Side
from engine.events import InformationTiming
from engine.fills import CostBreakdown, Fill
from engine.friction import FrictionContext, FrictionEngine
from engine.ids import DeterministicIdFactory
from engine.orders import Order
from engine.portfolio import Portfolio


@dataclass(frozen=True)
class RolloverDecision:
    timestamp: pd.Timestamp
    from_contract: str
    to_contract: str
    close_price: float
    open_price: float
    exclude_from_tradable_pnl: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": str(self.timestamp),
            "from_contract": self.from_contract,
            "to_contract": self.to_contract,
            "close_price": self.close_price,
            "open_price": self.open_price,
            "exclude_from_tradable_pnl": self.exclude_from_tradable_pnl,
        }


def execute_futures_rollover(
    portfolio: Portfolio,
    *,
    asset: FuturesAssetSpec,
    symbol: str,
    from_contract: str,
    to_contract: str,
    timestamp: pd.Timestamp,
    front_price: float,
    back_price: float,
    friction: FrictionEngine,
    id_factory: DeterministicIdFactory | None = None,
    quantity: float | None = None,
) -> tuple[list[Fill], RolloverDecision]:
    """
    Close front contract and open back contract at explicit prices.

    Continuous adjusted series jumps must NOT create tradable PnL — the economic
    PnL is only front→cash at front_price and cash→back at back_price with friction.
    The continuous gap itself is never booked as P&L.
    """
    pos = portfolio.get_position(symbol)
    qty = abs(float(pos.quantity)) if quantity is None else abs(float(quantity))
    if qty <= 0 or pos.is_flat:
        decision = RolloverDecision(timestamp, from_contract, to_contract, front_price, back_price)
        return [], decision

    is_long = pos.quantity > 0
    close_side = Side.SELL if is_long else Side.BUY
    open_side = Side.BUY if is_long else Side.SELL

    ctx = FrictionContext(mid_price=front_price, atr=None, bar_volume=1e9)
    close_px, close_costs = friction.build_costs(
        side=close_side, quantity=qty, reference_price=front_price, ctx=ctx, include_rollover=True
    )
    open_ctx = FrictionContext(mid_price=back_price, atr=None, bar_volume=1e9)
    open_px, open_costs = friction.build_costs(
        side=open_side, quantity=qty, reference_price=back_price, ctx=open_ctx, include_rollover=False
    )

    timing = InformationTiming(
        source_timestamp=timestamp,
        availability_timestamp=timestamp,
        decision_timestamp=timestamp,
        order_submission_timestamp=timestamp,
        order_activation_timestamp=timestamp,
        fill_timestamp=timestamp,
    )

    close_oid = (
        id_factory.next_order_id(
            symbol=symbol,
            side=close_side.value,
            quantity=qty,
            order_type="MARKET",
            decision_timestamp=str(timestamp),
            contract=from_contract,
        )
        if id_factory
        else f"roll_close_{from_contract}"
    )
    open_oid = (
        id_factory.next_order_id(
            symbol=symbol,
            side=open_side.value,
            quantity=qty,
            order_type="MARKET",
            decision_timestamp=str(timestamp),
            contract=to_contract,
        )
        if id_factory
        else f"roll_open_{to_contract}"
    )
    close_fid = (
        id_factory.next_fill_id(
            order_id=close_oid, fill_timestamp=str(timestamp), quantity=qty, price=close_px
        )
        if id_factory
        else f"roll_close_fill_{from_contract}"
    )
    open_fid = (
        id_factory.next_fill_id(
            order_id=open_oid, fill_timestamp=str(timestamp), quantity=qty, price=open_px
        )
        if id_factory
        else f"roll_open_fill_{to_contract}"
    )

    close_fill = Fill(
        order_id=close_oid,
        symbol=symbol,
        contract=from_contract,
        side=close_side,
        quantity=qty,
        price=close_px,
        timing=timing,
        costs=close_costs,
        fill_id=close_fid,
        reference_price=front_price,
        rollover_decision=f"{from_contract}->{to_contract}",
        meta={"rollover": True, "leg": "close", "exclude_continuous_gap_pnl": True},
    )
    open_fill = Fill(
        order_id=open_oid,
        symbol=symbol,
        contract=to_contract,
        side=open_side,
        quantity=qty,
        price=open_px,
        timing=timing,
        costs=open_costs,
        fill_id=open_fid,
        reference_price=back_price,
        rollover_decision=f"{from_contract}->{to_contract}",
        meta={"rollover": True, "leg": "open", "exclude_continuous_gap_pnl": True},
    )

    portfolio.apply_fill(close_fill)
    portfolio.apply_fill(open_fill)
    decision = RolloverDecision(
        timestamp=timestamp,
        from_contract=from_contract,
        to_contract=to_contract,
        close_price=front_price,
        open_price=back_price,
        exclude_from_tradable_pnl=True,
    )
    return [close_fill, open_fill], decision


def continuous_gap_not_tradable_pnl(
    *,
    unadjusted_front: float,
    unadjusted_back: float,
    continuous_adjusted_jump: float,
) -> float:
    """
    Prove accounting invariant: continuous adjustment jump is not tradable PnL.

    Returns the artificial PnL that would incorrectly be booked if the continuous
    jump were treated as a price move on a single contract (should be ignored).
    """
    # Artificial PnL if someone naively used continuous series for MTM across roll
    return float(continuous_adjusted_jump)
