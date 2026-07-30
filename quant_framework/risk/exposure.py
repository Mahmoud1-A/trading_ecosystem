"""Portfolio exposure helpers for the risk engine."""

from __future__ import annotations

from dataclasses import dataclass

from config.models import Side
from engine.orders import Order
from engine.portfolio import Portfolio


@dataclass(frozen=True)
class ExposureSnapshot:
    """Aggregate open exposure at a point in time."""

    open_positions: int
    symbol_exposure: float
    portfolio_exposure: float
    symbols_with_positions: tuple[str, ...]


def snapshot_exposure(portfolio: Portfolio, symbol: str | None = None) -> ExposureSnapshot:
    """Compute signed lot exposure across the book."""
    symbols: list[str] = []
    total = 0.0
    sym_exp = 0.0
    for sym, pos in portfolio.positions.items():
        if pos.is_flat:
            continue
        symbols.append(sym)
        qty = abs(float(pos.quantity))
        total += qty
        if symbol is not None and sym == symbol:
            sym_exp = qty
    if symbol is not None and symbol not in symbols:
        sym_exp = 0.0
    return ExposureSnapshot(
        open_positions=len(symbols),
        symbol_exposure=sym_exp,
        portfolio_exposure=total,
        symbols_with_positions=tuple(sorted(symbols)),
    )


def order_delta_exposure(order: Order, portfolio: Portfolio) -> float:
    """
    Signed change in absolute exposure if the order fully fills.

    Positive => increases gross exposure; negative => reduces; zero => flat/no-op.
    """
    pos = portfolio.get_position(order.symbol)
    current = float(pos.quantity)
    signed = order.quantity if order.side == Side.BUY else -order.quantity
    if order.reduce_only:
        # Closing/reducing only — never increases gross book exposure
        if current == 0:
            return 0.0
        if current > 0 and signed < 0:
            return -min(order.quantity, abs(current))
        if current < 0 and signed > 0:
            return -min(order.quantity, abs(current))
        return 0.0

    new_qty = current + signed
    before = abs(current)
    after = abs(new_qty)
    return after - before


def is_risk_increasing(order: Order, portfolio: Portfolio) -> bool:
    """True when the order would increase gross exposure or open a new position."""
    if order.reduce_only:
        return False
    delta = order_delta_exposure(order, portfolio)
    return delta > 1e-12


def is_risk_reducing(order: Order, portfolio: Portfolio) -> bool:
    """True when the order only reduces or closes exposure."""
    if order.reduce_only:
        return True
    delta = order_delta_exposure(order, portfolio)
    return delta < -1e-12
