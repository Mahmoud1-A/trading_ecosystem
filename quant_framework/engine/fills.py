"""Fill records with full cost attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.models import Side
from engine.events import InformationTiming, require_aware


@dataclass(frozen=True)
class CostBreakdown:
    """Per-fill friction attribution (currency units)."""

    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    financing_cost: float = 0.0
    rollover_cost: float = 0.0
    market_impact_cost: float = 0.0

    @property
    def total_friction(self) -> float:
        return (
            self.commission
            + self.spread_cost
            + self.slippage_cost
            + self.financing_cost
            + self.rollover_cost
            + self.market_impact_cost
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "commission": self.commission,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "financing_cost": self.financing_cost,
            "rollover_cost": self.rollover_cost,
            "market_impact_cost": self.market_impact_cost,
            "total_friction": self.total_friction,
        }


@dataclass(frozen=True)
class Fill:
    """Immutable fill on an explicit tradable contract."""

    order_id: str
    symbol: str
    contract: str
    side: Side
    quantity: float
    price: float
    timing: InformationTiming
    costs: CostBreakdown
    fill_id: str = ""
    reference_price: float | None = None
    mid_price: float | None = None
    liquidity_used: float = 0.0
    gap_through: bool = False
    partial: bool = False
    rollover_decision: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be > 0")
        if self.contract.strip() == "":
            raise ValueError("Fill requires explicit tradable contract")
        if self.timing.fill_timestamp is None:
            raise ValueError("Fill timing must include fill_timestamp")
        if not self.fill_id:
            import hashlib

            material = (
                f"{self.order_id}|{self.timing.fill_timestamp}|{self.quantity}|{self.price}|{self.contract}"
            )
            object.__setattr__(
                self, "fill_id", "fil_" + hashlib.sha256(material.encode()).hexdigest()[:24]
            )
        object.__setattr__(
            self,
            "timing",
            InformationTiming(
                source_timestamp=self.timing.source_timestamp,
                availability_timestamp=self.timing.availability_timestamp,
                decision_timestamp=self.timing.decision_timestamp,
                order_submission_timestamp=self.timing.order_submission_timestamp,
                order_activation_timestamp=self.timing.order_activation_timestamp,
                fill_timestamp=require_aware(self.timing.fill_timestamp, name="fill_timestamp"),
            ),
        )

    @property
    def fill_timestamp(self) -> pd.Timestamp:
        assert self.timing.fill_timestamp is not None
        return self.timing.fill_timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "contract": self.contract,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "fill_timestamp": str(self.fill_timestamp),
            "reference_price": self.reference_price,
            "gap_through": self.gap_through,
            "partial": self.partial,
            "rollover_decision": self.rollover_decision,
            "costs": self.costs.as_dict(),
            "meta": dict(self.meta),
        }
