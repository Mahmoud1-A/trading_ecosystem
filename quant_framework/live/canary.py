"""Canary live constraints — minimum size, limited symbols/sessions/risk."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class CanaryViolation(ValueError):
    pass


@dataclass(frozen=True)
class CanaryLimits:
    min_tradable_size: float = 1.0
    max_tradable_size: float = 1.0  # minimum tradable only by default
    allowed_symbols: tuple[str, ...] = ("ES",)
    allowed_sessions: tuple[str, ...] = ("rth",)
    max_daily_risk: float = 500.0
    hard_stop_loss: float = 250.0
    max_open_orders: int = 2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanaryController:
    limits: CanaryLimits = field(default_factory=CanaryLimits)
    daily_risk_used: float = 0.0
    open_orders: int = 0
    fill_comparisons: list[dict[str, Any]] = field(default_factory=list)

    def validate_order(
        self,
        *,
        symbol: str,
        session: str,
        size: float,
        projected_risk: float,
    ) -> None:
        if symbol not in self.limits.allowed_symbols:
            raise CanaryViolation(f"symbol {symbol} not in canary allowlist")
        if session not in self.limits.allowed_sessions:
            raise CanaryViolation(f"session {session} not allowed in canary")
        if size < self.limits.min_tradable_size or size > self.limits.max_tradable_size:
            raise CanaryViolation(
                f"size {size} outside canary [{self.limits.min_tradable_size}, {self.limits.max_tradable_size}]"
            )
        if self.daily_risk_used + projected_risk > self.limits.max_daily_risk:
            raise CanaryViolation("canary max_daily_risk exceeded")
        if self.open_orders >= self.limits.max_open_orders:
            raise CanaryViolation("canary max_open_orders exceeded")

    def record_risk(self, amount: float) -> None:
        self.daily_risk_used += amount
        if self.daily_risk_used >= self.limits.hard_stop_loss:
            raise CanaryViolation("canary hard_stop_loss breached — immediate rollback required")

    def record_fill_comparison(self, expected: dict[str, Any], actual: dict[str, Any]) -> None:
        self.fill_comparisons.append({"expected": expected, "actual": actual})

    def as_dict(self) -> dict[str, Any]:
        return {
            "limits": self.limits.as_dict(),
            "daily_risk_used": self.daily_risk_used,
            "open_orders": self.open_orders,
            "fill_comparisons": list(self.fill_comparisons),
        }
