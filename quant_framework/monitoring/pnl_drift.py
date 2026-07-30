"""PnL drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PnLDrift:
    expected_pnl: float
    actual_pnl: float
    drift: float
    drawdown: float
    daily_loss: float

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_pnl": self.expected_pnl,
            "actual_pnl": self.actual_pnl,
            "drift": self.drift,
            "drawdown": self.drawdown,
            "daily_loss": self.daily_loss,
        }


def measure_pnl_drift(
    *,
    expected_pnl: float,
    actual_pnl: float,
    drawdown: float = 0.0,
    daily_loss: float = 0.0,
) -> PnLDrift:
    return PnLDrift(
        expected_pnl=expected_pnl,
        actual_pnl=actual_pnl,
        drift=actual_pnl - expected_pnl,
        drawdown=drawdown,
        daily_loss=daily_loss,
    )
