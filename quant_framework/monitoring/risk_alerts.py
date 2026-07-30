"""Risk alerts for paper monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskAlerts:
    prop_breach_distance: float = 1.0
    daily_loss: float = 0.0
    drawdown: float = 0.0
    alerts: list[str] = field(default_factory=list)

    def update(
        self,
        *,
        prop_breach_distance: float,
        daily_loss: float,
        drawdown: float,
        daily_loss_limit: float = 0.05,
        drawdown_limit: float = 0.10,
    ) -> None:
        self.prop_breach_distance = prop_breach_distance
        self.daily_loss = daily_loss
        self.drawdown = drawdown
        self.alerts.clear()
        if prop_breach_distance < 0.1:
            self.alerts.append("prop_breach_near")
        if daily_loss > daily_loss_limit:
            self.alerts.append("daily_loss_breach")
        if drawdown > drawdown_limit:
            self.alerts.append("drawdown_breach")

    def as_dict(self) -> dict[str, Any]:
        return {
            "prop_breach_distance": self.prop_breach_distance,
            "daily_loss": self.daily_loss,
            "drawdown": self.drawdown,
            "alerts": list(self.alerts),
        }
