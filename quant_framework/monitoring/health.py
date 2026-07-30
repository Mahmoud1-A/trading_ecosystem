"""Paper / shadow runtime health."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeHealth:
    ok: bool = True
    broker_connected: bool = True
    kill_switch_engaged: bool = False
    recon_ok: bool = True
    missing_signals: int = 0
    duplicate_signals: int = 0
    rejected_orders: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "broker_connected": self.broker_connected,
            "kill_switch_engaged": self.kill_switch_engaged,
            "recon_ok": self.recon_ok,
            "missing_signals": self.missing_signals,
            "duplicate_signals": self.duplicate_signals,
            "rejected_orders": self.rejected_orders,
            "notes": list(self.notes),
        }
