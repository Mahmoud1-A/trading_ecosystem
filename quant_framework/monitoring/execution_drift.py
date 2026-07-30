"""Execution drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from brokers.protocol import BrokerFill


@dataclass
class ExecutionDrift:
    latency_drift_ms: float
    fill_rate_drift: float
    spread_drift: float
    slippage_drift: float
    cost_drift: float

    def as_dict(self) -> dict[str, float]:
        return {
            "latency_drift_ms": self.latency_drift_ms,
            "fill_rate_drift": self.fill_rate_drift,
            "spread_drift": self.spread_drift,
            "slippage_drift": self.slippage_drift,
            "cost_drift": self.cost_drift,
        }


def measure_execution_drift(
    *,
    expected_fills: Sequence[BrokerFill],
    actual_fills: Sequence[BrokerFill],
    expected_latency_ms: float = 0.0,
    actual_latency_ms: float = 0.0,
    expected_spread: float = 0.0,
    actual_spread: float = 0.0,
    expected_slippage: float = 0.0,
    actual_slippage: float = 0.0,
    expected_cost: float = 0.0,
    actual_cost: float = 0.0,
) -> ExecutionDrift:
    exp_qty = sum(f.quantity for f in expected_fills) or 1.0
    act_qty = sum(f.quantity for f in actual_fills)
    return ExecutionDrift(
        latency_drift_ms=actual_latency_ms - expected_latency_ms,
        fill_rate_drift=(act_qty / exp_qty) - 1.0,
        spread_drift=actual_spread - expected_spread,
        slippage_drift=actual_slippage - expected_slippage,
        cost_drift=actual_cost - expected_cost,
    )
