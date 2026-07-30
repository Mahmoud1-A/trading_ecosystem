"""System health and execution-quality API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from api.types import CounterKind, TypedCounter


@dataclass
class ExecutionQualityState:
    orders: int = 0
    fills: int = 0
    rejected_orders: int = 0
    partial_fills: int = 0
    latency_ms_sum: float = 0.0
    latency_samples: int = 0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    financing: float = 0.0
    rollover_cost: float = 0.0
    reconciliation_errors: int = 0

    def record_order(self, *, rejected: bool = False) -> None:
        self.orders += 1
        if rejected:
            self.rejected_orders += 1

    def record_fill(self, *, partial: bool = False, latency_ms: float = 0.0) -> None:
        self.fills += 1
        if partial:
            self.partial_fills += 1
        if latency_ms > 0:
            self.latency_ms_sum += latency_ms
            self.latency_samples += 1

    def record_costs(
        self,
        *,
        spread: float = 0.0,
        slippage: float = 0.0,
        commission: float = 0.0,
        financing: float = 0.0,
        rollover: float = 0.0,
    ) -> None:
        self.spread_cost += spread
        self.slippage_cost += slippage
        self.commission += commission
        self.financing += financing
        self.rollover_cost += rollover


@dataclass
class DataQualityState:
    provider_ok: bool = True
    missing_bars: int = 0
    stale_feeds: int = 0
    feature_failures: int = 0
    causality_tests_passed: int = 0
    causality_tests_failed: int = 0
    dataset_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class HealthAPI:
    execution: ExecutionQualityState = field(default_factory=ExecutionQualityState)
    data_quality: DataQualityState = field(default_factory=DataQualityState)
    system_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def execution_counters(self) -> list[TypedCounter]:
        e = self.execution
        avg_lat = e.latency_ms_sum / e.latency_samples if e.latency_samples else 0.0
        return [
            TypedCounter("orders", e.orders, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("fills", e.fills, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("rejected_orders", e.rejected_orders, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("partial_fills", e.partial_fills, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("avg_latency_ms", avg_lat, CounterKind.SNAPSHOT),
            TypedCounter("spread_cost", e.spread_cost, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("slippage_cost", e.slippage_cost, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("commission", e.commission, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("financing", e.financing, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("rollover_cost", e.rollover_cost, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("reconciliation_errors", e.reconciliation_errors, CounterKind.CUMULATIVE_EVENT),
        ]

    def data_quality_counters(self) -> list[TypedCounter]:
        d = self.data_quality
        return [
            TypedCounter("provider_healthy", int(d.provider_ok), CounterKind.SNAPSHOT),
            TypedCounter("missing_data", d.missing_bars, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("stale_data", d.stale_feeds, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("feature_failures", d.feature_failures, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("causality_passed", d.causality_tests_passed, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("causality_failed", d.causality_tests_failed, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("dataset_version_count", len(d.dataset_versions), CounterKind.SNAPSHOT),
        ]

    def system_counters(self) -> list[TypedCounter]:
        return [
            TypedCounter("system_ok", int(self.system_ok), CounterKind.SNAPSHOT),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_ok": self.system_ok,
            "notes": list(self.notes),
            "dataset_versions": dict(self.data_quality.dataset_versions),
            "execution": [c.as_dict() for c in self.execution_counters()],
            "data_quality": [c.as_dict() for c in self.data_quality_counters()],
            "system": [c.as_dict() for c in self.system_counters()],
        }
