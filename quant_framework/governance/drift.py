"""Distribution and metric drift detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class DriftSeverity(str, Enum):
    NONE = "NONE"
    MILD = "MILD"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class MetricSnapshot:
    realized_expectancy: float = 0.0
    rolling_sharpe: float = 0.0
    rolling_drawdown: float = 0.0
    execution_slippage: float = 0.0
    spread: float = 0.0
    fill_rate: float = 1.0
    rejected_orders: float = 0.0
    signal_frequency: float = 0.0
    holding_period: float = 0.0
    prop_breach_probability: float = 0.0
    data_quality: float = 1.0
    provider_quality: float = 1.0
    strategy_contribution: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "realized_expectancy": self.realized_expectancy,
            "rolling_sharpe": self.rolling_sharpe,
            "rolling_drawdown": self.rolling_drawdown,
            "execution_slippage": self.execution_slippage,
            "spread": self.spread,
            "fill_rate": self.fill_rate,
            "rejected_orders": self.rejected_orders,
            "signal_frequency": self.signal_frequency,
            "holding_period": self.holding_period,
            "prop_breach_probability": self.prop_breach_probability,
            "data_quality": self.data_quality,
            "provider_quality": self.provider_quality,
            "strategy_contribution": self.strategy_contribution,
        }


@dataclass
class DriftReport:
    severity: DriftSeverity
    metric_deltas: dict[str, float]
    distribution_ks: dict[str, float] = field(default_factory=dict)
    monitoring_failed: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "metric_deltas": dict(self.metric_deltas),
            "distribution_ks": dict(self.distribution_ks),
            "monitoring_failed": self.monitoring_failed,
            "reason": self.reason,
        }


def _ks_stat(a: Sequence[float], b: Sequence[float]) -> float:
    """Simple two-sample KS statistic (sup |F1-F2|)."""
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    if len(x) == 0 or len(y) == 0:
        return 1.0
    grid = np.sort(np.unique(np.concatenate([x, y])))
    fx = np.searchsorted(x, grid, side="right") / len(x)
    fy = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


@dataclass
class DriftMonitor:
    """Compare expected backtest / Vault / Paper / live distributions."""

    mild_sharpe_drop: float = 0.3
    severe_sharpe_drop: float = 0.7
    critical_sharpe_drop: float = 1.2
    critical_ks: float = 0.55
    severe_ks: float = 0.35

    def compare_metrics(
        self,
        expected: MetricSnapshot,
        observed: MetricSnapshot,
        *,
        distributions: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    ) -> DriftReport:
        try:
            exp = expected.as_dict()
            obs = observed.as_dict()
            deltas = {k: obs[k] - exp[k] for k in exp}
            ks_scores: dict[str, float] = {}
            if distributions:
                for name, (ref, live) in distributions.items():
                    ks_scores[name] = _ks_stat(ref, live)

            sharpe_drop = exp["rolling_sharpe"] - obs["rolling_sharpe"]
            dd_worse = obs["rolling_drawdown"] - exp["rolling_drawdown"]  # more negative is worse if dd stored negative
            max_ks = max(ks_scores.values()) if ks_scores else 0.0

            severity = DriftSeverity.NONE
            if sharpe_drop >= self.mild_sharpe_drop or max_ks >= 0.2:
                severity = DriftSeverity.MILD
            if sharpe_drop >= self.severe_sharpe_drop or max_ks >= self.severe_ks or dd_worse < -0.05:
                severity = DriftSeverity.SEVERE
            if (
                sharpe_drop >= self.critical_sharpe_drop
                or max_ks >= self.critical_ks
                or obs["prop_breach_probability"] > 0.5
                or obs["data_quality"] < 0.5
            ):
                severity = DriftSeverity.CRITICAL

            return DriftReport(severity=severity, metric_deltas=deltas, distribution_ks=ks_scores)
        except Exception as exc:  # noqa: BLE001 — monitoring failure → safe
            return DriftReport(
                severity=DriftSeverity.CRITICAL,
                metric_deltas={},
                monitoring_failed=True,
                reason=f"monitoring_failure:{exc}",
            )
