"""Demotion engine — map drift to decay states and risk caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from governance.degradation import RISK_MULTIPLIER, DecayState, allows_new_entries
from governance.drift import DriftReport, DriftSeverity


@dataclass
class DemotionDecision:
    previous: DecayState
    next_state: DecayState
    risk_multiplier: float
    new_entries_allowed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "previous": self.previous.value,
            "next_state": self.next_state.value,
            "risk_multiplier": self.risk_multiplier,
            "new_entries_allowed": self.new_entries_allowed,
            "reason": self.reason,
        }


# Ordered severity of states for monotonic demotion (never auto-upgrade past WATCH without revalidation)
_STATE_RANK = {
    DecayState.HEALTHY: 0,
    DecayState.WATCH: 1,
    DecayState.DEGRADED: 2,
    DecayState.REDUCE_ONLY: 3,
    DecayState.SUSPENDED: 4,
    DecayState.RETIRED: 5,
}


@dataclass
class DemotionEngine:
    def apply(self, current: DecayState, drift: DriftReport) -> DemotionDecision:
        if current is DecayState.RETIRED:
            return DemotionDecision(
                previous=current,
                next_state=DecayState.RETIRED,
                risk_multiplier=0.0,
                new_entries_allowed=False,
                reason="already_retired",
            )

        target = current
        reason = "stable"
        if drift.monitoring_failed:
            target = DecayState.SUSPENDED
            reason = "monitoring_failure_defaults_to_safe_state"
        elif drift.severity is DriftSeverity.CRITICAL:
            target = DecayState.SUSPENDED
            reason = "critical_drift_suspends_new_entries"
        elif drift.severity is DriftSeverity.SEVERE:
            target = DecayState.REDUCE_ONLY
            reason = "severe_degradation_reduce_only"
        elif drift.severity is DriftSeverity.MILD:
            target = DecayState.WATCH if current is DecayState.HEALTHY else DecayState.DEGRADED
            reason = "mild_drift_watch_or_degraded"

        # Monotonic demotion only (except staying)
        if _STATE_RANK[target] < _STATE_RANK[current]:
            target = current
            reason = "no_automatic_upgrade"

        # If currently DEGRADED and still mild/severe, stay or worsen
        if current is DecayState.DEGRADED and drift.severity is DriftSeverity.MILD:
            target = DecayState.DEGRADED
            reason = "degraded_remains_reduced_risk"

        return DemotionDecision(
            previous=current,
            next_state=target,
            risk_multiplier=RISK_MULTIPLIER[target],
            new_entries_allowed=allows_new_entries(target),
            reason=reason,
        )

    def allowed_risk(self, base_risk: float, state: DecayState) -> float:
        """Degradation reduces allowed risk."""
        return float(base_risk * RISK_MULTIPLIER[state])
