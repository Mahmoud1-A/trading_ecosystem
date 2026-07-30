"""Strategy decay / health states for continuous monitoring."""

from __future__ import annotations

from enum import Enum


class DecayState(str, Enum):
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    REDUCE_ONLY = "REDUCE_ONLY"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


# Risk multipliers by state — degraded strategies receive smaller risk
RISK_MULTIPLIER = {
    DecayState.HEALTHY: 1.0,
    DecayState.WATCH: 0.75,
    DecayState.DEGRADED: 0.40,
    DecayState.REDUCE_ONLY: 0.0,  # no new risk; flatten/reduce only
    DecayState.SUSPENDED: 0.0,
    DecayState.RETIRED: 0.0,
}


def allows_new_entries(state: DecayState) -> bool:
    return state in {DecayState.HEALTHY, DecayState.WATCH, DecayState.DEGRADED}


def allows_automatic_reactivation(state: DecayState) -> bool:
    """Retired strategies cannot automatically return."""
    return state is not DecayState.RETIRED
