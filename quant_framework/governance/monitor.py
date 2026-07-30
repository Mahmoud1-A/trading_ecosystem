"""Continuous monitoring controller — wires drift → demotion → safe defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from governance.audit import GovernanceAudit
from governance.degradation import DecayState
from governance.demotion import DemotionDecision, DemotionEngine
from governance.drift import DriftMonitor, DriftReport, MetricSnapshot
from governance.retirement import RetirementError, RetirementRegistry


@dataclass
class MonitoredStrategy:
    strategy_id: str
    lineage_id: str
    state: DecayState = DecayState.HEALTHY
    base_risk: float = 1.0
    expected: MetricSnapshot = field(default_factory=MetricSnapshot)
    last_drift: DriftReport | None = None
    last_demotion: DemotionDecision | None = None

    @property
    def allowed_risk(self) -> float:
        from governance.degradation import RISK_MULTIPLIER

        return self.base_risk * RISK_MULTIPLIER[self.state]

    @property
    def new_entries_allowed(self) -> bool:
        from governance.degradation import allows_new_entries

        return allows_new_entries(self.state)


@dataclass
class MonitoringController:
    """
    Continuous alpha-decay monitoring.

    Monitoring failures default to SUSPENDED (safe state).
    """

    drift_monitor: DriftMonitor = field(default_factory=DriftMonitor)
    demotion: DemotionEngine = field(default_factory=DemotionEngine)
    retirement: RetirementRegistry = field(default_factory=RetirementRegistry)
    audit: GovernanceAudit = field(default_factory=GovernanceAudit)
    strategies: dict[str, MonitoredStrategy] = field(default_factory=dict)

    def register(self, strategy: MonitoredStrategy) -> None:
        self.strategies[strategy.strategy_id] = strategy

    def update(
        self,
        strategy_id: str,
        observed: MetricSnapshot,
        *,
        distributions: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    ) -> DemotionDecision:
        strat = self.strategies[strategy_id]
        if self.retirement.is_retired(strategy_id):
            strat.state = DecayState.RETIRED
            decision = DemotionDecision(
                previous=DecayState.RETIRED,
                next_state=DecayState.RETIRED,
                risk_multiplier=0.0,
                new_entries_allowed=False,
                reason="retired",
            )
            strat.last_demotion = decision
            return decision

        drift = self.drift_monitor.compare_metrics(
            strat.expected, observed, distributions=distributions
        )
        strat.last_drift = drift
        decision = self.demotion.apply(strat.state, drift)
        try:
            self.retirement.assert_not_auto_reactivate(
                strategy_id, new_state=decision.next_state
            )
        except RetirementError:
            decision = DemotionDecision(
                previous=strat.state,
                next_state=DecayState.RETIRED,
                risk_multiplier=0.0,
                new_entries_allowed=False,
                reason="retired_blocks_reactivation",
            )
        strat.state = decision.next_state
        strat.last_demotion = decision
        self.audit.record(
            "demotion",
            {"strategy_id": strategy_id, **decision.as_dict(), "drift": drift.as_dict()},
        )
        return decision

    def retire(self, strategy_id: str, *, reason: str) -> None:
        strat = self.strategies[strategy_id]
        self.retirement.retire(
            strategy_id=strategy_id, lineage_id=strat.lineage_id, reason=reason
        )
        strat.state = DecayState.RETIRED
        self.audit.record("retirement", {"strategy_id": strategy_id, "reason": reason})

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategies": {
                sid: {
                    "state": s.state.value,
                    "allowed_risk": s.allowed_risk,
                    "new_entries_allowed": s.new_entries_allowed,
                    "lineage_id": s.lineage_id,
                }
                for sid, s in self.strategies.items()
            },
            "audit": self.audit.as_dict(),
        }
