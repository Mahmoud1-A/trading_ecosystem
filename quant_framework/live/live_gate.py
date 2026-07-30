"""Live order gate — final barrier before any live submission path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from live.approvals import ApprovalRegistry
from live.capital_ramp import CapitalRamp, CapitalStage
from live.deployment import Environment, LiveConfig, LiveEnableError
from live.readiness import ReadinessReport
from live.rollback import RollbackController
from live.incident_response import IncidentResponse


@dataclass
class LiveOrderGate:
    """
    Hard gate for live orders.

    This framework never auto-enables real-money trading. Even when armed,
    callers must pass through this gate which re-checks every safety condition.
    """

    config: LiveConfig
    ramp: CapitalRamp
    approvals: ApprovalRegistry
    readiness: ReadinessReport
    rollback: RollbackController
    incidents: IncidentResponse

    def assert_live_order_allowed(
        self,
        *,
        notional: float,
        credentials_scope: str,
    ) -> None:
        if self.config.environment is Environment.TEST:
            raise LiveEnableError("live orders impossible in test environment")
        if self.config.environment is Environment.PAPER:
            raise LiveEnableError("live orders impossible in paper environment")
        if credentials_scope != "live":
            raise LiveEnableError("test/paper credentials cannot send live orders")
        if not self.readiness.complete:
            raise LiveEnableError("readiness report incomplete")
        if not self.config.commit_approved:
            raise LiveEnableError("production mode cannot start from an unapproved commit")
        self.incidents.assert_not_safe()
        self.rollback.assert_new_risk_allowed()
        if not self.ramp.live_orders_armed:
            raise LiveEnableError("live orders impossible without manual authorization arming")
        self.approvals.require(
            config_hash=self.config.frozen_hash,
            git_commit=self.config.git_commit,
            stage=self.ramp.stage.value,
        )
        try:
            self.ramp.assert_within_capital(notional)
        except Exception as exc:
            raise LiveEnableError(str(exc)) from exc

        # Local vs broker risk limits — both must allow
        local_cap = float(self.config.local_risk_limits.get("max_notional", 0.0))
        broker_cap = float(self.config.broker_risk_limits.get("max_notional", 0.0))
        if notional > local_cap + 1e-9:
            raise LiveEnableError("local risk limit cannot be bypassed")
        if notional > broker_cap + 1e-9:
            raise LiveEnableError("broker risk limit cannot be bypassed")

        if self.ramp.stage is CapitalStage.CANARY_LIVE:
            from live.canary import CanaryViolation

            try:
                self.ramp.canary.validate_order(
                    symbol=self.ramp.canary.limits.allowed_symbols[0],
                    session=self.ramp.canary.limits.allowed_sessions[0],
                    size=self.ramp.canary.limits.min_tradable_size,
                    projected_risk=min(notional, self.ramp.canary.limits.max_daily_risk),
                )
            except CanaryViolation as exc:
                raise LiveEnableError(str(exc)) from exc

    def try_submit_live(self, *, notional: float, credentials_scope: str) -> dict[str, Any]:
        """
        Research stub — never places a real broker live order.

        Returns an audit record proving the gate was evaluated. Actual broker
        live submission is out of scope for this phase and remains disabled.
        """
        self.assert_live_order_allowed(notional=notional, credentials_scope=credentials_scope)
        return {
            "accepted_by_gate": True,
            "live_broker_submit": False,
            "reason": "phase10_readiness_only_no_automatic_live_trading",
            "stage": self.ramp.stage.value,
            "notional": notional,
            "capital_limit": self.ramp.capital_limit(),
        }
