"""Rollback — disables new risk immediately."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from live.capital_ramp import CapitalRamp, CapitalStage, StageTransition
from live.deployment import LiveEnableError


@dataclass
class RollbackEvent:
    timestamp: str
    from_stage: str
    to_stage: str
    reason: str
    new_risk_disabled: bool
    live_orders_disarmed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "reason": self.reason,
            "new_risk_disabled": self.new_risk_disabled,
            "live_orders_disarmed": self.live_orders_disarmed,
        }


@dataclass
class RollbackController:
    ramp: CapitalRamp
    new_risk_disabled: bool = False
    events: list[RollbackEvent] = field(default_factory=list)
    flatten_callback: Callable[[], None] | None = None

    def rollback(self, *, reason: str, to_stage: CapitalStage = CapitalStage.PAPER) -> RollbackEvent:
        """Immediate rollback: disarm live orders, disable new risk, drop stage."""
        prev = self.ramp.stage
        order = list(CapitalStage)
        if order.index(to_stage) > order.index(prev):
            raise LiveEnableError("rollback cannot increase capital stage")

        self.ramp.live_orders_armed = False
        self.new_risk_disabled = True
        self.ramp.stage = to_stage
        self.ramp.previous_stage_successful = False
        if self.flatten_callback is not None:
            self.flatten_callback()

        ts = datetime.now(tz=timezone.utc).isoformat()
        self.ramp.audit.append(
            StageTransition(
                from_stage=prev,
                to_stage=to_stage,
                timestamp=ts,
                approver="rollback_system",
                config_hash=self.ramp.config.frozen_hash,
                git_commit=self.ramp.config.git_commit,
                observation_hours=0.0,
                gates={"rollback": True, "new_risk_disabled": True},
            )
        )
        event = RollbackEvent(
            timestamp=ts,
            from_stage=prev.value,
            to_stage=to_stage.value,
            reason=reason,
            new_risk_disabled=True,
            live_orders_disarmed=True,
        )
        self.events.append(event)
        return event

    def assert_new_risk_allowed(self) -> None:
        if self.new_risk_disabled:
            raise LiveEnableError(
                "rollback disabled new risk — cannot open new live exposure"
            )
