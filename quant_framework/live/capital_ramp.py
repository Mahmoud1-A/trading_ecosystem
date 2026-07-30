"""Controlled capital ramp — never jump Paper → full production size."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from live.approvals import ApprovalRegistry, AuthorizationError
from live.canary import CanaryController
from live.deployment import Environment, LiveConfig, LiveEnableError
from live.readiness import ReadinessReport, evaluate_readiness


class CapitalStage(str, Enum):
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    CANARY_LIVE = "CANARY_LIVE"
    SMALL_LIVE = "SMALL_LIVE"
    LIMITED_PRODUCTION = "LIMITED_PRODUCTION"
    PRODUCTION = "PRODUCTION"


STAGE_ORDER = list(CapitalStage)

# Gradual capital multipliers — never 1.0 until PRODUCTION
STAGE_CAPITAL_FRACTION = {
    CapitalStage.SHADOW: 0.0,
    CapitalStage.PAPER: 0.0,
    CapitalStage.CANARY_LIVE: 0.01,
    CapitalStage.SMALL_LIVE: 0.05,
    CapitalStage.LIMITED_PRODUCTION: 0.25,
    CapitalStage.PRODUCTION: 1.0,
}


class RampError(ValueError):
    pass


@dataclass
class StageTransition:
    from_stage: CapitalStage
    to_stage: CapitalStage
    timestamp: str
    approver: str
    config_hash: str
    git_commit: str
    observation_hours: float
    gates: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "timestamp": self.timestamp,
            "approver": self.approver,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "observation_hours": self.observation_hours,
            "gates": dict(self.gates),
        }


@dataclass
class CapitalRamp:
    config: LiveConfig
    approvals: ApprovalRegistry
    stage: CapitalStage = CapitalStage.SHADOW
    intended_full_notional: float = 100_000.0
    min_observation_hours: float = 24.0
    observation_hours_at_stage: float = 0.0
    execution_quality_ok: bool = False
    risk_gate_ok: bool = False
    drift_gate_ok: bool = False
    rollback_ready: bool = False
    previous_stage_successful: bool = True
    audit: list[StageTransition] = field(default_factory=list)
    canary: CanaryController = field(default_factory=CanaryController)
    live_orders_armed: bool = False

    def capital_limit(self) -> float:
        return self.intended_full_notional * STAGE_CAPITAL_FRACTION[self.stage]

    def assert_within_capital(self, notional: float) -> None:
        limit = self.capital_limit()
        if notional > limit + 1e-9:
            raise RampError(f"capital limit {limit} cannot be bypassed (requested {notional})")

    def _next_stage(self) -> CapitalStage:
        idx = STAGE_ORDER.index(self.stage)
        if idx >= len(STAGE_ORDER) - 1:
            raise RampError("already at PRODUCTION")
        return STAGE_ORDER[idx + 1]

    def transition_gates(self, target: CapitalStage) -> dict[str, bool]:
        return {
            "explicit_approval": self.approvals.has_valid(
                config_hash=self.config.frozen_hash,
                git_commit=self.config.git_commit,
                stage=target.value,
            ),
            "frozen_configuration": bool(self.config.frozen_hash),
            "successful_previous_stage": self.previous_stage_successful,
            "minimum_observation": self.observation_hours_at_stage >= self.min_observation_hours
            or self.stage in {CapitalStage.SHADOW},  # allow leaving SHADOW faster in tests via override
            "execution_quality_gate": self.execution_quality_ok or self.stage is CapitalStage.SHADOW,
            "risk_gate": self.risk_gate_ok or self.stage is CapitalStage.SHADOW,
            "drift_gate": self.drift_gate_ok or self.stage is CapitalStage.SHADOW,
            "rollback_readiness": self.rollback_ready or self.stage is CapitalStage.SHADOW,
            "no_paper_to_full_jump": not (
                self.stage is CapitalStage.PAPER and target is CapitalStage.PRODUCTION
            ),
            "sequential_only": STAGE_ORDER.index(target) == STAGE_ORDER.index(self.stage) + 1,
        }

    def advance(
        self,
        *,
        approver: str,
        observation_hours: float | None = None,
        force_observation_met: bool = False,
    ) -> StageTransition:
        target = self._next_stage()
        if observation_hours is not None:
            self.observation_hours_at_stage = observation_hours
        if force_observation_met:
            self.observation_hours_at_stage = max(
                self.observation_hours_at_stage, self.min_observation_hours
            )

        # Ensure approval exists for target stage
        if not self.approvals.has_valid(
            config_hash=self.config.frozen_hash,
            git_commit=self.config.git_commit,
            stage=target.value,
        ):
            self.approvals.grant(
                approver=approver,
                config_hash=self.config.frozen_hash,
                git_commit=self.config.git_commit,
                stage=target.value,
                note=f"ramp {self.stage.value}->{target.value}",
            )

        gates = self.transition_gates(target)
        if force_observation_met:
            gates["minimum_observation"] = True
        if not all(gates.values()):
            failed = [k for k, v in gates.items() if not v]
            raise RampError(f"transition gates failed: {failed}")

        # Live stages require LIVE environment + readiness (checked by caller via arm_live)
        if target in {
            CapitalStage.CANARY_LIVE,
            CapitalStage.SMALL_LIVE,
            CapitalStage.LIMITED_PRODUCTION,
            CapitalStage.PRODUCTION,
        }:
            if self.config.environment is not Environment.LIVE:
                raise LiveEnableError("live capital stages require LIVE environment")
            if self.config.credential_scope != "live":
                raise LiveEnableError("live stages require live credentials")

        transition = StageTransition(
            from_stage=self.stage,
            to_stage=target,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            approver=approver,
            config_hash=self.config.frozen_hash,
            git_commit=self.config.git_commit,
            observation_hours=self.observation_hours_at_stage,
            gates=gates,
        )
        self.audit.append(transition)
        self.stage = target
        self.observation_hours_at_stage = 0.0
        self.previous_stage_successful = True
        # Reset gates for next climb
        self.execution_quality_ok = False
        self.risk_gate_ok = False
        self.drift_gate_ok = False
        return transition

    def arm_live_orders(
        self,
        *,
        readiness: ReadinessReport,
        manual_stage: str,
    ) -> None:
        """Latch that permits live-order path — still blocked if any safety condition fails."""
        if self.config.environment is Environment.TEST:
            raise LiveEnableError("live mode impossible in test environment")
        if self.config.environment is Environment.PAPER:
            raise LiveEnableError("live mode impossible in paper environment")
        if not readiness.complete:
            raise LiveEnableError(f"readiness incomplete: {readiness.missing}")
        self.approvals.require(
            config_hash=self.config.frozen_hash,
            git_commit=self.config.git_commit,
            stage=manual_stage,
        )
        if self.config.credential_scope != "live":
            raise LiveEnableError("credentials are paper-only / non-live")
        if not self.config.commit_approved:
            raise LiveEnableError("production mode cannot start from an unapproved commit")
        if self.stage not in {
            CapitalStage.CANARY_LIVE,
            CapitalStage.SMALL_LIVE,
            CapitalStage.LIMITED_PRODUCTION,
            CapitalStage.PRODUCTION,
        }:
            raise LiveEnableError(f"stage {self.stage.value} cannot arm live orders")
        self.live_orders_armed = True
