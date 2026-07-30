"""Live environment and deployment configuration (research readiness only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from registry.hashing import sha256_json


class Environment(str, Enum):
    TEST = "test"
    PAPER = "paper"
    LIVE = "live"


class LiveEnableError(PermissionError):
    """Raised when live order paths are blocked."""


@dataclass(frozen=True)
class LiveConfig:
    """
    Separate live configuration — never reuse paper/test settings.

    ``enable_live_orders`` defaults to False and remains False unless every
    readiness/approval gate passes at runtime.
    """

    environment: Environment
    config_id: str
    git_commit: str
    commit_approved: bool
    secret_manager: str  # e.g. "env", "vault_kv", "aws_secrets"
    credential_scope: str  # must be "live" for live env
    network_allowlist: tuple[str, ...]
    least_privilege_role: str
    broker_risk_limits: dict[str, float]
    local_risk_limits: dict[str, float]
    remote_kill_switch_endpoint: str
    disaster_recovery_plan_id: str
    rollback_plan_id: str
    incident_runbook_id: str
    enable_live_orders: bool = False
    feature_set_version: str = ""
    cost_model_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["environment"] = self.environment.value
        d["network_allowlist"] = list(self.network_allowlist)
        return d

    @property
    def frozen_hash(self) -> str:
        payload = self.as_dict()
        payload.pop("enable_live_orders", None)  # runtime latch not part of freeze identity
        return "livecfg_" + sha256_json(payload)[:24]


def build_live_config(
    *,
    environment: Environment,
    git_commit: str,
    commit_approved: bool,
    credential_scope: str,
    network_allowlist: tuple[str, ...] = ("broker.example.com",),
    **kwargs: Any,
) -> LiveConfig:
    cfg = LiveConfig(
        environment=environment,
        config_id="pending",
        git_commit=git_commit,
        commit_approved=commit_approved,
        secret_manager=kwargs.pop("secret_manager", "vault_kv"),
        credential_scope=credential_scope,
        network_allowlist=network_allowlist,
        least_privilege_role=kwargs.pop("least_privilege_role", "trading-submit-only"),
        broker_risk_limits=kwargs.pop("broker_risk_limits", {"max_notional": 10_000.0}),
        local_risk_limits=kwargs.pop("local_risk_limits", {"max_notional": 5_000.0}),
        remote_kill_switch_endpoint=kwargs.pop(
            "remote_kill_switch_endpoint", "https://kill.internal/v1/engage"
        ),
        disaster_recovery_plan_id=kwargs.pop("disaster_recovery_plan_id", "dr_v1"),
        rollback_plan_id=kwargs.pop("rollback_plan_id", "rb_v1"),
        incident_runbook_id=kwargs.pop("incident_runbook_id", "ir_v1"),
        enable_live_orders=False,
        feature_set_version=kwargs.pop("feature_set_version", "fs1"),
        cost_model_version=kwargs.pop("cost_model_version", "cost_v1"),
    )
    if kwargs:
        raise TypeError(f"unexpected live config fields: {sorted(kwargs)}")
    return LiveConfig(
        environment=cfg.environment,
        config_id=cfg.frozen_hash,
        git_commit=cfg.git_commit,
        commit_approved=cfg.commit_approved,
        secret_manager=cfg.secret_manager,
        credential_scope=cfg.credential_scope,
        network_allowlist=cfg.network_allowlist,
        least_privilege_role=cfg.least_privilege_role,
        broker_risk_limits=dict(cfg.broker_risk_limits),
        local_risk_limits=dict(cfg.local_risk_limits),
        remote_kill_switch_endpoint=cfg.remote_kill_switch_endpoint,
        disaster_recovery_plan_id=cfg.disaster_recovery_plan_id,
        rollback_plan_id=cfg.rollback_plan_id,
        incident_runbook_id=cfg.incident_runbook_id,
        enable_live_orders=False,
        feature_set_version=cfg.feature_set_version,
        cost_model_version=cfg.cost_model_version,
    )
