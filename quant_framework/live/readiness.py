"""Live readiness checklist — incomplete reports block live mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from live.approvals import ApprovalRegistry
from live.deployment import Environment, LiveConfig


REQUIRED_READINESS_ITEMS = (
    "separate_live_configuration",
    "separate_live_credentials",
    "manual_human_authorization",
    "explicit_environment_flag",
    "production_secret_manager",
    "network_allowlist",
    "least_privilege_credentials",
    "broker_side_risk_limits",
    "independent_local_risk_limits",
    "independent_remote_kill_switch",
    "complete_audit_log",
    "disaster_recovery_plan",
    "rollback_plan",
    "incident_response_runbook",
    "vault_passed",
    "paper_gates_passed",
    "reconciliation_clean",
    "kill_switch_tested",
)


@dataclass
class ReadinessReport:
    items: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def mark(self, item: str, ok: bool, note: str = "") -> None:
        if item not in REQUIRED_READINESS_ITEMS:
            raise KeyError(f"unknown readiness item {item!r}")
        self.items[item] = ok
        if note:
            self.notes[item] = note

    @property
    def complete(self) -> bool:
        return all(self.items.get(i, False) for i in REQUIRED_READINESS_ITEMS)

    @property
    def missing(self) -> list[str]:
        return [i for i in REQUIRED_READINESS_ITEMS if not self.items.get(i, False)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "items": dict(self.items),
            "missing": self.missing,
            "notes": dict(self.notes),
        }


def evaluate_readiness(
    config: LiveConfig,
    *,
    approvals: ApprovalRegistry,
    vault_passed: bool,
    paper_gates_passed: bool,
    reconciliation_clean: bool,
    kill_switch_tested: bool,
    audit_log_present: bool,
    stage_for_auth: str = "CANARY_LIVE",
) -> ReadinessReport:
    report = ReadinessReport()
    report.mark("separate_live_configuration", bool(config.config_id))
    report.mark(
        "separate_live_credentials",
        config.credential_scope == "live" and config.environment is Environment.LIVE,
    )
    report.mark(
        "manual_human_authorization",
        approvals.has_valid(
            config_hash=config.frozen_hash,
            git_commit=config.git_commit,
            stage=stage_for_auth,
        ),
    )
    report.mark("explicit_environment_flag", config.environment is Environment.LIVE)
    report.mark(
        "production_secret_manager",
        config.secret_manager in {"vault_kv", "aws_secrets", "gcp_secret_manager"},
    )
    report.mark("network_allowlist", len(config.network_allowlist) > 0)
    report.mark("least_privilege_credentials", bool(config.least_privilege_role))
    report.mark("broker_side_risk_limits", bool(config.broker_risk_limits))
    report.mark("independent_local_risk_limits", bool(config.local_risk_limits))
    report.mark("independent_remote_kill_switch", bool(config.remote_kill_switch_endpoint))
    report.mark("complete_audit_log", audit_log_present)
    report.mark("disaster_recovery_plan", bool(config.disaster_recovery_plan_id))
    report.mark("rollback_plan", bool(config.rollback_plan_id))
    report.mark("incident_response_runbook", bool(config.incident_runbook_id))
    report.mark("vault_passed", vault_passed)
    report.mark("paper_gates_passed", paper_gates_passed)
    report.mark("reconciliation_clean", reconciliation_clean)
    report.mark("kill_switch_tested", kill_switch_tested)
    return report
