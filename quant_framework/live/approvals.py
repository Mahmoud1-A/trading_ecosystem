"""Manual human authorization for live readiness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from registry.hashing import sha256_json


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class ManualApproval:
    approval_id: str
    approver: str
    config_hash: str
    git_commit: str
    stage: str
    timestamp: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approver": self.approver,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "note": self.note,
        }


@dataclass
class ApprovalRegistry:
    """Append-only manual approvals — required for every capital-ramp transition."""

    approvals: list[ManualApproval] = field(default_factory=list)

    def grant(
        self,
        *,
        approver: str,
        config_hash: str,
        git_commit: str,
        stage: str,
        note: str = "",
    ) -> ManualApproval:
        if not approver.strip():
            raise AuthorizationError("approver identity required")
        approval = ManualApproval(
            approval_id="appr_" + uuid4().hex[:12],
            approver=approver.strip(),
            config_hash=config_hash,
            git_commit=git_commit,
            stage=stage,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            note=note,
        )
        self.approvals.append(approval)
        return approval

    def has_valid(
        self,
        *,
        config_hash: str,
        git_commit: str,
        stage: str,
    ) -> bool:
        return any(
            a.config_hash == config_hash and a.git_commit == git_commit and a.stage == stage
            for a in self.approvals
        )

    def require(
        self,
        *,
        config_hash: str,
        git_commit: str,
        stage: str,
    ) -> ManualApproval:
        for a in reversed(self.approvals):
            if a.config_hash == config_hash and a.git_commit == git_commit and a.stage == stage:
                return a
        raise AuthorizationError(
            f"missing manual authorization for stage={stage} commit={git_commit}"
        )

    def audit_trail(self) -> list[dict[str, Any]]:
        return [a.as_dict() for a in self.approvals]
