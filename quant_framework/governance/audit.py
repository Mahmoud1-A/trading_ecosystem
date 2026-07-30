"""Governance audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditEvent:
    timestamp: str
    kind: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "kind": self.kind, "payload": dict(self.payload)}


@dataclass
class GovernanceAudit:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, kind: str, payload: dict[str, Any]) -> AuditEvent:
        ev = AuditEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            kind=kind,
            payload=payload,
        )
        self.events.append(ev)
        return ev

    def as_dict(self) -> dict[str, Any]:
        return {"events": [e.as_dict() for e in self.events], "count": len(self.events)}
