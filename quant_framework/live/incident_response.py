"""Incident response — broker failure safe state, kill switches, DR hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from live.capital_ramp import CapitalRamp
from live.deployment import LiveEnableError
from live.rollback import RollbackController
from runtime.kill_switch import KillSwitch
from runtime.state_store import RuntimeState, StateStore


@dataclass
class IncidentEvent:
    timestamp: str
    kind: str
    detail: str
    safe_state: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "detail": self.detail,
            "safe_state": self.safe_state,
        }


@dataclass
class IncidentResponse:
    """
    Runbook automation for live-readiness incidents.

    Local and remote kill switches are independent — either can force safe state.
    """

    ramp: CapitalRamp
    rollback: RollbackController
    local_kill: KillSwitch | None = None
    remote_kill_engaged: bool = False
    safe_state: bool = False
    events: list[IncidentEvent] = field(default_factory=list)
    state: RuntimeState | None = None
    store: StateStore | None = None

    def _log(self, kind: str, detail: str) -> IncidentEvent:
        ev = IncidentEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            kind=kind,
            detail=detail,
            safe_state=self.safe_state,
        )
        self.events.append(ev)
        if self.state is not None and self.store is not None:
            self.store.append_audit(self.state, ev.as_dict())
        return ev

    def engage_local_kill(self, *, reason: str = "incident") -> dict[str, Any]:
        self.safe_state = True
        self.ramp.live_orders_armed = False
        result: dict[str, Any] = {"local": True}
        if self.local_kill is not None:
            result.update(self.local_kill.engage(reason=reason))
        self._log("local_kill", reason)
        return result

    def engage_remote_kill(self, *, reason: str = "remote_signal") -> None:
        """Independent remote kill — does not require local kill switch object."""
        self.remote_kill_engaged = True
        self.safe_state = True
        self.ramp.live_orders_armed = False
        self._log("remote_kill", reason)

    def on_broker_failure(self, detail: str = "broker_disconnect") -> None:
        self.safe_state = True
        self.ramp.live_orders_armed = False
        self.rollback.rollback(reason=f"broker_failure:{detail}")
        self._log("broker_failure", detail)

    def assert_not_safe(self) -> None:
        if self.safe_state or self.remote_kill_engaged:
            raise LiveEnableError("incident safe state active — live orders blocked")
        if self.local_kill is not None and self.local_kill.engaged:
            raise LiveEnableError("local kill switch engaged")

    def preserve_state_for_recovery(self) -> dict[str, Any]:
        if self.state is None or self.store is None:
            return {"preserved": False}
        self.store.save(self.state)
        loaded = self.store.load()
        assert loaded is not None
        return {"preserved": True, "state": loaded.as_dict()}
