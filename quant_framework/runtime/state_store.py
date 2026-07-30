"""Persistent runtime state store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuntimeState:
    run_id: str
    mode: str
    client_order_ids: list[str] = field(default_factory=list)
    broker_order_map: dict[str, str] = field(default_factory=dict)
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    cash: float = 100_000.0
    signal_fingerprints: list[str] = field(default_factory=list)
    kill_switch_engaged: bool = False
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> RuntimeState:
        return RuntimeState(**payload)


@dataclass
class StateStore:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: RuntimeState) -> None:
        self.path.write_text(json.dumps(state.as_dict(), indent=2, default=str), encoding="utf-8")

    def load(self) -> RuntimeState | None:
        if not self.path.exists():
            return None
        return RuntimeState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))

    def append_audit(self, state: RuntimeState, event: dict[str, Any]) -> None:
        state.audit_log.append(event)
        self.save(state)
