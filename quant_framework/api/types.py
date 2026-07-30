"""Shared API / dashboard counter semantics (Phase 8)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CounterKind(str, Enum):
    """Every displayed counter must declare its semantics."""

    SESSION_EVENT = "session_event_count"
    CUMULATIVE_EVENT = "cumulative_event_count"
    SNAPSHOT = "current_snapshot"


@dataclass(frozen=True)
class TypedCounter:
    name: str
    value: int | float
    kind: CounterKind
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "kind": self.kind.value,
            "description": self.description,
        }


class BackendAuthority(str, Enum):
    LEGACY = "legacy"
    NEW = "new"
    PARALLEL = "parallel"
