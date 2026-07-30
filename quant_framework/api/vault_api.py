"""Vault control-plane API — status only; never exposes Vault frames to dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from api.types import CounterKind, TypedCounter


@dataclass
class VaultAPI:
    """Opaque Vault access telemetry for the control plane."""

    submissions: int = 0
    successes: int = 0
    failures: int = 0
    last_vault_version: str | None = None
    _lineages_accessed: set[str] = field(default_factory=set)

    def record_access(
        self,
        *,
        lineage_id: str,
        vault_version: str,
        success: bool,
    ) -> None:
        self.submissions += 1
        self.last_vault_version = vault_version
        self._lineages_accessed.add(lineage_id)
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def counters(self) -> list[TypedCounter]:
        return [
            TypedCounter("vault_submissions", self.submissions, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("vault_successes", self.successes, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("vault_failures", self.failures, CounterKind.CUMULATIVE_EVENT),
            TypedCounter(
                "distinct_lineages_accessed",
                len(self._lineages_accessed),
                CounterKind.SNAPSHOT,
            ),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "last_vault_version": self.last_vault_version,
            "counters": [c.as_dict() for c in self.counters()],
        }
