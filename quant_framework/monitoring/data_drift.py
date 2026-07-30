"""Data drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataDrift:
    missing_bars: int
    stale_seconds: float
    schema_changes: int
    halted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "missing_bars": self.missing_bars,
            "stale_seconds": self.stale_seconds,
            "schema_changes": self.schema_changes,
            "halted": self.halted,
        }
