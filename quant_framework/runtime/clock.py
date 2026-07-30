"""Runtime clock with skew detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RuntimeClock:
    """Monotonic research clock with optional broker offset."""

    broker_offset_ms: float = 0.0
    max_skew_ms: float = 5_000.0
    _last_sync: datetime | None = None
    skew_alarm: bool = False

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def now_iso(self) -> str:
        return self.now().isoformat()

    def sync_broker_time(self, broker_ts: datetime) -> float:
        local = self.now()
        if broker_ts.tzinfo is None:
            broker_ts = broker_ts.replace(tzinfo=timezone.utc)
        skew_ms = (local - broker_ts).total_seconds() * 1000.0
        self.broker_offset_ms = skew_ms
        self._last_sync = local
        self.skew_alarm = abs(skew_ms) > self.max_skew_ms
        return skew_ms
