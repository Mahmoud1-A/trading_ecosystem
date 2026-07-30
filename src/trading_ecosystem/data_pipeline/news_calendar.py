from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, load_yaml


def _parse_dt(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class NewsCalendar:
    def __init__(
        self,
        path: str | Path | None = None,
        block_before_minutes: int = 30,
        block_after_minutes: int = 30,
    ) -> None:
        cfg_path = Path(path) if path else CONFIG_DIR / "news_calendar.yaml"
        raw = load_yaml(cfg_path)
        self.events: list[dict[str, Any]] = []
        for event in raw.get("events", []):
            self.events.append(
                {
                    "name": event.get("name", "event"),
                    "datetime": _parse_dt(str(event["datetime"])),
                    "impact": str(event.get("impact", "high")).lower(),
                }
            )
        self.block_before = timedelta(minutes=block_before_minutes)
        self.block_after = timedelta(minutes=block_after_minutes)

    def blocking_event(self, ts: datetime, impact: str = "high") -> dict[str, Any] | None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        for event in self.events:
            if event["impact"] != impact and impact == "high":
                if event["impact"] != "high":
                    continue
            start = event["datetime"] - self.block_before
            end = event["datetime"] + self.block_after
            if start <= ts <= end:
                return event
        return None

    def is_blocked(self, ts: datetime) -> bool:
        return self.blocking_event(ts) is not None
