"""Market and system events for the event-driven execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import pandas as pd


class EventType(str, Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    CANCEL = "CANCEL"
    RISK = "RISK"
    SESSION_END = "SESSION_END"
    TIMER = "TIMER"


def require_aware(ts: datetime | pd.Timestamp, *, name: str = "timestamp") -> pd.Timestamp:
    """Normalize to pandas Timestamp and require timezone awareness."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware; got naive {t!r}")
    return t


@dataclass(frozen=True)
class InformationTiming:
    """
    Explicit information availability contract for a feature or signal.

    Rules:
    - No feature may use information unavailable at decision_timestamp.
    - A signal generated at the close of bar t may execute only at open(t+1) or later.
    - If latency pushes activation past the next open, use the first legally available
      price after order_activation_timestamp — never a retroactive open.
    """

    source_timestamp: pd.Timestamp
    availability_timestamp: pd.Timestamp
    decision_timestamp: pd.Timestamp
    order_submission_timestamp: pd.Timestamp | None = None
    order_activation_timestamp: pd.Timestamp | None = None
    fill_timestamp: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_timestamp", require_aware(self.source_timestamp, name="source_timestamp"))
        object.__setattr__(
            self, "availability_timestamp", require_aware(self.availability_timestamp, name="availability_timestamp")
        )
        object.__setattr__(
            self, "decision_timestamp", require_aware(self.decision_timestamp, name="decision_timestamp")
        )
        if self.order_submission_timestamp is not None:
            object.__setattr__(
                self,
                "order_submission_timestamp",
                require_aware(self.order_submission_timestamp, name="order_submission_timestamp"),
            )
        if self.order_activation_timestamp is not None:
            object.__setattr__(
                self,
                "order_activation_timestamp",
                require_aware(self.order_activation_timestamp, name="order_activation_timestamp"),
            )
        if self.fill_timestamp is not None:
            object.__setattr__(self, "fill_timestamp", require_aware(self.fill_timestamp, name="fill_timestamp"))

        if self.availability_timestamp < self.source_timestamp:
            raise ValueError("availability_timestamp cannot precede source_timestamp")
        if self.decision_timestamp < self.availability_timestamp:
            raise ValueError("decision_timestamp cannot precede availability_timestamp")
        if self.order_submission_timestamp is not None and self.order_submission_timestamp < self.decision_timestamp:
            raise ValueError("order_submission_timestamp cannot precede decision_timestamp")
        if (
            self.order_activation_timestamp is not None
            and self.order_submission_timestamp is not None
            and self.order_activation_timestamp < self.order_submission_timestamp
        ):
            raise ValueError("order_activation_timestamp cannot precede order_submission_timestamp")
        if (
            self.fill_timestamp is not None
            and self.order_activation_timestamp is not None
            and self.fill_timestamp < self.order_activation_timestamp
        ):
            raise ValueError("fill_timestamp cannot precede order_activation_timestamp")

    def with_submission(
        self,
        submission: datetime | pd.Timestamp,
        *,
        latency: timedelta = timedelta(0),
    ) -> InformationTiming:
        sub = require_aware(submission, name="order_submission_timestamp")
        act = sub + latency
        return InformationTiming(
            source_timestamp=self.source_timestamp,
            availability_timestamp=self.availability_timestamp,
            decision_timestamp=self.decision_timestamp,
            order_submission_timestamp=sub,
            order_activation_timestamp=act,
            fill_timestamp=self.fill_timestamp,
        )

    def with_fill(self, fill_ts: datetime | pd.Timestamp) -> InformationTiming:
        if self.order_activation_timestamp is None:
            raise ValueError("Cannot set fill before order activation is known")
        return InformationTiming(
            source_timestamp=self.source_timestamp,
            availability_timestamp=self.availability_timestamp,
            decision_timestamp=self.decision_timestamp,
            order_submission_timestamp=self.order_submission_timestamp,
            order_activation_timestamp=self.order_activation_timestamp,
            fill_timestamp=require_aware(fill_ts, name="fill_timestamp"),
        )


@dataclass(frozen=True)
class BarEvent:
    """Single OHLC bar presented to the event loop."""

    timestamp: pd.Timestamp  # bar open timestamp (timezone-aware)
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    contract: str
    bar_end: pd.Timestamp
    atr: float | None = None
    is_stale: bool = False
    is_session_open: bool = False
    is_session_close: bool = False
    bid: float | None = None
    ask: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", require_aware(self.timestamp, name="bar.timestamp"))
        object.__setattr__(self, "bar_end", require_aware(self.bar_end, name="bar.bar_end"))
        if self.high < self.low:
            raise ValueError("Bar high < low")
        if self.contract.strip() == "":
            raise ValueError("Bar must carry an explicit tradable contract id")
        if bool(self.meta.get("is_continuous_research", False)):
            raise ValueError(
                "Continuous research prices are not executable. "
                "Provide an explicit tradable contract bar."
            )


@dataclass(frozen=True)
class SignalEvent:
    """Strategy intent produced at decision time (typically bar close)."""

    timing: InformationTiming
    symbol: str
    side: str  # BUY / SELL / FLAT
    quantity: float
    signal_id: str = field(default_factory=lambda: str(uuid4()))
    stop_price: float | None = None
    target_price: float | None = None
    order_type: str = "MARKET"
    limit_price: float | None = None
    stop_trigger: float | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def decision_timestamp(self) -> pd.Timestamp:
        return self.timing.decision_timestamp


@dataclass(frozen=True)
class EngineEvent:
    """Envelope for the deterministic event queue."""

    event_type: EventType
    timestamp: pd.Timestamp
    payload: Any
    sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", require_aware(self.timestamp, name="event.timestamp"))
