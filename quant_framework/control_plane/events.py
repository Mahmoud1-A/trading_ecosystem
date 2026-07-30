"""Research event model and event sinks (Phase 12) — engine-agnostic."""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4


class EventType(str, Enum):
    RUN_CREATED = "RUN_CREATED"
    RUN_STARTED = "RUN_STARTED"
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    DATA_LOADED = "DATA_LOADED"
    DATA_VALIDATED = "DATA_VALIDATED"
    GENERATION_STARTED = "GENERATION_STARTED"
    CANDIDATE_GENERATED = "CANDIDATE_GENERATED"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    CANDIDATE_EVALUATION_STARTED = "CANDIDATE_EVALUATION_STARTED"
    CANDIDATE_EVALUATED = "CANDIDATE_EVALUATED"
    WFO_FOLD_STARTED = "WFO_FOLD_STARTED"
    WFO_FOLD_COMPLETED = "WFO_FOLD_COMPLETED"
    STRESS_TEST_STARTED = "STRESS_TEST_STARTED"
    STRESS_TEST_COMPLETED = "STRESS_TEST_COMPLETED"
    BEHAVIORAL_CLUSTER_UPDATED = "BEHAVIORAL_CLUSTER_UPDATED"
    FINALIST_SELECTED = "FINALIST_SELECTED"
    PORTFOLIO_BUILT = "PORTFOLIO_BUILT"
    PORTFOLIO_FROZEN = "PORTFOLIO_FROZEN"
    VAULT_STARTED = "VAULT_STARTED"
    VAULT_COMPLETED = "VAULT_COMPLETED"
    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    WARNING = "WARNING"
    ERROR = "ERROR"
    RUN_CANCELLED = "RUN_CANCELLED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    PROGRESS = "PROGRESS"


class EventSeverity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ResearchEvent:
    event_id: str
    run_id: str
    event_type: EventType
    timestamp: str
    stage: str
    severity: EventSeverity
    message: str
    progress: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "severity": self.severity.value,
            "message": self.message,
            "progress": self.progress,
            "payload": dict(self.payload),
            "seq": self.seq,
        }

    @staticmethod
    def create(
        *,
        run_id: str,
        event_type: EventType,
        message: str,
        stage: str = "",
        severity: EventSeverity = EventSeverity.INFO,
        progress: float | None = None,
        payload: dict[str, Any] | None = None,
        seq: int = 0,
    ) -> ResearchEvent:
        return ResearchEvent(
            event_id="evt_" + uuid4().hex[:16],
            run_id=run_id,
            event_type=event_type,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            stage=stage,
            severity=severity,
            message=message,
            progress=progress,
            payload=payload or {},
            seq=seq,
        )


class ResearchEventSink(ABC):
    """Protocol-style sink — quantitative engine must not import FastAPI."""

    @abstractmethod
    def emit(self, event: ResearchEvent) -> None: ...


class NullEventSink(ResearchEventSink):
    def emit(self, event: ResearchEvent) -> None:
        return None


class InMemoryEventSink(ResearchEventSink):
    def __init__(self) -> None:
        self.events: list[ResearchEvent] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._subscribers: list[Callable[[ResearchEvent], None]] = []

    def emit(self, event: ResearchEvent) -> None:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            self.events.append(event)
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:  # noqa: BLE001
                pass

    def subscribe(self, callback: Callable[[ResearchEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def since(self, after_seq: int = 0) -> list[ResearchEvent]:
        with self._lock:
            return [e for e in self.events if e.seq > after_seq]


class PersistentEventSink(ResearchEventSink):
    """Persist events to JSONL before notifying optional downstream."""

    def __init__(self, path: Path, downstream: ResearchEventSink | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.downstream = downstream
        self._lock = threading.Lock()
        self._seq = 0
        if self.path.exists():
            # Resume seq from last line
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self._seq = max(self._seq, int(json.loads(line).get("seq", 0)))
                except json.JSONDecodeError:
                    continue

    def emit(self, event: ResearchEvent) -> None:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.as_dict(), default=str) + "\n")
                fh.flush()
        if self.downstream is not None:
            self.downstream.emit(event)

    def load_all(self) -> list[ResearchEvent]:
        if not self.path.exists():
            return []
        out: list[ResearchEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            out.append(
                ResearchEvent(
                    event_id=raw["event_id"],
                    run_id=raw["run_id"],
                    event_type=EventType(raw["event_type"]),
                    timestamp=raw["timestamp"],
                    stage=raw.get("stage", ""),
                    severity=EventSeverity(raw.get("severity", "INFO")),
                    message=raw.get("message", ""),
                    progress=raw.get("progress"),
                    payload=dict(raw.get("payload") or {}),
                    seq=int(raw.get("seq", 0)),
                )
            )
        return out

    def since(self, after_seq: int = 0) -> list[ResearchEvent]:
        return [e for e in self.load_all() if e.seq > after_seq]


class CompositeEventSink(ResearchEventSink):
    def __init__(self, *sinks: ResearchEventSink) -> None:
        self.sinks = list(sinks)

    def emit(self, event: ResearchEvent) -> None:
        for s in self.sinks:
            s.emit(event)
