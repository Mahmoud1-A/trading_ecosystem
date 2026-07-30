"""Non-invasive progress hooks — engines emit via ResearchEventSink only."""

from __future__ import annotations

from typing import Protocol

from control_plane.events import ResearchEvent, ResearchEventSink


class ProgressCallback(Protocol):
    def __call__(self, event: ResearchEvent) -> None: ...


def emit_progress(sink: ResearchEventSink, event: ResearchEvent) -> None:
    """Forward a research event to a sink (NullEventSink is a no-op)."""
    sink.emit(event)
