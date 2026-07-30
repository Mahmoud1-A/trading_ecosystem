"""Simple bar scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Scheduler:
    """Invoke callbacks on each market bar / tick batch."""

    _handlers: list[Callable[[object], None]] = field(default_factory=list)

    def on_bar(self, handler: Callable[[object], None]) -> None:
        self._handlers.append(handler)

    def dispatch(self, bar: object) -> None:
        for h in self._handlers:
            h(bar)
