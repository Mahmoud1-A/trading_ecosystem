"""Portfolio pipeline dashboard panel."""

from __future__ import annotations

from typing import Any

from api.portfolio_api import PortfolioAPI
from api.types import CounterKind


def portfolio_pipeline_view(api: PortfolioAPI) -> dict[str, Any]:
    counters = api.counters()
    return {
        "panel": "portfolio_pipeline",
        "current_members": api.current_members(),
        "event_counts": [c.as_dict() for c in counters if c.kind is not CounterKind.SNAPSHOT],
        "snapshots": [c.as_dict() for c in counters if c.kind is CounterKind.SNAPSHOT],
    }
