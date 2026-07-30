"""Candidate discovery funnel dashboard panel."""

from __future__ import annotations

from typing import Any

from api.discovery_api import DiscoveryAPI
from api.types import CounterKind


def candidate_funnel_view(api: DiscoveryAPI) -> dict[str, Any]:
    counters = api.funnel_counters()
    events = [c.as_dict() for c in counters if c.kind is not CounterKind.SNAPSHOT]
    snapshots = [c.as_dict() for c in counters if c.kind is CounterKind.SNAPSHOT]
    return {
        "panel": "candidate_funnel",
        "event_counts": events,
        "snapshots": snapshots,
        # Explicit: never merge snapshots into funnel stages
        "funnel_stage_names": [
            c["name"] for c in events if c["name"] != "registry_total_trials" and c["name"] != "valid"
        ],
    }
