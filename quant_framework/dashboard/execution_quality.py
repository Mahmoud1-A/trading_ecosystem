"""Execution quality dashboard panel."""

from __future__ import annotations

from typing import Any

from api.health_api import HealthAPI


def execution_quality_view(api: HealthAPI) -> dict[str, Any]:
    return {
        "panel": "execution_quality",
        "counters": [c.as_dict() for c in api.execution_counters()],
    }
