"""Data quality dashboard panel."""

from __future__ import annotations

from typing import Any

from api.health_api import HealthAPI


def data_quality_view(api: HealthAPI) -> dict[str, Any]:
    return {
        "panel": "data_quality",
        "counters": [c.as_dict() for c in api.data_quality_counters()],
        "dataset_versions": dict(api.data_quality.dataset_versions),
        "provider_healthy": api.data_quality.provider_ok,
    }
