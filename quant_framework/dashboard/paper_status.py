"""Paper status dashboard panel."""

from __future__ import annotations

from typing import Any

from api.paper_api import PaperAPI


def paper_status_view(api: PaperAPI) -> dict[str, Any]:
    return {
        "panel": "paper_status",
        "status_by_id": {k: v.value for k, v in api.status_by_id.items()},
        "counters": [c.as_dict() for c in api.counters()],
    }
