"""System health dashboard panel."""

from __future__ import annotations

from typing import Any

from api.health_api import HealthAPI
from api.vault_api import VaultAPI


def system_health_view(health: HealthAPI, vault: VaultAPI | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "panel": "system_health",
        "system_ok": health.system_ok,
        "notes": list(health.notes),
        "counters": [c.as_dict() for c in health.system_counters()],
    }
    if vault is not None:
        payload["vault"] = vault.as_dict()
    return payload
