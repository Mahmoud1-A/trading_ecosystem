"""Append-only discovery trial ledger + vault exposure burn tracking."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from trading_ecosystem.discovery.universe import get_active_universe, processed_artifact


def trial_ledger_path(universe_id: str | None = None) -> Path:
    return processed_artifact("trial_ledger", universe_id=universe_id, ext=".jsonl")


def vault_exposure_path(universe_id: str | None = None) -> Path:
    return processed_artifact("vault_exposures", universe_id=universe_id)


def params_hash(params: dict[str, Any] | None) -> str:
    blob = json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def code_fingerprint(class_name: str, params: dict[str, Any] | None) -> str:
    return hashlib.sha256(f"{class_name}|{params_hash(params)}".encode()).hexdigest()[:16]


def append_trial(record: dict[str, Any], *, universe_id: str | None = None) -> Path:
    path = trial_ledger_path(universe_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("universe", universe_id or get_active_universe())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return path


def load_trial_scores(universe_id: str | None = None, *, limit: int = 5000) -> list[float]:
    path = trial_ledger_path(universe_id)
    if not path.exists():
        return []
    scores: list[float] = []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = row.get("oos_cagr")
        if s is None:
            s = (row.get("fitness") or {}).get("oos_cagr")
        if s is not None:
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                pass
    return scores


def register_vault_exposure(
    *,
    strategy_id: str,
    class_name: str,
    params: dict[str, Any] | None,
    vault_metrics: dict[str, Any],
    universe_id: str | None = None,
) -> dict[str, Any]:
    """Mark a strategy version as research-burned after vault open."""
    path = vault_exposure_path(universe_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"exposures": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {"exposures": []}
    version = code_fingerprint(class_name, params)
    exposure = {
        "vault_exposure_id": f"{strategy_id}:{version}",
        "strategy_id": strategy_id,
        "class_name": class_name,
        "params_hash": params_hash(params),
        "code_fingerprint": version,
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vault_metrics": vault_metrics,
        "burned": True,
        "note": "Any param/code change requires a new version; do not reuse this vault silently.",
    }
    data.setdefault("exposures", []).append(exposure)
    data["updated_at"] = exposure["opened_at"]
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return exposure


def is_vault_burned(strategy_id: str, class_name: str, params: dict[str, Any] | None, *, universe_id: str | None = None) -> bool:
    path = vault_exposure_path(universe_id)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    fp = code_fingerprint(class_name, params)
    for e in data.get("exposures") or []:
        if e.get("strategy_id") == strategy_id and e.get("code_fingerprint") == fp and e.get("burned"):
            return True
    return False
