"""Named discovery universes (ETF book, crypto book, …) with namespaced artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml

DEFAULT_UNIVERSE = "etf"
UNIVERSES_DIR = CONFIG_DIR / "universes"


def active_universe_path() -> Path:
    return data_dir() / "processed" / "active_universe.json"


def list_universe_ids() -> list[str]:
    if not UNIVERSES_DIR.exists():
        return [DEFAULT_UNIVERSE]
    ids = sorted(p.stem for p in UNIVERSES_DIR.glob("*.yaml"))
    return ids or [DEFAULT_UNIVERSE]


def load_universe(universe_id: str | None = None) -> dict[str, Any]:
    uid = (universe_id or get_active_universe()).strip().lower()
    path = UNIVERSES_DIR / f"{uid}.yaml"
    if path.exists():
        raw = load_yaml(path)
        symbols = list(raw.get("symbols") or [])
        return {
            "id": str(raw.get("id") or uid),
            "label": str(raw.get("label") or uid),
            "label_en": str(raw.get("label_en") or uid),
            "timeframe": str(raw.get("timeframe") or "1d"),
            "trade_default": bool(raw.get("trade_default", uid == DEFAULT_UNIVERSE)),
            "cost_profile": str(raw.get("cost_profile") or uid),
            "portfolio_mode": str(raw.get("portfolio_mode") or "aggregate_book"),
            "min_bars_for_champion": int(raw.get("min_bars_for_champion") or 0),
            "max_heroes": int(raw["max_heroes"]) if raw.get("max_heroes") is not None else None,
            "min_hero_oos_cagr": (
                float(raw["min_hero_oos_cagr"]) if raw.get("min_hero_oos_cagr") is not None else None
            ),
            "symbols": symbols,
            "path": str(path),
            "count": len(symbols),
        }

    # Fallback: legacy discovery.yaml / symbols.yaml (ETF)
    discovery_cfg = load_yaml(CONFIG_DIR / "discovery.yaml")
    symbols_cfg = load_yaml(CONFIG_DIR / "symbols.yaml")
    symbols = list(discovery_cfg.get("symbols") or symbols_cfg.get("symbols") or [])
    return {
        "id": DEFAULT_UNIVERSE,
        "label": "أصول تقليدية (ETFs)",
        "label_en": "Traditional assets (ETFs)",
        "timeframe": str(symbols_cfg.get("timeframe") or "1d"),
        "trade_default": True,
        "cost_profile": "etf",
        "symbols": symbols,
        "path": str(CONFIG_DIR / "discovery.yaml"),
        "count": len(symbols),
    }


def get_active_universe() -> str:
    # Prefer explicit selection file (control panel / --universe) over env default.
    path = active_universe_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            uid = str(raw.get("id") or "").strip().lower()
            if uid:
                return uid
        except Exception:  # noqa: BLE001
            pass
    env = (os.getenv("DISCOVERY_UNIVERSE") or "").strip().lower()
    if env:
        return env
    discovery_cfg = load_yaml(CONFIG_DIR / "discovery.yaml")
    return str(discovery_cfg.get("active_universe") or DEFAULT_UNIVERSE).strip().lower()


def set_active_universe(universe_id: str) -> dict[str, Any]:
    uid = universe_id.strip().lower()
    if uid not in list_universe_ids() and uid != DEFAULT_UNIVERSE:
        # Allow if file exists
        if not (UNIVERSES_DIR / f"{uid}.yaml").exists():
            raise ValueError(f"Unknown universe: {uid}. Known: {list_universe_ids()}")
    info = load_universe(uid)
    path = active_universe_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"id": uid, "updated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())}, indent=2),
        encoding="utf-8",
    )
    return info


def artifact_suffix(universe_id: str | None = None) -> str:
    """ETF keeps legacy unsuffixed filenames; others get _<id>."""
    uid = (universe_id or get_active_universe()).strip().lower()
    if uid in {"", DEFAULT_UNIVERSE, "default"}:
        return ""
    return f"_{uid}"


def processed_artifact(stem: str, *, universe_id: str | None = None, ext: str = ".json") -> Path:
    """stem without extension, e.g. 'discovery_portfolio_book'."""
    return data_dir() / "processed" / f"{stem}{artifact_suffix(universe_id)}{ext}"


def strategies_export_path(universe_id: str | None = None) -> Path:
    uid = (universe_id or get_active_universe()).strip().lower()
    if uid in {"", DEFAULT_UNIVERSE, "default"}:
        return CONFIG_DIR / "strategies.discovered.yaml"
    return CONFIG_DIR / f"strategies.discovered_{uid}.yaml"


def paper_live_path(universe_id: str | None = None) -> Path:
    uid = (universe_id or get_active_universe()).strip().lower()
    if uid in {"", DEFAULT_UNIVERSE, "default"}:
        return CONFIG_DIR / "strategies.paper.live.yaml"
    return CONFIG_DIR / f"strategies.paper.live_{uid}.yaml"


def list_universes_public() -> list[dict[str, Any]]:
    active = get_active_universe()
    out: list[dict[str, Any]] = []
    for uid in list_universe_ids():
        info = load_universe(uid)
        out.append(
            {
                **{k: info[k] for k in ("id", "label", "label_en", "timeframe", "count", "trade_default")},
                "active": info["id"] == active,
                "portfolio_book": str(processed_artifact("discovery_portfolio_book", universe_id=info["id"])),
            }
        )
    return out


def apply_universe_to_discovery_cfg(
    discovery_cfg: dict[str, Any],
    *,
    universe_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (cfg_with_symbols, universe_info)."""
    info = load_universe(universe_id)
    cfg = dict(discovery_cfg)
    cfg["symbols"] = list(info["symbols"])
    cfg["active_universe"] = info["id"]
    cfg["cost_profile"] = info.get("cost_profile") or info["id"]
    return cfg, info
