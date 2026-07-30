from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from trading_ecosystem.discovery.leaderboard import _primary_symbol, _slim_row, _sort_key
from trading_ecosystem.discovery.universe import processed_artifact

DEFAULT_BEST_PER_SECTOR = 3


def sector_best_path(universe_id: str | None = None) -> Path:
    return processed_artifact("discovery_best3_by_sector", universe_id=universe_id)


def load_sector_best(path: Path | None = None) -> dict[str, Any]:
    p = path or sector_best_path()
    if not p.exists():
        return {
            "updated_at": None,
            "best_per_sector": DEFAULT_BEST_PER_SECTOR,
            "sectors": {},
        }
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        "updated_at": raw.get("updated_at"),
        "best_per_sector": int(raw.get("best_per_sector") or DEFAULT_BEST_PER_SECTOR),
        "sectors": dict(raw.get("sectors") or {}),
    }


def save_sector_best(
    sectors: dict[str, list[dict[str, Any]]],
    *,
    path: Path | None = None,
    best_per_sector: int = DEFAULT_BEST_PER_SECTOR,
) -> Path:
    p = path or sector_best_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for sym in sorted(sectors.keys()):
        ranked = []
        for i, row in enumerate(sectors[sym][:best_per_sector], 1):
            item = dict(row)
            item["rank_in_sector"] = i
            item["sector"] = sym
            ranked.append(item)
        if ranked:
            cleaned[sym] = ranked
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_per_sector": best_per_sector,
        "sector_count": len(cleaned),
        "sectors": cleaned,
    }
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def update_sector_best(
    newcomers: list[dict[str, Any]],
    *,
    path: Path | None = None,
    best_per_sector: int = DEFAULT_BEST_PER_SECTOR,
    universe_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Independent registry: keep only the strongest N algorithms per symbol/sector.
    Not related to the global Top-50 absolute board.
    """
    if path is None and universe_id is not None:
        path = sector_best_path(universe_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    current = load_sector_best(path)
    by_symbol: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for sym, rows in (current.get("sectors") or {}).items():
        for row in rows:
            slim = _slim_row(row, seen_at=row.get("updated_at") or now)
            by_symbol[sym][slim["fingerprint"]] = slim

    for row in newcomers:
        slim = _slim_row(row, seen_at=now)
        sym = _primary_symbol(slim)
        if sym == "_NONE_":
            continue
        fp = slim["fingerprint"]
        prev = by_symbol[sym].get(fp)
        if prev is None:
            slim["first_seen_at"] = now
            by_symbol[sym][fp] = slim
        elif _sort_key(slim) > _sort_key(prev):
            slim["first_seen_at"] = prev.get("first_seen_at") or now
            by_symbol[sym][fp] = slim

    sectors: dict[str, list[dict[str, Any]]] = {}
    for sym, fp_map in by_symbol.items():
        ranked = sorted(fp_map.values(), key=_sort_key, reverse=True)[:best_per_sector]
        sectors[sym] = ranked

    save_sector_best(sectors, path=path, best_per_sector=best_per_sector)
    return sectors


def sector_best_public(sectors: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, list[dict[str, Any]]]:
    raw = sectors if sectors is not None else load_sector_best().get("sectors") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for sym, rows in sorted(raw.items()):
        out[sym] = []
        for i, row in enumerate(rows, 1):
            c = row.get("candidate") or {}
            fit = row.get("fitness") or {}
            out[sym].append(
                {
                    "rank_in_sector": row.get("rank_in_sector") or i,
                    "strategy_id": c.get("strategy_id"),
                    "class_name": c.get("class_name"),
                    "symbols": c.get("symbols") or [sym],
                    "source": c.get("source"),
                    "params": c.get("params") or {},
                    "passed": fit.get("passed"),
                    "oos_cagr": fit.get("oos_cagr"),
                    "oos_max_dd": fit.get("oos_max_dd"),
                    "score": fit.get("score"),
                    "oos_trades": fit.get("oos_trades"),
                    "updated_at": row.get("updated_at"),
                }
            )
    return out
