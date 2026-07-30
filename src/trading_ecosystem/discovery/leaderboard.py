from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from trading_ecosystem.discovery.universe import processed_artifact

TOP50_SIZE = 50
DEFAULT_SEATS_PER_SYMBOL = 3
DEFAULT_POOL_SIZE = 400


def leaderboard_path(universe_id: str | None = None) -> Path:
    return processed_artifact("discovery_top50", universe_id=universe_id)


def _sort_key(row: dict[str, Any]) -> tuple:
    fit = row.get("fitness") or {}
    return (
        1 if fit.get("passed") else 0,
        float(fit.get("oos_cagr") or -1e9),
        float(fit.get("score") or -1e9),
        -float(fit.get("oos_max_dd") or 1.0),
    )


def _fingerprint(candidate: dict[str, Any]) -> str:
    params = candidate.get("params") or {}
    return (
        f"{candidate.get('class_name')}|"
        f"{','.join(candidate.get('symbols') or [])}|"
        f"{sorted(params.items())}"
    )


def _primary_symbol(row: dict[str, Any]) -> str:
    c = row.get("candidate") or {}
    syms = list(c.get("symbols") or [])
    return str(syms[0]) if syms else "_NONE_"


def _slim_row(row: dict[str, Any], *, seen_at: str | None = None) -> dict[str, Any]:
    c = row.get("candidate") or {}
    fit = row.get("fitness") or {}
    full = row.get("full_metrics") or {}
    return {
        "candidate": {
            "strategy_id": c.get("strategy_id"),
            "class_name": c.get("class_name"),
            "symbols": list(c.get("symbols") or []),
            "params": dict(c.get("params") or {}),
            "source": c.get("source"),
            "meta": dict(c.get("meta") or {}),
        },
        "fitness": {
            "passed": fit.get("passed"),
            "score": fit.get("score"),
            "oos_cagr": fit.get("oos_cagr"),
            "oos_max_dd": fit.get("oos_max_dd"),
            "oos_mar": fit.get("oos_mar"),
            "oos_trades": fit.get("oos_trades"),
            "hit_target_cagr": fit.get("hit_target_cagr"),
            "reason": fit.get("reason"),
        },
        "full_metrics": {
            "cagr": full.get("cagr"),
            "max_drawdown": full.get("max_drawdown"),
            "total_return": full.get("total_return"),
            "num_trades": full.get("num_trades"),
        }
        if full
        else {},
        "first_seen_at": row.get("first_seen_at") or seen_at,
        "updated_at": seen_at or row.get("updated_at"),
        "fingerprint": _fingerprint(c),
    }


def select_with_sector_seats(
    pool: list[dict[str, Any]],
    *,
    size: int = TOP50_SIZE,
    seats_per_symbol: int = DEFAULT_SEATS_PER_SYMBOL,
) -> list[dict[str, Any]]:
    """
    Guarantee up to `seats_per_symbol` strongest algos per symbol/sector,
    then fill remaining slots by global rank.
    """
    if seats_per_symbol < 1:
        return sorted(pool, key=_sort_key, reverse=True)[:size]

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_symbol[_primary_symbol(row)].append(row)

    selected: list[dict[str, Any]] = []
    selected_fps: set[str] = set()

    # Pass 1: reserved seats per symbol (strongest first within symbol)
    for sym in sorted(by_symbol.keys()):
        ranked_sym = sorted(by_symbol[sym], key=_sort_key, reverse=True)
        for row in ranked_sym[:seats_per_symbol]:
            fp = row.get("fingerprint") or _fingerprint(row.get("candidate") or {})
            if fp in selected_fps:
                continue
            item = dict(row)
            item["sector"] = sym
            item["sector_seat"] = True
            selected.append(item)
            selected_fps.add(fp)

    # Pass 2: fill remaining globally
    global_ranked = sorted(pool, key=_sort_key, reverse=True)
    for row in global_ranked:
        if len(selected) >= size:
            break
        fp = row.get("fingerprint") or _fingerprint(row.get("candidate") or {})
        if fp in selected_fps:
            continue
        item = dict(row)
        item["sector"] = _primary_symbol(row)
        item["sector_seat"] = False
        selected.append(item)
        selected_fps.add(fp)

    # Final display order: global strength, but keep seat metadata
    selected.sort(key=_sort_key, reverse=True)
    return selected[:size]


def load_top50(path: Path | None = None) -> dict[str, Any]:
    p = path or leaderboard_path()
    if not p.exists():
        return {
            "updated_at": None,
            "size": TOP50_SIZE,
            "count": 0,
            "seats_per_symbol": DEFAULT_SEATS_PER_SYMBOL,
            "entries": [],
            "pool": [],
        }
    raw = json.loads(p.read_text(encoding="utf-8"))
    entries = list(raw.get("entries") or raw.get("elites") or [])
    pool = list(raw.get("pool") or entries)
    return {
        "updated_at": raw.get("updated_at"),
        "size": int(raw.get("size") or TOP50_SIZE),
        "count": int(raw.get("count") or len(entries)),
        "seats_per_symbol": int(raw.get("seats_per_symbol") or DEFAULT_SEATS_PER_SYMBOL),
        "by_symbol": dict(raw.get("by_symbol") or {}),
        "entries": entries,
        "pool": pool,
    }


def save_top50(
    entries: list[dict[str, Any]],
    path: Path | None = None,
    *,
    size: int = TOP50_SIZE,
    pool: list[dict[str, Any]] | None = None,
    seats_per_symbol: int = DEFAULT_SEATS_PER_SYMBOL,
) -> Path:
    p = path or leaderboard_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    ranked = []
    for i, row in enumerate(entries[:size], 1):
        item = dict(row)
        item["rank"] = i
        ranked.append(item)

    by_symbol: dict[str, int] = defaultdict(int)
    for row in ranked:
        by_symbol[_primary_symbol(row)] += 1

    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size": size,
        "count": len(ranked),
        "seats_per_symbol": seats_per_symbol,
        "by_symbol": dict(sorted(by_symbol.items())),
        "entries": ranked,
        "pool": list(pool or ranked)[:DEFAULT_POOL_SIZE],
    }
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def _behavioral_cluster_key(row: dict[str, Any]) -> str:
    """Collapse near-duplicate GapFade twins into one behavioral cluster."""
    c = row.get("candidate") or {}
    cls = str(c.get("class_name") or "")
    sym = _primary_symbol(row)
    params = c.get("params") or {}
    # Coarse buckets — not exact params
    if cls == "GapFade":
        gap = float(params.get("gap_pct") or 0.0)
        hold = int(params.get("hold_bars") or 0)
        gap_bucket = round(gap, 3)
        return f"{cls}|{sym}|g{gap_bucket}|h{hold}"
    if cls == "BreakoutATR":
        lb = int(params.get("lookback") or 0)
        return f"{cls}|{sym}|lb{lb}"
    return _fingerprint(c)


def behavioral_dedupe(rows: list[dict[str, Any]], *, max_per_cluster: int = 2) -> list[dict[str, Any]]:
    """Keep at most N representatives per behavioral cluster (strongest first)."""
    ranked = sorted(rows, key=_sort_key, reverse=True)
    counts: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for row in ranked:
        key = _behavioral_cluster_key(row)
        if counts[key] >= max_per_cluster:
            continue
        item = dict(row)
        item["behavior_cluster"] = key
        out.append(item)
        counts[key] += 1
    return out


def update_top50(
    newcomers: list[dict[str, Any]],
    *,
    path: Path | None = None,
    size: int = TOP50_SIZE,
    seats_per_symbol: int = 0,
    pool_size: int = DEFAULT_POOL_SIZE,
    universe_id: str | None = None,
    max_per_behavior_cluster: int = 2,
) -> list[dict[str, Any]]:
    """Merge newcomers into pool, then keep absolute global Top-N (no sector quotas)."""
    if path is None and universe_id is not None:
        path = leaderboard_path(universe_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board = load_top50(path)
    by_fp: dict[str, dict[str, Any]] = {}

    for row in board.get("pool") or board.get("entries") or []:
        slim = _slim_row(row, seen_at=row.get("updated_at") or now)
        by_fp[slim["fingerprint"]] = slim

    for row in newcomers:
        slim = _slim_row(row, seen_at=now)
        fp = slim["fingerprint"]
        prev = by_fp.get(fp)
        if prev is None:
            slim["first_seen_at"] = now
            by_fp[fp] = slim
        elif _sort_key(slim) > _sort_key(prev):
            slim["first_seen_at"] = prev.get("first_seen_at") or now
            by_fp[fp] = slim

    pool = sorted(by_fp.values(), key=_sort_key, reverse=True)[:pool_size]
    pool = behavioral_dedupe(pool, max_per_cluster=max(1, int(max_per_behavior_cluster)))
    # Absolute global board — sector best-3 lives in sector_best.py separately.
    if seats_per_symbol and seats_per_symbol > 0:
        ranked = select_with_sector_seats(pool, size=size, seats_per_symbol=seats_per_symbol)
    else:
        ranked = [{**row, "sector": _primary_symbol(row), "sector_seat": False} for row in pool[:size]]
    save_top50(
        ranked,
        path=path,
        size=size,
        pool=pool,
        seats_per_symbol=0,
    )
    return ranked


def top50_public(entries: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Compact rows for API/dashboard."""
    rows = entries if entries is not None else load_top50().get("entries") or []
    out = []
    for i, row in enumerate(rows[:TOP50_SIZE], 1):
        c = row.get("candidate") or {}
        fit = row.get("fitness") or {}
        out.append(
            {
                "rank": row.get("rank") or i,
                "strategy_id": c.get("strategy_id"),
                "class_name": c.get("class_name"),
                "symbols": c.get("symbols") or [],
                "sector": row.get("sector") or _primary_symbol(row),
                "sector_seat": bool(row.get("sector_seat")),
                "source": c.get("source"),
                "params": c.get("params") or {},
                "passed": fit.get("passed"),
                "oos_cagr": fit.get("oos_cagr"),
                "oos_max_dd": fit.get("oos_max_dd"),
                "score": fit.get("score"),
                "oos_trades": fit.get("oos_trades"),
                "hit_target_cagr": fit.get("hit_target_cagr"),
                "updated_at": row.get("updated_at"),
            }
        )
    return out
