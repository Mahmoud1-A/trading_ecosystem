from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.backtest.metrics import compute_metrics
from trading_ecosystem.common.config import data_dir
from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.leaderboard import _primary_symbol
from trading_ecosystem.discovery.portfolio_select import select_sector_champions
from trading_ecosystem.discovery.universe import get_active_universe, processed_artifact
from trading_ecosystem.execution.costs import load_execution_costs
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategy


def portfolio_book_path(universe_id: str | None = None) -> Path:
    return processed_artifact("discovery_portfolio_book", universe_id=universe_id)


def book_objective(
    metrics: dict[str, Any] | None,
    *,
    prop_max_total_dd: float = 0.10,
) -> float:
    """
    Score the combined book for profit under risk.
    Prefer high MAR / CAGR while staying near Prop total-DD.
    """
    if not metrics:
        return -1e18
    cagr = float(metrics.get("cagr") or 0.0)
    dd = float(metrics.get("max_drawdown") or 1.0)
    if dd <= 1e-12:
        mar = cagr * 100.0
    else:
        mar = float(metrics.get("mar") if metrics.get("mar") is not None else cagr / dd)
    # Heavy penalty if book DD exceeds prop total DD (realized equity path).
    over = max(0.0, dd - prop_max_total_dd)
    return mar + 2.0 * cagr - 80.0 * over - 5.0 * dd


def load_portfolio_book(path: Path | None = None) -> dict[str, Any]:
    p = path or portfolio_book_path()
    if not p.exists():
        return {
            "updated_at": None,
            "members": [],
            "metrics": {},
            "score": None,
            "best_ever": None,
        }
    return json.loads(p.read_text(encoding="utf-8"))


def save_portfolio_book(payload: dict[str, Any], path: Path | None = None) -> Path:
    p = path or portfolio_book_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return p


def _sector_of(row: dict[str, Any]) -> str:
    return str(row.get("sector") or _primary_symbol(row))


def _slim_member(row: dict[str, Any]) -> dict[str, Any]:
    c = row.get("candidate") or {}
    fit = row.get("fitness") or {}
    sector = _sector_of(row)
    return {
        "candidate": {
            "strategy_id": c.get("strategy_id"),
            "class_name": c.get("class_name"),
            "symbols": list(c.get("symbols") or []),
            "params": dict(c.get("params") or {}),
            "source": c.get("source"),
            "meta": c.get("meta") or {},
        },
        "fitness": {
            "passed": fit.get("passed"),
            "score": fit.get("score"),
            "oos_cagr": fit.get("oos_cagr"),
            "oos_max_dd": fit.get("oos_max_dd"),
            "oos_mar": fit.get("oos_mar"),
            "oos_trades": fit.get("oos_trades"),
            "reason": fit.get("reason"),
        },
        "full_metrics": row.get("full_metrics") or {},
        "sector": sector,
    }


def members_to_map(members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in members:
        sym = _sector_of(row)
        if sym and sym != "_NONE_":
            out[sym] = _slim_member(row)
    return out


def map_to_members(by_sector: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [_slim_member(by_sector[s]) for s in sorted(by_sector.keys())]


def run_combined_book(
    members: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    *,
    starting_equity: float,
    universe_id: str | None = None,
) -> dict[str, Any]:
    strategies = []
    for e in members:
        c = e.get("candidate") or {}
        if not c.get("class_name") or not c.get("strategy_id"):
            continue
        strategies.append(
            build_strategy(
                c["class_name"],
                c["strategy_id"],
                list(c.get("symbols") or []),
                dict(c.get("params") or {}),
            )
        )
    if not strategies:
        return {}
    needed = sorted({s for st in strategies for s in st.symbols})
    sub = {s: panel.get(s, []) for s in needed}
    costs = load_execution_costs(universe_id=universe_id or get_active_universe())
    port = BacktestEngine(
        strategies=strategies,
        risk_manager=MasterRiskManager(),
        starting_equity=starting_equity,
        costs=costs,
    ).run(sub)
    metrics = port.metrics.as_dict()
    metrics["members"] = len(strategies)
    metrics["mode"] = "aggregate_book"
    metrics["risk_rejects"] = port.risk_rejects
    metrics["cost_profile"] = costs.profile
    return metrics


def run_equal_weight_champions(
    members: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    *,
    starting_equity: float,
    universe_id: str | None = None,
) -> tuple[dict[str, Any], list[tuple]]:
    """
    Realistic crypto sleeve: one champion per coin, equal capital per sleeve.
    Each strategy is backtested alone on equity/N; portfolio equity = sum of sleeves.
    """
    built: list[Any] = []
    for e in members:
        c = e.get("candidate") or {}
        if not c.get("class_name") or not c.get("strategy_id"):
            continue
        built.append(
            (
                e,
                build_strategy(
                    c["class_name"],
                    c["strategy_id"],
                    list(c.get("symbols") or []),
                    dict(c.get("params") or {}),
                ),
            )
        )
    if not built:
        return {}, []

    n = len(built)
    sleeve_eq = float(starting_equity) / n
    costs = load_execution_costs(universe_id=universe_id or get_active_universe())
    by_ts: dict[Any, float] = defaultdict(float)
    risk_rejects = 0
    member_cagrs: list[float] = []
    member_pfs: list[float] = []
    member_wrs: list[float] = []
    total_trades = 0

    for _row, strat in built:
        needed = sorted(set(strat.symbols))
        sub = {s: panel.get(s, []) for s in needed}
        result = BacktestEngine(
            strategies=[strat],
            risk_manager=MasterRiskManager(),
            starting_equity=sleeve_eq,
            costs=costs,
        ).run(sub)
        for ts, eq in result.equity_curve:
            by_ts[ts] += float(eq)
        risk_rejects += int(result.risk_rejects)
        member_cagrs.append(float(result.metrics.cagr))
        member_pfs.append(float(result.metrics.profit_factor))
        member_wrs.append(float(result.metrics.win_rate))
        total_trades += int(result.metrics.num_trades)

    curve = sorted(by_ts.items(), key=lambda x: x[0])
    metrics = compute_metrics(curve, []).as_dict()
    metrics["members"] = n
    metrics["mode"] = "sector_champions"
    metrics["allocation"] = "equal_weight"
    metrics["sleeve_equity"] = sleeve_eq
    metrics["risk_rejects"] = risk_rejects
    metrics["cost_profile"] = costs.profile
    metrics["avg_member_cagr"] = float(sum(member_cagrs) / n) if member_cagrs else 0.0
    metrics["num_trades"] = total_trades
    metrics["profit_factor"] = float(sum(member_pfs) / n) if member_pfs else 0.0
    metrics["win_rate"] = float(sum(member_wrs) / n) if member_wrs else 0.0
    return metrics, curve


def build_champions_book(
    *,
    sector_map: dict[str, list[dict[str, Any]]],
    panel: dict[str, list[Bar]],
    starting_equity: float,
    prop_max_total_dd: float = 0.10,
    require_passed: bool = True,
    rank: int = 1,
    universe_id: str | None = None,
    min_bars: int = 0,
    max_heroes: int | None = None,
    min_oos_cagr: float | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """
    Per-symbol champions, optionally sparsified to top-K heroes.
    Equal-weight sleeve metrics — no free multi-asset weight genome.
    """
    sparse = max_heroes is not None and int(max_heroes) > 0
    if progress_cb:
        progress_cb(
            "selecting sparse heroes…" if sparse else "selecting per-symbol champions…",
            0,
            1,
        )

    selected = select_sector_champions(
        sector_map,
        rank=rank,
        require_passed=require_passed,
    )
    if min_bars > 0:
        filtered: list[dict[str, Any]] = []
        for row in selected:
            sym = _sector_of(row)
            bars = panel.get(sym) or []
            if len(bars) >= int(min_bars):
                filtered.append(row)
        selected = filtered

    if min_oos_cagr is not None:
        floor = float(min_oos_cagr)
        selected = [
            r
            for r in selected
            if float((r.get("fitness") or {}).get("oos_cagr") or -1e9) >= floor
        ]

    if sparse:
        k = max(1, int(max_heroes))
        # Marginal contribution: greedily add the candidate that most improves book_objective
        pool = sorted(
            selected,
            key=lambda r: (
                float((r.get("fitness") or {}).get("oos_cagr") or -1e9),
                float((r.get("fitness") or {}).get("score") or -1e9),
            ),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        best_score = -1e18
        # Cap family concentration
        family_counts: dict[str, int] = defaultdict(int)
        max_per_family = max(1, k // 2)
        for _ in range(k):
            best_row = None
            best_try = best_score
            for row in pool:
                if row in chosen:
                    continue
                fam = str((row.get("candidate") or {}).get("class_name") or "")
                if family_counts[fam] >= max_per_family and len(chosen) > 0:
                    continue
                trial = chosen + [row]
                mem = [_slim_member(r) for r in trial]
                try:
                    m, _ = run_equal_weight_champions(
                        mem, panel, starting_equity=starting_equity, universe_id=universe_id
                    )
                    sc = book_objective(m, prop_max_total_dd=prop_max_total_dd)
                except Exception:  # noqa: BLE001
                    sc = float((row.get("fitness") or {}).get("oos_cagr") or -1e9)
                if sc > best_try:
                    best_try = sc
                    best_row = row
            if best_row is None:
                # fallback: next by individual OOS
                for row in pool:
                    if row not in chosen:
                        best_row = row
                        break
            if best_row is None:
                break
            chosen.append(best_row)
            fam = str((best_row.get("candidate") or {}).get("class_name") or "")
            family_counts[fam] += 1
            best_score = best_try
        selected = chosen

    members = [_slim_member(r) for r in selected]
    mode_name = "sparse_heroes" if sparse else "sector_champions"
    if progress_cb:
        progress_cb(f"equal-weight backtest on {len(members)} {mode_name}…", 0, 1)

    metrics, _curve = run_equal_weight_champions(
        members,
        panel,
        starting_equity=starting_equity,
        universe_id=universe_id,
    )
    score = book_objective(metrics, prop_max_total_dd=prop_max_total_dd)
    metrics = dict(metrics or {})
    metrics["mode"] = mode_name
    metrics["book_score"] = score
    metrics["swaps_tried"] = 0
    metrics["swaps_accepted"] = 0
    metrics["max_heroes"] = int(max_heroes) if sparse else None
    metrics["pool_size"] = len(sector_map)

    locked = load_portfolio_book(portfolio_book_path(universe_id))
    best_ever = locked.get("best_ever")
    prev_best = float(best_ever["score"]) if best_ever and best_ever.get("score") is not None else None
    prev_mode = ((best_ever or {}).get("metrics") or {}).get("mode")
    if (
        best_ever is None
        or prev_mode not in {mode_name, "sector_champions", "sparse_heroes"}
        or prev_best is None
        or score > prev_best
    ):
        best_ever = {
            "score": score,
            "metrics": metrics,
            "members": members,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    payload = {
        "universe": universe_id,
        "members": members,
        "metrics": metrics,
        "score": score,
        "best_ever": best_ever,
        "objective": f"{mode_name}_equal_weight",
    }
    save_portfolio_book(payload, portfolio_book_path(universe_id))
    if progress_cb:
        progress_cb(f"{mode_name} book saved", 1, 1)
    return members, metrics, payload


def _challenge_pool(
    *,
    sector_map: dict[str, list[dict[str, Any]]],
    evaluated: list[dict[str, Any]],
    current: dict[str, dict[str, Any]],
    require_passed: bool,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(row: dict[str, Any]) -> None:
        fit = row.get("fitness") or {}
        if require_passed and not fit.get("passed"):
            return
        c = row.get("candidate") or {}
        sid = str(c.get("strategy_id") or "")
        fp = f"{_sector_of(row)}|{c.get('class_name')}|{sorted((c.get('params') or {}).items())}"
        if fp in seen:
            return
        seen.add(fp)
        slim = _slim_member(row)
        cur = current.get(slim["sector"])
        if cur and (cur.get("candidate") or {}).get("strategy_id") == sid:
            return
        pool.append(slim)

    for row in evaluated:
        _add(row)
    for _sym, rows in sector_map.items():
        for row in rows[:3]:
            _add(row)

    pool.sort(
        key=lambda r: (
            float((r.get("fitness") or {}).get("oos_cagr") or -1e9),
            float((r.get("fitness") or {}).get("score") or -1e9),
        ),
        reverse=True,
    )
    return pool


def improve_aggregate_book(
    *,
    sector_map: dict[str, list[dict[str, Any]]],
    evaluated: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    starting_equity: float,
    prop_max_total_dd: float = 0.10,
    require_passed: bool = True,
    max_swap_trials: int = 24,
    locked: dict[str, Any] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
    workers: int = 1,
    universe_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """
    Optimize the combined sector book for aggregate profit/risk.

    Starts from the locked book (if any), else from #1 per sector.
    Accepts a sector swap only when the combined-book objective improves.
    """
    book_path = portfolio_book_path(universe_id)
    locked = locked if locked is not None else load_portfolio_book(book_path)
    current = members_to_map(list(locked.get("members") or []))

    if not current:
        # Prefer historical best_ever only when members were not explicitly cleared.
        # Do NOT resurrect from discovery_report.json — that blocked fresh-search resets
        # and kept showing the old GapFade sleeve (e.g. freeze ~71%) on the UI.
        be_members = ((locked.get("best_ever") or {}).get("members")) or []
        if be_members and locked.get("members") is None:
            # legacy payloads without members key
            current = members_to_map(list(be_members))
        if not current:
            seed = select_sector_champions(
                sector_map,
                rank=1,
                require_passed=require_passed,
            )
            current = members_to_map(seed)

    if progress_cb:
        progress_cb("baseline combined book backtest…", 0, max_swap_trials)

    members = map_to_members(current)
    metrics = (
        run_combined_book(
            members, panel, starting_equity=starting_equity, universe_id=universe_id
        )
        if members
        else {}
    )
    best_score = book_objective(metrics, prop_max_total_dd=prop_max_total_dd)
    accepted = 0
    tried = 0

    challenges = _challenge_pool(
        sector_map=sector_map,
        evaluated=evaluated,
        current=current,
        require_passed=require_passed,
    )
    # Also try filling missing sectors (as challenges) — accepted only if book improves.
    for sym, rows in sector_map.items():
        if sym in current:
            continue
        for row in rows[:1]:
            fit = row.get("fitness") or {}
            if require_passed and not fit.get("passed"):
                continue
            challenges.append(_slim_member(row))

    # Score many swaps in parallel vs frozen book; accept the single best improvement.
    from trading_ecosystem.discovery.parallel import score_swaps_parallel

    remaining = [c for c in challenges if c.get("sector") and c.get("sector") != "_NONE_"]
    batch = remaining[:max_swap_trials]
    planned = len(batch)
    n_workers = max(1, int(workers))
    if batch:
        if progress_cb:
            progress_cb(
                f"book swaps parallel {planned} trials on {n_workers} workers…",
                0,
                planned,
            )
        scored = score_swaps_parallel(
            members=members,
            challenges=batch,
            panel=panel,
            starting_equity=starting_equity,
            prop_max_total_dd=prop_max_total_dd,
            workers=n_workers,
            progress_cb=progress_cb,
            universe_id=universe_id,
        )
        tried = len(batch)
        if scored:
            best = max(scored, key=lambda r: float(r.get("score") or -1e18))
            trial_score = float(best.get("score") or -1e18)
            if trial_score > best_score + 1e-9:
                members = list(best.get("members") or members)
                current = members_to_map(members)
                metrics = dict(best.get("metrics") or {})
                best_score = trial_score
                accepted = 1

    metrics = dict(metrics or {})
    metrics["mode"] = "aggregate_book"
    metrics["members"] = len(members)
    metrics["book_score"] = best_score
    metrics["swaps_tried"] = tried
    metrics["swaps_accepted"] = accepted

    best_ever = locked.get("best_ever")
    prev_best_score = None
    if best_ever and best_ever.get("score") is not None:
        prev_best_score = float(best_ever["score"])
    if prev_best_score is None or best_score > prev_best_score:
        best_ever = {
            "score": best_score,
            "metrics": metrics,
            "members": members,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    payload = {
        "universe": universe_id,
        "members": members,
        "metrics": metrics,
        "score": best_score,
        "best_ever": best_ever,
        "objective": "aggregate_book_mar_cagr_prop",
    }
    save_portfolio_book(payload, path=book_path)
    return members, metrics, payload
