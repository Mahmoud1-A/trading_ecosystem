from __future__ import annotations

import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Callable

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.runner import evaluate_candidate
from trading_ecosystem.discovery.search_space import Candidate, candidate_from_dict

logger = logging.getLogger(__name__)

_PANEL: dict[str, list[Bar]] | None = None
_DISCOVERY_CFG: dict[str, Any] | None = None
_PROP: dict[str, Any] | None = None
_STARTING_EQUITY: float = 100_000.0
_PROP_MAX_DD: float = 0.10


def default_workers(configured: int | None = None) -> int:
    """
    Parallel backtest workers capped at logical CPU count.
    Reserves 2 cores for OS / te-monitor so eval never oversubscribes.
    """
    cpu = max(1, os.cpu_count() or 4)
    budget = max(1, cpu - 2)  # highest sustained throughput on this machine
    if configured is not None and int(configured) > 0:
        return max(1, min(int(configured), cpu, budget))
    return max(2, budget)


def _init_worker(
    panel: dict[str, list[Bar]],
    discovery_cfg: dict[str, Any],
    prop: dict[str, Any],
    starting_equity: float,
    prop_max_total_dd: float = 0.10,
) -> None:
    global _PANEL, _DISCOVERY_CFG, _PROP, _STARTING_EQUITY, _PROP_MAX_DD
    _PANEL = panel
    _DISCOVERY_CFG = discovery_cfg
    _PROP = prop
    _STARTING_EQUITY = float(starting_equity)
    _PROP_MAX_DD = float(prop_max_total_dd)


def _eval_candidate_task(cand_dict: dict[str, Any]) -> dict[str, Any]:
    assert _PANEL is not None and _DISCOVERY_CFG is not None and _PROP is not None
    cand = candidate_from_dict(cand_dict)
    try:
        return evaluate_candidate(
            cand,
            _PANEL,
            discovery_cfg=_DISCOVERY_CFG,
            prop=_PROP,
            starting_equity=_STARTING_EQUITY,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate": cand_dict,
            "fitness": {
                "passed": False,
                "score": -1e9,
                "reason": f"error:{exc}",
                "oos_cagr": 0.0,
                "oos_max_dd": 0.0,
                "oos_mar": 0.0,
                "oos_trades": 0,
                "positive_oos_folds": 0,
                "hit_target_cagr": False,
                "halted_or_prop_breach": False,
                "fold_metrics": [],
            },
            "full_metrics": {},
            "full_equity_curve": [],
        }


def _score_swap_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Score one sector swap against a frozen book baseline (worker)."""
    from trading_ecosystem.discovery.aggregate_book import (
        book_objective,
        map_to_members,
        members_to_map,
        run_combined_book,
    )

    assert _PANEL is not None
    current = members_to_map(list(payload["members"]))
    chal = payload["challenge"]
    sector = chal["sector"]
    trial = dict(current)
    trial[sector] = chal
    members = map_to_members(trial)
    metrics = run_combined_book(
        members,
        _PANEL,
        starting_equity=_STARTING_EQUITY,
        universe_id=payload.get("universe_id"),
    )
    score = book_objective(metrics, prop_max_total_dd=_PROP_MAX_DD)
    return {
        "sector": sector,
        "challenge": chal,
        "score": score,
        "metrics": metrics,
        "members": members,
    }


def evaluate_candidates_parallel(
    candidates: list[Candidate],
    panel: dict[str, list[Bar]],
    *,
    discovery_cfg: dict[str, Any],
    prop: dict[str, Any],
    starting_equity: float,
    workers: int,
    progress_cb: Callable[[int, int, dict[str, Any] | None], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a batch of candidates using multiple processes."""
    if not candidates:
        return []
    n_workers = max(1, min(int(workers), len(candidates)))
    if n_workers == 1:
        out: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates, 1):
            try:
                row = evaluate_candidate(
                    cand,
                    panel,
                    discovery_cfg=discovery_cfg,
                    prop=prop,
                    starting_equity=starting_equity,
                )
            except Exception as exc:  # noqa: BLE001
                row = {
                    "candidate": {
                        "strategy_id": cand.strategy_id,
                        "class_name": cand.class_name,
                        "symbols": list(cand.symbols),
                        "params": dict(cand.params),
                        "source": cand.source,
                        "meta": dict(cand.meta),
                    },
                    "fitness": {
                        "passed": False,
                        "score": -1e9,
                        "reason": f"error:{exc}",
                        "oos_cagr": 0.0,
                        "oos_max_dd": 0.0,
                        "oos_mar": 0.0,
                        "oos_trades": 0,
                        "positive_oos_folds": 0,
                        "hit_target_cagr": False,
                        "halted_or_prop_breach": False,
                        "fold_metrics": [],
                    },
                    "full_metrics": {},
                    "full_equity_curve": [],
                }
            out.append(row)
            if progress_cb:
                progress_cb(i, len(candidates), row)
        return out

    payloads = [
        {
            "strategy_id": c.strategy_id,
            "class_name": c.class_name,
            "symbols": list(c.symbols),
            "params": dict(c.params),
            "source": c.source,
            "meta": dict(c.meta),
        }
        for c in candidates
    ]
    results: list[dict[str, Any] | None] = [None] * len(payloads)
    done = 0
    if progress_cb:
        progress_cb(0, len(payloads), None)

    # Per-future stall timeout (seconds). Avoids multi-hour hangs if a worker dies.
    stall_timeout = 180.0
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(panel, discovery_cfg, prop, starting_equity, 0.10),
        ) as pool:
            futures = {
                pool.submit(_eval_candidate_task, p): idx for idx, p in enumerate(payloads)
            }
            pending = set(futures.keys())
            while pending:
                finished, pending = wait(
                    pending, timeout=stall_timeout, return_when=FIRST_COMPLETED
                )
                if not finished:
                    logger.error(
                        "Parallel eval stalled for %.0fs with %s pending — falling back sequential",
                        stall_timeout,
                        len(pending),
                    )
                    for fut in list(pending):
                        fut.cancel()
                        idx = futures[fut]
                        if results[idx] is None:
                            # Evaluate remaining in-process so the generation can continue.
                            cand = candidate_from_dict(payloads[idx])
                            try:
                                row = evaluate_candidate(
                                    cand,
                                    panel,
                                    discovery_cfg=discovery_cfg,
                                    prop=prop,
                                    starting_equity=starting_equity,
                                )
                            except Exception as exc:  # noqa: BLE001
                                row = {
                                    "candidate": payloads[idx],
                                    "fitness": {
                                        "passed": False,
                                        "score": -1e9,
                                        "reason": f"stall_fallback:{exc}",
                                        "oos_cagr": 0.0,
                                        "oos_max_dd": 0.0,
                                        "oos_mar": 0.0,
                                        "oos_trades": 0,
                                        "positive_oos_folds": 0,
                                        "hit_target_cagr": False,
                                        "halted_or_prop_breach": False,
                                        "fold_metrics": [],
                                    },
                                    "full_metrics": {},
                                    "full_equity_curve": [],
                                }
                            results[idx] = row
                            done += 1
                            if progress_cb:
                                progress_cb(done, len(payloads), row)
                    break
                for fut in finished:
                    idx = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("parallel candidate failed")
                        p = payloads[idx]
                        row = {
                            "candidate": p,
                            "fitness": {
                                "passed": False,
                                "score": -1e9,
                                "reason": f"error:{exc}",
                                "oos_cagr": 0.0,
                                "oos_max_dd": 0.0,
                                "oos_mar": 0.0,
                                "oos_trades": 0,
                                "positive_oos_folds": 0,
                                "hit_target_cagr": False,
                                "halted_or_prop_breach": False,
                                "fold_metrics": [],
                            },
                            "full_metrics": {},
                            "full_equity_curve": [],
                        }
                    results[idx] = row
                    done += 1
                    if progress_cb:
                        progress_cb(done, len(payloads), row)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Process pool failed (%s) — sequential fallback", exc)
        return evaluate_candidates_parallel(
            candidates,
            panel,
            discovery_cfg=discovery_cfg,
            prop=prop,
            starting_equity=starting_equity,
            workers=1,
            progress_cb=progress_cb,
        )
    return [r for r in results if r is not None]


def score_swaps_parallel(
    *,
    members: list[dict[str, Any]],
    challenges: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    starting_equity: float,
    prop_max_total_dd: float,
    workers: int,
    progress_cb: Callable[[str, int, int], None] | None = None,
    universe_id: str | None = None,
) -> list[dict[str, Any]]:
    """Score many sector swaps vs the same frozen book.

    Book backtests are heavy; keep worker count modest and fall back sequential on stall.
    """
    if not challenges:
        return []
    # Cap book workers — full-book sims are heavy; stay below eval budget on Windows.
    n_workers = max(1, min(int(workers), 8, len(challenges)))
    payloads = [
        {"members": members, "challenge": c, "universe_id": universe_id} for c in challenges
    ]

    def _sequential() -> list[dict[str, Any]]:
        _init_worker(panel, {}, {}, starting_equity, prop_max_total_dd)
        out_local: list[dict[str, Any]] = []
        for i, p in enumerate(payloads, 1):
            if progress_cb:
                progress_cb(
                    f"book swap trial {i}/{len(payloads)} on {p['challenge'].get('sector')}",
                    i - 1,
                    len(payloads),
                )
            out_local.append(_score_swap_task(p))
        return out_local

    if n_workers == 1:
        return _sequential()

    out: list[dict[str, Any] | None] = [None] * len(payloads)
    done = 0
    stall_timeout = 300.0
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(panel, {}, {}, starting_equity, prop_max_total_dd),
        ) as pool:
            futures = {
                pool.submit(_score_swap_task, p): idx for idx, p in enumerate(payloads)
            }
            pending = set(futures.keys())
            while pending:
                finished, pending = wait(
                    pending, timeout=stall_timeout, return_when=FIRST_COMPLETED
                )
                if not finished:
                    logger.error(
                        "Book swap pool stalled (%.0fs, pending=%s) — sequential fallback",
                        stall_timeout,
                        len(pending),
                    )
                    for fut in list(pending):
                        fut.cancel()
                    # Fill any missing via sequential for remaining indices
                    missing = [i for i, r in enumerate(out) if r is None]
                    _init_worker(panel, {}, {}, starting_equity, prop_max_total_dd)
                    for i in missing:
                        if progress_cb:
                            progress_cb(
                                f"book swap fallback {done + 1}/{len(payloads)}",
                                done,
                                len(payloads),
                            )
                        out[i] = _score_swap_task(payloads[i])
                        done += 1
                    break
                for fut in finished:
                    idx = futures[fut]
                    try:
                        out[idx] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("parallel book swap failed")
                        out[idx] = {
                            "sector": challenges[idx].get("sector"),
                            "challenge": challenges[idx],
                            "score": -1e18,
                            "metrics": {},
                            "members": members,
                            "error": str(exc),
                        }
                    done += 1
                    if progress_cb:
                        sec = challenges[idx].get("sector")
                        progress_cb(
                            f"book swap scored {done}/{len(payloads)} ({sec})",
                            done,
                            len(payloads),
                        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Book swap pool failed (%s) — sequential", exc)
        return _sequential()
    return [r for r in out if r is not None]
