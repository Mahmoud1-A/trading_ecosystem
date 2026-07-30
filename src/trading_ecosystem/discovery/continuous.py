from __future__ import annotations

import json
import logging
import signal
import time
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml
from trading_ecosystem.discovery.anomaly import scan_anomalies
from trading_ecosystem.discovery.evolve import breed_generation
from trading_ecosystem.discovery.leaderboard import TOP50_SIZE, leaderboard_path, load_top50, top50_public, update_top50
from trading_ecosystem.discovery.sector_best import (
    DEFAULT_BEST_PER_SECTOR,
    load_sector_best,
    sector_best_path,
    sector_best_public,
    update_sector_best,
)
from trading_ecosystem.discovery.aggregate_book import improve_aggregate_book
from trading_ecosystem.discovery.parallel import default_workers, evaluate_candidates_parallel
from trading_ecosystem.discovery.report import (
    candidates_to_strategies_yaml,
    finalize_report,
    write_report,
    write_strategies_yaml,
)
from trading_ecosystem.discovery.runner import _load_panel
from trading_ecosystem.discovery.search_space import Candidate, candidate_from_dict, expand_family_candidates
from trading_ecosystem.discovery.universe import (
    apply_universe_to_discovery_cfg,
    processed_artifact,
    set_active_universe,
    strategies_export_path,
)
from trading_ecosystem.monitoring.state_store import MONITOR

logger = logging.getLogger(__name__)

_STOP = False


def _request_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True
    logger.warning("Stop signal received — finishing current candidate then exiting")


def _stop_flag_path() -> Path:
    return data_dir() / "processed" / "discovery_stop.flag"


def _migrate_legacy_hof() -> None:
    """One-time pull from old hall-of-fame file into top50 if present."""
    legacy = data_dir() / "processed" / "discovery_hall_of_fame.json"
    if not legacy.exists():
        return
    try:
        raw = json.loads(legacy.read_text(encoding="utf-8"))
        elites = list(raw.get("elites") or [])
        if elites:
            update_top50(elites, size=TOP50_SIZE)
            update_sector_best(elites, best_per_sector=DEFAULT_BEST_PER_SECTOR)
    except Exception:  # noqa: BLE001
        logger.exception("Failed migrating legacy hall of fame")


def _seed_population(discovery_cfg: dict[str, Any], panel: dict, rng_seed: int) -> list[Candidate]:
    fam = expand_family_candidates(discovery_cfg, max_per_family=12, seed=rng_seed)
    anom = scan_anomalies(panel, top_k=12)
    # Prefer diversity for seed
    mixed = fam[:20] + anom[:12]
    return mixed


def run_continuous_discovery(
    *,
    start: str = "2018-01-01",
    end: str | None = None,
    batch_size: int | None = None,
    max_generations: int | None = None,
    sleep_sec: float | None = None,
    export_path: str | Path | None = None,
    report_path: str | Path | None = None,
    universe: str | None = None,
) -> dict[str, Any]:
    """
    24/7 evolutionary discovery loop.
    Stop with Ctrl+C or by creating data/processed/discovery_stop.flag
    """
    global _STOP
    _STOP = False
    if _stop_flag_path().exists():
        _stop_flag_path().unlink()

    try:
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)
    except Exception:  # noqa: BLE001 — not all platforms allow
        pass

    if universe:
        set_active_universe(universe)

    discovery_cfg = load_yaml(CONFIG_DIR / "discovery.yaml")
    discovery_cfg, uni = apply_universe_to_discovery_cfg(discovery_cfg, universe_id=universe)
    cont = discovery_cfg.get("continuous") or {}
    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    timeframe = str(uni.get("timeframe") or "1d")
    starting_equity = float(prop.get("starting_equity", 100_000))

    batch = int(batch_size if batch_size is not None else cont.get("batch_size", 24))
    elite_size = int(cont.get("elite_size", 12))
    workers = default_workers(cont.get("workers"))
    board_size = int(discovery_cfg.get("leaderboard_size", TOP50_SIZE))
    per_sector = int(discovery_cfg.get("best_per_sector", DEFAULT_BEST_PER_SECTOR))
    pause = float(sleep_sec if sleep_sec is not None else cont.get("sleep_sec_between_generations", 2))
    seed = int(discovery_cfg.get("random_seed", 42))
    logger.info(
        "Continuous discovery universe=%s symbols=%s workers=%s batch=%s",
        uni["id"],
        uni["count"],
        workers,
        batch,
    )

    import random

    rng = random.Random(seed + int(time.time()) % 10_000)

    symbols = list(uni["symbols"])
    panel = _load_panel(symbols, start=start, end=end, timeframe=timeframe)

    _migrate_legacy_hof() if uni["id"] == "etf" else None
    top_rows = list(load_top50(leaderboard_path(uni["id"])).get("entries") or [])
    # Guard: never keep symbols outside the active universe (prevents ETF↔crypto bleed).
    allowed = set(uni["symbols"])
    def _in_universe(row: dict[str, Any]) -> bool:
        syms = ((row.get("candidate") or {}).get("symbols") or [])
        return bool(syms) and all(s in allowed for s in syms)

    top_rows = [r for r in top_rows if _in_universe(r)]
    if top_rows:
        from trading_ecosystem.discovery.leaderboard import save_top50

        save_top50(top_rows, path=leaderboard_path(uni["id"]), size=board_size)
    # Breeding pool = top elite_size from the top-50 registry
    elites_rows = top_rows[:elite_size]
    elite_candidates = [
        candidate_from_dict(r["candidate"]) for r in elites_rows if r.get("candidate")
    ]

    generation = 0
    total_evaluated = 0
    best_ever: dict[str, Any] | None = top_rows[0] if top_rows else None
    final_board: list[dict[str, Any]] = top_rows

    MONITOR.discovery_start(
        phase="continuous",
        total=batch,
        message=f"Continuous evolution [{uni['id']}] — invent / mutate / crossover under Prop",
    )
    MONITOR.discovery_patch(
        top50=top50_public(top_rows),
        best3_by_sector=sector_best_public(
            load_sector_best(sector_best_path(uni["id"])).get("sectors") or {}
        ),
        mode="continuous",
        workers=workers,
        universe=uni["id"],
        universe_label=uni.get("label_en") or uni["id"],
    )
    MONITOR.push_alert(
        "info",
        f"Continuous discovery started — universe={uni['id']} workers={workers}",
    )

    out_report = (
        Path(report_path)
        if report_path
        else processed_artifact("discovery_report", universe_id=uni["id"])
    )
    export = Path(export_path) if export_path else strategies_export_path(uni["id"])

    try:
        while not _STOP:
            if _stop_flag_path().exists():
                logger.info("Stop flag detected")
                break
            if max_generations is not None and generation >= max_generations:
                break

            generation += 1
            if generation == 1 and not elite_candidates:
                batch_candidates = _seed_population(discovery_cfg, panel, seed + generation)
                batch_candidates = batch_candidates[:batch]
            else:
                batch_candidates = breed_generation(
                    elite_candidates,
                    discovery_cfg,
                    panel,
                    generation=generation,
                    batch_size=batch,
                    rng=rng,
                )

            MONITOR.discovery_progress(
                current=0,
                total=len(batch_candidates),
                strategy_id=f"generation_{generation}",
                class_name="Evolution",
                symbols=[],
                source="continuous",
                message=(
                    f"Generation {generation}: evolving {len(batch_candidates)} candidates "
                    f"(mutate/crossover/invent)"
                ),
            )
            best_payload = None
            if best_ever:
                best_payload = {
                    "strategy_id": best_ever.get("candidate", {}).get("strategy_id"),
                    "class_name": best_ever.get("candidate", {}).get("class_name"),
                    "symbols": best_ever.get("candidate", {}).get("symbols"),
                    "oos_cagr": (best_ever.get("fitness") or {}).get("oos_cagr"),
                    "score": (best_ever.get("fitness") or {}).get("score"),
                    "passed": (best_ever.get("fitness") or {}).get("passed"),
                }
            MONITOR.discovery_patch(
                mode="continuous",
                generation=generation,
                total_evaluated=total_evaluated,
                best_ever=best_payload,
            )

            MONITOR.discovery_progress(
                current=0,
                total=len(batch_candidates),
                strategy_id=f"generation_{generation}",
                class_name="ParallelEval",
                symbols=[],
                source="continuous",
                message=(
                    f"Gen {generation}: evaluating {len(batch_candidates)} candidates "
                    f"on {workers} workers…"
                ),
            )

            def _cand_progress(done: int, total: int, row: dict[str, Any] | None) -> None:
                c = (row or {}).get("candidate") or {}
                fit = (row or {}).get("fitness") or {}
                MONITOR.discovery_progress(
                    current=done,
                    total=total,
                    strategy_id=str(c.get("strategy_id") or "parallel"),
                    class_name=str(c.get("class_name") or "ParallelEval"),
                    symbols=list(c.get("symbols") or []),
                    source=str(c.get("source") or "continuous"),
                    passed=None,
                    oos_cagr=fit.get("oos_cagr"),
                    message=(
                        f"Gen {generation}: parallel {done}/{total} "
                        f"({workers} workers) — {c.get('class_name')} on "
                        f"{','.join(c.get('symbols') or [])}"
                    ),
                )
                MONITOR.discovery_patch(
                    mode="continuous",
                    generation=generation,
                    total_evaluated=total_evaluated + done,
                )

            evaluated = evaluate_candidates_parallel(
                batch_candidates,
                panel,
                discovery_cfg=discovery_cfg,
                prop=prop,
                starting_equity=starting_equity,
                workers=workers,
                progress_cb=_cand_progress,
            )
            total_evaluated += len(evaluated)

            from trading_ecosystem.discovery.leaderboard import behavioral_dedupe

            unique_batch = behavioral_dedupe(evaluated, max_per_cluster=2)
            MONITOR.discovery_tally_batch(
                generated=len(batch_candidates),
                evaluated_rows=evaluated,
                behaviorally_unique=len(unique_batch),
            )

            if not evaluated:
                break

            top_rows = update_top50(evaluated, size=board_size, universe_id=uni["id"])
            sector_map = update_sector_best(
                evaluated, best_per_sector=per_sector, universe_id=uni["id"]
            )
            elites_rows = top_rows[:elite_size]
            elite_candidates = [
                candidate_from_dict(r["candidate"]) for r in elites_rows if r.get("candidate")
            ]
            if top_rows:
                best_ever = top_rows[0]
            MONITOR.discovery_patch(
                top50=top50_public(top_rows),
                best3_by_sector=sector_best_public(sector_map),
            )

            # Portfolio construction: crypto = per-coin champions; ETF = aggregate swaps.
            portfolio_mode = str(
                uni.get("portfolio_mode")
                or discovery_cfg.get("portfolio_mode")
                or "aggregate_book"
            ).strip().lower()
            prop_dd = float(prop.get("max_total_drawdown_pct", 10.0)) / 100.0

            def _book_progress(msg: str, cur: int, total: int) -> None:
                MONITOR.discovery_progress(
                    current=cur,
                    total=max(total, 1),
                    strategy_id=portfolio_mode,
                    class_name="ChampionsBook" if "champion" in portfolio_mode else "AggregateBook",
                    symbols=[],
                    source="continuous",
                    message=f"Gen {generation}: {msg}",
                )

            if portfolio_mode in {"sector_champions", "champions", "per_coin", "sparse_heroes", "heroes"}:
                from trading_ecosystem.discovery.aggregate_book import build_champions_book

                hero_n = uni.get("max_heroes")
                if portfolio_mode in {"sparse_heroes", "heroes"} and hero_n is None:
                    hero_n = 3
                MONITOR.discovery_progress(
                    current=len(batch_candidates),
                    total=len(batch_candidates),
                    strategy_id=portfolio_mode,
                    class_name="SparseHeroes" if portfolio_mode in {"sparse_heroes", "heroes"} else "ChampionsBook",
                    symbols=[],
                    source="continuous",
                    message=f"Gen {generation}: building {portfolio_mode} sleeve…",
                )
                selected, portfolio_metrics, book_payload = build_champions_book(
                    sector_map=sector_map,
                    panel=panel,
                    starting_equity=starting_equity,
                    prop_max_total_dd=prop_dd,
                    require_passed=bool(discovery_cfg.get("portfolio_require_passed", True)),
                    rank=int(discovery_cfg.get("portfolio_sector_rank", 1)),
                    universe_id=uni["id"],
                    min_bars=int(uni.get("min_bars_for_champion") or 0),
                    max_heroes=int(hero_n) if hero_n is not None else None,
                    min_oos_cagr=uni.get("min_hero_oos_cagr"),
                    progress_cb=_book_progress,
                )
            else:
                MONITOR.discovery_progress(
                    current=len(batch_candidates),
                    total=len(batch_candidates),
                    strategy_id="aggregate_book",
                    class_name="AggregateBook",
                    symbols=[],
                    source="continuous",
                    message=f"Gen {generation}: optimizing combined book (swap trials)…",
                )
                book_cap = int(cont.get("book_workers") or 8)
                book_workers = max(1, min(book_cap, int(workers), 8))
                selected, portfolio_metrics, book_payload = improve_aggregate_book(
                    sector_map=sector_map,
                    evaluated=evaluated,
                    panel=panel,
                    starting_equity=starting_equity,
                    prop_max_total_dd=prop_dd,
                    require_passed=bool(discovery_cfg.get("portfolio_require_passed", True)),
                    max_swap_trials=int(discovery_cfg.get("portfolio_max_swap_trials", 12)),
                    progress_cb=_book_progress,
                    workers=book_workers,
                    universe_id=uni["id"],
                )

            MONITOR.discovery_progress(
                current=len(batch_candidates),
                total=len(batch_candidates),
                strategy_id="stability_gate",
                class_name="StabilityGate",
                symbols=[],
                source="continuous",
                message=f"Gen {generation}: stability gate + readiness report…",
            )
            from trading_ecosystem.discovery.stability import evaluate_portfolio_readiness

            # Fingerprint cache skips full re-sim when book members unchanged.
            every_n = max(1, int(cont.get("readiness_every_n_gens", 1)))
            skip_unchanged = bool(cont.get("readiness_skip_unchanged", True))
            if not skip_unchanged:
                force_ready = True
            elif every_n > 1 and generation % every_n == 0:
                force_ready = True
            else:
                force_ready = False
            readiness = evaluate_portfolio_readiness(
                selected,
                panel,
                universe_id=uni["id"],
                starting_equity=starting_equity,
                portfolio_metrics=portfolio_metrics,
                prop_rules=prop,
                force=force_ready,
            )

            MONITOR.discovery_tally_batch(
                generated=0,
                evaluated_rows=[],
                behaviorally_unique=0,
                book_members=len(selected),
            )
            # Vault pipeline (unique book fingerprints) — not the same as preliminary eligible
            vault_rep = readiness.get("vault")
            vault_check = next(
                (
                    c
                    for c in (readiness.get("checks") or [])
                    if c.get("name") == "vault_non_negative_cagr"
                ),
                None,
            )
            tested = vault_rep is not None and not readiness.get("skipped_resim")
            # If skipped_resim but vault already in cached readiness, count as tested once via fp
            if readiness.get("skipped_resim") and vault_rep is not None:
                tested = True
            passed: bool | None = None
            if tested:
                if vault_check is not None:
                    passed = bool(vault_check.get("ok"))
                elif isinstance(vault_rep, dict) and vault_rep.get("ok") is False:
                    passed = False
                elif isinstance(vault_rep, dict):
                    vm = vault_rep.get("book_metrics") or {}
                    passed = float(vm.get("cagr") or 0.0) >= 0.0
            MONITOR.record_vault_pipeline(
                fingerprint=str(readiness.get("fingerprint") or ""),
                eligible=bool(readiness.get("ready") or (readiness.get("promotion") or {}).get("allow_freeze_paper")),
                tested=bool(tested and readiness.get("ready")),
                passed=passed if (tested and readiness.get("ready")) else None,
                skipped=bool(readiness.get("skipped_resim")),
            )

            report = finalize_report(
                top_rows,
                [
                    {
                        "candidate": e["candidate"],
                        "fitness": e["fitness"],
                        "full_metrics": e.get("full_metrics", {}),
                        "sector": e.get("sector") or ((e.get("candidate") or {}).get("symbols") or [None])[0],
                    }
                    for e in selected
                    if e.get("candidate")
                ],
                portfolio_metrics,
                target_cagr=float(discovery_cfg.get("target_cagr", 0.30)),
                prop_rules=prop,
                top_n=int(discovery_cfg.get("top_n_report", 20)),
            )
            report["summary"]["mode"] = "continuous"
            report["summary"]["generation"] = generation
            report["summary"]["total_evaluated"] = total_evaluated
            report["summary"]["top50_count"] = len(top_rows)
            report["summary"]["sector_count"] = len(sector_map)
            report["summary"]["portfolio_mode"] = (portfolio_metrics or {}).get("mode") or portfolio_mode
            report["summary"]["portfolio_members"] = len(selected)
            report["summary"]["book_score"] = (portfolio_metrics or {}).get("book_score")
            report["summary"]["swaps_accepted"] = (portfolio_metrics or {}).get("swaps_accepted")
            report["summary"]["allocation"] = (portfolio_metrics or {}).get("allocation")
            report["readiness"] = readiness
            report["summary"]["readiness_level"] = readiness.get("level")
            report["summary"]["ready_for_paper"] = bool(
                (readiness.get("promotion") or {}).get("allow_freeze_paper")
            )
            from trading_ecosystem.execution.costs import load_execution_costs

            cost_info = load_execution_costs(
                profile=discovery_cfg.get("cost_profile"),
                universe_id=uni["id"],
            ).as_dict()
            report["summary"]["cost_profile"] = cost_info
            report["summary"]["universe"] = uni["id"]
            if portfolio_metrics is not None:
                portfolio_metrics = dict(portfolio_metrics)
                portfolio_metrics["cost_profile"] = cost_info.get("profile")
                portfolio_metrics["taker_fee_bps"] = cost_info.get("taker_fee_bps")
                portfolio_metrics["slippage_bps"] = cost_info.get("slippage_bps")
            be = (book_payload or {}).get("best_ever") or {}
            if be.get("metrics"):
                report["summary"]["best_book_cagr"] = (be.get("metrics") or {}).get("cagr")
                report["summary"]["best_book_max_dd"] = (be.get("metrics") or {}).get("max_drawdown")
                report["summary"]["best_book_mar"] = (be.get("metrics") or {}).get("mar")
            report["top50"] = top50_public(top_rows)
            report["best3_by_sector"] = sector_best_public(sector_map)
            for item in report["top_candidates"]:
                item.pop("full_equity_curve", None)
            write_report(report, out_report)
            write_strategies_yaml(
                candidates_to_strategies_yaml([{"candidate": e["candidate"]} for e in selected]),
                export,
            )

            best_cagr = (best_ever.get("fitness") or {}).get("oos_cagr") if best_ever else None
            book_cagr = (portfolio_metrics or {}).get("cagr")
            book_dd = (portfolio_metrics or {}).get("max_drawdown")
            MONITOR.push_alert(
                "info",
                (
                    f"Gen {generation} done | universe={uni['id']} costs={cost_info.get('profile')} "
                    f"slip={cost_info.get('slippage_bps')}bps taker={cost_info.get('taker_fee_bps')}bps | "
                    f"book_cagr={book_cagr} book_dd={book_dd} "
                    f"swaps={(portfolio_metrics or {}).get('swaps_accepted')} "
                    f"total_eval={total_evaluated} best_indiv_oos={best_cagr} "
                    f"readiness={readiness.get('level')}"
                ),
            )
            MONITOR.discovery_patch(
                status="running",
                mode="continuous",
                generation=generation,
                total_evaluated=total_evaluated,
                last_summary=report["summary"],
                top50=top50_public(top_rows),
                best3_by_sector=sector_best_public(sector_map),
                portfolio_metrics=portfolio_metrics or {},
                readiness=readiness,
                selected_portfolio=[
                    {
                        "strategy_id": (e.get("candidate") or {}).get("strategy_id"),
                        "class_name": (e.get("candidate") or {}).get("class_name"),
                        "symbols": (e.get("candidate") or {}).get("symbols") or [],
                        "sector": e.get("sector")
                        or ((e.get("candidate") or {}).get("symbols") or [None])[0],
                        "oos_cagr": (e.get("fitness") or {}).get("oos_cagr"),
                        "oos_max_dd": (e.get("fitness") or {}).get("oos_max_dd"),
                        "passed": (e.get("fitness") or {}).get("passed"),
                        "score": (e.get("fitness") or {}).get("score"),
                    }
                    for e in selected
                    if e.get("candidate")
                ],
                message=f"Generation {generation} complete — sleeping {pause}s then evolving again",
                universe=uni["id"],
                universe_label=uni.get("label_en") or uni["id"],
                cost_profile=cost_info,
                best_ever={
                    "strategy_id": (best_ever or {}).get("candidate", {}).get("strategy_id"),
                    "class_name": (best_ever or {}).get("candidate", {}).get("class_name"),
                    "symbols": (best_ever or {}).get("candidate", {}).get("symbols"),
                    "params": (best_ever or {}).get("candidate", {}).get("params"),
                    "oos_cagr": best_cagr,
                    "score": (best_ever or {}).get("fitness", {}).get("score"),
                    "passed": (best_ever or {}).get("fitness", {}).get("passed"),
                    "source": (best_ever or {}).get("candidate", {}).get("source"),
                },
            )

            logger.info(
                "Generation %s complete total_evaluated=%s best_oos_cagr=%s",
                generation,
                total_evaluated,
                best_cagr,
            )
            if max_generations is not None and generation >= max_generations:
                break
            # Sleep in small slices so stop is responsive
            slept = 0.0
            while slept < pause and not _STOP and not _stop_flag_path().exists():
                time.sleep(min(0.5, pause - slept))
                slept += 0.5

    finally:
        final_board = list(load_top50(leaderboard_path(uni["id"])).get("entries") or [])
        summary = {
            "mode": "continuous",
            "universe": uni["id"],
            "generation": generation,
            "total_evaluated": total_evaluated,
            "top50_count": len(final_board),
            "best_strategy_id": (best_ever or {}).get("candidate", {}).get("strategy_id"),
            "best_oos_cagr": (best_ever or {}).get("fitness", {}).get("oos_cagr") if best_ever else None,
            "target_reached": bool(
                best_ever
                and (best_ever.get("fitness") or {}).get("oos_cagr", 0)
                >= float(discovery_cfg.get("target_cagr", 0.30))
            ),
        }
        MONITOR.discovery_patch(
            top50=top50_public(final_board),
            best3_by_sector=sector_best_public(
                load_sector_best(sector_best_path(uni["id"])).get("sectors") or {}
            ),
            universe=uni["id"],
            universe_label=uni.get("label_en") or uni["id"],
        )
        MONITOR.discovery_finish(summary)
        MONITOR.push_alert("info", f"Continuous discovery stopped: {summary}")

    return {
        "summary": summary,
        "top50": top50_public(final_board),
    }
