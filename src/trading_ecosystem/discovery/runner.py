from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml
from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.data_pipeline.provider import YFinanceProvider
from trading_ecosystem.data_pipeline.store import ParquetBarStore
from trading_ecosystem.discovery.anomaly import scan_anomalies
from trading_ecosystem.discovery.fitness import evaluate_folds
from trading_ecosystem.discovery.leaderboard import TOP50_SIZE, top50_public, update_top50
from trading_ecosystem.discovery.sector_best import (
    DEFAULT_BEST_PER_SECTOR,
    sector_best_public,
    update_sector_best,
)
from trading_ecosystem.discovery.aggregate_book import improve_aggregate_book
from trading_ecosystem.discovery.report import (
    candidates_to_strategies_yaml,
    finalize_report,
    write_report,
    write_strategies_yaml,
)
from trading_ecosystem.discovery.search_space import Candidate, expand_family_candidates
from trading_ecosystem.discovery.walk_forward import build_folds, ensure_aware, filter_panel
from trading_ecosystem.execution.costs import ExecutionCosts, load_execution_costs
from trading_ecosystem.monitoring.state_store import MONITOR
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.base import Strategy
from trading_ecosystem.strategies.registry import build_strategy

logger = logging.getLogger(__name__)


def _costs_from_cfg(discovery_cfg: dict[str, Any]) -> ExecutionCosts:
    return load_execution_costs(
        profile=discovery_cfg.get("cost_profile"),
        universe_id=discovery_cfg.get("active_universe"),
    )


def _parse_end(end: str | None) -> str | None:
    return end


def _warmup_strategy(
    strategy: Strategy,
    full_panel: dict[str, list[Bar]],
    fold_start: datetime,
    warmup_bars: int = 120,
) -> None:
    """Seed indicator history with bars strictly before fold_start (no trading)."""
    start = ensure_aware(fold_start)
    prior: list[Bar] = []
    for sym in strategy.symbols:
        for b in full_panel.get(sym, []):
            if ensure_aware(b.ts) < start:
                prior.append(b)
    prior.sort(key=lambda b: (b.ts, b.symbol))
    for b in prior[-warmup_bars:]:
        strategy._history.setdefault(b.symbol, []).append(b)


def _load_panel(
    symbols: list[str],
    *,
    start: str,
    end: str | None,
    timeframe: str,
    ingest_missing: bool = True,
) -> dict[str, list[Bar]]:
    store = ParquetBarStore()
    panel = store.load_panel(symbols, timeframe=timeframe, start=start, end=end)
    missing = [s for s, bars in panel.items() if not bars]
    if missing and ingest_missing:
        logger.info("Fetching missing symbols for discovery: %s", missing)
        bars = YFinanceProvider().fetch_bars(missing, start=start, end=end, timeframe=timeframe)
        store.write_bars(bars)
        panel = store.load_panel(symbols, timeframe=timeframe, start=start, end=end)
    return panel


def _candidate_dict(c: Candidate) -> dict[str, Any]:
    return {
        "strategy_id": c.strategy_id,
        "class_name": c.class_name,
        "symbols": list(c.symbols),
        "params": dict(c.params),
        "source": c.source,
        "meta": dict(c.meta),
    }


def evaluate_candidate(
    candidate: Candidate,
    panel: dict[str, list[Bar]],
    *,
    discovery_cfg: dict[str, Any],
    prop: dict[str, Any],
    starting_equity: float,
) -> dict[str, Any]:
    wf = discovery_cfg.get("walk_forward") or {}
    needed = {s: panel.get(s, []) for s in candidate.symbols}
    folds = build_folds(
        needed,
        train_ratio=float(wf.get("train_ratio", 0.60)),
        validate_ratio=float(wf.get("validate_ratio", 0.20)),
        holdout_ratio=float(wf.get("holdout_ratio", 0.20)),
        use_yearly_rolls=bool(wf.get("use_yearly_rolls", True)),
        min_bars_per_fold=int(wf.get("min_bars_per_fold", 60)),
        vault_ratio=float(wf.get("vault_ratio", 0.20)),
        seal_vault=bool(wf.get("seal_vault", True)),
    )
    fold_results = []
    costs = _costs_from_cfg(discovery_cfg)
    for fold in folds:
        # Vault is promotion-only — do not simulate during discovery selection.
        if fold.kind in {"vault", "vault_year"}:
            continue
        # Skip pure train for fitness OOS scoring, but keep for transparency
        sub = filter_panel(needed, fold.start, fold.end)
        if not any(sub.values()):
            continue
        strategy = build_strategy(
            candidate.class_name,
            candidate.strategy_id,
            candidate.symbols,
            candidate.params,
        )
        _warmup_strategy(strategy, needed, fold.start)
        engine = BacktestEngine(
            strategies=[strategy],
            risk_manager=MasterRiskManager(),
            starting_equity=starting_equity,
            costs=costs,
        )
        result = engine.run(sub)
        fold_results.append((fold.name, fold.kind, result))

    fitness = evaluate_folds(
        fold_results,
        max_total_dd_pct=float(prop["max_total_drawdown_pct"]),
        max_daily_dd_pct=float(prop["max_daily_drawdown_pct"]),
        target_cagr=float(discovery_cfg.get("target_cagr", 0.30)),
        min_trades=int(discovery_cfg.get("min_trades", 8)),
        min_positive_oos_folds=int(discovery_cfg.get("min_positive_oos_folds", 1)),
    )

    try:
        from trading_ecosystem.discovery.trial_ledger import append_trial, code_fingerprint
        from trading_ecosystem.discovery.universe import get_active_universe

        append_trial(
            {
                "strategy_id": candidate.strategy_id,
                "class_name": candidate.class_name,
                "symbols": list(candidate.symbols),
                "params": dict(candidate.params),
                "code_hash": code_fingerprint(candidate.class_name, candidate.params),
                "seed": (candidate.meta or {}).get("seed") if getattr(candidate, "meta", None) else None,
                "fold_ids": [n for n, k, _ in fold_results],
                "oos_cagr": fitness.oos_cagr,
                "oos_max_dd": fitness.oos_max_dd,
                "passed": fitness.passed,
                "score": fitness.score,
                "competitors_in_batch": (candidate.meta or {}).get("competitors_in_batch")
                if getattr(candidate, "meta", None)
                else None,
                "source": getattr(candidate, "source", None),
            },
            universe_id=str(discovery_cfg.get("active_universe") or get_active_universe()),
        )
    except Exception:  # noqa: BLE001
        pass

    # Full-sample equity for correlation / portfolio (still under MRM)
    strategy = build_strategy(
        candidate.class_name,
        candidate.strategy_id,
        candidate.symbols,
        candidate.params,
    )
    needed_full = {s: panel.get(s, []) for s in candidate.symbols}
    full = BacktestEngine(
        strategies=[strategy],
        risk_manager=MasterRiskManager(),
        starting_equity=starting_equity,
        costs=costs,
    ).run(needed_full)

    return {
        "candidate": _candidate_dict(candidate),
        "fitness": fitness.as_dict(),
        "full_metrics": full.metrics.as_dict(),
        "full_equity_curve": full.equity_curve,
        "risk_rejects_full": full.risk_rejects,
    }


def run_discovery(
    *,
    start: str = "2018-01-01",
    end: str | None = None,
    phase: str = "all",
    export_path: str | Path | None = None,
    report_path: str | Path | None = None,
    max_candidates: int | None = None,
    universe: str | None = None,
) -> dict[str, Any]:
    from trading_ecosystem.discovery.universe import (
        apply_universe_to_discovery_cfg,
        processed_artifact,
        set_active_universe,
        strategies_export_path,
    )

    if universe:
        set_active_universe(universe)

    discovery_cfg = load_yaml(CONFIG_DIR / "discovery.yaml")
    discovery_cfg, uni = apply_universe_to_discovery_cfg(discovery_cfg, universe_id=universe)
    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    timeframe = str(uni.get("timeframe") or "1d")
    starting_equity = float(prop.get("starting_equity", 100_000))

    symbols = list(uni["symbols"])
    panel = _load_panel(symbols, start=start, end=_parse_end(end), timeframe=timeframe)

    candidates: list[Candidate] = []
    phase_l = phase.lower()
    if phase_l in {"families", "all"}:
        fam = expand_family_candidates(
            discovery_cfg,
            seed=int(discovery_cfg.get("random_seed", 42)),
        )
        candidates.extend(fam)
        logger.info("Family candidates: %s", len(fam))
    if phase_l in {"anomaly", "all"}:
        anom = scan_anomalies(
            panel,
            top_k=int(discovery_cfg.get("anomaly_top_patterns", 40)),
        )
        candidates.extend(anom)
        logger.info("Anomaly candidates: %s", len(anom))

    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    MONITOR.discovery_start(
        phase=phase_l,
        total=len(candidates),
        message=f"Starting discovery phase={phase_l} candidates={len(candidates)}",
    )
    MONITOR.push_alert("info", f"Discovery started phase={phase_l} total={len(candidates)}")

    evaluated: list[dict[str, Any]] = []
    try:
        for i, cand in enumerate(candidates, 1):
            logger.info(
                "Evaluating %s/%s %s %s %s",
                i,
                len(candidates),
                cand.source,
                cand.class_name,
                cand.strategy_id,
            )
            MONITOR.discovery_progress(
                current=i - 1,
                total=len(candidates),
                strategy_id=cand.strategy_id,
                class_name=cand.class_name,
                symbols=cand.symbols,
                source=cand.source,
                message=f"Trying {cand.class_name} on {','.join(cand.symbols)} ({i}/{len(candidates)})",
            )
            try:
                row = evaluate_candidate(
                    cand,
                    panel,
                    discovery_cfg=discovery_cfg,
                    prop=prop,
                    starting_equity=starting_equity,
                )
            except Exception as exc:  # noqa: BLE001 — keep discovery loop alive
                logger.exception("Candidate failed: %s", cand.strategy_id)
                row = {
                    "candidate": _candidate_dict(cand),
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
                    "risk_rejects_full": 0,
                }
            evaluated.append(row)
            fit = row.get("fitness") or {}
            MONITOR.discovery_progress(
                current=i,
                total=len(candidates),
                strategy_id=cand.strategy_id,
                class_name=cand.class_name,
                symbols=cand.symbols,
                source=cand.source,
                passed=bool(fit.get("passed")),
                oos_cagr=fit.get("oos_cagr"),
            )
    except Exception as exc:  # noqa: BLE001
        MONITOR.discovery_finish(error=f"Discovery aborted: {exc}")
        raise

    MONITOR.discovery_progress(
        current=len(candidates),
        total=len(candidates),
        strategy_id="portfolio_select",
        class_name="PortfolioSelect",
        symbols=[],
        source=phase_l,
        message="Selecting portfolio and writing report…",
    )

    board_size = int(discovery_cfg.get("leaderboard_size", TOP50_SIZE))
    per_sector = int(discovery_cfg.get("best_per_sector", DEFAULT_BEST_PER_SECTOR))
    top_rows = update_top50(evaluated, size=board_size, universe_id=uni["id"])
    sector_map = update_sector_best(
        evaluated, best_per_sector=per_sector, universe_id=uni["id"]
    )

    prop_dd = float(prop.get("max_total_drawdown_pct", 10.0)) / 100.0
    from trading_ecosystem.discovery.parallel import default_workers

    portfolio_mode = str(
        uni.get("portfolio_mode")
        or discovery_cfg.get("portfolio_mode")
        or "aggregate_book"
    ).strip().lower()
    if portfolio_mode in {"sector_champions", "champions", "per_coin", "sparse_heroes", "heroes"}:
        from trading_ecosystem.discovery.aggregate_book import build_champions_book

        hero_n = uni.get("max_heroes")
        if portfolio_mode in {"sparse_heroes", "heroes"} and hero_n is None:
            hero_n = 3
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
        )
    else:
        selected, portfolio_metrics, book_payload = improve_aggregate_book(
            sector_map=sector_map,
            evaluated=evaluated,
            panel=panel,
            starting_equity=starting_equity,
            prop_max_total_dd=prop_dd,
            require_passed=bool(discovery_cfg.get("portfolio_require_passed", True)),
            max_swap_trials=int(discovery_cfg.get("portfolio_max_swap_trials", 24)),
            workers=default_workers((discovery_cfg.get("continuous") or {}).get("workers")),
            universe_id=uni["id"],
        )

    from trading_ecosystem.discovery.stability import evaluate_portfolio_readiness

    readiness = evaluate_portfolio_readiness(
        selected,
        panel,
        universe_id=uni["id"],
        starting_equity=starting_equity,
        portfolio_metrics=portfolio_metrics,
        prop_rules=prop,
    )

    report = finalize_report(
        evaluated,
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
    be = (book_payload or {}).get("best_ever") or {}
    if be.get("metrics"):
        report["summary"]["best_book_cagr"] = (be.get("metrics") or {}).get("cagr")
        report["summary"]["best_book_max_dd"] = (be.get("metrics") or {}).get("max_drawdown")
        report["summary"]["best_book_mar"] = (be.get("metrics") or {}).get("mar")
    report["top50"] = top50_public(top_rows)
    report["best3_by_sector"] = sector_best_public(sector_map)

    # Strip bulky equity curves from persisted top candidates
    for item in report["top_candidates"]:
        item.pop("full_equity_curve", None)

    out_report = (
        Path(report_path)
        if report_path
        else processed_artifact("discovery_report", universe_id=uni["id"])
    )
    write_report(report, out_report)

    export = Path(export_path) if export_path else strategies_export_path(uni["id"])
    payload = candidates_to_strategies_yaml(
        [{"candidate": e["candidate"]} for e in selected]
    )
    write_strategies_yaml(payload, export)

    MONITOR.discovery_patch(
        top50=top50_public(top_rows),
        best3_by_sector=sector_best_public(sector_map),
        readiness=readiness,
        universe=uni["id"],
    )
    MONITOR.discovery_finish(report["summary"])
    MONITOR.push_alert(
        "info",
        (
            f"Discovery done evaluated={report['summary']['candidates_evaluated']} "
            f"passed={report['summary']['passed_prop_and_stability']} "
            f"target_hit={report['summary']['target_reached']} "
            f"best_passed_cagr={report['summary']['best_passed_oos_cagr']}"
        ),
    )
    logger.info("Wrote report %s and strategies %s", out_report, export)
    return report
