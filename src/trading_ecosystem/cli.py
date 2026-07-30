from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.config import CONFIG_DIR, load_yaml
from trading_ecosystem.common.logging_setup import setup_logging
from trading_ecosystem.data_pipeline.provider import YFinanceProvider
from trading_ecosystem.data_pipeline.store import ParquetBarStore
from trading_ecosystem.execution.alpaca_gateway import AlpacaPaperGateway
from trading_ecosystem.execution.broker import PaperSimBroker
from trading_ecosystem.execution.runtime import PaperRuntime
from trading_ecosystem.monitoring.alerts import send_telegram
from trading_ecosystem.monitoring.state_store import MONITOR
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategies

logger = logging.getLogger(__name__)


def _parse_common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trading Ecosystem CLI")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    return p


def ingest(argv: list[str] | None = None) -> None:
    load_dotenv()
    setup_logging()
    parser = _parse_common()
    parser.add_argument(
        "--universe",
        default=None,
        help="Universe id from configs/universes (etf, crypto). Default: all listed universes.",
    )
    args = parser.parse_args(argv)
    from trading_ecosystem.discovery.universe import list_universe_ids, load_universe

    provider = YFinanceProvider()
    store = ParquetBarStore()
    if args.universe:
        ids = [args.universe.strip().lower()]
    else:
        ids = list_universe_ids()
    all_counts: dict[str, Any] = {}
    for uid in ids:
        uni = load_universe(uid)
        symbols = list(uni["symbols"])
        timeframe = str(uni.get("timeframe") or "1d")
        logger.info(
            "Fetching universe=%s (%s symbols) via yfinance (%s -> %s)",
            uid,
            len(symbols),
            args.start,
            args.end,
        )
        bars = provider.fetch_bars(symbols, start=args.start, end=args.end, timeframe=timeframe)
        counts = store.write_bars(bars)
        all_counts[uid] = counts
        logger.info("Wrote parquet bars for %s: %s", uid, counts)
    print(json.dumps(all_counts, indent=2))


def backtest(argv: list[str] | None = None) -> None:
    load_dotenv()
    setup_logging()
    parser = _parse_common()
    parser.add_argument("--ingest-first", action="store_true")
    args = parser.parse_args(argv)

    if args.ingest_first:
        ingest(["--start", args.start] + (["--end", args.end] if args.end else []))

    symbols_cfg = load_yaml(CONFIG_DIR / "symbols.yaml")
    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    store = ParquetBarStore()
    strategies = build_strategies()
    needed = sorted({s for st in strategies for s in st.symbols})
    panel = store.load_panel(needed, timeframe=str(symbols_cfg.get("timeframe", "1d")), start=args.start, end=args.end)
    missing = [s for s, bars in panel.items() if not bars]
    if missing:
        logger.info("Missing local data for %s — fetching", missing)
        bars = YFinanceProvider().fetch_bars(missing, start=args.start, end=args.end)
        store.write_bars(bars)
        panel = store.load_panel(needed, start=args.start, end=args.end)

    from trading_ecosystem.execution.costs import load_execution_costs

    costs = load_execution_costs(universe_id="etf")
    engine = BacktestEngine(
        strategies=strategies,
        risk_manager=MasterRiskManager(),
        starting_equity=float(prop.get("starting_equity", 100_000)),
        costs=costs,
    )
    result = engine.run(panel)
    metrics = result.metrics.as_dict()
    out = {
        "metrics": metrics,
        "fills": len(result.fills),
        "risk_rejects": result.risk_rejects,
    }
    print(json.dumps(out, indent=2))

    MONITOR.update(
        equity=result.equity_curve[-1][1] if result.equity_curve else None,
        daily_dd_pct=0.0,
        total_dd_pct=metrics["max_drawdown"] * 100,
        agents={s.strategy_id: {"symbols": s.symbols} for s in strategies},
        last_bar_ts=str(result.equity_curve[-1][0]) if result.equity_curve else None,
    )
    MONITOR.push_alert("info", f"Backtest complete CAGR={metrics['cagr']:.2%} MaxDD={metrics['max_drawdown']:.2%}")


def discover(argv: list[str] | None = None) -> None:
    import multiprocessing as mp

    # Required on Windows when ProcessPoolExecutor spawns workers.
    mp.freeze_support()
    load_dotenv()
    setup_logging()
    parser = _parse_common()
    parser.add_argument(
        "--phase",
        choices=["families", "anomaly", "all"],
        default="all",
        help="Discovery phase to run (ignored in --continuous)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="24/7 evolve loop: invent, mutate, crossover elites under Prop",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="With --continuous: stop after N generations (default: run forever)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="With --continuous: candidates per generation",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Path for strategies.discovered.yaml (default: configs/strategies.discovered.yaml)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for discovery_report.json (default: data/processed/discovery_report.json)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap for faster smoke runs (batch mode)",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="Discovery universe: etf | etf_momentum | crypto (configs/universes/). Default: active",
    )
    args = parser.parse_args(argv)

    if args.continuous:
        from trading_ecosystem.discovery.continuous import run_continuous_discovery

        result = run_continuous_discovery(
            start=args.start,
            end=args.end,
            batch_size=args.batch_size,
            max_generations=args.generations,
            export_path=args.export,
            report_path=args.report,
            universe=args.universe,
        )
        print(json.dumps(result["summary"], indent=2, default=str))
        return

    from trading_ecosystem.discovery.runner import run_discovery

    report = run_discovery(
        start=args.start,
        end=args.end,
        phase=args.phase,
        export_path=args.export,
        report_path=args.report,
        max_candidates=args.max_candidates,
        universe=args.universe,
    )
    print(json.dumps(report["summary"], indent=2))


def paper(argv: list[str] | None = None) -> None:
    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Paper-trade the locked aggregate book on fake money (sim or Alpaca paper)"
    )
    parser.add_argument("--broker", choices=["sim", "alpaca", "shadow"], default="sim")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Strategies YAML (default: configs/strategies.paper.yaml after freeze)",
    )
    parser.add_argument(
        "--no-freeze-book",
        action="store_true",
        help="Do not re-freeze; use existing strategies.paper.yaml / --strategies",
    )
    parser.add_argument(
        "--replay-days",
        type=int,
        default=60,
        help="Replay last N daily bars before live cycles (0=skip)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running and pick up new bars (discovery can stay running)",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=3600.0,
        help="Sleep between live cycles when --loop is set",
    )
    parser.add_argument(
        "--live-max",
        type=int,
        default=None,
        help="Max strategies in live sleeve (default: discovery.paper_live_size or Prop max positions)",
    )
    parser.add_argument(
        "--full-book",
        action="store_true",
        help="Trade the full frozen book instead of the Prop-sized live sleeve",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Refuse freeze/start unless stability gate level is ready_for_paper or ready_for_live",
    )
    parser.add_argument(
        "--reset-alpaca",
        action="store_true",
        help="Flatten all Alpaca paper positions before starting (paper money only)",
    )
    parser.add_argument(
        "--skip-latest",
        action="store_true",
        help="Do not trade the current last daily bar; wait only for newer bars",
    )
    parser.add_argument(
        "--order-stagger-sec",
        type=float,
        default=1.0,
        help="Pause between Alpaca fills to avoid burst orders (default 1s)",
    )
    args = parser.parse_args(argv)

    from trading_ecosystem.execution.paper_loop import run_paper_session

    summary = run_paper_session(
        strategies_path=args.strategies,
        broker=args.broker,
        replay_days=int(args.replay_days),
        loop=bool(args.loop),
        interval_sec=float(args.interval_sec),
        freeze=not bool(args.no_freeze_book),
        use_live_sleeve=not bool(args.full_book),
        live_max=args.live_max,
        require_ready=bool(args.require_ready),
        reset_alpaca=bool(args.reset_alpaca),
        trade_latest=not bool(args.skip_latest),
        order_stagger_sec=float(args.order_stagger_sec),
    )
    print(json.dumps(summary, indent=2, default=str))


def monitor(argv: list[str] | None = None) -> None:
    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser(description="Start monitoring dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    import uvicorn

    from trading_ecosystem.monitoring.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    backtest()
