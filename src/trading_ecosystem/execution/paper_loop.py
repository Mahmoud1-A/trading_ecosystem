from __future__ import annotations

import atexit
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml
from trading_ecosystem.common.contracts import Bar, Position
from trading_ecosystem.data_pipeline.store import ParquetBarStore
from trading_ecosystem.discovery.aggregate_book import load_portfolio_book
from trading_ecosystem.discovery.universe import DEFAULT_UNIVERSE, paper_live_path
from trading_ecosystem.discovery.report import candidates_to_strategies_yaml, write_strategies_yaml
from trading_ecosystem.execution.alpaca_gateway import AlpacaPaperGateway
from trading_ecosystem.execution.broker import PaperSimBroker, ShadowBroker
from trading_ecosystem.execution.runtime import PaperRuntime
from trading_ecosystem.monitoring.state_store import MONITOR
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategies

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def acquire_paper_lock(broker: str) -> Path:
    """Ensure only one te-paper process per broker (prevents Alpaca order storms)."""
    path = data_dir() / "processed" / f"te-paper-{broker}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    my_pid = os.getpid()
    if path.exists():
        try:
            old = int(path.read_text(encoding="utf-8").strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            old = 0
        if old and old != my_pid and _pid_alive(old):
            raise RuntimeError(
                f"Another te-paper --broker {broker} is already running (pid={old}). "
                "Stop it before starting a new session."
            )
    path.write_text(f"{my_pid}\n{datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")

    def _release() -> None:
        try:
            if path.exists():
                cur = int(path.read_text(encoding="utf-8").strip().splitlines()[0])
                if cur == my_pid:
                    path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_release)
    return path


def sync_book_from_alpaca(runtime: PaperRuntime) -> dict[str, Any]:
    """
    Hydrate local book from Alpaca account + positions.
    Maps broker symbols onto sleeve strategy_ids (one strategy per symbol).
    """
    broker = runtime.broker
    if not hasattr(broker, "get_account") or not getattr(broker, "configured", False):
        return {}
    acct = broker.get_account()  # type: ignore[attr-defined]
    positions = broker.get_positions()  # type: ignore[attr-defined]
    symbol_to_sid: dict[str, str] = {}
    for st in runtime.strategies:
        for sym in st.symbols:
            symbol_to_sid.setdefault(sym, st.strategy_id)

    eq = float(acct.get("equity") or runtime.book.state.equity)
    cash = float(acct.get("cash") if acct.get("cash") is not None else runtime.book.state.cash)
    bp = acct.get("buying_power")
    runtime.book.state.equity = eq
    runtime.book.state.cash = cash
    runtime.book.state.buying_power = float(bp) if bp is not None else None
    runtime.book.state.peak_equity = max(runtime.book.state.peak_equity, eq)
    runtime.book.state.positions.clear()
    for p in positions:
        sid = symbol_to_sid.get(p.symbol, p.strategy_id or "broker")
        key = runtime.book.state.position_key(sid, p.symbol)
        runtime.book.state.positions[key] = Position(
            symbol=p.symbol,
            strategy_id=sid,
            qty=float(p.qty),
            avg_price=float(p.avg_price),
            unrealized_pnl=float(p.unrealized_pnl or 0.0),
        )
    logger.info(
        "Alpaca hydrate equity=%.2f cash=%.2f bp=%s positions=%s gross=%.2f",
        eq,
        cash,
        bp,
        len(runtime.book.state.positions),
        runtime.book.state.gross_exposure(),
    )
    return {
        "equity": eq,
        "cash": cash,
        "buying_power": bp,
        "positions": len(runtime.book.state.positions),
    }


def flatten_alpaca_paper(*, reason: str = "manual") -> dict[str, Any]:
    """Close all Alpaca paper positions (paper money reset)."""
    gw = AlpacaPaperGateway()
    if not gw.configured:
        raise RuntimeError("Alpaca not configured")
    before = gw.sync()
    results = gw.flatten()
    time.sleep(2)
    after = gw.sync()
    logger.info("Flattened Alpaca paper (%s): orders=%s", reason, len(results))
    return {"before": before, "orders": len(results), "after": after}


def paper_last_seen_path(broker: str = "sim") -> Path:
    return data_dir() / "processed" / f"paper_last_seen_{broker}.json"


def load_last_seen(broker: str) -> dict[str, str]:
    path = paper_last_seen_path(broker)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (raw or {}).items()}
    except Exception:  # noqa: BLE001
        return {}


def save_last_seen(broker: str, last_seen: dict[str, str]) -> None:
    path = paper_last_seen_path(broker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(last_seen, indent=2, sort_keys=True), encoding="utf-8")


def paper_strategies_path() -> Path:
    return CONFIG_DIR / "strategies.paper.yaml"


def paper_live_strategies_path() -> Path:
    """Executable live sleeve (subset) — discovery book stays full elsewhere."""
    return CONFIG_DIR / "strategies.paper.live.yaml"


def paper_state_path(broker: str = "sim") -> Path:
    if broker == "alpaca":
        name = "paper_state_alpaca.json"
    elif broker == "shadow":
        name = "paper_state_shadow.json"
    else:
        name = "paper_state_sim.json"
    return data_dir() / "processed" / name


def annualize_return(start_equity: float, equity: float, days: float) -> float | None:
    if start_equity <= 0 or equity <= 0 or days < 1:
        return None
    return float((equity / start_equity) ** (365.0 / days) - 1.0)


def load_paper_states() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for broker in ("sim", "alpaca"):
        p = paper_state_path(broker)
        if p.exists():
            try:
                out[broker] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
    # Legacy single file
    legacy = data_dir() / "processed" / "paper_state.json"
    if "sim" not in out and legacy.exists():
        try:
            out["sim"] = json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return out


def select_live_sleeve(
    members: list[dict[str, Any]],
    *,
    max_strategies: int,
) -> list[dict[str, Any]]:
    """
    Pick an executable sleeve sized to Prop concurrent-position capacity.
    Prefer highest OOS CAGR, one strategy per sector/symbol.
    """
    ranked = sorted(
        members,
        key=lambda m: (
            float((m.get("fitness") or {}).get("oos_cagr") or -1e9),
            float((m.get("fitness") or {}).get("score") or -1e9),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_sectors: set[str] = set()
    for row in ranked:
        if len(selected) >= max_strategies:
            break
        c = row.get("candidate") or {}
        sector = str(row.get("sector") or (c.get("symbols") or ["_NONE_"])[0])
        if sector in seen_sectors:
            continue
        seen_sectors.add(sector)
        selected.append(row)
    return selected


def freeze_book_to_paper_yaml(
    path: Path | None = None,
    *,
    live_max: int | None = None,
    universe: str | None = None,
    require_ready: bool = False,
) -> Path:
    """
    Snapshot full locked book + a smaller live sleeve for execution.
    Default trade universe is ETF so crypto discovery does not overwrite the live sleeve.
    Set require_ready=True to enforce the stability gate (ready_for_paper / ready_for_live).
    """
    trade_uni = (universe or DEFAULT_UNIVERSE).strip().lower()
    out = path or paper_strategies_path()
    from trading_ecosystem.discovery.aggregate_book import portfolio_book_path
    from trading_ecosystem.discovery.stability import load_readiness

    readiness = load_readiness(trade_uni)
    if require_ready and not (readiness.get("promotion") or {}).get("allow_freeze_paper"):
        raise RuntimeError(
            f"Universe={trade_uni} not ready for paper freeze "
            f"(level={readiness.get('level')}: {readiness.get('message')})"
        )

    book = load_portfolio_book(portfolio_book_path(trade_uni))
    members = list(book.get("members") or [])
    if not members:
        discovered = CONFIG_DIR / "strategies.discovered.yaml"
        if discovered.exists() and trade_uni == DEFAULT_UNIVERSE and not require_ready:
            text = discovered.read_text(encoding="utf-8")
            out.write_text(text, encoding="utf-8")
            logger.info("Froze paper strategies from %s -> %s", discovered, out)
            return out
        raise RuntimeError(
            f"No locked portfolio book for universe={trade_uni} to freeze"
        )

    payload = candidates_to_strategies_yaml(members)
    write_strategies_yaml(payload, out)

    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    discovery_cfg = load_yaml(CONFIG_DIR / "discovery.yaml")
    max_live = int(
        live_max
        if live_max is not None
        else discovery_cfg.get("paper_live_size")
        or prop.get("max_concurrent_positions")
        or 10
    )
    live_members = select_live_sleeve(members, max_strategies=max_live)
    live_path = paper_live_path(trade_uni) if trade_uni != DEFAULT_UNIVERSE else paper_live_strategies_path()
    write_strategies_yaml(candidates_to_strategies_yaml(live_members), live_path)

    meta = {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(portfolio_book_path(trade_uni).name),
        "universe": trade_uni,
        "members_full": len(members),
        "members_live": len(live_members),
        "live_path": str(live_path),
        "readiness_level": readiness.get("level"),
        "ready_for_paper": bool((readiness.get("promotion") or {}).get("allow_freeze_paper")),
        "require_ready": require_ready,
        "live_ids": [
            (m.get("candidate") or {}).get("strategy_id") for m in live_members
        ],
        "live_sectors": [
            m.get("sector") or ((m.get("candidate") or {}).get("symbols") or [None])[0]
            for m in live_members
        ],
        "book_score": book.get("score"),
        "book_cagr": (book.get("metrics") or {}).get("cagr"),
        "book_max_dd": (book.get("metrics") or {}).get("max_drawdown"),
    }
    meta_path = data_dir() / "processed" / "paper_freeze_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        from trading_ecosystem.monitoring.state_store import MONITOR

        MONITOR.record_paper_promotion()
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        "Froze full book=%s and live sleeve=%s (max=%s) -> %s / %s",
        len(members),
        len(live_members),
        max_live,
        out,
        live_path,
    )
    return out


def _save_paper_state(
    runtime: PaperRuntime,
    *,
    broker_name: str,
    starting_equity: float,
    started_at: datetime,
    days_elapsed: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    acct = runtime.book.state
    equity = float(acct.equity)
    cash = float(acct.cash)
    positions = len(acct.positions)

    # Prefer live Alpaca account numbers when connected.
    if broker_name == "alpaca" and hasattr(runtime.broker, "get_account"):
        try:
            ba = runtime.broker.get_account()  # type: ignore[attr-defined]
            if ba.get("configured") and ba.get("equity") is not None:
                equity = float(ba["equity"])
                cash = float(ba.get("cash") or cash)
                positions = len(runtime.broker.get_positions())  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpaca account refresh failed: %s", exc)

    now = datetime.now(timezone.utc)
    if days_elapsed is None:
        days_elapsed = max(1.0, (now - started_at).total_seconds() / 86400.0)
    ann = annualize_return(starting_equity, equity, float(days_elapsed))

    prev: dict[str, Any] = {}
    p = paper_state_path(broker_name)
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}

    ann_min = prev.get("ann_return_min")
    ann_max = prev.get("ann_return_max")
    if ann is not None:
        ann_min = ann if ann_min is None else min(float(ann_min), ann)
        ann_max = ann if ann_max is None else max(float(ann_max), ann)

    payload = {
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "broker": broker_name,
        "starting_equity": starting_equity,
        "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days_elapsed": round(float(days_elapsed), 2),
        "equity": equity,
        "cash": cash,
        "period_return": (equity / starting_equity - 1.0) if starting_equity else None,
        "ann_return": ann,
        "ann_return_min": ann_min,
        "ann_return_max": ann_max,
        "daily_dd_pct": acct.daily_drawdown_pct(),
        "total_dd_pct": acct.total_drawdown_pct(),
        "halted": acct.halted,
        "daily_halted": acct.daily_halted,
        "positions": positions,
        "strategies": len(runtime.strategies),
    }
    if extra:
        payload.update(extra)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Load disk first so we don't wipe a running discovery process's dashboard fields.
    MONITOR.load()
    paper_map = dict((MONITOR.snapshot().get("paper") or {}))
    paper_map[broker_name] = {
        "equity": equity,
        "cash": cash,
        "ann_return": ann,
        "ann_return_min": ann_min,
        "ann_return_max": ann_max,
        "period_return": payload.get("period_return"),
        "days_elapsed": payload.get("days_elapsed"),
        "positions": positions,
        "updated_at": payload["updated_at"],
    }
    MONITOR.update(
        equity=equity if broker_name == "alpaca" else (MONITOR.snapshot().get("equity") or equity),
        cash=cash if broker_name == "alpaca" else (MONITOR.snapshot().get("cash") or cash),
        daily_dd_pct=acct.daily_drawdown_pct(),
        total_dd_pct=acct.total_drawdown_pct(),
        halted=acct.halted,
        daily_halted=acct.daily_halted,
        agents={s.strategy_id: {"symbols": s.symbols} for s in runtime.strategies},
        last_bar_ts=payload["updated_at"],
        paper=paper_map,
    )
    MONITOR.push_alert(
        "info",
        f"Paper ({broker_name}) equity={equity:,.2f} ann="
        f"{(ann * 100) if ann is not None else float('nan'):.1f}% "
        f"min/max="
        f"{(float(ann_min) * 100) if ann_min is not None else float('nan'):.1f}%/"
        f"{(float(ann_max) * 100) if ann_max is not None else float('nan'):.1f}%",
    )
    return payload


def _unique_day_bars(
    store: ParquetBarStore,
    symbols: list[str],
    *,
    last_n_days: int | None,
) -> list[Bar]:
    by_day: dict[str, list[Bar]] = {}
    for sym in symbols:
        bars = store.load_bars(sym, "1d")
        if last_n_days and last_n_days > 0:
            bars = bars[-last_n_days:]
        for b in bars:
            day = b.ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
            by_day.setdefault(day, []).append(b)
    days = sorted(by_day.keys())
    ordered: list[Bar] = []
    for day in days:
        # Stable symbol order within day
        ordered.extend(sorted(by_day[day], key=lambda x: x.symbol))
    return ordered


def build_paper_runtime(
    *,
    strategies_path: str | Path | None,
    broker_name: str,
    starting_equity: float,
) -> tuple[PaperRuntime, str]:
    strategies = build_strategies(strategies_path) if strategies_path else build_strategies()
    if not strategies:
        raise RuntimeError("No strategies loaded for paper trading")
    risk = MasterRiskManager()
    store = ParquetBarStore()
    if broker_name == "alpaca":
        broker: Any = AlpacaPaperGateway()
        broker.sync()
    elif broker_name == "shadow":
        broker = ShadowBroker(starting_cash=starting_equity)
    else:
        broker = PaperSimBroker(cash=starting_equity)
    runtime = PaperRuntime(
        strategies=strategies,
        risk=risk,
        broker=broker,
        store=store,
        starting_equity=starting_equity,
    )
    return runtime, broker_name


def run_paper_session(
    *,
    strategies_path: str | Path | None = None,
    broker: str = "sim",
    replay_days: int = 60,
    loop: bool = False,
    interval_sec: float = 3600.0,
    freeze: bool = True,
    use_live_sleeve: bool = True,
    live_max: int | None = None,
    require_ready: bool = False,
    reset_alpaca: bool = False,
    trade_latest: bool = True,
    order_stagger_sec: float = 1.0,
) -> dict[str, Any]:
    """
    Paper-trade on sim/Alpaca paper money.
    Discovery keeps the full book; execution uses a Prop-sized live sleeve by default.
    Set require_ready=True to enforce the stability gate before freeze.
    Set reset_alpaca=True to flatten all Alpaca paper positions before starting.
    trade_latest=True processes the newest stored daily bar once (persisted last_seen
    prevents re-spray on restart). Set False to wait only for bars newer than now.
    """
    acquire_paper_lock(broker)
    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    starting_equity = float(prop.get("starting_equity", 100_000))
    started_at = datetime.now(timezone.utc)

    if reset_alpaca and broker == "alpaca":
        flatten_alpaca_paper(reason="session_reset")

    # Always refresh freeze artifacts when requested (full + live sleeve).
    if freeze or not paper_strategies_path().exists():
        freeze_book_to_paper_yaml(live_max=live_max, require_ready=require_ready)

    if strategies_path:
        path = Path(strategies_path)
    elif use_live_sleeve and paper_live_strategies_path().exists():
        path = paper_live_strategies_path()
    else:
        path = paper_strategies_path()

    # Never replay historical bars into Alpaca (would spam real paper orders).
    effective_replay = 0 if broker == "alpaca" else int(replay_days)

    runtime, broker_name = build_paper_runtime(
        strategies_path=path,
        broker_name=broker,
        starting_equity=starting_equity,
    )
    if broker_name == "alpaca" and hasattr(runtime.broker, "configured"):
        if not runtime.broker.configured:  # type: ignore[attr-defined]
            raise RuntimeError("Alpaca client not configured — check ALPACA_* in .env")
        try:
            synced = sync_book_from_alpaca(runtime)
            if synced.get("equity") is not None:
                starting_equity = float(synced["equity"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpaca account sync skipped: %s", exc)

    symbols = sorted({s for st in runtime.strategies for s in st.symbols})
    logger.info(
        "Paper session broker=%s strategies=%s file=%s symbols=%s replay_days=%s loop=%s sleeve=%s",
        broker_name,
        len(runtime.strategies),
        path,
        len(symbols),
        effective_replay,
        loop,
        use_live_sleeve,
    )

    # Avoid double-counting warmed last bars during replay: clear then feed chronologically.
    for st in runtime.strategies:
        for sym in st.symbols:
            bars = runtime.store.load_bars(sym, "1d")
            keep = (
                max(0, len(bars) - max(effective_replay, 1))
                if effective_replay > 0
                else max(0, len(bars) - 1)
            )
            st._history[sym] = bars[:keep]

    processed = 0
    approved = 0
    rejected = 0
    # Persist across restarts: trade unread latest bars once, never re-spray same day.
    last_seen: dict[str, str] = load_last_seen(broker_name)
    if effective_replay > 0:
        bars = _unique_day_bars(runtime.store, symbols, last_n_days=effective_replay)
        logger.info("Replaying %s bars across %s symbols", len(bars), len(symbols))
        for bar in bars:
            results = runtime.on_bar(bar)
            processed += 1
            last_seen[bar.symbol] = bar.ts.isoformat()
            for r in results:
                if (r.get("decision") or {}).get("approved"):
                    approved += 1
                else:
                    rejected += 1
        save_last_seen(broker_name, last_seen)
        _save_paper_state(
            runtime,
            broker_name=broker_name,
            starting_equity=starting_equity,
            started_at=started_at,
            days_elapsed=float(effective_replay),
            extra={
                "mode": "replay",
                "replay_days": effective_replay,
                "bars_processed": processed,
                "intents_approved": approved,
                "intents_rejected": rejected,
                "strategies_file": str(path),
            },
        )

    if loop or broker_name == "alpaca":
        if trade_latest:
            logger.info(
                "Persisted last_seen=%s symbols — will process unread latest daily bars once "
                "(stagger=%.1fs between fills)",
                len(last_seen),
                order_stagger_sec,
            )
        else:
            for symbol in symbols:
                bars = runtime.store.load_bars(symbol, "1d")
                if bars:
                    last_seen[symbol] = bars[-1].ts.isoformat()
            save_last_seen(broker_name, last_seen)
            logger.info(
                "Seeded last_seen for %s symbols (awaiting newer bars only)",
                len(last_seen),
            )

    cycles = 0
    last_payload: dict[str, Any] = {}
    while True:
        cycles += 1
        cycle_approved = 0
        cycle_rejected = 0
        if broker_name == "alpaca":
            try:
                sync_book_from_alpaca(runtime)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Alpaca re-hydrate failed: %s", exc)
        for symbol in symbols:
            bars = runtime.store.load_bars(symbol, "1d")
            if not bars:
                continue
            bar = bars[-1]
            ts_key = bar.ts.isoformat()
            if last_seen.get(symbol) == ts_key:
                continue
            for st in runtime.strategies:
                hist = st._history.get(symbol, [])
                if hist and hist[-1].ts == bar.ts:
                    st._history[symbol] = hist[:-1]
            results = runtime.on_bar(bar)
            last_seen[symbol] = ts_key
            save_last_seen(broker_name, last_seen)
            placed = False
            for r in results:
                if r.get("broker_rejected"):
                    cycle_rejected += 1
                    MONITOR.push_decision(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "source": "broker",
                            "approved": False,
                            "reason": (r.get("order") or {}).get("error") or "broker_rejected",
                            "symbol": (r.get("intent") or {}).get("symbol"),
                            "strategy_id": (r.get("intent") or {}).get("strategy_id"),
                        }
                    )
                elif (r.get("decision") or {}).get("approved"):
                    cycle_approved += 1
                    placed = True
                    MONITOR.push_decision(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "source": "mrm",
                            "approved": True,
                            "reason": (r.get("decision") or {}).get("reason"),
                            "symbol": (r.get("intent") or {}).get("symbol"),
                            "strategy_id": (r.get("intent") or {}).get("strategy_id"),
                            "qty": (r.get("intent") or {}).get("qty"),
                        }
                    )
                else:
                    cycle_rejected += 1
                    MONITOR.push_decision(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "source": "mrm",
                            "approved": False,
                            "reason": (r.get("decision") or {}).get("reason"),
                            "symbol": (r.get("intent") or {}).get("symbol"),
                            "strategy_id": (r.get("intent") or {}).get("strategy_id"),
                        }
                    )
            if placed and order_stagger_sec > 0 and broker_name == "alpaca":
                time.sleep(float(order_stagger_sec))
                try:
                    sync_book_from_alpaca(runtime)
                except Exception:  # noqa: BLE001
                    pass
        days_elapsed = None
        if effective_replay > 0:
            days_elapsed = float(effective_replay) + max(
                0.0, (datetime.now(timezone.utc) - started_at).total_seconds() / 86400.0
            )
        last_payload = _save_paper_state(
            runtime,
            broker_name=broker_name,
            starting_equity=starting_equity,
            started_at=started_at,
            days_elapsed=days_elapsed,
            extra={
                "mode": "live_cycle" if loop else "single_cycle",
                "cycle": cycles,
                "intents_approved": cycle_approved,
                "intents_rejected": cycle_rejected,
                "strategies_file": str(path),
                "trade_latest": trade_latest,
            },
        )
        summary = {
            "broker": broker_name,
            "equity": last_payload.get("equity"),
            "cash": last_payload.get("cash"),
            "ann_return": last_payload.get("ann_return"),
            "ann_return_min": last_payload.get("ann_return_min"),
            "ann_return_max": last_payload.get("ann_return_max"),
            "positions": last_payload.get("positions"),
            "strategies": len(runtime.strategies),
            "replay_bars": processed,
            "cycle": cycles,
            "strategies_file": str(path),
        }
        if not loop:
            return summary
        logger.info(
            "Paper loop sleep %.0fs equity=%s ann=%s",
            interval_sec,
            last_payload.get("equity"),
            last_payload.get("ann_return"),
        )
        time.sleep(max(5.0, float(interval_sec)))
