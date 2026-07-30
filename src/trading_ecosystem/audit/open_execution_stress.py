"""
Decisive open-execution stress test on the SAME GapFade signals.

Uses daily bars for gap signal (prior close vs session open), then Alpaca 1-minute
bars for fills at: first printable after 09:30, 09:31, 09:35, 09:45, VWAP 5m,
and next-day open. Each fill path adds spread + slippage + fees + partial-fill model.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml
from trading_ecosystem.data_pipeline.store import ParquetBarStore
from trading_ecosystem.strategies.gap_fade import GapFade
from trading_ecosystem.strategies.registry import build_strategies

ET = ZoneInfo("America/New_York")


@dataclass
class OpenCosts:
    spread_bps: float = 2.0  # half-spread charged adverse
    slippage_bps: float = 5.0
    fee_bps: float = 0.0  # ETF usually 0
    commission_per_share: float = 0.0
    partial_fill_frac: float = 0.55  # first slice
    partial_worse_bps: float = 3.0  # second slice adverse vs first


@dataclass
class SignalEvent:
    strategy_id: str
    symbol: str
    signal_date: date
    gap: float
    prior_close: float
    session_open: float
    stop: float
    risk_fraction: float
    hold_bars: int = 5
    atr_stop_mult: float = 0.02


@dataclass
class ScenarioResult:
    name: str
    n_signals: int = 0
    n_filled: int = 0
    n_missed: int = 0
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    avg_slip_bps: float = 0.0
    notes: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _adverse_price(mid: float, side: str, bps: float) -> float:
    if mid <= 0:
        return mid
    slip = mid * (bps / 10_000.0)
    return mid + slip if side == "buy" else mid - slip


def _apply_costs(mid: float, side: str, costs: OpenCosts) -> tuple[float, float]:
    """Return (fill_price, total_cost_bps_estimate)."""
    half_spread = costs.spread_bps / 2.0
    total_bps = half_spread + costs.slippage_bps + costs.fee_bps
    px = _adverse_price(mid, side, total_bps)
    return px, total_bps


def _partial_fill_price(mid: float, side: str, costs: OpenCosts, next_mid: float | None) -> tuple[float, float]:
    """VWAP of two slices: first at mid-cost, second worse."""
    p1, bps1 = _apply_costs(mid, side, costs)
    if next_mid is None or next_mid <= 0:
        return p1, bps1
    p2_mid = _adverse_price(next_mid, side, costs.partial_worse_bps)
    p2, bps2 = _apply_costs(p2_mid, side, costs)
    f = max(0.05, min(0.95, costs.partial_fill_frac))
    px = f * p1 + (1.0 - f) * p2
    bps = f * bps1 + (1.0 - f) * (bps2 + costs.partial_worse_bps)
    return px, bps


def detect_gapfade_signals(
    *,
    start: str = "2022-01-01",
    end: str | None = None,
    gap_pct_floor: float | None = None,
) -> list[SignalEvent]:
    """
    Same signal definition as Post-Open GapFade: gap = open/prior_close - 1,
    ATR/stop from prior bars only. Uses strategy params from live sleeve.
    """
    strategies = build_strategies(CONFIG_DIR / "strategies.paper.live.yaml")
    store = ParquetBarStore()
    events: list[SignalEvent] = []
    for st in strategies:
        if not isinstance(st, GapFade):
            continue
        sym = st.symbols[0]
        bars = store.load_bars(sym, timeframe="1d", start=start, end=end)
        if len(bars) < 40:
            continue
        gap_pct = float(st.params.get("gap_pct", 0.002))
        if gap_pct_floor is not None:
            gap_pct = max(gap_pct, float(gap_pct_floor))
        atr_period = int(st.params.get("atr_period", 14))
        atr_mult = float(st.params.get("atr_stop_mult", 1.5))
        risk_fraction = float(st.params.get("risk_fraction", 0.0075))
        hold_bars = int(st.params.get("hold_bars", 5))
        # simple ATR on prior closes/highs/lows
        for i in range(atr_period + 2, len(bars)):
            prev = bars[i - 1]
            bar = bars[i]
            if prev.close <= 0:
                continue
            gap = (bar.open - prev.close) / prev.close
            if gap > -gap_pct:
                continue
            # ATR from bars before today
            trs = []
            for j in range(i - atr_period, i):
                b0, b1 = bars[j - 1], bars[j]
                tr = max(b1.high - b1.low, abs(b1.high - b0.close), abs(b1.low - b0.close))
                trs.append(tr)
            atr = sum(trs) / len(trs) if trs else 0.0
            if atr <= 0:
                continue
            stop = float(bar.open) - atr_mult * atr
            d = bar.ts.astimezone(timezone.utc).date() if bar.ts.tzinfo else bar.ts.date()
            events.append(
                SignalEvent(
                    strategy_id=st.strategy_id,
                    symbol=sym,
                    signal_date=d,
                    gap=float(gap),
                    prior_close=float(prev.close),
                    session_open=float(bar.open),
                    stop=float(stop),
                    risk_fraction=risk_fraction,
                    hold_bars=hold_bars,
                    atr_stop_mult=atr_mult,
                )
            )
    return events


def _alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient
    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca API keys missing")
    return StockHistoricalDataClient(key, secret)


def fetch_opening_minutes(
    symbols: list[str],
    day: date,
    *,
    client: Any = None,
    minutes: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch 1-minute bars for RTH open window on a calendar day (ET)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = client or _alpaca_client()
    start_et = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    end_et = start_et + timedelta(minutes=minutes)
    # also need next day open for next-day scenario — caller fetches separately
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_et,
        end=end_et,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX,
    )
    barset = client.get_stock_bars(req)
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
    data = getattr(barset, "data", None) or {}
    if hasattr(barset, "__iter__") and not data:
        # BarSet mapping
        try:
            data = dict(barset)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            data = {}
    for sym, bars in (data.items() if hasattr(data, "items") else []):
        rows = []
        for b in bars:
            ts = b.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_et = ts.astimezone(ET)
            rows.append(
                {
                    "ts": ts_et,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(getattr(b, "volume", 0) or 0),
                    "vwap": float(getattr(b, "vwap", 0) or 0) or None,
                }
            )
        out[str(sym)] = sorted(rows, key=lambda r: r["ts"])
    return out


def fetch_next_day_open(symbol: str, day: date, *, client: Any = None) -> float | None:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = client or _alpaca_client()
    # find next session: try day+1 .. day+5
    for add in range(1, 6):
        d = day + timedelta(days=add)
        start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
        end_et = start_et + timedelta(minutes=5)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_et,
            end=end_et,
            adjustment=Adjustment.SPLIT,
            feed=DataFeed.IEX,
        )
        try:
            barset = client.get_stock_bars(req)
        except Exception:  # noqa: BLE001
            continue
        data = getattr(barset, "data", {}) or {}
        bars = data.get(symbol) or []
        if bars:
            return float(bars[0].open)
    return None


def _bar_at_or_after(rows: list[dict[str, Any]], hh: int, mm: int) -> dict[str, Any] | None:
    target = None
    for r in rows:
        ts: datetime = r["ts"]
        if (ts.hour, ts.minute) >= (hh, mm):
            return r
        target = r
    return None


def _first_bar(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _vwap_first_n(rows: list[dict[str, Any]], n_minutes: int = 5) -> float | None:
    if not rows:
        return None
    start = rows[0]["ts"]
    end = start + timedelta(minutes=n_minutes)
    window = [r for r in rows if r["ts"] < end]
    if not window:
        return None
    # Prefer exchange VWAP if present on last bar; else volume-weighted close
    if window[-1].get("vwap"):
        # Approximate session open VWAP from bar vwaps * volume
        num = 0.0
        den = 0.0
        for r in window:
            v = r["volume"] or 0.0
            px = r.get("vwap") or r["close"]
            num += px * v
            den += v
        if den > 0:
            return num / den
    num = sum(r["close"] * max(r["volume"], 1.0) for r in window)
    den = sum(max(r["volume"], 1.0) for r in window)
    return num / den if den else window[-1]["close"]


def _intraday_stop_after_entry(
    rows: list[dict[str, Any]],
    entry_ts: datetime | None,
    stop: float,
) -> float | None:
    """If any minute low after entry hits stop, return pessimistic fill."""
    for r in rows:
        if entry_ts is not None and r["ts"] < entry_ts:
            continue
        if r["low"] <= stop:
            return float(min(stop, r["open"]))
    return None


def _exit_hold_bars(
    *,
    symbol: str,
    signal_date: date,
    entry_px: float,
    entry_ts: datetime | None,
    stop: float,
    hold_bars: int,
    open_rows: list[dict[str, Any]],
    daily_bars: list[Any],
) -> tuple[float, str]:
    """
    Mirror GapFade exits: stop or held >= hold_bars on daily bars.
    On the entry session, also honor intraday stop after the fill timestamp
    (more realistic than daily OHLC-only for tight stops).
    """
    hit = _intraday_stop_after_entry(open_rows, entry_ts, stop)
    if hit is not None:
        return hit, "intraday_stop"

    # Map daily bars by date
    by_day: dict[date, Any] = {}
    for b in daily_bars:
        d = b.ts.astimezone(timezone.utc).date() if b.ts.tzinfo else b.ts.date()
        by_day[d] = b

    # Entry day close if we survived the open window
    d0 = by_day.get(signal_date)
    held = 0
    if d0 is not None:
        held = 1
        if float(d0.low) <= stop:
            return float(min(stop, d0.open)), "day0_stop"
        if held >= hold_bars:
            return float(d0.close), "day0_hold"

    # Subsequent sessions
    d = signal_date
    for _ in range(1, max(hold_bars, 1) + 5):
        d = d + timedelta(days=1)
        b = by_day.get(d)
        if b is None:
            continue
        held += 1
        if float(b.low) <= stop:
            return float(min(stop, b.open)), f"day{held}_stop"
        if held >= hold_bars:
            return float(b.close), f"day{held}_hold"
    return entry_px, "no_exit_data"


SCENARIOS = [
    ("first_after_0930", "first"),
    ("t_0931", (9, 31)),
    ("t_0935", (9, 35)),
    ("t_0945", (9, 45)),
    ("vwap_5m", "vwap5"),
    ("next_day_open", "next_open"),
    # Optimistic reference: fill exactly at daily session open (often untradeable)
    ("daily_open_print", "daily_open"),
]


def run_open_execution_stress(
    *,
    start: str = "2024-01-01",
    end: str | None = None,
    max_signals: int = 400,
    costs: OpenCosts | None = None,
    sleep_sec: float = 0.15,
) -> dict[str, Any]:
    costs = costs or OpenCosts()
    signals = detect_gapfade_signals(start=start, end=end)
    # Deduplicate by symbol+date (one entry per symbol day)
    uniq: dict[tuple[str, date], SignalEvent] = {}
    for s in signals:
        key = (s.symbol, s.signal_date)
        if key not in uniq or abs(s.gap) > abs(uniq[key].gap):
            uniq[key] = s
    signals = sorted(uniq.values(), key=lambda x: x.signal_date)
    if max_signals and len(signals) > max_signals:
        # Time-stratified sample (not only the latest month)
        step = max(1, len(signals) // max_signals)
        sampled = signals[::step][:max_signals]
        # ensure we keep chronological order and cover the full span
        if sampled[-1].signal_date < signals[-1].signal_date:
            sampled[-1] = signals[-1]
        signals = sampled

    client = _alpaca_client()
    starting = float(load_yaml(CONFIG_DIR / "prop_rules.yaml").get("starting_equity", 100_000))
    store = ParquetBarStore()

    curves: dict[str, list[tuple]] = {name: [] for name, _ in SCENARIOS}
    equity: dict[str, float] = {name: starting for name, _ in SCENARIOS}
    filled: dict[str, int] = {name: 0 for name, _ in SCENARIOS}
    missed: dict[str, int] = {name: 0 for name, _ in SCENARIOS}
    slip_acc: dict[str, list[float]] = {name: [] for name, _ in SCENARIOS}
    peak: dict[str, float] = {name: starting for name, _ in SCENARIOS}
    max_dd: dict[str, float] = {name: 0.0 for name, _ in SCENARIOS}
    wins: dict[str, int] = {name: 0 for name, _ in SCENARIOS}
    exit_reasons: dict[str, dict[str, int]] = {name: {} for name, _ in SCENARIOS}
    trade_rets: dict[str, list[float]] = {name: [] for name, _ in SCENARIOS}

    # Prefetch minute opens only for signal days (next-day open uses daily parquet)
    days_needed: set[date] = {s.signal_date for s in signals}
    syms_by_day: dict[date, set[str]] = {}
    for s in signals:
        syms_by_day.setdefault(s.signal_date, set()).add(s.symbol)

    cache_day: dict[date, dict[str, list[dict[str, Any]]]] = {}
    day_list = sorted(days_needed)
    print(f"prefetch {len(day_list)} signal days for {len(signals)} signals…", flush=True)
    for i, day in enumerate(day_list):
        day_syms = sorted(syms_by_day.get(day) or [])
        if not day_syms:
            continue
        try:
            cache_day[day] = fetch_opening_minutes(day_syms, day, client=client, minutes=60)
        except Exception as exc:  # noqa: BLE001
            cache_day[day] = {s: [] for s in day_syms}
            print(f"fetch fail {day}: {exc}", flush=True)
        if sleep_sec:
            time.sleep(min(sleep_sec, 0.05))
        if (i + 1) % 25 == 0:
            print(f"fetched open windows {i+1}/{len(day_list)} days…", flush=True)

    # Daily bars cache for multi-day exits
    daily_cache: dict[str, list[Any]] = {}
    min_d = min(s.signal_date for s in signals) - timedelta(days=5)
    max_d = max(s.signal_date for s in signals) + timedelta(days=20)

    def daily_for(symbol: str) -> list[Any]:
        if symbol not in daily_cache:
            daily_cache[symbol] = store.load_bars(
                symbol,
                timeframe="1d",
                start=min_d.isoformat(),
                end=max_d.isoformat(),
            )
        return daily_cache[symbol]

    def next_session_from_daily(symbol: str, day: date) -> tuple[float | None, date | None]:
        bars = daily_for(symbol)
        for b in bars:
            d = b.ts.astimezone(timezone.utc).date() if b.ts.tzinfo else b.ts.date()
            if d > day:
                return float(b.open), d
        return None, None

    for idx, sig in enumerate(signals):
        rows = (cache_day.get(sig.signal_date) or {}).get(sig.symbol) or []
        nd_open, nd_date = next_session_from_daily(sig.symbol, sig.signal_date)
        d_bars = daily_for(sig.symbol)

        for name, kind in SCENARIOS:
            mid = None
            next_mid = None
            entry_ts: datetime | None = None
            entry_day = sig.signal_date
            open_rows_for_exit = rows

            if kind == "first":
                b = _first_bar(rows)
                mid = b["open"] if b else None
                entry_ts = b["ts"] if b else None
                if b and len(rows) > 1:
                    next_mid = rows[1]["open"]
            elif kind == "vwap5":
                mid = _vwap_first_n(rows, 5)
                if rows:
                    entry_ts = rows[0]["ts"]
                    next_mid = rows[min(5, len(rows) - 1)]["close"]
            elif kind == "next_open":
                mid = nd_open
                entry_day = nd_date or sig.signal_date
                open_rows_for_exit = []  # no minute window; daily stop/hold only
                entry_ts = None
            elif kind == "daily_open":
                mid = sig.session_open
                entry_ts = datetime(
                    sig.signal_date.year, sig.signal_date.month, sig.signal_date.day, 9, 30, tzinfo=ET
                )
            else:
                hh, mm = kind  # type: ignore[misc]
                b = _bar_at_or_after(rows, hh, mm)
                mid = b["open"] if b else None
                entry_ts = b["ts"] if b else None
                if b:
                    try:
                        i = rows.index(b)
                    except ValueError:
                        i = -1
                    if i >= 0 and i + 1 < len(rows):
                        next_mid = rows[i + 1]["open"]

            if mid is None or mid <= 0:
                missed[name] += 1
                curves[name].append(
                    (datetime.combine(sig.signal_date, datetime.min.time(), tzinfo=timezone.utc), equity[name])
                )
                continue

            entry, slip_bps = _partial_fill_price(
                float(mid), "buy", costs, float(next_mid) if next_mid else None
            )
            exit_raw, reason = _exit_hold_bars(
                symbol=sig.symbol,
                signal_date=entry_day,
                entry_px=entry,
                entry_ts=entry_ts,
                stop=sig.stop,
                hold_bars=sig.hold_bars,
                open_rows=open_rows_for_exit,
                daily_bars=d_bars,
            )
            exit_px, _ = _apply_costs(float(exit_raw), "sell", costs)
            exit_reasons[name][reason] = exit_reasons[name].get(reason, 0) + 1

            risk_ps = max(entry - sig.stop, entry * 0.0005)
            # Cap size: never risk more than risk_fraction of equity on stop distance
            qty = max(1.0, (equity[name] * sig.risk_fraction) / risk_ps)
            qty = float(int(qty))
            # Hard notional cap 25% equity to avoid blowups with microscopic stops
            max_qty = max(1.0, (equity[name] * 0.25) / entry)
            qty = min(qty, float(int(max_qty)))
            if qty <= 0:
                missed[name] += 1
                continue
            pnl = (exit_px - entry) * qty
            pnl -= costs.commission_per_share * qty * 2
            ret = pnl / equity[name] if equity[name] else 0.0
            trade_rets[name].append(ret)
            if pnl > 0:
                wins[name] += 1
            equity[name] += pnl
            filled[name] += 1
            slip_acc[name].append(slip_bps)
            peak[name] = max(peak[name], equity[name])
            dd = (peak[name] - equity[name]) / peak[name] if peak[name] > 0 else 0.0
            max_dd[name] = max(max_dd[name], dd)
            curves[name].append(
                (datetime.combine(sig.signal_date, datetime.min.time(), tzinfo=timezone.utc), equity[name])
            )

        if (idx + 1) % 50 == 0:
            print(f"priced {idx+1}/{len(signals)} signals…", flush=True)

    span_days = max(1, (signals[-1].signal_date - signals[0].signal_date).days)
    results: dict[str, Any] = {}
    for name, _ in SCENARIOS:
        curve = curves[name]
        total_ret = equity[name] / starting - 1.0
        # Period-annualized (not calendar CAGR from dense trade stamps)
        years = span_days / 365.25
        if years > 0 and equity[name] > 0:
            cagr = (equity[name] / starting) ** (1.0 / years) - 1.0
        else:
            cagr = -1.0 if total_ret < 0 else 0.0
        avg_slip = sum(slip_acc[name]) / len(slip_acc[name]) if slip_acc[name] else 0.0
        n_f = filled[name]
        win_rate = wins[name] / n_f if n_f else 0.0
        avg_trade = sum(trade_rets[name]) / n_f if n_f else 0.0
        results[name] = asdict(
            ScenarioResult(
                name=name,
                n_signals=len(signals),
                n_filled=n_f,
                n_missed=missed[name],
                total_return=total_ret,
                cagr=cagr,
                max_drawdown=max_dd[name],
                avg_slip_bps=avg_slip,
                notes="same GapFade signals; Alpaca IEX 1m open; hold_bars exit; SPLIT",
                detail={
                    "final_equity": equity[name],
                    "starting_equity": starting,
                    "win_rate": win_rate,
                    "avg_trade_ret": avg_trade,
                    "exit_reasons": exit_reasons[name],
                    "span_days": span_days,
                },
            )
        )

    verdict = _verdict(results)
    # Param sanity for interpretation
    atr_mults = [s.atr_stop_mult for s in signals]
    holds = [s.hold_bars for s in signals]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_signals": len(signals),
        "start": start,
        "end": end,
        "span_days": span_days,
        "signal_date_range": [str(signals[0].signal_date), str(signals[-1].signal_date)],
        "costs": asdict(costs),
        "feed": "Alpaca IEX 1Min, adjustment=SPLIT",
        "param_notes": {
            "avg_atr_stop_mult": sum(atr_mults) / len(atr_mults),
            "min_atr_stop_mult": min(atr_mults),
            "max_atr_stop_mult": max(atr_mults),
            "avg_hold_bars": sum(holds) / len(holds),
            "warning": "live GapFade atr_stop_mult is extremely tight (~0.01-0.05 ATR); "
            "intraday stop checks will stop out most same-day entries",
        },
        "scenarios": results,
        "verdict": verdict,
        "interpretation": {
            "graceful_decay": "edge real if period return declines smoothly 09:31→09:35→09:45 but stays positive",
            "phantom_open": "edge fake if daily_open_print strong but 09:31 already near-zero/negative",
        },
    }


def _verdict(results: dict[str, Any]) -> dict[str, Any]:
    def tot(name: str) -> float:
        return float((results.get(name) or {}).get("total_return") or 0.0)

    def cagr(name: str) -> float:
        return float((results.get(name) or {}).get("cagr") or 0.0)

    first = tot("first_after_0930")
    t31 = tot("t_0931")
    t35 = tot("t_0935")
    t45 = tot("t_0945")
    nxt = tot("next_day_open")
    daily = tot("daily_open_print")

    # Graceful decay on period returns (same signal set)
    graceful = (
        t31 > 0.05
        and t35 > 0.0
        and t45 > -0.05
        and t31 >= t35 >= t45 - 0.02
        and (t31 - t45) < 0.35
    )
    phantom = daily > 0.10 and t31 < 0.02 and (t35 < 0.0 or first < 0.02)
    if phantom:
        label = "phantom_or_unharvestable_open"
    elif graceful:
        label = "likely_real_edge_with_time_decay"
    elif max(first, t31, t35, daily) < 0.02:
        label = "weak_or_no_edge_under_realistic_open_fills"
    else:
        label = "mixed_inconclusive"
    return {
        "label": label,
        "total_first": first,
        "total_0931": t31,
        "total_0935": t35,
        "total_0945": t45,
        "total_next_day": nxt,
        "total_daily_open_print": daily,
        "cagr_0931": cagr("t_0931"),
        "cagr_0935": cagr("t_0935"),
        "cagr_0945": cagr("t_0945"),
    }


def write_report(payload: dict[str, Any], out_dir: Path | None = None) -> tuple[Path, Path]:
    out = out_dir or (data_dir() / "processed" / "audit")
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    jp = out / f"open_execution_stress_{ts}.json"
    mp = out / "open_execution_stress_report.md"
    latest = out / "open_execution_stress_latest.json"
    text = json.dumps(payload, indent=2, default=str)
    jp.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    v = payload.get("verdict") or {}
    pn = payload.get("param_notes") or {}
    lines = [
        "# Open-execution stress (same GapFade signals)",
        "",
        f"Generated: `{payload.get('generated_at')}`",
        f"Signals: **{payload.get('n_signals')}** | Span: `{payload.get('signal_date_range')}` "
        f"({payload.get('span_days')}d) | Feed: `{payload.get('feed')}`",
        f"Verdict: **{v.get('label')}**",
        "",
        "## Scenario results (period return primary)",
    ]
    for name, row in (payload.get("scenarios") or {}).items():
        d = row.get("detail") or {}
        lines.append(
            f"- `{name}`: Total={row['total_return']:.2%}  Ann≈{row['cagr']:.2%}  "
            f"MaxDD={row['max_drawdown']:.2%}  win={d.get('win_rate', 0):.1%}  "
            f"filled={row['n_filled']} missed={row['n_missed']}  "
            f"avg_slip≈{row['avg_slip_bps']:.1f}bps  exits={d.get('exit_reasons')}"
        )
    lines.extend(
        [
            "",
            "## Reading the result",
            "- Graceful decay 09:31→09:45 while staying positive → harvestable edge.",
            "- Strong `daily_open_print` then collapse by 09:31 → open print likely unharvestable/phantom.",
            "",
            f"Param notes: `{pn}`",
            f"Costs model: `{payload.get('costs')}`",
        ]
    )
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp
