"""Forensic re-test of the frozen live book (~71% claim)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from trading_ecosystem.backtest.engine import BacktestEngine, BacktestResult
from trading_ecosystem.backtest.metrics import compute_metrics
from trading_ecosystem.common.config import CONFIG_DIR, data_dir, load_yaml
from trading_ecosystem.common.contracts import Bar, Fill, OrderIntent, OrderSide
from trading_ecosystem.data_pipeline.store import ParquetBarStore
from trading_ecosystem.execution.costs import ExecutionCosts, load_execution_costs
from trading_ecosystem.portfolio.state import PortfolioBook
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategies


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    impact: dict[str, Any] = field(default_factory=dict)


def _metrics_dict(result: BacktestResult) -> dict[str, Any]:
    m = result.metrics.as_dict()
    return {
        "cagr": m["cagr"],
        "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "num_trades": m["num_trades"],
        "profit_factor": m["profit_factor"],
        "win_rate": m["win_rate"],
        "fills": len(result.fills),
        "risk_rejects": result.risk_rejects,
    }


def _scale_costs(base: ExecutionCosts, mult: float) -> ExecutionCosts:
    return ExecutionCosts(
        slippage_bps=float(base.slippage_bps) * mult,
        maker_fee_bps=float(base.maker_fee_bps) * mult,
        taker_fee_bps=float(base.taker_fee_bps) * mult,
        commission_per_share=float(base.commission_per_share) * mult,
        default_liquidity=base.default_liquidity,
        profile=f"{base.profile}_x{mult:g}",
    )


class IndependentEngine:
    """Second executor: next-open / worst fill / SOD equity sizing."""

    def __init__(
        self,
        strategies: list[Any],
        starting_equity: float,
        costs: ExecutionCosts,
        fill_mode: str = "next_open",
        equity_sizing: str = "sod",
    ) -> None:
        self.strategies = strategies
        self.starting_equity = starting_equity
        self.costs = costs
        self.fill_mode = fill_mode
        self.equity_sizing = equity_sizing
        self.risk = MasterRiskManager()

    def run(self, bars_by_symbol: dict[str, list[Bar]]) -> BacktestResult:
        events: list[Bar] = []
        for bars in bars_by_symbol.values():
            events.extend(bars)
        events.sort(key=lambda b: (b.ts, b.symbol))

        book = PortfolioBook(self.starting_equity)
        equity_curve: list[tuple] = []
        fills: list[Fill] = []
        trade_pnls: list[float] = []
        entry_px: dict[str, float] = {}
        rejects = 0
        last_prices: dict[str, float] = {}
        sod_equity = self.starting_equity
        last_day = None
        pending: list[tuple[OrderIntent, float | None]] = []

        for bar in events:
            day = bar.ts.date() if hasattr(bar.ts, "date") else bar.ts
            if last_day is None:
                last_day = day
            if day != last_day:
                sod_equity = book.state.equity
                last_day = day

            book.maybe_roll_day(bar.ts)
            last_prices[bar.symbol] = bar.close
            book.mark_to_market(last_prices)
            self.risk.update_halt_flags(book.state)

            still: list[tuple[OrderIntent, float | None]] = []
            for intent, stop in pending:
                if intent.symbol != bar.symbol:
                    still.append((intent, stop))
                    continue
                if self.fill_mode == "next_open":
                    mid = float(bar.open)
                elif self.fill_mode == "worst":
                    mid = float(bar.high) if intent.side == OrderSide.BUY else float(bar.low)
                else:
                    mid = float(bar.close)
                px = self.costs.fill_price(mid, intent.side, "taker")
                decision = self.risk.evaluate(intent, book.state, mid)
                if not decision.approved:
                    rejects += 1
                    continue
                intent.qty = decision.adjusted_qty if decision.adjusted_qty is not None else intent.qty
                commission = self.costs.commission(intent.qty, px, "taker")
                fill = Fill(
                    intent_id=intent.intent_id,
                    strategy_id=intent.strategy_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    qty=intent.qty,
                    price=px,
                    ts=bar.ts,
                    commission=commission,
                    slippage=abs(px) * (self.costs.slippage_bps / 10_000.0),
                )
                book.apply_fill(fill, stop_price=stop)
                fills.append(fill)
                k = book.state.position_key(intent.strategy_id, intent.symbol)
                if intent.side == OrderSide.BUY:
                    entry_px[k] = fill.price
                elif intent.side == OrderSide.SELL and k in entry_px:
                    trade_pnls.append((fill.price - entry_px[k]) * fill.qty - fill.commission)
                    entry_px.pop(k, None)
            pending = still

            for key, pos in list(book.state.positions.items()):
                if pos.symbol != bar.symbol or pos.qty <= 0:
                    continue
                stop = pos.stop_price or book.stops.get(key)
                if stop is not None and bar.low <= stop:
                    intent = OrderIntent(
                        strategy_id=pos.strategy_id,
                        symbol=pos.symbol,
                        side=OrderSide.SELL,
                        qty=abs(pos.qty),
                        ts=bar.ts,
                        reduce_only=True,
                        meta={"reason": "stop_hit"},
                    )
                    mid = float(bar.low) if self.fill_mode == "worst" else float(stop)
                    px = self.costs.fill_price(mid, OrderSide.SELL, "taker")
                    commission = self.costs.commission(intent.qty, px, "taker")
                    fill = Fill(
                        intent_id=intent.intent_id,
                        strategy_id=intent.strategy_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        price=px,
                        ts=bar.ts,
                        commission=commission,
                        slippage=abs(px) * (self.costs.slippage_bps / 10_000.0),
                    )
                    book.apply_fill(fill)
                    fills.append(fill)
                    if key in entry_px:
                        trade_pnls.append((fill.price - entry_px[key]) * fill.qty - fill.commission)
                        entry_px.pop(key, None)

            for strategy in self.strategies:
                if bar.symbol not in strategy.symbols:
                    continue
                strategy.params["equity_hint"] = sod_equity if self.equity_sizing == "sod" else book.state.equity
                key = book.state.position_key(strategy.strategy_id, bar.symbol)
                position = book.state.positions.get(key)
                for intent in strategy.on_bar(bar, position):
                    if self.fill_mode in {"next_open", "worst"} and intent.side == OrderSide.BUY and not intent.reduce_only:
                        pending.append((intent, intent.stop_price))
                        continue
                    mid = float(bar.close)
                    if self.fill_mode == "worst":
                        mid = float(bar.high) if intent.side == OrderSide.BUY else float(bar.low)
                    decision = self.risk.evaluate(intent, book.state, mid)
                    if not decision.approved:
                        rejects += 1
                        continue
                    intent.qty = decision.adjusted_qty if decision.adjusted_qty is not None else intent.qty
                    px = self.costs.fill_price(mid, intent.side, "taker")
                    commission = self.costs.commission(intent.qty, px, "taker")
                    fill = Fill(
                        intent_id=intent.intent_id,
                        strategy_id=intent.strategy_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        price=px,
                        ts=bar.ts,
                        commission=commission,
                        slippage=abs(px) * (self.costs.slippage_bps / 10_000.0),
                    )
                    book.apply_fill(fill, stop_price=intent.stop_price)
                    fills.append(fill)
                    k = book.state.position_key(intent.strategy_id, intent.symbol)
                    if intent.side == OrderSide.BUY:
                        entry_px[k] = fill.price
                    elif intent.side == OrderSide.SELL and k in entry_px:
                        trade_pnls.append((fill.price - entry_px[k]) * fill.qty - fill.commission)
                        entry_px.pop(k, None)

            book.mark_to_market(last_prices)
            equity_curve.append((bar.ts, book.state.equity))

        return BacktestResult(
            metrics=compute_metrics(equity_curve, trade_pnls),
            equity_curve=equity_curve,
            fills=fills,
            risk_rejects=rejects,
            audit=list(self.risk.audit),
        )


def _run_book(
    strategies: list[Any],
    panel: dict[str, list[Bar]],
    *,
    costs: ExecutionCosts,
    starting_equity: float,
    fill_mode: str = "close",
    equity_sizing: str = "mtm",
) -> BacktestResult:
    if fill_mode == "close" and equity_sizing == "mtm":
        return BacktestEngine(
            strategies=strategies,
            risk_manager=MasterRiskManager(),
            starting_equity=starting_equity,
            costs=costs,
        ).run(panel)
    return IndependentEngine(
        strategies=strategies,
        starting_equity=starting_equity,
        costs=costs,
        fill_mode=fill_mode,
        equity_sizing=equity_sizing,
    ).run(panel)


def _concentration(result: BacktestResult) -> dict[str, Any]:
    open_px: dict[tuple[str, str], tuple[float, float]] = {}
    trades: list[tuple[str, float, Any]] = []
    for f in result.fills:
        key = (f.strategy_id, f.symbol)
        if f.side == OrderSide.BUY:
            open_px[key] = (float(f.price), float(f.qty))
        elif f.side == OrderSide.SELL and key in open_px:
            ep, qty = open_px.pop(key)
            pnl = (float(f.price) - ep) * min(qty, float(f.qty)) - float(f.commission)
            trades.append((f.symbol, pnl, f.ts))
    if not trades:
        return {"best_5_trades_share": None, "best_symbol_share": None, "best_5_days_share": None}
    pnls = sorted((p for _, p, _ in trades), reverse=True)
    total = sum(pnls)
    by_sym: dict[str, float] = defaultdict(float)
    by_day: dict[Any, float] = defaultdict(float)
    for sym, p, ts in trades:
        by_sym[sym] += p
        by_day[ts.date() if hasattr(ts, "date") else ts] += p
    denom = total if abs(total) > 1e-9 else 1.0
    return {
        "best_5_trades_share": sum(pnls[:5]) / denom,
        "best_symbol_share": (max(by_sym.values()) / denom) if by_sym else None,
        "best_5_days_share": sum(sorted(by_day.values(), reverse=True)[:5]) / denom,
        "best_symbol": max(by_sym, key=by_sym.get) if by_sym else None,
        "gross_trade_pnl": total,
    }


def _symbol_coverage(panel: dict[str, list[Bar]]) -> dict[str, Any]:
    spans = {}
    for sym, bars in panel.items():
        if not bars:
            spans[sym] = None
            continue
        spans[sym] = {"first": str(bars[0].ts), "last": str(bars[-1].ts), "n": len(bars)}
    firsts = [bars[0].ts for bars in panel.values() if bars]
    lasts = [bars[-1].ts for bars in panel.values() if bars]
    return {
        "per_symbol": spans,
        "common_start": str(max(firsts)) if firsts else None,
        "common_end": str(min(lasts)) if lasts else None,
        "staggered_starts": len({str(bars[0].ts.date()) for bars in panel.values() if bars}) > 1,
    }


def run_forensic(
    *,
    strategies_path: Any = None,
    start: str = "2018-01-01",
    end: str | None = None,
) -> dict[str, Any]:
    from pathlib import Path

    prop = load_yaml(CONFIG_DIR / "prop_rules.yaml")
    starting = float(prop.get("starting_equity", 100_000))
    path = Path(strategies_path) if strategies_path else (CONFIG_DIR / "strategies.paper.live.yaml")
    strategies = build_strategies(path)
    needed = sorted({s for st in strategies for s in st.symbols})
    panel = ParquetBarStore().load_panel(needed, timeframe="1d", start=start, end=end)
    base_costs = load_execution_costs(universe_id="etf")

    checks: list[CheckResult] = [
        CheckResult(
            name="same_bar_open_signal_close_fill",
            status="warn",
            detail=(
                "Legacy risk: open-gap signal with same-bar close fill. "
                "Current GapFade uses post_open + next_bar_open; compare base vs delayed scenarios."
            ),
            impact={"severity": "high_if_legacy"},
        ),
        CheckResult(
            name="costs_applied_entry_and_exit",
            status="pass",
            detail=f"ExecutionCosts slip={base_costs.slippage_bps}bps via fill_price on every fill",
        ),
        CheckResult(
            name="price_adjustment_policy",
            status="warn",
            detail="Yahoo auto_adjust=True default; store supports raw series suffix + provenance sidecars",
        ),
    ]
    coverage = _symbol_coverage(panel)
    checks.append(
        CheckResult(
            name="symbol_history_coverage",
            status="warn" if coverage.get("staggered_starts") else "pass",
            detail="ETF history start dates may differ; full-panel allows staggered entry",
            impact={k: coverage[k] for k in ("common_start", "common_end", "staggered_starts")},
        )
    )

    scenarios: dict[str, Any] = {}
    base = _run_book(strategies, panel, costs=base_costs, starting_equity=starting)
    scenarios["base_costs"] = _metrics_dict(base)
    scenarios["base_concentration"] = _concentration(base)

    delayed = _run_book(
        strategies, panel, costs=base_costs, starting_equity=starting, fill_mode="next_open", equity_sizing="sod"
    )
    scenarios["next_open_delayed_entry"] = _metrics_dict(delayed)
    scenarios["next_open_vs_base_cagr_delta"] = float(delayed.metrics.cagr) - float(base.metrics.cagr)

    for mult, key in [(2.0, "costs_2x"), (4.0, "costs_4x")]:
        scenarios[key] = _metrics_dict(
            _run_book(strategies, panel, costs=_scale_costs(base_costs, mult), starting_equity=starting)
        )

    scenarios["worst_reasonable_fill"] = _metrics_dict(
        _run_book(strategies, panel, costs=base_costs, starting_equity=starting, fill_mode="worst", equity_sizing="sod")
    )
    scenarios["sod_equity_sizing"] = _metrics_dict(
        _run_book(strategies, panel, costs=base_costs, starting_equity=starting, fill_mode="close", equity_sizing="sod")
    )

    best_sym = scenarios["base_concentration"].get("best_symbol")
    if best_sym:
        trimmed = [s for s in strategies if best_sym not in s.symbols]
        if trimmed:
            scenarios["remove_best_symbol"] = {
                **_metrics_dict(_run_book(trimmed, panel, costs=base_costs, starting_equity=starting)),
                "removed": best_sym,
            }

    scenarios["remove_best_5_trades_proxy"] = {
        "note": "Share of gross trade PnL from best 5 round-trips",
        "best_5_trades_share": scenarios["base_concentration"].get("best_5_trades_share"),
    }

    base_cagr = float(base.metrics.cagr)
    delayed_cagr = float(delayed.metrics.cagr)
    c2 = float(scenarios["costs_2x"]["cagr"])
    if delayed_cagr < 0.15 or c2 < 0.20:
        trust = "untrusted_pending_repromotion"
    elif delayed_cagr < base_cagr * 0.5:
        trust = "degraded_pending_repromotion"
    else:
        trust = "trusted_pending_promotion"

    checks.append(
        CheckResult(
            name="trust_verdict",
            status="fail" if trust.startswith("untrusted") else ("warn" if trust.startswith("degraded") else "pass"),
            detail=trust,
            impact={"base_cagr": base_cagr, "next_open_cagr": delayed_cagr, "costs_2x_cagr": c2},
        )
    )

    freeze_meta: dict[str, Any] = {}
    fm = data_dir() / "processed" / "paper_freeze_meta.json"
    if fm.exists():
        freeze_meta = json.loads(fm.read_text(encoding="utf-8"))

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategies_path": str(path),
        "members": len(strategies),
        "symbols": needed,
        "freeze_meta": {
            "book_cagr": freeze_meta.get("book_cagr"),
            "book_max_dd": freeze_meta.get("book_max_dd"),
            "frozen_at": freeze_meta.get("frozen_at"),
        },
        "checks": [asdict(c) for c in checks],
        "scenarios": scenarios,
        "trust": trust,
        "notes": [
            "Legacy engine fill=close after open-gap signal is structural look-ahead for GapFade.",
            "next_open_delayed_entry uses an independent simplified executor.",
            "Does not mutate strategies.paper.live.yaml.",
        ],
    }


def write_forensic_report(report: dict[str, Any], out_dir: Any = None) -> tuple[Any, Any]:
    from pathlib import Path

    out = Path(out_dir) if out_dir else (data_dir() / "processed" / "audit")
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = out / f"forensic_freeze71_{ts}.json"
    md_path = out / "forensic_freeze71_report.md"
    payload = json.dumps(report, indent=2, default=str)
    json_path.write_text(payload, encoding="utf-8")
    (out / "forensic_freeze71_latest.json").write_text(payload, encoding="utf-8")

    lines = [
        "# Forensic freeze (~71%) report",
        "",
        f"Generated: `{report.get('generated_at')}`",
        f"Trust: **{report.get('trust')}**",
        "",
        "## Freeze meta",
        f"- Claimed book CAGR: {((report.get('freeze_meta') or {}).get('book_cagr'))}",
        f"- Claimed MaxDD: {((report.get('freeze_meta') or {}).get('book_max_dd'))}",
        "",
        "## Scenario CAGRs",
    ]
    for k, v in (report.get("scenarios") or {}).items():
        if isinstance(v, dict) and "cagr" in v:
            lines.append(
                f"- `{k}`: CAGR={v['cagr']:.4%} MaxDD={v.get('max_drawdown', 0):.4%} trades={v.get('num_trades')}"
            )
    lines.extend(["", "## Checks"])
    for c in report.get("checks") or []:
        lines.append(f"- **{c['name']}** [{c['status']}]: {c['detail']}")
    lines.extend(["", "## Notes"])
    for n in report.get("notes") or []:
        lines.append(f"- {n}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
