from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from trading_ecosystem.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_ecosystem.common.contracts import Bar, Fill, OrderIntent, OrderSide
from trading_ecosystem.execution.costs import ExecutionCosts
from trading_ecosystem.portfolio.state import PortfolioBook
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    equity_curve: list[tuple]
    fills: list[Fill]
    risk_rejects: int
    audit: list[Any] = field(default_factory=list)


class BacktestEngine:
    """Event-driven backtest sharing strategy + MRM path with live."""

    def __init__(
        self,
        strategies: list[Strategy],
        risk_manager: MasterRiskManager,
        starting_equity: float = 100_000.0,
        commission_per_share: float = 0.0,
        slippage_bps: float = 2.0,
        costs: ExecutionCosts | None = None,
    ) -> None:
        self.strategies = strategies
        self.risk = risk_manager
        self.starting_equity = starting_equity
        self.costs = costs or ExecutionCosts(
            slippage_bps=float(slippage_bps),
            commission_per_share=float(commission_per_share),
            profile="legacy",
        )
        self.commission_per_share = self.costs.commission_per_share
        self.slippage_bps = self.costs.slippage_bps

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
        pending: list[tuple[OrderIntent, float | None]] = []

        for bar in events:
            book.maybe_roll_day(bar.ts)
            last_prices[bar.symbol] = bar.close
            book.mark_to_market(last_prices)
            self.risk.update_halt_flags(book.state)
            if hasattr(self.risk, "on_mark"):
                self.risk.on_mark(book.state)

            # Flush pending intents scheduled for this symbol (e.g. next_bar_open)
            still: list[tuple[OrderIntent, float | None]] = []
            for intent, stop in pending:
                if intent.symbol != bar.symbol:
                    still.append((intent, stop))
                    continue
                mid = float(bar.open)
                intent = intent.model_copy(update={"ts": bar.ts})
                decision = self.risk.evaluate(intent, book.state, mid)
                if not decision.approved:
                    rejects += 1
                    continue
                qty = decision.adjusted_qty if decision.adjusted_qty is not None else intent.qty
                intent = intent.model_copy(update={"qty": qty})
                px = self.costs.fill_price(mid, intent.side, "taker")
                fill = self._execute(intent, px, book, stop_price=stop)
                if fill:
                    fills.append(fill)
                    self._track_pnl(fill, intent, entry_px, trade_pnls, book)
                    if hasattr(self.risk, "on_fill"):
                        self.risk.on_fill(book.state, fill)
            pending = still

            # Stop checks for open positions on this symbol
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
                        meta={"reason": "stop_hit", "liquidity": "taker"},
                    )
                    fill = self._execute(intent, stop, book)
                    if fill:
                        fills.append(fill)
                        self._track_pnl(fill, intent, entry_px, trade_pnls, book)

            for strategy in self.strategies:
                if bar.symbol not in strategy.symbols:
                    continue
                strategy.params["equity_hint"] = book.state.equity
                key = book.state.position_key(strategy.strategy_id, bar.symbol)
                position = book.state.positions.get(key)
                intents = strategy.on_bar(bar, position)
                for intent in intents:
                    fill_mode = None
                    if isinstance(intent.meta, dict):
                        fill_mode = intent.meta.get("fill_mode")
                    if fill_mode == "next_bar_open" and intent.side == OrderSide.BUY and not intent.reduce_only:
                        pending.append((intent, intent.stop_price))
                        continue
                    decision = self.risk.evaluate(intent, book.state, bar.close)
                    if not decision.approved:
                        rejects += 1
                        continue
                    qty = decision.adjusted_qty if decision.adjusted_qty is not None else intent.qty
                    intent.qty = qty
                    liq = None
                    if isinstance(intent.meta, dict):
                        liq = intent.meta.get("liquidity")
                    px = self.costs.fill_price(bar.close, intent.side, liq)
                    fill = self._execute(intent, px, book, stop_price=intent.stop_price)
                    if fill:
                        fills.append(fill)
                        self._track_pnl(fill, intent, entry_px, trade_pnls, book)
                        if hasattr(self.risk, "on_fill"):
                            self.risk.on_fill(book.state, fill)

            book.mark_to_market(last_prices)
            equity_curve.append((bar.ts, book.state.equity))

        if hasattr(self.risk, "end_of_day"):
            self.risk.end_of_day(book.state)

        metrics = compute_metrics(equity_curve, trade_pnls)
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            fills=fills,
            risk_rejects=rejects,
            audit=list(self.risk.audit),
        )

    def _track_pnl(
        self,
        fill: Fill,
        intent: OrderIntent,
        entry_px: dict[str, float],
        trade_pnls: list[float],
        book: PortfolioBook,
    ) -> None:
        k = book.state.position_key(intent.strategy_id, intent.symbol)
        if intent.side == OrderSide.BUY:
            entry_px[k] = fill.price
        elif intent.side == OrderSide.SELL and k in entry_px:
            trade_pnls.append((fill.price - entry_px[k]) * fill.qty - fill.commission)
            entry_px.pop(k, None)

    def _execute(
        self,
        intent: OrderIntent,
        price: float,
        book: PortfolioBook,
        stop_price: float | None = None,
    ) -> Fill | None:
        if intent.qty <= 0:
            return None
        liq = None
        if isinstance(intent.meta, dict):
            liq = intent.meta.get("liquidity")
        if intent.meta and intent.meta.get("reason") == "stop_hit":
            liq = "taker"
            price = self.costs.fill_price(price, intent.side, liq)
        commission = self.costs.commission(intent.qty, price, liq)
        slip_amt = abs(price) * (self.costs.slippage_bps / 10_000.0)
        fill = Fill(
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            price=price,
            ts=intent.ts,
            commission=commission,
            slippage=slip_amt,
        )
        book.apply_fill(fill, stop_price=stop_price or intent.stop_price)
        return fill
