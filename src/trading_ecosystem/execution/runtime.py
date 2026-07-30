from __future__ import annotations

import logging
from datetime import datetime, timezone

from trading_ecosystem.common.contracts import Bar, OrderSide
from trading_ecosystem.data_pipeline.store import ParquetBarStore
from trading_ecosystem.execution.broker import BrokerGateway
from trading_ecosystem.portfolio.state import PortfolioBook
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.base import Strategy

logger = logging.getLogger(__name__)


class PaperRuntime:
    """Live/paper loop: latest bars -> strategies -> MRM -> broker."""

    def __init__(
        self,
        strategies: list[Strategy],
        risk: MasterRiskManager,
        broker: BrokerGateway,
        store: ParquetBarStore | None = None,
        starting_equity: float = 100_000.0,
    ) -> None:
        self.strategies = strategies
        self.risk = risk
        self.broker = broker
        self.store = store or ParquetBarStore()
        self.book = PortfolioBook(starting_equity)
        # Warm history from store
        self._warm_history()

    def _warm_history(self) -> None:
        for strategy in self.strategies:
            for symbol in strategy.symbols:
                bars = self.store.load_bars(symbol, "1d")
                strategy._history[symbol] = bars[-300:]

    def on_bar(self, bar: Bar) -> list[dict]:
        self.book.maybe_roll_day(bar.ts)
        self.book.mark_to_market({bar.symbol: bar.close})
        self.risk.update_halt_flags(self.book.state)

        results: list[dict] = []
        for strategy in self.strategies:
            if bar.symbol not in strategy.symbols:
                continue
            strategy.params["equity_hint"] = self.book.state.equity
            key = self.book.state.position_key(strategy.strategy_id, bar.symbol)
            position = self.book.state.positions.get(key)
            intents = strategy.on_bar(bar, position)
            for intent in intents:
                decision = self.risk.evaluate(intent, self.book.state, bar.close)
                if not decision.approved:
                    results.append({"intent": intent.model_dump(), "decision": decision.model_dump()})
                    continue
                if decision.adjusted_qty is not None:
                    intent.qty = decision.adjusted_qty
                # Keep local sim prices for PaperSimBroker
                if hasattr(self.broker, "set_price"):
                    self.broker.set_price(bar.symbol, bar.close)  # type: ignore[attr-defined]
                order = self.broker.place_order(intent)
                if order.get("rejected") or str(order.get("status", "")).lower() == "rejected":
                    results.append(
                        {
                            "intent": intent.model_dump(),
                            "decision": decision.model_dump(),
                            "order": {k: v for k, v in order.items() if k != "fill"},
                            "broker_rejected": True,
                        }
                    )
                    continue
                # Mirror fills into local book when sim/dry provides fill
                fill = order.get("fill")
                if fill is not None:
                    self.book.apply_fill(fill, stop_price=intent.stop_price)
                elif order.get("dry_run"):
                    # Optimistic local book update for dry-run only
                    from trading_ecosystem.common.contracts import Fill

                    fill_obj = Fill(
                        intent_id=intent.intent_id,
                        strategy_id=intent.strategy_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        price=bar.close,
                        ts=intent.ts if intent.ts.tzinfo else datetime.now(timezone.utc),
                    )
                    self.book.apply_fill(fill_obj, stop_price=intent.stop_price)
                elif str(order.get("status", "")).lower() == "submitted":
                    # Real broker (Alpaca): optimistic fill so MRM sees exposure in-cycle;
                    # paper_loop re-hydrates from broker each cycle for truth.
                    from trading_ecosystem.common.contracts import Fill

                    fill_obj = Fill(
                        intent_id=intent.intent_id,
                        strategy_id=intent.strategy_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        price=bar.close,
                        ts=intent.ts if intent.ts.tzinfo else datetime.now(timezone.utc),
                    )
                    self.book.apply_fill(fill_obj, stop_price=intent.stop_price)
                    # Reduce buying_power estimate immediately when known
                    if self.book.state.buying_power is not None:
                        self.book.state.buying_power = max(
                            0.0,
                            float(self.book.state.buying_power) - intent.qty * bar.close,
                        )
                results.append(
                    {
                        "intent": intent.model_dump(),
                        "decision": decision.model_dump(),
                        "order": {k: v for k, v in order.items() if k != "fill"},
                    }
                )
        return results

    def recover(self) -> dict:
        synced = self.broker.sync()
        logger.info("broker sync: %s", synced.get("account"))
        return synced
