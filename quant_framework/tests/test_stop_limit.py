"""
Phase 5.5-D acceptance tests — true stop-limit orders.

A stop-limit is NOT a stop-market. The stop only activates a limit order; the fill
requires the limit price to be legally available. A gap beyond the limit must leave
the order triggered and unfilled rather than silently marketizing.
"""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config import default_es_futures
from config.cost_model import CostModel
from config.models import OrderStatus, OrderType, Side
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.portfolio import Portfolio


TZ = ZoneInfo("America/Chicago")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def _zero_cost(participation: float = 1.0) -> CostModel:
    return CostModel(
        version="stop_limit_test",
        commission_per_contract=0.0,
        minimum_commission=0.0,
        fixed_spread_ticks=0.0,
        dynamic_spread_enabled=False,
        slippage_ticks_mean=0.0,
        slippage_ticks_std=0.0,
        volatility_dependent_slippage=False,
        time_of_day_slippage=False,
        liquidity_dependent_slippage=False,
        overnight_swap_enabled=False,
        participation_rate_cap=participation,
    )


def _bar(
    open_ts: str,
    o: float,
    h: float,
    low: float,
    c: float,
    *,
    volume: float = 10_000.0,
    session_close: bool = False,
) -> BarEvent:
    start = _ts(open_ts)
    return BarEvent(
        timestamp=start,
        open=o,
        high=h,
        low=low,
        close=c,
        volume=volume,
        symbol="ES",
        contract="ESH24",
        bar_end=start + pd.Timedelta(minutes=5),
        is_session_close=session_close,
    )


def _engine(participation: float = 1.0) -> EventExecutionEngine:
    return EventExecutionEngine(
        default_es_futures(),
        _zero_cost(participation),
        starting_equity=100_000.0,
        exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
    )


def _stop_limit_signal(
    bar: BarEvent,
    *,
    side: str,
    stop: float,
    limit: float,
    qty: float = 1.0,
) -> SignalEvent:
    return SignalEvent(
        timing=InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.bar_end,
            decision_timestamp=bar.bar_end,
        ),
        symbol="ES",
        side=side,
        quantity=qty,
        order_type="STOP_LIMIT",
        stop_trigger=stop,
        limit_price=limit,
        signal_id="sig_stop_limit",
    )


def _submit_on_first_bar(**kwargs: float | str):
    def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
        if i != 0:
            return None
        return _stop_limit_signal(bar, **kwargs)  # type: ignore[arg-type]

    return signal_fn


class TestStopLimitIsNotStopMarket:
    def test_trigger_without_fill_when_limit_not_available(self) -> None:
        """
        Buy stop 101 with limit 100.9: the trigger happens at 101, which is worse
        than the limit, so no fill may occur even though the stop fired.
        """
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            # Rises through the stop but never trades back down to the limit
            _bar("2024-01-02 08:35", 100.95, 102.0, 100.95, 101.8),
            _bar("2024-01-02 08:40", 101.8, 102.5, 101.5, 102.0),
        ]
        result = engine.run(bars, _submit_on_first_bar(side="BUY", stop=101.0, limit=100.9))

        order = [o for o in result.orders if o.order_type == OrderType.STOP_LIMIT][0]
        assert order.meta["stop_triggered"] is True
        assert order.meta["stop_trigger_price"] == pytest.approx(101.0)
        assert order.meta.get("stop_limit_gapped_beyond_limit") is True
        assert result.fills == []
        assert order.status == OrderStatus.ACTIVE
        assert order.filled_quantity == pytest.approx(0.0)
        assert result.portfolio.get_position("ES").is_flat
        # The trigger is recorded, not swallowed
        assert any(e["type"] == "stop_limit_triggered" for e in result.risk_events)

    def test_trigger_then_fill_on_a_later_bar_at_the_limit(self) -> None:
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            _bar("2024-01-02 08:35", 100.95, 102.0, 100.95, 101.8),  # triggers, no fill
            _bar("2024-01-02 08:40", 101.5, 101.6, 100.5, 100.8),  # trades back to the limit
        ]
        result = engine.run(bars, _submit_on_first_bar(side="BUY", stop=101.0, limit=100.9))

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.meta["pricing_mode"] == "stop_limit_resting"
        assert fill.reference_price == pytest.approx(100.9)
        assert fill.price == pytest.approx(100.9)
        assert fill.fill_timestamp == bars[2].timestamp
        order = [o for o in result.orders if o.order_type == OrderType.STOP_LIMIT][0]
        assert order.status == OrderStatus.FILLED
        assert result.portfolio.get_position("ES").quantity == pytest.approx(1.0)

    def test_immediate_fill_when_limit_covers_the_trigger(self) -> None:
        """Limit above the stop: the trigger price is acceptable, so it fills at the trigger."""
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            _bar("2024-01-02 08:35", 100.2, 101.5, 100.1, 101.4),
            _bar("2024-01-02 08:40", 101.4, 101.8, 101.2, 101.6),
        ]
        result = engine.run(bars, _submit_on_first_bar(side="BUY", stop=101.0, limit=101.25))

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.meta["pricing_mode"] == "stop_limit_trigger"
        # Filled at the stop, never marketized to the bar close of 101.4
        assert fill.reference_price == pytest.approx(101.0)
        assert fill.price == pytest.approx(101.0)
        assert fill.price <= 101.25

    def test_gap_beyond_limit_leaves_order_unfilled(self) -> None:
        """A runaway gap must not be filled as a market order at the gap price."""
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            _bar("2024-01-02 08:35", 110.0, 112.0, 109.5, 111.0),  # gap far above the limit
            _bar("2024-01-02 08:40", 111.0, 113.0, 110.5, 112.0),
            _bar("2024-01-02 08:45", 112.0, 114.0, 111.0, 113.0),
        ]
        result = engine.run(bars, _submit_on_first_bar(side="BUY", stop=101.0, limit=101.5))

        order = [o for o in result.orders if o.order_type == OrderType.STOP_LIMIT][0]
        assert order.meta["stop_triggered"] is True
        assert order.meta["stop_trigger_price"] == pytest.approx(110.0)
        assert order.meta["stop_limit_gapped_beyond_limit"] is True
        assert result.fills == []
        assert result.portfolio.get_position("ES").is_flat
        assert result.portfolio.cash == pytest.approx(100_000.0)

    def test_sell_stop_limit_gap_below_limit_unfilled(self) -> None:
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            _bar("2024-01-02 08:35", 90.0, 90.5, 88.0, 89.0),  # gaps below the sell limit
            _bar("2024-01-02 08:40", 89.0, 89.5, 87.0, 88.0),
        ]
        result = engine.run(bars, _submit_on_first_bar(side="SELL", stop=99.0, limit=98.5))

        order = [o for o in result.orders if o.order_type == OrderType.STOP_LIMIT][0]
        assert order.meta["stop_triggered"] is True
        assert order.meta["stop_trigger_price"] == pytest.approx(90.0)
        assert order.meta["stop_limit_gapped_beyond_limit"] is True
        assert result.fills == []

    def test_expires_at_session_close_after_trigger(self) -> None:
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            _bar("2024-01-02 08:35", 100.95, 102.0, 100.95, 101.8),  # triggers, no fill
            _bar("2024-01-02 08:40", 101.8, 102.5, 101.5, 102.0, session_close=True),
            _bar("2024-01-02 08:45", 102.0, 102.5, 100.0, 100.5),  # would have filled
        ]
        result = engine.run(bars, _submit_on_first_bar(side="BUY", stop=101.0, limit=100.9))

        order = [o for o in result.orders if o.order_type == OrderType.STOP_LIMIT][0]
        assert order.meta["stop_triggered"] is True
        assert order.status == OrderStatus.EXPIRED
        assert order.reject_reason == "session_close"
        assert [s for s, _ in order.status_history][-1] == OrderStatus.EXPIRED
        # Expiry happens at the session-close bar end, before the next bar could fill it
        assert order.status_history[-1][1] == bars[2].bar_end
        assert result.fills == []

    def test_cancellation_after_partial_fill_blocks_the_remainder(self) -> None:
        """
        Participation cap of 10% of a 20-lot bar allows 2 of 5 contracts, then the
        remainder is cancelled and must never fill on later bars.
        """
        engine = _engine(participation=0.10)
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0, volume=20.0),
            _bar("2024-01-02 08:35", 100.2, 101.5, 100.1, 101.4, volume=20.0),
            _bar("2024-01-02 08:40", 101.4, 101.8, 100.9, 101.6, volume=20.0),
            _bar("2024-01-02 08:45", 101.6, 102.0, 100.9, 101.8, volume=20.0),
        ]
        cancelled_at: list[pd.Timestamp] = []

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i == 0:
                return _stop_limit_signal(bar, side="BUY", stop=101.0, limit=101.25, qty=5.0)
            if i == 1:
                # Cancel the remainder right after the first partial fill
                for order in list(engine._working):
                    if order.order_type == OrderType.STOP_LIMIT:
                        assert order.status == OrderStatus.PARTIALLY_FILLED
                        assert engine.cancel_order(order.order_id, bar.bar_end) is True
                        cancelled_at.append(bar.bar_end)
            return None

        result = engine.run(bars, signal_fn)

        order = [o for o in result.orders if o.order_type == OrderType.STOP_LIMIT][0]
        assert cancelled_at, "the remainder was never cancelled"
        assert order.status == OrderStatus.CANCELLED
        assert order.filled_quantity == pytest.approx(2.0)
        assert order.remaining_quantity == pytest.approx(3.0)
        # Only the partial fill exists — later bars cannot resurrect the remainder
        assert len(result.fills) == 1
        assert result.fills[0].quantity == pytest.approx(2.0)
        assert result.fills[0].partial is True
        assert result.portfolio.get_position("ES").quantity == pytest.approx(2.0)
        assert engine._working == []

    def test_stop_limit_never_reuses_stop_market_pricing(self) -> None:
        """
        Regression guard: the same gap that a STOP order fills at the gap price must
        leave an otherwise identical STOP_LIMIT unfilled.
        """
        bars = [
            _bar("2024-01-02 08:30", 100.0, 100.5, 99.5, 100.0),
            _bar("2024-01-02 08:35", 110.0, 112.0, 109.5, 111.0),
            _bar("2024-01-02 08:40", 111.0, 113.0, 110.5, 112.0),
        ]

        def stop_market_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i != 0:
                return None
            return SignalEvent(
                timing=InformationTiming(
                    source_timestamp=bar.timestamp,
                    availability_timestamp=bar.bar_end,
                    decision_timestamp=bar.bar_end,
                ),
                symbol="ES",
                side="BUY",
                quantity=1.0,
                order_type="STOP",
                stop_trigger=101.0,
                signal_id="sig_stop",
            )

        stop_result = _engine().run(bars, stop_market_fn)
        limit_result = _engine().run(bars, _submit_on_first_bar(side="BUY", stop=101.0, limit=101.5))

        assert len(stop_result.fills) == 1
        assert stop_result.fills[0].reference_price == pytest.approx(110.0)
        assert stop_result.fills[0].gap_through is True
        assert limit_result.fills == []
        assert stop_result.fills[0].side == Side.BUY
