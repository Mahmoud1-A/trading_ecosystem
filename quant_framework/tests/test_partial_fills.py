"""
Phase 5.5-C acceptance tests — partial fills and bar participation capacity.

Bar volume is a shared, finite resource: the participation cap applies per bar and
is consumed across every order competing in that bar. Fixtures use small, exact
volumes so every expected quantity is hand-checkable.
"""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config import default_es_futures
from config.cost_model import CostModel
from config.models import OrderStatus, OrderType
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.portfolio import Portfolio


TZ = ZoneInfo("America/Chicago")

# 10% of bar volume may be consumed, so a 30-lot bar supports exactly 3 contracts.
PARTICIPATION = 0.10
BAR_VOLUME = 30.0
BAR_CAPACITY = PARTICIPATION * BAR_VOLUME  # 3.0


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def _cost(participation: float = PARTICIPATION) -> CostModel:
    return CostModel(
        version="partial_fill_test",
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
    price: float = 100.0,
    *,
    volume: float = BAR_VOLUME,
    stale: bool = False,
) -> BarEvent:
    start = _ts(open_ts)
    return BarEvent(
        timestamp=start,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=volume,
        symbol="ES",
        contract="ESH24",
        bar_end=start + pd.Timedelta(minutes=5),
        is_stale=stale,
    )


def _engine(participation: float = PARTICIPATION) -> EventExecutionEngine:
    return EventExecutionEngine(
        default_es_futures(),
        _cost(participation),
        starting_equity=1_000_000.0,
        exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
    )


def _buy(bar: BarEvent, qty: float, *, signal_id: str = "sig") -> SignalEvent:
    return SignalEvent(
        timing=InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.bar_end,
            decision_timestamp=bar.bar_end,
        ),
        symbol="ES",
        side="BUY",
        quantity=qty,
        signal_id=signal_id,
    )


def _bars(n: int, *, volume: float = BAR_VOLUME, price: float = 100.0) -> list[BarEvent]:
    return [
        _bar(
            str(_ts("2024-01-02 08:30") + pd.Timedelta(minutes=5 * i))[:16],
            price,
            volume=volume,
        )
        for i in range(n)
    ]


class TestFillQuantityBounds:
    def test_fill_never_exceeds_remaining_quantity(self) -> None:
        """Requested 2 contracts against 3 of capacity: only 2 may fill."""
        engine = _engine()
        bars = _bars(3)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 2.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        assert len(result.fills) == 1
        assert result.fills[0].quantity == pytest.approx(2.0)
        assert result.fills[0].partial is False
        assert result.orders[0].status == OrderStatus.FILLED
        assert result.orders[0].filled_quantity == pytest.approx(2.0)
        assert result.orders[0].remaining_quantity == pytest.approx(0.0)

    def test_fill_never_exceeds_participation_capacity(self) -> None:
        """Requested 10 contracts, capacity 3 per bar: the first fill is exactly 3."""
        engine = _engine()
        bars = _bars(2)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 10.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        assert len(result.fills) == 1
        assert result.fills[0].quantity == pytest.approx(BAR_CAPACITY)
        assert result.fills[0].partial is True
        assert result.fills[0].liquidity_used == pytest.approx(BAR_CAPACITY)

    def test_capacity_is_shared_across_orders_in_the_same_bar(self) -> None:
        """
        Two 2-lot orders activate on the same bar with only 3 of capacity: the first
        takes 2, the second may take only the remaining 1.
        """
        engine = _engine()
        bars = _bars(3)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> list[SignalEvent] | None:
            if i != 0:
                return None
            return [_buy(bar, 2.0, signal_id="a"), _buy(bar, 2.0, signal_id="b")]

        result = engine.run(bars, signal_fn)
        first_bar_fills = [f for f in result.fills if f.fill_timestamp == bars[1].timestamp]
        assert sum(f.quantity for f in first_bar_fills) == pytest.approx(BAR_CAPACITY)
        assert sorted(f.quantity for f in first_bar_fills) == pytest.approx([1.0, 2.0])
        # The second order's remainder fills on the next bar's fresh capacity
        assert sum(f.quantity for f in result.fills) == pytest.approx(4.0)
        assert result.portfolio.get_position("ES").quantity == pytest.approx(4.0)


class TestLifecycleTransitions:
    def test_active_to_partially_filled_to_filled(self) -> None:
        """7 contracts over 3-per-bar capacity: 3 + 3 + 1 across three bars."""
        engine = _engine()
        bars = _bars(5)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 7.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        order = result.orders[0]
        statuses = [s for s, _ in order.status_history]
        assert statuses == [
            OrderStatus.CREATED,
            OrderStatus.SUBMITTED,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
        ]
        assert [f.quantity for f in result.fills] == pytest.approx([3.0, 3.0, 1.0])
        assert [f.partial for f in result.fills] == [True, True, False]
        assert order.filled_quantity == pytest.approx(7.0)
        assert order.remaining_quantity == pytest.approx(0.0)

    def test_cancelled_remainder_cannot_fill_later(self) -> None:
        engine = _engine()
        bars = _bars(5)
        cancelled = False

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int):
            nonlocal cancelled
            if i == 0:
                return _buy(bar, 9.0)
            if i == 1 and not cancelled:
                order = engine._working[0]
                assert order.status == OrderStatus.PARTIALLY_FILLED
                assert engine.cancel_order(order.order_id, bar.bar_end, reason="risk_pull") is True
                cancelled = True
            return None

        result = engine.run(bars, signal_fn)
        order = result.orders[0]
        assert order.status == OrderStatus.CANCELLED
        assert order.reject_reason == "risk_pull"
        assert order.filled_quantity == pytest.approx(3.0)
        assert order.remaining_quantity == pytest.approx(6.0)
        # Three more bars of capacity existed but produced nothing
        assert len(result.fills) == 1
        assert result.portfolio.get_position("ES").quantity == pytest.approx(3.0)
        assert engine._working == []


class TestStaleAndRepeatedBars:
    def test_stale_volume_cannot_create_fills(self) -> None:
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30"),
            _bar("2024-01-02 08:35", stale=True),
            _bar("2024-01-02 08:40"),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 2.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        assert result.fills == []
        assert result.orders[0].status == OrderStatus.REJECTED
        assert result.orders[0].reject_reason == "stale_bar"
        assert result.portfolio.get_position("ES").is_flat

    def test_zero_volume_bar_yields_no_liquidity(self) -> None:
        engine = _engine()
        bars = [
            _bar("2024-01-02 08:30"),
            _bar("2024-01-02 08:35", volume=0.0),
            _bar("2024-01-02 08:40"),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 2.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        assert result.fills == []
        assert result.orders[0].reject_reason == "insufficient_liquidity"

    def test_repeated_bars_never_overfill_the_order(self) -> None:
        """Twenty bars of capacity must still fill exactly the 7 requested contracts."""
        engine = _engine()
        bars = _bars(20)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 7.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        assert sum(f.quantity for f in result.fills) == pytest.approx(7.0)
        assert result.orders[0].filled_quantity == pytest.approx(7.0)
        assert result.orders[0].status == OrderStatus.FILLED
        assert result.portfolio.get_position("ES").quantity == pytest.approx(7.0)

    def test_per_bar_capacity_resets_each_bar(self) -> None:
        engine = _engine()
        bars = _bars(4)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            return _buy(bar, 6.0) if i == 0 else None

        result = engine.run(bars, signal_fn)
        by_bar = {f.fill_timestamp: f.quantity for f in result.fills}
        assert by_bar[bars[1].timestamp] == pytest.approx(BAR_CAPACITY)
        assert by_bar[bars[2].timestamp] == pytest.approx(BAR_CAPACITY)
        assert len(by_bar) == 2


class TestPortfolioReconciliation:
    def test_portfolio_quantity_equals_cumulative_fills(self) -> None:
        engine = _engine()
        bars = _bars(8)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> list[SignalEvent] | None:
            if i == 0:
                return [_buy(bar, 5.0, signal_id="a")]
            if i == 3:
                return [_buy(bar, 4.0, signal_id="b")]
            return None

        result = engine.run(bars, signal_fn)
        cumulative = sum(
            f.quantity if f.side.value == "BUY" else -f.quantity for f in result.fills
        )
        assert result.portfolio.get_position("ES").quantity == pytest.approx(cumulative)
        assert cumulative == pytest.approx(9.0)
        # Order-level accounting matches fill-level accounting
        for order in result.orders:
            order_fills = [f for f in result.fills if f.order_id == order.order_id]
            assert order.filled_quantity == pytest.approx(sum(f.quantity for f in order_fills))
            assert order.quantity == pytest.approx(
                order.filled_quantity + (order.remaining_quantity or 0.0)
            )

    def test_no_fill_exceeds_bar_capacity_across_whole_run(self) -> None:
        engine = _engine()
        bars = _bars(10)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> list[SignalEvent] | None:
            if i in {0, 1, 2}:
                return [_buy(bar, 4.0, signal_id=f"s{i}")]
            return None

        result = engine.run(bars, signal_fn)
        per_bar: dict[pd.Timestamp, float] = {}
        for fill in result.fills:
            per_bar[fill.fill_timestamp] = per_bar.get(fill.fill_timestamp, 0.0) + fill.quantity
        assert per_bar, "expected fills"
        for ts, qty in per_bar.items():
            assert qty <= BAR_CAPACITY + 1e-9, f"bar {ts} overfilled: {qty}"
        assert sum(per_bar.values()) == pytest.approx(12.0)
        assert result.portfolio.get_position("ES").quantity == pytest.approx(12.0)
