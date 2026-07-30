"""
Phase 5.5-E acceptance tests — CFD overnight financing.

Financing is accrued only when a broker rollover boundary is crossed while a
position is held. It is recorded separately from commission / spread / slippage and
it reduces cash, and therefore net PnL and equity.
"""

from __future__ import annotations

from datetime import time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.asset_spec import CFDAssetSpec, SessionHours
from config.cost_model import CostModel
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.financing import CFDFinancingEngine
from engine.portfolio import Portfolio


UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")

CONTRACT_SIZE = 100.0
LONG_RATE = 0.0002   # positive => the long holder pays
SHORT_RATE = 0.0001  # positive => the short holder pays


def _cfd(timezone: str = "UTC") -> CFDAssetSpec:
    return CFDAssetSpec(
        broker_symbol="US500",
        contract_size=CONTRACT_SIZE,
        leverage=20.0,
        broker_spread_markup=0.0,
        overnight_swap_long=LONG_RATE,
        overnight_swap_short=SHORT_RATE,
        margin_rate=0.05,
        stop_out_level=0.5,
        tick_size=0.1,
        min_lot=0.01,
        lot_step=0.01,
        broker_trading_hours=SessionHours(
            timezone=timezone,
            open_time=time(0, 0),
            close_time=time(23, 59, 59),
            overnight=True,
        ),
    )


def _cost_model() -> CostModel:
    return CostModel(
        version="financing_test",
        commission_per_contract=0.0,
        minimum_commission=0.0,
        fixed_spread_ticks=0.0,
        dynamic_spread_enabled=False,
        slippage_ticks_mean=0.0,
        slippage_ticks_std=0.0,
        volatility_dependent_slippage=False,
        time_of_day_slippage=False,
        liquidity_dependent_slippage=False,
        overnight_swap_enabled=True,
        participation_rate_cap=1.0,
    )


def _portfolio_with_position(qty: float, price: float = 100.0) -> Portfolio:
    p = Portfolio(starting_equity=100_000.0)
    p.cash = 100_000.0
    p.set_multiplier("US500", CONTRACT_SIZE)
    pos = p.get_position("US500")
    pos.quantity = qty
    pos.avg_price = price
    pos.multiplier = CONTRACT_SIZE
    pos.contract = "US500"
    return p


def _bar(ts: pd.Timestamp, price: float = 100.0, *, minutes: int = 5) -> BarEvent:
    return BarEvent(
        timestamp=ts,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price,
        volume=100_000.0,
        symbol="US500",
        contract="US500",
        bar_end=ts + pd.Timedelta(minutes=minutes),
    )


class TestRolloverBoundaryDetection:
    def test_first_mark_never_accrues(self) -> None:
        eng = CFDFinancingEngine(asset=_cfd())
        crossed = eng.rollover_boundaries_crossed(None, pd.Timestamp("2024-01-02 23:00", tz=UTC))
        assert crossed == []

    def test_no_overnight_means_no_financing(self) -> None:
        eng = CFDFinancingEngine(asset=_cfd())
        p = _portfolio_with_position(2.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-02 14:00", tz=UTC),
            curr_ts=pd.Timestamp("2024-01-02 21:55", tz=UTC),
            mark_price=100.0,
            symbol="US500",
        )
        assert events == []
        assert p.cash == pytest.approx(100_000.0)
        assert p.get_position("US500").financing_paid == pytest.approx(0.0)

    def test_single_rollover_debits_long(self) -> None:
        """2 lots x 100 contract size x 100 price = 20,000 notional; 0.0002 => 4.00."""
        eng = CFDFinancingEngine(asset=_cfd())
        p = _portfolio_with_position(2.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-02 21:55", tz=UTC),  # Tuesday
            curr_ts=pd.Timestamp("2024-01-02 22:05", tz=UTC),
            mark_price=100.0,
            symbol="US500",
        )
        assert len(events) == 1
        evt = events[0]
        assert evt.side == "LONG"
        assert evt.triple_swap is False
        assert evt.notional == pytest.approx(20_000.0)
        assert evt.amount == pytest.approx(4.0)
        assert p.cash == pytest.approx(100_000.0 - 4.0)
        assert p.get_position("US500").financing_paid == pytest.approx(4.0)

    def test_single_rollover_debits_short_at_short_rate(self) -> None:
        eng = CFDFinancingEngine(asset=_cfd())
        p = _portfolio_with_position(-2.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-02 21:55", tz=UTC),
            curr_ts=pd.Timestamp("2024-01-02 22:05", tz=UTC),
            mark_price=100.0,
            symbol="US500",
        )
        assert len(events) == 1
        assert events[0].side == "SHORT"
        assert events[0].amount == pytest.approx(2.0)  # 20,000 * 0.0001
        assert p.cash == pytest.approx(100_000.0 - 2.0)

    def test_flat_position_accrues_nothing(self) -> None:
        eng = CFDFinancingEngine(asset=_cfd())
        p = _portfolio_with_position(0.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-02 21:55", tz=UTC),
            curr_ts=pd.Timestamp("2024-01-02 22:05", tz=UTC),
            mark_price=100.0,
            symbol="US500",
        )
        assert events == []
        assert p.cash == pytest.approx(100_000.0)

    def test_multiple_rollovers_in_one_gap(self) -> None:
        """A weekend gap crosses Fri/Sat/Sun boundaries: three accruals."""
        eng = CFDFinancingEngine(asset=_cfd())
        p = _portfolio_with_position(1.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-05 21:00", tz=UTC),  # Friday
            curr_ts=pd.Timestamp("2024-01-08 09:00", tz=UTC),  # Monday
            mark_price=100.0,
            symbol="US500",
        )
        dates = [str(e.timestamp.date()) for e in events]
        assert dates == ["2024-01-05", "2024-01-06", "2024-01-07"]
        # 1 lot -> 10,000 notional -> 2.00 per night, none of them a Wednesday
        assert all(e.triple_swap is False for e in events)
        assert sum(e.amount for e in events) == pytest.approx(6.0)
        assert p.cash == pytest.approx(100_000.0 - 6.0)

    def test_triple_swap_day_charges_three_nights(self) -> None:
        """2024-01-03 is a Wednesday: the 22:00 boundary carries a 3x swap."""
        eng = CFDFinancingEngine(asset=_cfd())
        p = _portfolio_with_position(1.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-03 21:55", tz=UTC),
            curr_ts=pd.Timestamp("2024-01-03 22:05", tz=UTC),
            mark_price=100.0,
            symbol="US500",
        )
        assert len(events) == 1
        assert events[0].timestamp.weekday() == 2
        assert events[0].triple_swap is True
        assert events[0].rate == pytest.approx(LONG_RATE * 3)
        assert events[0].amount == pytest.approx(6.0)  # 10,000 * 0.0002 * 3
        assert p.cash == pytest.approx(100_000.0 - 6.0)

    def test_negative_rate_is_a_credit(self) -> None:
        asset = _cfd().model_copy(update={"overnight_swap_long": -0.0002})
        eng = CFDFinancingEngine(asset=asset)
        p = _portfolio_with_position(1.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-01-02 21:55", tz=UTC),
            curr_ts=pd.Timestamp("2024-01-02 22:05", tz=UTC),
            mark_price=100.0,
            symbol="US500",
        )
        assert events[0].amount == pytest.approx(-2.0)
        assert p.cash == pytest.approx(100_000.0 + 2.0)


class TestDaylightSavingBoundaries:
    def test_boundary_tracks_local_clock_across_dst_change(self) -> None:
        """
        US DST starts 2024-03-10. The broker's 22:00 local boundary is 03:00 UTC on
        03-10 (EST) and 02:00 UTC on 03-11 (EDT) — one boundary per trading day, and
        the UTC offset shifts rather than an extra or missing accrual appearing.
        """
        eng = CFDFinancingEngine(asset=_cfd("America/New_York"))
        p = _portfolio_with_position(1.0)
        events = eng.accrue(
            p,
            prev_ts=pd.Timestamp("2024-03-09 12:00", tz=NY),
            curr_ts=pd.Timestamp("2024-03-11 23:30", tz=NY),
            mark_price=100.0,
            symbol="US500",
        )
        local = [e.timestamp.tz_convert(NY) for e in events]
        assert [str(t.date()) for t in local] == ["2024-03-09", "2024-03-10", "2024-03-11"]
        assert {t.hour for t in local} == {22}
        utc_hours = [e.timestamp.tz_convert(UTC).hour for e in events]
        assert utc_hours == [3, 2, 2]
        assert len(events) == 3
        assert p.cash == pytest.approx(100_000.0 - 6.0)


class TestEngineWiring:
    @staticmethod
    def _run(bars: list[BarEvent], qty: float = 2.0):
        engine = EventExecutionEngine(
            _cfd(),
            _cost_model(),
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
        )

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i != 0:
                return None
            return SignalEvent(
                timing=InformationTiming(
                    source_timestamp=bar.timestamp,
                    availability_timestamp=bar.bar_end,
                    decision_timestamp=bar.bar_end,
                ),
                symbol="US500",
                side="BUY",
                quantity=qty,
                signal_id="sig_cfd",
            )

        return engine, engine.run(bars, signal_fn)

    def test_engine_creates_financing_engine_for_cfd(self) -> None:
        engine, _ = self._run([_bar(pd.Timestamp("2024-01-02 12:00", tz=UTC))] * 1)
        assert isinstance(engine.financing, CFDFinancingEngine)

    def test_engine_accrues_across_rollover_and_reduces_equity(self) -> None:
        start = pd.Timestamp("2024-01-02 21:45", tz=UTC)
        bars = [_bar(start + pd.Timedelta(minutes=5 * i)) for i in range(6)]
        engine, result = self._run(bars, qty=2.0)

        # Position opened at 100 with a zero-cost model, so only financing moves cash
        assert result.portfolio.get_position("US500").quantity == pytest.approx(2.0)
        assert len(result.financing_events) == 1
        evt = result.financing_events[0]
        assert evt.amount == pytest.approx(4.0)
        assert result.portfolio.cash == pytest.approx(100_000.0 - 4.0)
        assert result.portfolio.equity() == pytest.approx(100_000.0 - 4.0)
        # Recorded separately: no fill-level cost component was touched
        assert all(f.costs.financing_cost == pytest.approx(0.0) for f in result.fills)
        assert all(f.costs.commission == pytest.approx(0.0) for f in result.fills)
        assert all(f.costs.spread_cost == pytest.approx(0.0) for f in result.fills)
        assert all(f.costs.slippage_cost == pytest.approx(0.0) for f in result.fills)
        assert result.portfolio.get_position("US500").financing_paid == pytest.approx(4.0)
        assert any(e["type"] == "financing_accrual" for e in result.risk_events)

    def test_engine_no_rollover_no_financing(self) -> None:
        start = pd.Timestamp("2024-01-02 12:00", tz=UTC)
        bars = [_bar(start + pd.Timedelta(minutes=5 * i)) for i in range(6)]
        _engine_obj, result = self._run(bars, qty=2.0)
        assert result.financing_events == []
        assert result.portfolio.cash == pytest.approx(100_000.0)

    def test_engine_accrues_multiple_rollovers(self) -> None:
        # Hourly bars from Tue 20:00 to Fri 20:00 cross exactly three 22:00 boundaries
        start = pd.Timestamp("2024-01-02 20:00", tz=UTC)
        bars = [_bar(start + pd.Timedelta(hours=i), minutes=60) for i in range(3 * 24)]
        _engine_obj, result = self._run(bars, qty=1.0)
        assert len(result.financing_events) == 3
        # 2024-01-03 is a Wednesday triple swap: 2 + 6 + 2 = 10
        assert [e.triple_swap for e in result.financing_events] == [False, True, False]
        assert sum(e.amount for e in result.financing_events) == pytest.approx(10.0)
        assert result.portfolio.cash == pytest.approx(100_000.0 - 10.0)

    def test_financing_reduces_net_pnl_on_round_trip(self) -> None:
        start = pd.Timestamp("2024-01-02 21:45", tz=UTC)
        bars = [_bar(start + pd.Timedelta(minutes=5 * i)) for i in range(8)]
        engine = EventExecutionEngine(
            _cfd(),
            _cost_model(),
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
        )

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            timing = InformationTiming(
                source_timestamp=bar.timestamp,
                availability_timestamp=bar.bar_end,
                decision_timestamp=bar.bar_end,
            )
            if i == 0:
                return SignalEvent(
                    timing=timing, symbol="US500", side="BUY", quantity=2.0, signal_id="in"
                )
            if i == 5:
                return SignalEvent(
                    timing=timing, symbol="US500", side="FLAT", quantity=2.0, signal_id="out"
                )
            return None

        result = engine.run(bars, signal_fn)
        assert result.portfolio.get_position("US500").is_flat
        assert len(result.financing_events) == 1
        # Flat round trip at an unchanged price: the only loss is the swap
        gross = sum(t.gross_pnl for t in result.portfolio.trades)
        assert gross == pytest.approx(0.0)
        assert result.portfolio.cash == pytest.approx(100_000.0 - 4.0)
        assert result.portfolio.equity() == pytest.approx(100_000.0 - 4.0)
