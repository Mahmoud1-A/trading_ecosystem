"""Phase 2 acceptance tests — execution timing, lifecycle, friction, stale rejection."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config import default_es_futures, default_futures_cost_model
from config.cost_model import CostModel
from config.models import IntrabarAmbiguityPolicy, OrderStatus, OrderType, Side
from engine import (
    BarEvent,
    EventExecutionEngine,
    ExecutionConfig,
    FrictionContext,
    FrictionEngine,
    InformationTiming,
    Order,
    OrderLifecycleError,
    Portfolio,
    SignalEvent,
    bars_from_frame,
)
from engine.orders import Order as OrderCls


TZ = ZoneInfo("America/Chicago")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def _bar(
    open_ts: str,
    o: float,
    h: float,
    l: float,
    c: float,
    *,
    volume: float = 10_000.0,
    contract: str = "ESH24",
    stale: bool = False,
    atr: float | None = 2.0,
    freq_minutes: int = 5,
) -> BarEvent:
    start = _ts(open_ts)
    return BarEvent(
        timestamp=start,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
        symbol="ES",
        contract=contract,
        bar_end=start + pd.Timedelta(minutes=freq_minutes),
        atr=atr,
        is_stale=stale,
    )


def _close_signal(bar: BarEvent, side: str, qty: float = 1.0, **kwargs: object) -> SignalEvent:
    timing = InformationTiming(
        source_timestamp=bar.timestamp,
        availability_timestamp=bar.bar_end,
        decision_timestamp=bar.bar_end,
    )
    return SignalEvent(
        timing=timing,
        symbol="ES",
        side=side,
        quantity=qty,
        **kwargs,  # type: ignore[arg-type]
    )


class TestInformationTimingContract:
    def test_signal_at_close_t_never_fills_before_t1_open(self) -> None:
        bars = [
            _bar("2024-01-02 08:30", 100, 101, 99, 100.5),
            _bar("2024-01-02 08:35", 100.5, 102, 100, 101),
            _bar("2024-01-02 08:40", 101, 103, 100.5, 102),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i == 0:
                return _close_signal(bar, "BUY", 1.0)
            return None

        engine = EventExecutionEngine(
            default_es_futures(),
            default_futures_cost_model(),
            exec_config=ExecutionConfig(
                latency=timedelta(0),
                random_seed=1,
            ),
        )
        # Zero out slip noise for cleaner assertions: use fixed zero-slip model
        engine.friction = FrictionEngine(
            CostModel(
                version="test",
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
                participation_rate_cap=1.0,
            ),
            default_es_futures(),
            seed=1,
        )
        result = engine.run(bars, signal_fn)
        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.fill_timestamp >= bars[1].timestamp
        assert fill.fill_timestamp < bars[0].bar_end or fill.fill_timestamp >= bars[1].timestamp
        # Must not fill during bar 0
        assert fill.fill_timestamp >= bars[1].timestamp
        assert fill.meta["pricing_mode"] == "bar_open"
        assert fill.reference_price == pytest.approx(bars[1].open)

    def test_latency_never_receives_retroactive_open(self) -> None:
        bars = [
            _bar("2024-01-02 08:30", 100, 101, 99, 100),
            _bar("2024-01-02 08:35", 110, 112, 109, 111),  # gapped open
            _bar("2024-01-02 08:40", 111, 113, 110, 112),
        ]
        # Latency pushes activation 1 minute after next bar open
        latency = timedelta(minutes=1)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i == 0:
                return _close_signal(bar, "BUY", 1.0)
            return None

        engine = EventExecutionEngine(
            default_es_futures(),
            CostModel(
                version="test",
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
                participation_rate_cap=1.0,
            ),
            exec_config=ExecutionConfig(latency=latency, random_seed=0),
        )
        result = engine.run(bars, signal_fn)
        assert len(result.fills) == 1
        fill = result.fills[0]
        # Activation = max(bar0.close + 1m, bar1.open) = bar1.open + 1m
        assert fill.timing.order_activation_timestamp == bars[1].timestamp + latency
        assert fill.meta["pricing_mode"] == "post_open_no_retroactive"
        # Must NOT use the already-elapsed open of 110
        assert fill.reference_price == pytest.approx(bars[1].close)
        assert fill.reference_price != pytest.approx(bars[1].open)

    def test_timing_monotonicity_enforced(self) -> None:
        with pytest.raises(ValueError, match="availability_timestamp"):
            InformationTiming(
                source_timestamp=_ts("2024-01-02 08:35"),
                availability_timestamp=_ts("2024-01-02 08:30"),
                decision_timestamp=_ts("2024-01-02 08:35"),
            )


class TestOrderLifecycle:
    def test_deterministic_transitions(self) -> None:
        timing = InformationTiming(
            source_timestamp=_ts("2024-01-02 08:30"),
            availability_timestamp=_ts("2024-01-02 08:35"),
            decision_timestamp=_ts("2024-01-02 08:35"),
        )
        order = Order(
            symbol="ES",
            contract="ESH24",
            side=Side.BUY,
            quantity=2.0,
            order_type=OrderType.MARKET,
            timing=timing,
        )
        assert order.status == OrderStatus.CREATED
        order.submit(_ts("2024-01-02 08:35"), latency=timedelta(0))
        assert order.status == OrderStatus.SUBMITTED
        assert order.activation_timestamp == _ts("2024-01-02 08:35")
        order.activate(_ts("2024-01-02 08:40"))
        assert order.status == OrderStatus.ACTIVE
        st = order.apply_fill_qty(1.0, _ts("2024-01-02 08:40"))
        assert st == OrderStatus.PARTIALLY_FILLED
        st = order.apply_fill_qty(1.0, _ts("2024-01-02 08:40"))
        assert st == OrderStatus.FILLED

    def test_illegal_transition_rejected(self) -> None:
        timing = InformationTiming(
            source_timestamp=_ts("2024-01-02 08:30"),
            availability_timestamp=_ts("2024-01-02 08:35"),
            decision_timestamp=_ts("2024-01-02 08:35"),
        )
        order = Order(
            symbol="ES",
            contract="ESH24",
            side=Side.BUY,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            timing=timing,
            limit_price=100.0,
        )
        order.reject(_ts("2024-01-02 08:35"), "test")
        with pytest.raises(OrderLifecycleError):
            order.activate(_ts("2024-01-02 08:40"))

    def test_cannot_activate_before_activation_timestamp(self) -> None:
        timing = InformationTiming(
            source_timestamp=_ts("2024-01-02 08:30"),
            availability_timestamp=_ts("2024-01-02 08:35"),
            decision_timestamp=_ts("2024-01-02 08:35"),
        )
        order = Order(
            symbol="ES",
            contract="ESH24",
            side=Side.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            timing=timing,
        )
        order.submit(_ts("2024-01-02 08:35"), latency=timedelta(minutes=5))
        with pytest.raises(OrderLifecycleError, match="before order_activation"):
            order.activate(_ts("2024-01-02 08:36"))


class TestFrictionCosts:
    def test_commission_spread_slippage_deducted(self) -> None:
        asset = default_es_futures()
        model = CostModel(
            version="test_cost",
            commission_per_contract=2.5,
            minimum_commission=2.5,
            fixed_spread_ticks=2.0,  # 2 ticks => 0.5 points full; half=0.25
            dynamic_spread_enabled=False,
            slippage_ticks_mean=1.0,  # 0.25 points
            slippage_ticks_std=0.0,
            volatility_dependent_slippage=False,
            time_of_day_slippage=False,
            liquidity_dependent_slippage=False,
            overnight_swap_enabled=False,
            participation_rate_cap=1.0,
        )
        eng = FrictionEngine(model, asset, seed=0)
        ctx = FrictionContext(mid_price=100.0, atr=None, bar_volume=10_000)
        fill_px, costs = eng.build_costs(side=Side.BUY, quantity=1.0, reference_price=100.0, ctx=ctx)
        # half spread = 1 tick = 0.25; slip = 1 tick = 0.25; fill = 100.5
        assert fill_px == pytest.approx(100.5)
        assert costs.commission == pytest.approx(2.5)
        # point value 50: spread_cost = 0.25 * 1 * 50 = 12.5; slip = 12.5
        assert costs.spread_cost == pytest.approx(12.5)
        assert costs.slippage_cost == pytest.approx(12.5)

        bars = [
            _bar("2024-01-02 08:30", 100, 101, 99, 100),
            _bar("2024-01-02 08:35", 100, 101, 99, 100),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i == 0:
                return _close_signal(bar, "BUY", 1.0)
            return None

        engine = EventExecutionEngine(
            asset,
            model,
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
        )
        result = engine.run(bars, signal_fn)
        assert len(result.fills) == 1
        f = result.fills[0]
        assert f.costs.commission == pytest.approx(2.5)
        assert f.costs.spread_cost == pytest.approx(12.5)
        assert f.costs.slippage_cost == pytest.approx(12.5)
        # Cash reduced by total friction on open
        assert result.portfolio.cash == pytest.approx(100_000.0 - f.costs.total_friction)

    def test_trade_report_shows_cost_components_on_round_trip(self) -> None:
        asset = default_es_futures()
        model = CostModel(
            version="test_cost",
            commission_per_contract=2.5,
            minimum_commission=2.5,
            fixed_spread_ticks=0.0,
            dynamic_spread_enabled=False,
            slippage_ticks_mean=0.0,
            slippage_ticks_std=0.0,
            volatility_dependent_slippage=False,
            time_of_day_slippage=False,
            liquidity_dependent_slippage=False,
            overnight_swap_enabled=False,
            participation_rate_cap=1.0,
        )
        bars = [
            _bar("2024-01-02 08:30", 100, 101, 99, 100),
            _bar("2024-01-02 08:35", 100, 101, 99, 100),
            _bar("2024-01-02 08:40", 110, 111, 109, 110),
            _bar("2024-01-02 08:45", 110, 111, 109, 110),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i == 0:
                return _close_signal(bar, "BUY", 1.0)
            if i == 2:
                return _close_signal(bar, "FLAT", 1.0)
            return None

        engine = EventExecutionEngine(
            asset, model, exec_config=ExecutionConfig(random_seed=0)
        )
        result = engine.run(bars, signal_fn)
        assert len(result.portfolio.trades) == 1
        trade = result.portfolio.trades[0]
        assert trade.gross_pnl == pytest.approx((110 - 100) * 50)
        assert trade.commission == pytest.approx(5.0)  # entry + exit
        assert "net_pnl" in trade.as_dict()
        assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.commission)


class TestGapThroughStop:
    def test_gap_through_stop_is_conservative(self) -> None:
        asset = default_es_futures()
        model = CostModel(
            version="test",
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
            participation_rate_cap=1.0,
        )
        # Bar 0 signal buy; bar 1 fills at open 100; bar 2 gaps through sell-stop at 95 with open 90
        bars = [
            _bar("2024-01-02 08:30", 100, 101, 99, 100),
            _bar("2024-01-02 08:35", 100, 101, 99, 100),
            _bar("2024-01-02 08:40", 90, 91, 89, 90),  # gap down through stop
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i != 0:
                return None
            return _close_signal(bar, "BUY", 1.0, stop_price=95.0)

        engine = EventExecutionEngine(
            asset, model, exec_config=ExecutionConfig(random_seed=0)
        )
        result = engine.run(bars, signal_fn)
        # Entry fill + stop exit fill
        assert len(result.fills) >= 2
        exit_fill = result.fills[-1]
        assert exit_fill.gap_through is True
        # Conservative: fill at open 90, not at stop 95
        assert exit_fill.reference_price == pytest.approx(90.0)
        assert exit_fill.reference_price < 95.0


class TestStaleData:
    def test_stale_bar_cannot_generate_fills(self) -> None:
        asset = default_es_futures()
        model = CostModel(
            version="test",
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
            participation_rate_cap=1.0,
        )
        bars = [
            _bar("2024-01-02 08:30", 100, 101, 99, 100),
            _bar("2024-01-02 08:35", 100, 101, 99, 100, stale=True),
            _bar("2024-01-02 08:40", 100, 101, 99, 100),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            if i == 0:
                return _close_signal(bar, "BUY", 1.0)
            return None

        engine = EventExecutionEngine(
            asset,
            model,
            exec_config=ExecutionConfig(stale_rejects_fills=True, random_seed=0),
        )
        result = engine.run(bars, signal_fn)
        # Order activates on stale bar 1 and is rejected — no fill
        assert result.fills == []
        assert any(o.status == OrderStatus.REJECTED for o in result.orders)
        assert any(o.reject_reason == "stale_bar" for o in result.orders)


class TestContinuousNotExecutable:
    def test_continuous_research_bar_rejected(self) -> None:
        with pytest.raises(ValueError, match="Continuous research"):
            BarEvent(
                timestamp=_ts("2024-01-02 08:30"),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1000,
                symbol="ES",
                contract="ES_CONT",
                bar_end=_ts("2024-01-02 08:35"),
                meta={"is_continuous_research": True},
            )

    def test_fill_requires_explicit_contract(self) -> None:
        timing = InformationTiming(
            source_timestamp=_ts("2024-01-02 08:30"),
            availability_timestamp=_ts("2024-01-02 08:35"),
            decision_timestamp=_ts("2024-01-02 08:35"),
        )
        with pytest.raises(ValueError, match="tradable contract"):
            OrderCls(
                symbol="ES",
                contract="",
                side=Side.BUY,
                quantity=1.0,
                order_type=OrderType.MARKET,
                timing=timing,
            )


class TestBarsFromFrame:
    def test_bars_from_validated_frame(self) -> None:
        idx = pd.date_range("2024-01-02 08:30", periods=3, freq="5min", tz=TZ)
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
                "volume": [100, 100, 100],
            },
            index=idx,
        )
        events = bars_from_frame(df, symbol="ES", contract="ESH24", freq="5min")
        assert len(events) == 3
        assert events[0].contract == "ESH24"
        assert events[0].bar_end == idx[0] + pd.Timedelta(minutes=5)
