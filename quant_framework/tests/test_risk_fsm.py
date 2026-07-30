"""Phase 3 acceptance tests — prop-firm risk FSM."""

from __future__ import annotations

from datetime import time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config import default_es_futures, default_prop_profile
from config.cost_model import CostModel
from config.models import OrderStatus, OrderType, RiskState, Side
from config.prop_profile import PropProfile
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.orders import Order
from engine.portfolio import Portfolio
from risk import PropRiskFSM, is_risk_increasing, is_risk_reducing


TZ = ZoneInfo("America/New_York")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def _profile(**overrides: object) -> PropProfile:
    base = default_prop_profile("challenge")
    data = base.model_dump()
    data.update(overrides)
    return PropProfile(**data)


def _portfolio(equity: float = 100_000.0) -> Portfolio:
    p = Portfolio(starting_equity=equity)
    p.cash = equity
    return p


def _order(side: Side, qty: float = 1.0, *, reduce_only: bool = False) -> Order:
    timing = InformationTiming(
        source_timestamp=_ts("2024-01-02 10:00"),
        availability_timestamp=_ts("2024-01-02 10:05"),
        decision_timestamp=_ts("2024-01-02 10:05"),
    )
    return Order(
        symbol="ES",
        contract="ESH24",
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        timing=timing,
        reduce_only=reduce_only,
    )


def _zero_cost_model() -> CostModel:
    return CostModel(
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


def _bar(open_ts: str, close: float) -> BarEvent:
    start = _ts(open_ts)
    return BarEvent(
        timestamp=start,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10_000,
        symbol="ES",
        contract="ESH24",
        bar_end=start + pd.Timedelta(minutes=5),
    )


class TestSoftBreachTransitions:
    def test_normal_to_caution_to_reduce_only(self) -> None:
        profile = _profile(
            internal_soft_daily_limit=-0.01,
            prop_hard_daily_loss_limit=-0.015,
            soft_to_reduce_only_ratio=0.85,
        )
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)

        # -0.8% — still NORMAL
        p = _portfolio(99_200.0)
        fsm.on_mark(p, _ts("2024-01-02 10:00"))
        assert fsm.state == RiskState.NORMAL

        # -1.0% — CAUTION
        p.cash = 99_000.0
        fsm.on_mark(p, _ts("2024-01-02 10:05"))
        assert fsm.state == RiskState.CAUTION

        # -1.425% — progress 0.85 -> REDUCE_ONLY
        p.cash = 98_575.0
        fsm.on_mark(p, _ts("2024-01-02 10:10"))
        assert fsm.state == RiskState.REDUCE_ONLY

        transitions = [e.to_state for e in fsm.events]
        assert RiskState.CAUTION in transitions
        assert RiskState.REDUCE_ONLY in transitions


class TestReduceOnlyGating:
    def test_rejects_risk_increasing_allows_reducing(self) -> None:
        profile = _profile()
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        fsm.state = RiskState.REDUCE_ONLY

        portfolio = _portfolio(100_000.0)
        pos = portfolio.get_position("ES")
        pos.quantity = 2.0
        pos.avg_price = 100.0
        pos.contract = "ESH24"

        buy = _order(Side.BUY, 1.0)
        sell_reduce = _order(Side.SELL, 1.0, reduce_only=True)
        flat = _order(Side.SELL, 2.0, reduce_only=True)

        assert is_risk_increasing(buy, portfolio)
        assert is_risk_reducing(sell_reduce, portfolio)

        rej = fsm.evaluate_order(buy, portfolio)
        assert rej.allowed is False
        assert rej.reason == "reduce_only"

        ok = fsm.evaluate_order(sell_reduce, portfolio)
        assert ok.allowed is True

        ok_flat = fsm.evaluate_order(flat, portfolio)
        assert ok_flat.allowed is True

    def test_normal_flat_exit_allowed_when_symbol_exposure_full(self) -> None:
        """Regression: full max_symbol_exposure must not block reduce-only FLAT.

        Pre-fix, size_by_risk_budget returned exposure_cap with remaining=0,
        so exits never filled → n_trades=0 while open MTM invented MaxDD.
        """
        profile = _profile(max_symbol_exposure=1.0, max_portfolio_exposure=1.0)
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        assert fsm.state == RiskState.NORMAL

        portfolio = _portfolio(100_000.0)
        pos = portfolio.get_position("ES")
        pos.quantity = 1.0
        pos.avg_price = 100.0
        pos.contract = "ESH24"

        # New risk-increasing entry must still be blocked
        buy = _order(Side.BUY, 1.0)
        rej = fsm.evaluate_order(buy, portfolio)
        assert rej.allowed is False

        flat = _order(Side.SELL, 1.0, reduce_only=True)
        ok = fsm.evaluate_order(flat, portfolio)
        assert ok.allowed is True
        assert ok.adjusted_quantity == pytest.approx(1.0)
        assert ok.reason == "risk_reducing_exit"


class TestHardBreachFlattenHalt:
    def test_hard_breach_flattens_then_halts(self) -> None:
        profile = _profile(prop_hard_daily_loss_limit=-0.015)
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio(98_400.0)  # -1.6% daily
        fsm.on_mark(p, _ts("2024-01-02 11:00"))
        assert fsm.state == RiskState.FLATTEN

        # Simulate flat book
        pos = p.get_position("ES")
        pos.quantity = 0.0
        fsm.complete_flatten(_ts("2024-01-02 11:05"), p)
        assert fsm.state == RiskState.HALTED

    def test_engine_cancels_pending_and_flattens_on_hard_breach(self) -> None:
        """
        Full hard-breach sequence driven by a real market loss (no injected cash).

        ES multiplier is 50, so a 1-contract long entered at 100 and marked at 68
        loses 1,600 = -1.6% of 100,000, breaching the -1.5% hard daily limit.
        """
        profile = _profile(
            prop_hard_daily_loss_limit=-0.015,
            internal_soft_daily_limit=-0.010,
            max_symbol_exposure=5.0,
            max_portfolio_exposure=10.0,
            include_unrealized_pnl=True,
        )
        risk = PropRiskFSM(profile, starting_equity=100_000.0)
        engine = EventExecutionEngine(
            default_es_futures(),
            _zero_cost_model(),
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
            risk_manager=risk,
        )

        bars = [
            _bar("2024-01-02 09:30", 100.0),
            _bar("2024-01-02 09:35", 100.0),
            _bar("2024-01-02 09:40", 68.0),  # -1.6% unrealized -> hard breach
            _bar("2024-01-02 09:45", 68.0),  # flatten market order fills here
            _bar("2024-01-02 09:50", 68.0),  # post-halt signal must be refused
            _bar("2024-01-02 09:55", 68.0),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> list[SignalEvent] | None:
            timing = InformationTiming(
                source_timestamp=bar.timestamp,
                availability_timestamp=bar.bar_end,
                decision_timestamp=bar.bar_end,
            )
            if i == 0:
                return [
                    SignalEvent(timing=timing, symbol="ES", side="BUY", quantity=1.0),
                    # Resting limit far below the market: still pending at breach time
                    SignalEvent(
                        timing=timing,
                        symbol="ES",
                        side="BUY",
                        quantity=1.0,
                        order_type="LIMIT",
                        limit_price=50.0,
                        reason="resting_pending",
                    ),
                ]
            if i == 4:
                # Attempted re-entry after HALTED
                return [SignalEvent(timing=timing, symbol="ES", side="BUY", quantity=1.0)]
            return None

        result = engine.run(bars, signal_fn)

        # 1) Transitions pass through FLATTEN and settle in HALTED
        states = [e.to_state for e in risk.events]
        assert RiskState.FLATTEN in states
        assert RiskState.HALTED in states
        assert states.index(RiskState.FLATTEN) < states.index(RiskState.HALTED)
        assert risk.state == RiskState.HALTED

        # 2) The pending resting limit order was cancelled, not left working
        limits = [o for o in result.orders if o.order_type == OrderType.LIMIT]
        assert len(limits) == 1
        assert limits[0].status == OrderStatus.CANCELLED
        assert limits[0].reject_reason == "hard_breach_cancel"
        assert engine._working == []
        assert not [
            o
            for o in result.orders
            if o.status
            in {OrderStatus.SUBMITTED, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}
        ]

        # 3) Book is flat
        assert result.portfolio.get_position("ES").quantity == pytest.approx(0.0)
        assert all(p.is_flat for p in result.portfolio.positions.values())

        # 4) Flatten fills exist and closed the long
        flatten_orders = [o for o in result.orders if o.meta.get("flatten")]
        assert len(flatten_orders) == 1
        flatten_ids = {o.order_id for o in flatten_orders}
        flatten_fills = [f for f in result.fills if f.order_id in flatten_ids]
        assert len(flatten_fills) == 1
        assert flatten_fills[0].side == Side.SELL
        assert flatten_fills[0].quantity == pytest.approx(1.0)
        assert flatten_orders[0].status == OrderStatus.FILLED

        # 5) No new order is accepted once HALTED
        assert any(r["reason"] == "risk_halted" for r in result.rejected_signals)
        entry_orders = [
            o
            for o in result.orders
            if o.order_type == OrderType.MARKET and not o.meta.get("flatten")
        ]
        assert len(entry_orders) == 1  # only the pre-breach entry

        # 6) Equity and realized PnL reconcile exactly (zero-cost model)
        realized = sum(t.net_pnl for t in result.portfolio.trades)
        assert realized == pytest.approx((68.0 - 100.0) * 50.0)
        assert result.portfolio.cash == pytest.approx(100_000.0 + realized)
        assert result.portfolio.equity() == pytest.approx(result.portfolio.cash)
        assert result.portfolio.equity() == pytest.approx(98_400.0)
        assert risk.daily_pnl_pct(result.portfolio.equity()) == pytest.approx(-0.016)

    def test_no_pending_order_survives_hard_breach_cancel(self) -> None:
        profile = _profile(prop_hard_daily_loss_limit=-0.015, cancel_pending_orders_on_hard_breach=True)
        risk = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio(98_400.0)
        risk.on_mark(p, _ts("2024-01-02 11:00"))
        assert risk.state == RiskState.FLATTEN
        assert risk.should_cancel_pending() is True
        assert risk.should_flatten_positions() is True


class TestUnrealizedPnlModes:
    """P5.5-B: the same market move must be judged differently under each mode."""

    @staticmethod
    def _run(include_unrealized: bool):
        profile = _profile(
            prop_hard_daily_loss_limit=-0.015,
            internal_soft_daily_limit=-0.010,
            max_symbol_exposure=5.0,
            max_portfolio_exposure=10.0,
            include_unrealized_pnl=include_unrealized,
        )
        risk = PropRiskFSM(profile, starting_equity=100_000.0)
        engine = EventExecutionEngine(
            default_es_futures(),
            _zero_cost_model(),
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(latency=timedelta(0), random_seed=0),
            risk_manager=risk,
        )
        bars = [
            _bar("2024-01-02 09:30", 100.0),
            _bar("2024-01-02 09:35", 100.0),
            _bar("2024-01-02 09:40", 68.0),
            _bar("2024-01-02 09:45", 68.0),
            _bar("2024-01-02 09:50", 68.0),
        ]

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
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
            )

        result = engine.run(bars, signal_fn)
        return risk, result

    def test_unrealized_included_breaches_and_halts(self) -> None:
        risk, result = self._run(True)
        assert risk.state == RiskState.HALTED
        assert result.portfolio.get_position("ES").quantity == pytest.approx(0.0)
        assert risk.mark_equity(result.portfolio) == pytest.approx(98_400.0)

    def test_unrealized_excluded_does_not_breach(self) -> None:
        risk, result = self._run(False)
        # Cash is untouched by the 1,600 mark-to-market loss under this mode
        assert result.portfolio.cash == pytest.approx(100_000.0)
        assert risk.mark_equity(result.portfolio) == pytest.approx(100_000.0)
        assert risk.state == RiskState.NORMAL
        # Position stays open — no flatten was triggered
        assert result.portfolio.get_position("ES").quantity == pytest.approx(1.0)
        # But the true marked equity is still down: the mode choice, not the market,
        # is what suppressed the breach.
        assert result.portfolio.equity() == pytest.approx(98_400.0)

    def test_mark_equity_and_budgets_follow_the_mode(self) -> None:
        profile_incl = _profile(include_unrealized_pnl=True, prop_hard_daily_loss_limit=-0.015)
        profile_excl = _profile(include_unrealized_pnl=False, prop_hard_daily_loss_limit=-0.015)
        p = _portfolio(100_000.0)
        pos = p.get_position("ES")
        pos.quantity = 1.0
        pos.avg_price = 100.0
        pos.multiplier = 50.0
        pos.contract = "ESH24"
        p.set_multiplier("ES", 50.0)
        p.last_marks["ES"] = 68.0

        incl = PropRiskFSM(profile_incl, starting_equity=100_000.0)
        excl = PropRiskFSM(profile_excl, starting_equity=100_000.0)
        assert incl.mark_equity(p) == pytest.approx(98_400.0)
        assert excl.mark_equity(p) == pytest.approx(100_000.0)

        # Hard budget room is consumed only when unrealized PnL counts
        assert incl.remaining_hard_daily_budget(p) == pytest.approx(0.0)
        assert excl.remaining_hard_daily_budget(p) == pytest.approx(1_500.0)
        assert excl.remaining_soft_daily_budget(p) == pytest.approx(1_000.0)


class TestManualLockAndReset:
    def test_manual_lock_blocks_trading(self) -> None:
        profile = _profile()
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio()
        fsm.set_manual_lock(_ts("2024-01-02 12:00"), p)
        assert fsm.state == RiskState.MANUAL_LOCK

        dec = fsm.evaluate_order(_order(Side.BUY), p)
        assert dec.allowed is False
        assert dec.reason == "manual_lock"

        # Does not auto-resume on mark
        fsm.on_mark(p, _ts("2024-01-02 13:00"))
        assert fsm.state == RiskState.MANUAL_LOCK

    def test_authorized_reset_required_to_resume(self) -> None:
        profile = _profile()
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio()
        fsm.set_manual_lock(_ts("2024-01-02 12:00"), p)
        assert fsm.authorized_reset(_ts("2024-01-02 12:05"), p) is True
        assert fsm.state == RiskState.NORMAL

    def test_halted_requires_authorized_reset(self) -> None:
        profile = _profile(prop_hard_daily_loss_limit=-0.015)
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio(98_400.0)
        fsm.on_mark(p, _ts("2024-01-02 11:00"))
        fsm.complete_flatten(_ts("2024-01-02 11:05"), p)
        assert fsm.state == RiskState.HALTED

        # Still breached — reset denied
        assert fsm.authorized_reset(_ts("2024-01-02 11:10"), p) is False

        # Recover equity above hard limit
        p.cash = 99_000.0
        assert fsm.authorized_reset(_ts("2024-01-02 11:15"), p) is True
        assert fsm.state == RiskState.NORMAL


class TestDailyReset:
    def test_daily_reset_at_configured_timezone_and_time(self) -> None:
        profile = _profile(
            daily_reset_timezone="America/New_York",
            daily_reset_time=time(17, 0),
            internal_soft_daily_limit=-0.01,
        )
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio(99_000.0)

        # Before reset — CAUTION
        fsm.on_mark(p, _ts("2024-01-02 16:00"))
        assert fsm.state == RiskState.CAUTION
        assert fsm.daily_pnl_pct(p.equity()) == pytest.approx(-0.01)

        # Cross 17:00 ET reset boundary (same calendar day after reset)
        fsm.on_mark(p, _ts("2024-01-02 17:05"))
        assert fsm.daily_pnl_pct(p.equity()) == pytest.approx(0.0)
        assert fsm.state == RiskState.NORMAL
        assert any(e.reason.value == "daily_reset" for e in fsm.events)

    def test_reset_only_on_boundary_not_every_bar(self) -> None:
        profile = _profile(daily_reset_timezone="America/New_York", daily_reset_time=time(17, 0))
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        p = _portfolio(100_000.0)
        fsm.on_mark(p, _ts("2024-01-02 10:00"))
        resets_before = sum(1 for e in fsm.events if e.reason.value == "daily_reset")
        fsm.on_mark(p, _ts("2024-01-02 10:05"))
        resets_after = sum(1 for e in fsm.events if e.reason.value == "daily_reset")
        assert resets_after == resets_before


class TestCautionSizeReduction:
    def test_caution_reduces_allowed_risk(self) -> None:
        profile = _profile(internal_risk_budget=0.5, max_symbol_exposure=10.0, max_portfolio_exposure=10.0)
        fsm = PropRiskFSM(profile, starting_equity=100_000.0)
        fsm.state = RiskState.CAUTION
        p = _portfolio()

        dec = fsm.evaluate_order(_order(Side.BUY, 4.0), p)
        assert dec.allowed is True
        assert dec.adjusted_quantity == pytest.approx(2.0)
        assert dec.reason == "caution_size_reduction"
