"""
Phase 5.5-F acceptance tests — futures rollover.

Two invariants are enforced here:

1. Fills always reference an explicit tradable contract; the continuous
   back-adjusted series is a signal input only.
2. A synthetic back-adjustment jump can never become tradable PnL. Rolling a
   position between a front month at 100 and a back month at 110 must leave
   equity untouched even though the continuous series jumps by 10 points.
"""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.asset_spec import default_es_futures
from config.cost_model import CostModel
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.friction import FrictionEngine
from engine.ids import DeterministicIdFactory
from engine.portfolio import Portfolio
from engine.rollover import (
    RolloverDecision,
    continuous_gap_not_tradable_pnl,
    execute_futures_rollover,
)


CHI = ZoneInfo("America/Chicago")
MULT = 50.0
FRONT = "ESH24"
BACK = "ESM24"


def _zero_cost() -> CostModel:
    return CostModel(
        version="roll_zero",
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
        futures_rollover_cost_ticks=0.0,
        participation_rate_cap=1.0,
    )


def _ts(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-03-01 09:00", tz=CHI) + pd.Timedelta(minutes=minute)


def _bar(
    i: int,
    price: float,
    contract: str,
    *,
    meta: dict[str, object] | None = None,
) -> BarEvent:
    return BarEvent(
        timestamp=_ts(i * 5),
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=50_000.0,
        symbol="ES",
        contract=contract,
        bar_end=_ts(i * 5 + 5),
        meta=dict(meta or {}),
    )


def _engine(*, cost: CostModel | None = None) -> EventExecutionEngine:
    return EventExecutionEngine(
        default_es_futures(FRONT),
        cost or _zero_cost(),
        starting_equity=100_000.0,
        exec_config=ExecutionConfig(latency=timedelta(0), random_seed=7),
        id_factory=DeterministicIdFactory(run_id="roll", candidate_id="c1", fold_id=0),
    )


def _roll_bars(front_price: float = 100.0, back_price: float = 110.0) -> list[BarEvent]:
    """
    Four front-month bars at ``front_price`` then four back-month bars at
    ``back_price``. The roll bar carries the two explicit roll prices.
    """
    bars = [_bar(i, front_price, FRONT) for i in range(4)]
    bars.append(
        _bar(
            4,
            back_price,
            BACK,
            meta={"roll_front_price": front_price, "roll_back_price": back_price},
        )
    )
    bars.extend(_bar(i, back_price, BACK) for i in range(5, 8))
    return bars


def _buy_on_first_bar(qty: float = 2.0):
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
            quantity=qty,
            signal_id="entry",
        )

    return signal_fn


def _portfolio(qty: float, price: float, contract: str = FRONT) -> Portfolio:
    p = Portfolio(starting_equity=100_000.0)
    p.set_multiplier("ES", MULT)
    pos = p.get_position("ES")
    pos.quantity = qty
    pos.avg_price = price
    pos.multiplier = MULT
    pos.contract = contract
    p.last_marks["ES"] = price
    return p


class TestRolloverPrimitive:
    def test_long_roll_moves_contract_and_reprices_at_back_month(self) -> None:
        p = _portfolio(2.0, 100.0)
        friction = FrictionEngine(_zero_cost(), default_es_futures(FRONT), seed=0)
        fills, decision = execute_futures_rollover(
            p,
            asset=default_es_futures(FRONT),
            symbol="ES",
            from_contract=FRONT,
            to_contract=BACK,
            timestamp=_ts(20),
            front_price=100.0,
            back_price=110.0,
            friction=friction,
        )
        assert [f.contract for f in fills] == [FRONT, BACK]
        assert [f.side.value for f in fills] == ["SELL", "BUY"]
        pos = p.get_position("ES")
        assert pos.quantity == pytest.approx(2.0)
        assert pos.avg_price == pytest.approx(110.0)
        assert pos.contract == BACK
        assert isinstance(decision, RolloverDecision)
        assert decision.exclude_from_tradable_pnl is True
        assert decision.from_contract == FRONT and decision.to_contract == BACK

    def test_short_roll_preserves_signed_exposure(self) -> None:
        p = _portfolio(-3.0, 100.0)
        friction = FrictionEngine(_zero_cost(), default_es_futures(FRONT), seed=0)
        fills, _ = execute_futures_rollover(
            p,
            asset=default_es_futures(FRONT),
            symbol="ES",
            from_contract=FRONT,
            to_contract=BACK,
            timestamp=_ts(20),
            front_price=100.0,
            back_price=95.0,
            friction=friction,
        )
        assert [f.side.value for f in fills] == ["BUY", "SELL"]
        pos = p.get_position("ES")
        assert pos.quantity == pytest.approx(-3.0)
        assert pos.avg_price == pytest.approx(95.0)
        assert pos.contract == BACK

    def test_flat_position_is_not_rolled(self) -> None:
        p = _portfolio(0.0, 0.0)
        friction = FrictionEngine(_zero_cost(), default_es_futures(FRONT), seed=0)
        fills, decision = execute_futures_rollover(
            p,
            asset=default_es_futures(FRONT),
            symbol="ES",
            from_contract=FRONT,
            to_contract=BACK,
            timestamp=_ts(20),
            front_price=100.0,
            back_price=110.0,
            friction=friction,
        )
        assert fills == []
        assert decision.from_contract == FRONT
        assert p.cash == pytest.approx(100_000.0)

    def test_roll_fills_are_tagged_for_pnl_exclusion(self) -> None:
        p = _portfolio(1.0, 100.0)
        friction = FrictionEngine(_zero_cost(), default_es_futures(FRONT), seed=0)
        fills, _ = execute_futures_rollover(
            p,
            asset=default_es_futures(FRONT),
            symbol="ES",
            from_contract=FRONT,
            to_contract=BACK,
            timestamp=_ts(20),
            front_price=100.0,
            back_price=110.0,
            friction=friction,
        )
        assert all(f.meta["rollover"] is True for f in fills)
        assert all(f.meta["exclude_continuous_gap_pnl"] is True for f in fills)
        assert [f.meta["leg"] for f in fills] == ["close", "open"]
        assert all(f.rollover_decision == f"{FRONT}->{BACK}" for f in fills)

    def test_rollover_friction_is_attributed_to_rollover_cost(self) -> None:
        cost = _zero_cost().model_copy(update={"futures_rollover_cost_ticks": 1.0})
        p = _portfolio(2.0, 100.0)
        friction = FrictionEngine(cost, default_es_futures(FRONT), seed=0)
        fills, _ = execute_futures_rollover(
            p,
            asset=default_es_futures(FRONT),
            symbol="ES",
            from_contract=FRONT,
            to_contract=BACK,
            timestamp=_ts(20),
            front_price=100.0,
            back_price=110.0,
            friction=friction,
        )
        # 1 tick * 0.25 tick size * 2 contracts * 50 multiplier = 25.0, close leg only
        assert fills[0].costs.rollover_cost == pytest.approx(25.0)
        assert fills[1].costs.rollover_cost == pytest.approx(0.0)
        assert p.cash == pytest.approx(100_000.0 - 25.0)


class TestSyntheticJumpIsNotTradablePnl:
    def test_roll_across_ten_point_contango_creates_no_pnl(self) -> None:
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 110.0), _buy_on_first_bar(2.0))
        pos = result.portfolio.get_position("ES")

        assert len(result.rollover_decisions) == 1
        assert pos.quantity == pytest.approx(2.0)
        assert pos.contract == BACK
        assert pos.avg_price == pytest.approx(110.0)
        # Entry at 100 on the front month, roll at 100/110, back month flat at 110:
        # the 10-point continuous jump contributes nothing.
        assert result.portfolio.cash == pytest.approx(100_000.0)
        assert result.portfolio.equity() == pytest.approx(100_000.0)
        assert result.portfolio.equity_curve[-1][1] == pytest.approx(100_000.0)

    def test_equity_is_flat_across_the_roll_bar(self) -> None:
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 110.0), _buy_on_first_bar(2.0))
        equities = [eq for _, eq in result.portfolio.equity_curve]
        assert all(eq == pytest.approx(100_000.0) for eq in equities)

    def test_backwardation_roll_also_creates_no_pnl(self) -> None:
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 88.0), _buy_on_first_bar(2.0))
        assert result.portfolio.get_position("ES").avg_price == pytest.approx(88.0)
        assert result.portfolio.equity() == pytest.approx(100_000.0)

    def test_naive_continuous_mtm_would_have_booked_the_jump(self) -> None:
        """Contrast: the same 10-point jump on a single contract is 1,000 of fake PnL."""
        artificial = continuous_gap_not_tradable_pnl(
            unadjusted_front=100.0,
            unadjusted_back=110.0,
            continuous_adjusted_jump=(110.0 - 100.0) * 2 * MULT,
        )
        assert artificial == pytest.approx(1_000.0)
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 110.0), _buy_on_first_bar(2.0))
        assert result.portfolio.equity() - 100_000.0 == pytest.approx(0.0)
        assert result.portfolio.equity() - 100_000.0 != pytest.approx(artificial)

    def test_real_post_roll_move_is_still_captured(self) -> None:
        bars = _roll_bars(100.0, 110.0)
        # Back month rallies 2 points after the roll settles
        bars[-1] = _bar(7, 112.0, BACK)
        engine = _engine()
        result = engine.run(bars, _buy_on_first_bar(2.0))
        # (112 - 110) * 2 * 50 = 200 of genuine PnL on the back contract
        assert result.portfolio.equity() == pytest.approx(100_200.0)

    def test_roll_trade_report_has_zero_gross_pnl(self) -> None:
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 110.0), _buy_on_first_bar(2.0))
        roll_trades = [t for t in result.portfolio.trades if t.contract == FRONT]
        assert len(roll_trades) == 1
        assert roll_trades[0].gross_pnl == pytest.approx(0.0)
        assert roll_trades[0].net_pnl == pytest.approx(0.0)


class TestExplicitContracts:
    def test_all_fills_carry_the_contract_that_was_tradable(self) -> None:
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 110.0), _buy_on_first_bar(2.0))
        contracts = [f.contract for f in result.fills]
        assert contracts == [FRONT, FRONT, BACK]
        assert all(c.strip() for c in contracts)

    def test_orders_after_the_roll_target_the_back_contract(self) -> None:
        bars = _roll_bars(100.0, 110.0)

        def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
            timing = InformationTiming(
                source_timestamp=bar.timestamp,
                availability_timestamp=bar.bar_end,
                decision_timestamp=bar.bar_end,
            )
            if i == 0:
                return SignalEvent(
                    timing=timing, symbol="ES", side="BUY", quantity=1.0, signal_id="in"
                )
            if i == 5:
                return SignalEvent(
                    timing=timing, symbol="ES", side="BUY", quantity=1.0, signal_id="add"
                )
            return None

        engine = _engine()
        result = engine.run(bars, signal_fn)
        entry = next(o for o in result.orders if o.parent_signal_id == "in")
        add = next(o for o in result.orders if o.parent_signal_id == "add")
        assert entry.contract == FRONT
        assert add.contract == BACK
        post_roll = [f for f in result.fills if f.timing.fill_timestamp >= _ts(25)]
        assert post_roll
        assert all(f.contract == BACK for f in post_roll)

    def test_continuous_research_bars_are_rejected_as_executable(self) -> None:
        with pytest.raises(ValueError, match="not executable"):
            _bar(0, 100.0, FRONT, meta={"is_continuous_research": True})

    def test_no_roll_when_contract_never_changes(self) -> None:
        bars = [_bar(i, 100.0, FRONT) for i in range(6)]
        engine = _engine()
        result = engine.run(bars, _buy_on_first_bar(1.0))
        assert result.rollover_decisions == []
        assert all(f.contract == FRONT for f in result.fills)

    def test_roll_is_skipped_while_flat(self) -> None:
        engine = _engine()
        result = engine.run(_roll_bars(100.0, 110.0), None)
        assert result.rollover_decisions == []
        assert result.portfolio.cash == pytest.approx(100_000.0)

    def test_roll_can_be_disabled(self) -> None:
        engine = EventExecutionEngine(
            default_es_futures(FRONT),
            _zero_cost(),
            starting_equity=100_000.0,
            exec_config=ExecutionConfig(
                latency=timedelta(0), random_seed=7, auto_roll_futures=False
            ),
            id_factory=DeterministicIdFactory(run_id="roll", candidate_id="c1", fold_id=0),
        )
        result = engine.run(_roll_bars(100.0, 110.0), _buy_on_first_bar(2.0))
        assert result.rollover_decisions == []
        # Without the roll the position is still marked on the front month's basis,
        # so the synthetic jump leaks into equity — exactly what the roll prevents.
        assert result.portfolio.equity() == pytest.approx(101_000.0)
