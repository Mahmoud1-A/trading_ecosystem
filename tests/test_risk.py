from datetime import datetime, timezone

from trading_ecosystem.common.contracts import AccountState, OrderIntent, OrderSide
from trading_ecosystem.risk.master import MasterRiskManager


def _account(equity: float = 100_000.0) -> AccountState:
    return AccountState(
        equity=equity,
        cash=equity,
        starting_equity=equity,
        peak_equity=equity,
        day_start_equity=equity,
        trading_day=datetime.now(timezone.utc).date(),
    )


def test_rejects_when_daily_dd_breached():
    mrm = MasterRiskManager()
    acct = _account()
    acct.equity = 94_000  # 6% daily DD vs 5% limit
    intent = OrderIntent(
        strategy_id="s1",
        symbol="SPY",
        side=OrderSide.BUY,
        qty=10,
        ts=datetime(2023, 6, 15, 15, 0, tzinfo=timezone.utc),
        stop_price=400.0,
    )
    decision = mrm.evaluate(intent, acct, mark_price=420.0)
    assert decision.approved is False
    assert "daily_dd" in decision.reason


def test_reduce_only_allowed_on_daily_halt():
    mrm = MasterRiskManager()
    acct = _account()
    acct.daily_halted = True
    intent = OrderIntent(
        strategy_id="s1",
        symbol="SPY",
        side=OrderSide.SELL,
        qty=10,
        ts=datetime(2023, 6, 15, 15, 0, tzinfo=timezone.utc),
        reduce_only=True,
    )
    decision = mrm.evaluate(intent, acct, mark_price=420.0)
    assert decision.approved is True


def test_slot_cap_keeps_room_for_forty_positions():
    """With 40 max positions and 100% gross, each new entry ~2.5% equity."""
    mrm = MasterRiskManager()
    acct = _account(100_000.0)
    intent = OrderIntent(
        strategy_id="s1",
        symbol="EEM",
        side=OrderSide.BUY,
        qty=50_000,  # would be huge without caps
        ts=datetime(2023, 6, 15, 15, 0, tzinfo=timezone.utc),
        stop_price=63.0,
    )
    decision = mrm.evaluate(intent, acct, mark_price=64.0)
    assert decision.approved is True
    qty = decision.adjusted_qty if decision.adjusted_qty is not None else intent.qty
    # 2.5% of 100k = 2500 notional → ~39 shares at $64
    assert qty <= 40
    assert qty * 64.0 <= 100_000.0 * 0.026


def test_rejects_when_buying_power_zero():
    mrm = MasterRiskManager()
    acct = _account()
    acct.buying_power = 0.0
    intent = OrderIntent(
        strategy_id="s1",
        symbol="SPY",
        side=OrderSide.BUY,
        qty=10,
        ts=datetime(2023, 6, 15, 15, 0, tzinfo=timezone.utc),
        stop_price=400.0,
    )
    decision = mrm.evaluate(intent, acct, mark_price=420.0)
    assert decision.approved is False
    assert "buying_power" in decision.reason
