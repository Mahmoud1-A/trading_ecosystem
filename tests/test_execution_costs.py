from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.contracts import OrderSide
from trading_ecosystem.execution.costs import ExecutionCosts, load_execution_costs
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.donchian_atr import DonchianATR
from tests.test_strategy_and_backtest import _synth_bars


def test_load_cost_profiles():
    etf = load_execution_costs(profile="etf")
    crypto = load_execution_costs(profile="crypto")
    assert etf.taker_fee_bps == 0.0
    assert crypto.taker_fee_bps == 25.0
    assert crypto.maker_fee_bps == 15.0
    assert crypto.slippage_bps >= etf.slippage_bps


def test_taker_fee_and_slippage_reduce_cash():
    costs = ExecutionCosts(
        slippage_bps=10.0,
        maker_fee_bps=0.0,
        taker_fee_bps=25.0,
        default_liquidity="taker",
        profile="test",
    )
    mid = 100.0
    buy_px = costs.fill_price(mid, OrderSide.BUY, "taker")
    assert buy_px > mid
    fee = costs.commission(10.0, buy_px, "taker")
    assert abs(fee - (10.0 * buy_px * 0.0025)) < 1e-9

    maker_px = costs.fill_price(mid, OrderSide.BUY, "maker")
    assert mid < maker_px < buy_px
    maker_fee = costs.commission(10.0, maker_px, "maker")
    assert maker_fee < fee


def test_backtest_with_crypto_fees_lower_equity_than_zero_cost():
    strat = DonchianATR(
        "t1",
        ["TEST"],
        {"channel_period": 20, "atr_period": 14, "atr_stop_mult": 2.5, "risk_fraction": 0.01},
    )
    bars = {"TEST": _synth_bars()}
    free = BacktestEngine(
        [strat],
        MasterRiskManager(),
        starting_equity=100_000,
        costs=ExecutionCosts(slippage_bps=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
    ).run(bars)
    costly = BacktestEngine(
        [
            DonchianATR(
                "t1",
                ["TEST"],
                {"channel_period": 20, "atr_period": 14, "atr_stop_mult": 2.5, "risk_fraction": 0.01},
            )
        ],
        MasterRiskManager(),
        starting_equity=100_000,
        costs=load_execution_costs(profile="crypto"),
    ).run(bars)
    if free.fills and costly.fills:
        free_eq = free.equity_curve[-1][1]
        cost_eq = costly.equity_curve[-1][1]
        assert cost_eq <= free_eq + 1e-6
        assert any(f.commission > 0 for f in costly.fills)
