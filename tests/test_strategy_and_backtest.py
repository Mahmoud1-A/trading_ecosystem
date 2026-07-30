from datetime import datetime, timedelta, timezone

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.donchian_atr import DonchianATR


def _synth_bars(n: int = 120, breakout_at: int = 80) -> list[Bar]:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars: list[Bar] = []
    price = 100.0
    for i in range(n):
        if i < breakout_at:
            # range-bound
            o = price
            h = price + 1.0
            l = price - 1.0
            c = price + (0.1 if i % 2 == 0 else -0.1)
            price = c
        else:
            # strong uptrend breakout
            o = price
            c = price + 2.5
            h = c + 0.5
            l = o - 0.2
            price = c
        bars.append(
            Bar(
                symbol="TEST",
                ts=start + timedelta(days=i),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1_000_000,
                timeframe="1d",
            )
        )
    return bars


def test_donchian_emits_entry_after_breakout():
    strat = DonchianATR(
        "t1",
        ["TEST"],
        {"channel_period": 20, "atr_period": 14, "atr_stop_mult": 2.5, "risk_fraction": 0.01},
    )
    bars = _synth_bars()
    entered = False
    pos = None
    for bar in bars:
        intents = strat.on_bar(bar, pos)
        if intents and intents[0].side.value == "buy":
            entered = True
            break
    assert entered


def test_backtest_runs_end_to_end():
    strat = DonchianATR(
        "t1",
        ["TEST"],
        {"channel_period": 20, "atr_period": 14, "atr_stop_mult": 2.5, "risk_fraction": 0.01},
    )
    engine = BacktestEngine([strat], MasterRiskManager(), starting_equity=100_000)
    result = engine.run({"TEST": _synth_bars()})
    assert result.equity_curve
    assert result.metrics.num_trades >= 0
    assert isinstance(result.metrics.cagr, float)
