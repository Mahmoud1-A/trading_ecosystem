"""GapFade must not fill using the same open that formed the gap signal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.execution.costs import ExecutionCosts
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.gap_fade import GapFade


def _gap_panel(n: int = 80) -> dict[str, list[Bar]]:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars: list[Bar] = []
    px = 100.0
    for i in range(n):
        # Create a clear down-gap on day 50
        if i == 50:
            o = px * 0.97
            c = o * 1.01
        else:
            o = px
            c = px * 1.001
        bars.append(
            Bar(
                symbol="TEST",
                ts=start + timedelta(days=i),
                open=o,
                high=max(o, c) * 1.01,
                low=min(o, c) * 0.99,
                close=c,
                volume=1e6,
                timeframe="1d",
            )
        )
        px = c
    return {"TEST": bars}


def test_gapfade_signal_does_not_use_close_for_stop_on_entry():
    strat = GapFade(
        "gf1",
        ["TEST"],
        {"gap_pct": 0.01, "hold_bars": 5, "atr_period": 14, "atr_stop_mult": 1.5, "risk_fraction": 0.01},
    )
    panel = _gap_panel()
    # Warm history
    for b in panel["TEST"][:49]:
        strat.on_bar(b, None)
    sig_bar = panel["TEST"][50]
    # Manually append path via on_bar
    intents = strat.on_bar(sig_bar, None)
    assert intents, "expected entry intent on gap day"
    intent = intents[0]
    assert intent.meta.get("execution_model") == "post_open"
    assert intent.meta.get("fill_mode") == "next_bar_open"
    assert intent.stop_price is not None
    assert intent.stop_price < sig_bar.open


def test_engine_defers_gapfade_entry_to_next_open():
    strat = GapFade(
        "gf1",
        ["TEST"],
        {"gap_pct": 0.01, "hold_bars": 5, "atr_period": 14, "atr_stop_mult": 1.5, "risk_fraction": 0.01},
    )
    panel = _gap_panel()
    result = BacktestEngine(
        strategies=[strat],
        risk_manager=MasterRiskManager(),
        starting_equity=100_000,
        costs=ExecutionCosts(slippage_bps=5.0, profile="etf"),
    ).run(panel)
    # First buy should not be priced at the gap-day open
    buys = [f for f in result.fills if f.side.value == "buy"]
    assert buys, "expected at least one buy fill"
    gap_day = panel["TEST"][50]
    next_day = panel["TEST"][51]
    # Fill should occur on/after next bar (open+/-slip), not at gap open
    assert buys[0].ts >= next_day.ts or abs(buys[0].price - gap_day.open) > 0.5
    # Stronger: fill timestamp is next day
    assert buys[0].ts == next_day.ts
