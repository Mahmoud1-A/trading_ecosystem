from datetime import datetime, timezone

from trading_ecosystem.common.contracts import OrderIntent, OrderSide
from trading_ecosystem.execution.broker import PaperSimBroker
from trading_ecosystem.execution.alpaca_gateway import AlpacaPaperGateway


def test_paper_sim_fill_and_sync():
    broker = PaperSimBroker(cash=50_000)
    broker.set_price("SPY", 400.0)
    intent = OrderIntent(
        strategy_id="s1",
        symbol="SPY",
        side=OrderSide.BUY,
        qty=10,
        ts=datetime.now(timezone.utc),
    )
    rec = broker.place_order(intent)
    assert rec["status"] == "filled"
    assert len(broker.get_positions()) == 1
    synced = broker.sync()
    assert synced["account"]["equity"] > 0


def test_alpaca_dry_run_without_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    gw = AlpacaPaperGateway()
    assert gw.configured is False
    intent = OrderIntent(
        strategy_id="s1",
        symbol="SPY",
        side=OrderSide.BUY,
        qty=1,
        ts=datetime.now(timezone.utc),
    )
    rec = gw.place_order(intent)
    assert rec.get("dry_run") is True
