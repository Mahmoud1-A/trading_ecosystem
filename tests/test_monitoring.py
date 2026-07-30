from fastapi.testclient import TestClient

from trading_ecosystem.monitoring.app import create_app
from trading_ecosystem.monitoring.state_store import MONITOR


def test_health_and_state():
    MONITOR.update(equity=100000.0, daily_dd_pct=1.2, total_dd_pct=3.4, halted=False)
    MONITOR.discovery_start(phase="families", total=10, message="Testing discovery UI")
    MONITOR.discovery_progress(
        current=3,
        total=10,
        strategy_id="breakout_spy_001",
        class_name="BreakoutATR",
        symbols=["SPY"],
        source="families",
        passed=True,
        oos_cagr=0.04,
    )
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "ok"
    state = client.get("/api/state").json()
    assert state["equity"] == 100000.0
    assert state["discovery"]["status"] == "running"
    assert state["discovery"]["current"] == 3
    disc = client.get("/api/discovery").json()
    assert disc["eta_human"] is not None
    html = client.get("/")
    assert html.status_code == 200
    assert "Trading Ecosystem Monitor" in html.text
    assert "Discovery" in html.text
    assert "Trying algorithms now" in html.text or "Running" in html.text
    assert "BreakoutATR" in html.text
    assert "/control" in html.text
    readiness = client.get("/api/readiness")
    assert readiness.status_code == 200
    body = readiness.json()
    assert "level" in body
    assert "checks" in body
    ctrl = client.get("/control")
    assert ctrl.status_code == 200
    assert "لوحة التداول" in ctrl.text
    status = client.get("/api/control/status")
    assert status.status_code == 200
    assert "prop" in status.json()
    denied = client.post("/api/control/paper/stop")
    assert denied.status_code in (401, 403)
