from datetime import datetime, timedelta, timezone

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.stability import (
    classify_years,
    evaluate_portfolio_readiness,
)


def _year_bars(symbol: str, year: int, start: float, end: float, n: int = 60) -> list[Bar]:
    bars: list[Bar] = []
    for i in range(n):
        t = i / max(1, n - 1)
        px = start + (end - start) * t
        ts = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=i * 5)
        bars.append(
            Bar(
                symbol=symbol,
                ts=ts,
                open=px,
                high=px * 1.01,
                low=px * 0.99,
                close=px,
                volume=1_000_000,
                timeframe="1d",
            )
        )
    return bars


def test_classify_years_bull_bear_sideways():
    bars = (
        _year_bars("SPY", 2019, 100.0, 120.0)  # +20% bull
        + _year_bars("SPY", 2020, 120.0, 100.0)  # -16.7% bear
        + _year_bars("SPY", 2021, 100.0, 103.0)  # +3% sideways
    )
    labels = classify_years(bars, bull_year_return=0.10, bear_year_return=-0.10)
    assert labels[2019]["label"] == "bull"
    assert labels[2020]["label"] == "bear"
    assert labels[2021]["label"] == "sideways"
    assert labels[2019]["benchmark_return"] > 0.10
    assert labels[2020]["benchmark_return"] < -0.10


def test_empty_book_readiness_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trading_ecosystem.discovery.stability.readiness_path",
        lambda universe_id=None: tmp_path / f"readiness_{universe_id or 'etf'}.json",
    )
    payload = evaluate_portfolio_readiness([], {}, universe_id="etf")
    assert payload["ready"] is False
    assert payload["level"] == "not_ready"
    assert payload["checks"][0]["name"] == "has_members"
    assert not payload["checks"][0]["ok"]
    assert (tmp_path / "readiness_etf.json").exists()


def test_readiness_skips_resim_when_fingerprint_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "readiness_etf.json"
    monkeypatch.setattr(
        "trading_ecosystem.discovery.stability.readiness_path",
        lambda universe_id=None: path,
    )
    members = [
        {
            "candidate": {
                "strategy_id": "a1",
                "class_name": "DonchianATR",
                "symbols": ["BTC-USD"],
            },
            "fitness": {"passed": True, "oos_cagr": 0.1},
        }
    ]
    first = {
        "ready": False,
        "level": "candidate",
        "universe": "etf",
        "checks": [{"name": "min_members", "ok": False, "hard": True, "detail": "1"}],
        "fingerprint": "a1|DonchianATR|BTC-USD",
        "metrics": {"cagr": 0.01},
        "promotion": {"allow_freeze_paper": False, "allow_live": False},
    }
    path.write_text(__import__("json").dumps(first), encoding="utf-8")

    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("should not re-sim")

    monkeypatch.setattr(
        "trading_ecosystem.discovery.stability._run_book_with_curve",
        boom,
    )
    out = evaluate_portfolio_readiness(
        members,
        {},
        universe_id="etf",
        portfolio_metrics={"cagr": 0.02, "book_score": 0.5},
    )
    assert out.get("skipped_resim") is True
    assert out["metrics"]["cagr"] == 0.02
    assert calls["n"] == 0
