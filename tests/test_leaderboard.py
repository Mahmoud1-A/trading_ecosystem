from __future__ import annotations

from collections import Counter

from trading_ecosystem.discovery.leaderboard import (
    load_top50,
    select_with_sector_seats,
    top50_public,
    update_top50,
)


def _row(sid: str, cagr: float, passed: bool = True, sym: str = "QQQ", lookback: int = 20) -> dict:
    return {
        "candidate": {
            "strategy_id": sid,
            "class_name": "ZScoreRevert",
            "symbols": [sym],
            "params": {
                "lookback": lookback,
                "entry_z": 2.0,
                "exit_z": 0.5,
                "risk_fraction": 0.005,
            },
            "source": "test",
            "meta": {},
        },
        "fitness": {
            "passed": passed,
            "score": cagr * 100,
            "oos_cagr": cagr,
            "oos_max_dd": 0.03,
            "oos_mar": 1.0,
            "oos_trades": 12,
            "hit_target_cagr": cagr >= 0.3,
            "reason": "ok",
        },
        "full_metrics": {
            "cagr": cagr,
            "max_drawdown": 0.03,
            "total_return": 0.2,
            "num_trades": 12,
        },
    }


def test_update_top50_keeps_best_fifty(tmp_path):
    path = tmp_path / "top50.json"
    newcomers = [
        _row(f"s{i}", 0.01 + i * 0.001, sym="QQQ" if i % 2 == 0 else "SPY", lookback=10 + i)
        for i in range(60)
    ]
    ranked = update_top50(newcomers, path=path, size=50, seats_per_symbol=3)
    assert len(ranked) == 50
    board = load_top50(path)
    assert board["count"] == 50
    pub = top50_public(board["entries"])
    assert pub[0]["rank"] == 1
    assert len(pub) == 50


def test_update_top50_replaces_weaker_same_fingerprint(tmp_path):
    path = tmp_path / "top50.json"
    weak = _row("a", 0.02)
    strong = _row("a2", 0.08)
    update_top50([weak], path=path, size=50, seats_per_symbol=3)
    ranked = update_top50([strong], path=path, size=50, seats_per_symbol=3)
    assert len(ranked) == 1
    assert ranked[0]["fitness"]["oos_cagr"] == 0.08
    assert ranked[0]["candidate"]["strategy_id"] == "a2"


def test_sector_seats_guarantee_three_per_symbol():
    pool = []
    # SLV dominates with many strong algos
    for i in range(20):
        pool.append(_row(f"slv{i}", 0.20 - i * 0.001, sym="SLV", lookback=10 + i))
    # Weaker symbols still deserve seats
    for i in range(5):
        pool.append(_row(f"spy{i}", 0.02 - i * 0.001, sym="SPY", lookback=30 + i))
        pool.append(_row(f"gld{i}", 0.015 - i * 0.001, sym="GLD", lookback=40 + i))
        pool.append(_row(f"uso{i}", 0.01 - i * 0.001, sym="USO", lookback=50 + i))

    selected = select_with_sector_seats(pool, size=50, seats_per_symbol=3)
    counts = Counter(
        (r.get("candidate") or {}).get("symbols", ["?"])[0] for r in selected
    )
    assert counts["SLV"] >= 3
    assert counts["SPY"] >= 3
    assert counts["GLD"] >= 3
    assert counts["USO"] >= 3
    # Seat-marked rows include the reserved ones
    assert sum(1 for r in selected if r.get("sector_seat")) >= 12
