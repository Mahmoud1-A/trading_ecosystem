from __future__ import annotations

from trading_ecosystem.discovery.sector_best import load_sector_best, update_sector_best


def _row(sid: str, cagr: float, sym: str, lookback: int = 20) -> dict:
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
            "passed": True,
            "score": cagr * 100,
            "oos_cagr": cagr,
            "oos_max_dd": 0.03,
            "oos_mar": 1.0,
            "oos_trades": 10,
            "hit_target_cagr": False,
            "reason": "ok",
        },
        "full_metrics": {},
    }


def test_best3_per_sector_independent(tmp_path):
    path = tmp_path / "best3.json"
    rows = []
    for i in range(10):
        rows.append(_row(f"slv{i}", 0.2 - i * 0.01, "SLV", lookback=10 + i))
    for i in range(5):
        rows.append(_row(f"spy{i}", 0.05 - i * 0.005, "SPY", lookback=30 + i))
    sectors = update_sector_best(rows, path=path, best_per_sector=3)
    assert len(sectors["SLV"]) == 3
    assert len(sectors["SPY"]) == 3
    assert sectors["SLV"][0]["fitness"]["oos_cagr"] >= sectors["SLV"][2]["fitness"]["oos_cagr"]
    board = load_sector_best(path)
    assert board["best_per_sector"] == 3
    assert set(board["sectors"].keys()) == {"SLV", "SPY"}
