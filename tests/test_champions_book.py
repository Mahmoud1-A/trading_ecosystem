from trading_ecosystem.discovery.portfolio_select import select_sector_champions


def test_select_sector_champions_one_per_symbol():
    sectors = {
        "BTC-USD": [
            {
                "candidate": {"strategy_id": "a", "class_name": "DonchianATR", "symbols": ["BTC-USD"]},
                "fitness": {"passed": True, "oos_cagr": 0.1, "score": 1},
            },
            {
                "candidate": {"strategy_id": "b", "class_name": "BreakoutATR", "symbols": ["BTC-USD"]},
                "fitness": {"passed": True, "oos_cagr": 0.2, "score": 2},
            },
        ],
        "ETH-USD": [
            {
                "candidate": {"strategy_id": "c", "class_name": "GapFade", "symbols": ["ETH-USD"]},
                "fitness": {"passed": True, "oos_cagr": 0.05, "score": 1},
            }
        ],
        "SOL-USD": [
            {
                "candidate": {"strategy_id": "d", "class_name": "ZScoreRevert", "symbols": ["SOL-USD"]},
                "fitness": {"passed": False, "oos_cagr": 0.5, "score": 9},
            }
        ],
    }
    selected = select_sector_champions(sectors, rank=1, require_passed=True)
    ids = {(r.get("candidate") or {}).get("strategy_id") for r in selected}
    assert ids == {"a", "c"}
    assert all((r.get("fitness") or {}).get("passed") for r in selected)
