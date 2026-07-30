from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.anomaly import scan_anomalies
from trading_ecosystem.discovery.aggregate_book import book_objective
from trading_ecosystem.discovery.portfolio_select import select_portfolio, select_sector_champions
from trading_ecosystem.discovery.report import candidates_to_strategies_yaml, write_strategies_yaml
from trading_ecosystem.discovery.search_space import expand_family_candidates
from trading_ecosystem.strategies.registry import available_strategies, build_strategy


def _trending_bars(symbol: str = "SPY", n: int = 260) -> list[Bar]:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    out: list[Bar] = []
    px = 100.0
    for i in range(n):
        ts = start + timedelta(days=i)
        # mild uptrend with noise
        px *= 1.0015
        gap_open = px * (0.97 if i % 37 == 0 else 1.0)
        out.append(
            Bar(
                symbol=symbol,
                ts=ts,
                open=gap_open,
                high=px * 1.01,
                low=px * 0.99,
                close=px,
                volume=1_000_000,
            )
        )
    return out


def test_registry_has_new_families():
    names = set(available_strategies())
    assert {"DonchianATR", "BreakoutATR", "GapFade", "ZScoreRevert", "SeasonalityWindow"} <= names


def test_expand_family_candidates_respects_cap():
    cfg = {
        "symbols": ["SPY"],
        "max_candidates_per_family": 5,
        "random_seed": 1,
        "families": {
            "BreakoutATR": {
                "enabled": True,
                "grids": {
                    "lookback": [10, 20, 40],
                    "atr_period": [14],
                    "atr_stop_mult": [2.0, 2.5],
                    "risk_fraction": [0.0075],
                },
            }
        },
    }
    cands = expand_family_candidates(cfg, max_per_family=5, seed=1)
    assert len(cands) == 5
    assert all(c.class_name == "BreakoutATR" for c in cands)


def test_anomaly_scanner_emits_candidates():
    panel = {"SPY": _trending_bars()}
    cands = scan_anomalies(panel, top_k=10)
    assert isinstance(cands, list)
    # May be empty on synthetic path; if present must map to known classes
    for c in cands:
        build_strategy(c.class_name, c.strategy_id, c.symbols, c.params)


def test_portfolio_export_yaml(tmp_path: Path):
    evaluated = [
        {
            "candidate": {
                "strategy_id": "breakout_spy_000",
                "class_name": "BreakoutATR",
                "symbols": ["SPY"],
                "params": {"lookback": 20, "atr_period": 14, "atr_stop_mult": 2.5, "risk_fraction": 0.0075},
            },
            "fitness": {"passed": True, "score": 10.0, "oos_cagr": 0.1},
            "full_equity_curve": [
                (datetime(2020, 1, 1, tzinfo=timezone.utc), 100_000.0),
                (datetime(2020, 6, 1, tzinfo=timezone.utc), 105_000.0),
                (datetime(2021, 1, 1, tzinfo=timezone.utc), 110_000.0),
            ],
        },
        {
            "candidate": {
                "strategy_id": "gap_tlt_000",
                "class_name": "GapFade",
                "symbols": ["TLT"],
                "params": {"gap_pct": 0.015, "hold_bars": 2, "atr_period": 14, "atr_stop_mult": 2.0, "risk_fraction": 0.005},
            },
            "fitness": {"passed": True, "score": 8.0, "oos_cagr": 0.08},
            "full_equity_curve": [
                (datetime(2020, 1, 1, tzinfo=timezone.utc), 100_000.0),
                (datetime(2020, 6, 1, tzinfo=timezone.utc), 99_000.0),
                (datetime(2021, 1, 1, tzinfo=timezone.utc), 108_000.0),
            ],
        },
    ]
    selected = select_portfolio(evaluated, max_size=2, max_correlation=0.99)
    assert selected
    payload = candidates_to_strategies_yaml([{"candidate": e["candidate"]} for e in selected])
    path = tmp_path / "strategies.discovered.yaml"
    write_strategies_yaml(payload, path)
    text = path.read_text(encoding="utf-8")
    assert "BreakoutATR" in text or "GapFade" in text


def test_book_objective_prefers_better_mar_under_prop():
    weak = {"cagr": 0.20, "max_drawdown": 0.22, "mar": 0.90}
    strong = {"cagr": 0.30, "max_drawdown": 0.05, "mar": 6.0}
    assert book_objective(strong, prop_max_total_dd=0.10) > book_objective(
        weak, prop_max_total_dd=0.10
    )


def test_select_sector_champions_one_per_sector():
    sectors = {
        "SPY": [
            {
                "candidate": {"strategy_id": "spy_1", "class_name": "GapFade", "symbols": ["SPY"], "params": {}},
                "fitness": {"passed": True, "score": 9.0, "oos_cagr": 0.12},
                "sector": "SPY",
            },
            {
                "candidate": {"strategy_id": "spy_2", "class_name": "GapFade", "symbols": ["SPY"], "params": {}},
                "fitness": {"passed": True, "score": 5.0, "oos_cagr": 0.05},
                "sector": "SPY",
            },
        ],
        "TLT": [
            {
                "candidate": {"strategy_id": "tlt_1", "class_name": "ZScoreRevert", "symbols": ["TLT"], "params": {}},
                "fitness": {"passed": True, "score": 7.0, "oos_cagr": 0.08},
                "sector": "TLT",
            },
        ],
        "UNG": [
            {
                "candidate": {"strategy_id": "ung_fail", "class_name": "GapFade", "symbols": ["UNG"], "params": {}},
                "fitness": {"passed": False, "score": 1.0, "oos_cagr": 0.4},
                "sector": "UNG",
            },
        ],
    }
    selected = select_sector_champions(sectors, rank=1, require_passed=True)
    ids = [e["candidate"]["strategy_id"] for e in selected]
    assert ids == ["spy_1", "tlt_1"]
    assert "ung_fail" not in ids
