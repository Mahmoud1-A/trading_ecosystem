from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.evolve import breed_generation, crossover_candidates, mutate_candidate
from trading_ecosystem.discovery.search_space import Candidate


def _cand(sid: str, cls: str = "ZScoreRevert", sym: str = "QQQ", **params) -> Candidate:
    base = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "risk_fraction": 0.0075}
    base.update(params)
    return Candidate(strategy_id=sid, class_name=cls, symbols=[sym], params=base, source="test")


def test_mutate_changes_params_or_symbol():
    rng = random.Random(1)
    parent = _cand("p1")
    child = mutate_candidate(parent, symbols=["SPY", "QQQ", "GLD"], rng=rng, generation=2, seq=1)
    assert child.source == "mutate"
    assert child.class_name == parent.class_name
    assert child.key() != parent.key() or child.symbols != parent.symbols


def test_crossover_has_both_parents_meta():
    rng = random.Random(2)
    a = _cand("a", lookback=10, entry_z=2.5)
    b = _cand("b", lookback=40, entry_z=1.5, sym="SPY")
    child = crossover_candidates(a, b, symbols=["SPY", "QQQ"], rng=rng, generation=3, seq=7)
    assert child.source == "crossover"
    assert set(child.meta.get("parents") or []) == {"a", "b"}


def test_breed_generation_returns_batch():
    rng = random.Random(3)
    elites = [_cand("e1"), _cand("e2", sym="SPY", lookback=40)]
    cfg = {
        "symbols": ["SPY", "QQQ"],
        "max_candidates_per_family": 4,
        "families": {
            "ZScoreRevert": {
                "enabled": True,
                "grids": {
                    "lookback": [10, 20],
                    "entry_z": [2.0],
                    "exit_z": [0.5],
                    "risk_fraction": [0.005],
                },
            }
        },
    }
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars = []
    px = 100.0
    for i in range(120):
        px *= 1.001
        bars.append(
            Bar(
                symbol="QQQ",
                ts=start + timedelta(days=i),
                open=px,
                high=px * 1.01,
                low=px * 0.99,
                close=px,
                volume=1e6,
            )
        )
    panel = {"QQQ": bars, "SPY": bars}
    kids = breed_generation(elites, cfg, panel, generation=1, batch_size=10, rng=rng)
    assert 1 <= len(kids) <= 10
    assert any(c.source in {"mutate", "crossover", "invent", "invent_anomaly"} for c in kids)


def test_invent_respects_family_weights():
    from collections import Counter

    from trading_ecosystem.discovery.evolve import invent_candidates

    rng = random.Random(11)
    cfg = {
        "symbols": ["SMH", "GDX"],
        "continuous": {"invent_family_weights": {"GapFade": 0.8, "BreakoutATR": 0.2}},
        "families": {
            "GapFade": {
                "enabled": True,
                "grids": {
                    "gap_pct": [0.002, 0.003],
                    "hold_bars": [3, 5],
                    "risk_fraction": [0.005],
                    "atr_period": [14],
                    "atr_stop_mult": [1.5],
                },
            },
            "BreakoutATR": {
                "enabled": True,
                "grids": {
                    "lookback": [10, 20],
                    "atr_period": [14],
                    "atr_stop_mult": [2.0],
                    "risk_fraction": [0.005],
                },
            },
        },
    }
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars = []
    px = 100.0
    for i in range(80):
        o = px
        c = px * (0.99 if i % 7 == 0 else 1.001)
        bars.append(
            Bar(
                symbol="SMH",
                ts=start + timedelta(days=i),
                open=o * 0.985 if i % 7 == 0 else o,
                high=max(o, c) * 1.01,
                low=min(o, c) * 0.99,
                close=c,
                volume=1e6,
            )
        )
        px = c
    panel = {"SMH": bars, "GDX": bars}
    invented = invent_candidates(cfg, panel, rng=rng, generation=1, n_random=40, n_anomaly=0)
    counts = Counter(c.class_name for c in invented if c.source == "invent")
    assert counts["GapFade"] > counts["BreakoutATR"]
    assert all(0.0005 <= float(c.params["gap_pct"]) <= 0.012 for c in invented if c.class_name == "GapFade")


def test_gapfade_mutate_clamps_gap_pct():
    rng = random.Random(4)
    parent = Candidate(
        strategy_id="p",
        class_name="GapFade",
        symbols=["SMH"],
        params={
            "gap_pct": 0.002,
            "hold_bars": 5,
            "atr_period": 20,
            "atr_stop_mult": 1.5,
            "risk_fraction": 0.0075,
        },
        source="test",
    )
    child = mutate_candidate(parent, symbols=["SMH", "GDX"], rng=rng, generation=1, seq=0)
    assert 0.0005 <= float(child.params["gap_pct"]) <= 0.012
    assert 1 <= int(child.params["hold_bars"]) <= 15
