from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.walk_forward import build_folds, filter_panel, ratio_folds


def _bars(n: int = 300, symbol: str = "SPY") -> list[Bar]:
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    out: list[Bar] = []
    px = 100.0
    for i in range(n):
        ts = start + timedelta(days=i)
        px += 0.05
        out.append(
            Bar(
                symbol=symbol,
                ts=ts,
                open=px,
                high=px + 1,
                low=px - 1,
                close=px,
                volume=1_000_000,
            )
        )
    return out


def test_ratio_folds_cover_full_range_without_overlap_leakage():
    panel = {"SPY": _bars(300)}
    folds = ratio_folds(panel, train_ratio=0.6, validate_ratio=0.2, holdout_ratio=0.2, min_bars_per_fold=40)
    assert len(folds) == 3
    train, val, hold = folds
    assert train.end < val.start
    assert val.end < hold.start
    assert train.kind == "train"
    assert hold.kind == "holdout"


def test_filter_panel_excludes_future_bars():
    panel = {"SPY": _bars(100)}
    folds = ratio_folds(panel, min_bars_per_fold=20)
    train = folds[0]
    sub = filter_panel(panel, train.start, train.end)
    assert sub["SPY"]
    assert all(b.ts <= train.end for b in sub["SPY"])
    assert all(b.ts >= train.start for b in sub["SPY"])


def test_build_folds_includes_yearly_oos():
    panel = {"SPY": _bars(1200)}  # ~3+ years
    folds = build_folds(panel, use_yearly_rolls=True, min_bars_per_fold=40)
    kinds = {f.kind for f in folds}
    assert "train" in kinds
    assert "oos" in kinds
