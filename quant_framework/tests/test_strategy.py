"""Phase 4 acceptance tests — strategy protocol and mean reversion."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from alpha import MeanReversionStrategy, StrategyState
from alpha.protocol import StrategyProtocol
from config.strategy_config import RegimeParams, StrategyParams
from engine.events import InformationTiming
from engine.portfolio import Portfolio


TZ = ZoneInfo("America/Chicago")


def _synthetic_bars(n: int = 120, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz=TZ)
    px = 4800.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": px - 0.25,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "volume": rng.integers(500, 5000, n).astype(float),
        }
    )
    return df


class TestStrategyProtocol:
    def test_mean_reversion_implements_protocol(self) -> None:
        s = MeanReversionStrategy(StrategyParams())
        assert isinstance(s, StrategyProtocol)

    def test_required_features_not_empty(self) -> None:
        s = MeanReversionStrategy(StrategyParams())
        feats = s.required_features()
        assert "vwap" in feats
        assert "combined_z" not in feats  # derived at compute time

    def test_params_not_hardcoded(self) -> None:
        params = StrategyParams(z_entry=1.5, z_exit=0.1, lookback=15)
        s = MeanReversionStrategy(params)
        assert s.params.z_entry == 1.5
        assert s.params.lookback == 15

    def test_compute_features_adds_availability_timestamps(self) -> None:
        s = MeanReversionStrategy(StrategyParams(use_regime_filter=False))
        bars = _synthetic_bars(80)
        feat = s.compute_features(bars)
        assert "availability_timestamp" in feat.columns
        assert "combined_z" in feat.columns
        assert "z_vwap" in feat.columns
        assert feat["availability_timestamp"].notna().all()

    def test_no_lookahead_decision_at_or_after_availability(self) -> None:
        s = MeanReversionStrategy(StrategyParams(use_regime_filter=False, z_entry=0.5))
        bars = _synthetic_bars(100)
        feat = s.compute_features(bars)
        portfolio = Portfolio(100_000.0)
        state = StrategyState()
        violations = 0
        for i, (_, row) in enumerate(feat.iterrows()):
            sigs = s.generate_signals(row, bar_index=i, portfolio=portfolio, state=state)
            for sig in sigs:
                if sig.timing.decision_timestamp < sig.timing.availability_timestamp:
                    violations += 1
        assert violations == 0

    def test_entry_signal_when_z_extreme(self) -> None:
        params = StrategyParams(z_entry=1.0, z_exit=0.1, use_regime_filter=False)
        s = MeanReversionStrategy(params)
        bars = _synthetic_bars(60)
        feat = s.compute_features(bars)
        portfolio = Portfolio(100_000.0)
        state = StrategyState()
        entries = 0
        for i, (_, row) in enumerate(feat.iterrows()):
            if abs(float(row.get("combined_z", 0))) >= params.z_entry:
                sigs = s.generate_signals(row, bar_index=i, portfolio=portfolio, state=state)
                entries += sum(1 for sig in sigs if sig.side in {"BUY", "SELL"})
        assert entries >= 1

    def test_session_end_flattens(self) -> None:
        s = MeanReversionStrategy(StrategyParams(session_flatten=True))
        portfolio = Portfolio(100_000.0)
        pos = portfolio.get_position("ES")
        pos.quantity = 1.0
        pos.contract = "ESH24"
        state = StrategyState()
        sigs = s.on_session_end(pd.Timestamp("2024-01-02 15:00", tz=TZ), portfolio, state)
        assert len(sigs) == 1
        assert sigs[0].side == "FLAT"
