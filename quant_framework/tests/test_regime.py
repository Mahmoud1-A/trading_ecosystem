"""Phase 4 acceptance tests — regime detection and information timing."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.models import RegimeLabel, Timeframe
from config.strategy_config import RegimeParams
from regime import RegimeDetector


TZ = ZoneInfo("America/Chicago")


def _ltf_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz=TZ)
    trend = np.linspace(0, 20, n)
    noise = rng.normal(0, 1, n)
    px = 4800 + trend + noise
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": px - 0.2,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "volume": rng.integers(800, 4000, n).astype(float),
        }
    )


class TestRegimeDetector:
    def test_htf_features_have_availability_timestamp(self) -> None:
        det = RegimeDetector(RegimeParams(htf=Timeframe.H1))
        bars = _ltf_bars()
        htf = det.compute_htf_features(bars)
        assert "availability_timestamp" in htf.columns
        assert "adx" in htf.columns
        assert htf["availability_timestamp"].notna().any()

    def test_regime_mapped_to_ltf_without_lookahead(self) -> None:
        det = RegimeDetector(RegimeParams(htf=Timeframe.H1, trade_only_range=False))
        bars = _ltf_bars()
        mapped = det.detect(bars)
        assert "regime" in mapped.columns
        assert "regime_availability_timestamp" in mapped.columns
        # Every row must have regime available at or before decision
        if "bar_end" in mapped.columns:
            decision = mapped["bar_end"]
        else:
            decision = mapped.index + pd.Timedelta(minutes=5)
        has_regime = mapped["regime_availability_timestamp"].notna()
        valid = (~has_regime) | (mapped["regime_availability_timestamp"] <= decision)
        assert valid.all()

    def test_classify_trend_up_on_strong_slope(self) -> None:
        det = RegimeDetector(RegimeParams(adx_trend_threshold=10.0))
        row = pd.Series({"adx": 30.0, "atr_pct": 50.0, "ma_slope": 0.01, "range_compression": 1.0})
        assert det.classify_row(row) == RegimeLabel.TREND_UP

    def test_trade_only_range_blocks_trend(self) -> None:
        det = RegimeDetector(RegimeParams(trade_only_range=True))
        assert det.is_tradeable(RegimeLabel.RANGE) is True
        assert det.is_tradeable(RegimeLabel.TREND_UP) is False

    def test_unknown_when_adx_missing(self) -> None:
        det = RegimeDetector(RegimeParams())
        row = pd.Series({"adx": np.nan})
        assert det.classify_row(row) == RegimeLabel.UNKNOWN
