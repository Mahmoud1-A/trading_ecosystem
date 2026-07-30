from datetime import datetime, timezone
from pathlib import Path

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.data_pipeline.news_calendar import NewsCalendar
from trading_ecosystem.data_pipeline.store import ParquetBarStore


def test_parquet_roundtrip(tmp_path: Path):
    store = ParquetBarStore(root=tmp_path)
    bars = [
        Bar(
            symbol="SPY",
            ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=100,
            high=101,
            low=99,
            close=100.5,
            volume=10,
            timeframe="1d",
        ),
        Bar(
            symbol="SPY",
            ts=datetime(2024, 1, 3, tzinfo=timezone.utc),
            open=100.5,
            high=102,
            low=100,
            close=101,
            volume=11,
            timeframe="1d",
        ),
    ]
    counts = store.write_bars(bars)
    assert counts["SPY:1d"] == 2
    loaded = store.load_bars("SPY", "1d")
    assert len(loaded) == 2
    assert loaded[1].close == 101


def test_news_window_blocks():
    cal = NewsCalendar(block_before_minutes=30, block_after_minutes=30)
    # Use a known event from config if present; otherwise inject via object
    if not cal.events:
        cal.events = [
            {
                "name": "NFP",
                "datetime": datetime(2025, 8, 1, 12, 30, tzinfo=timezone.utc),
                "impact": "high",
            }
        ]
    event = cal.events[0]
    ts = event["datetime"]
    assert cal.is_blocked(ts)
    far = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert not cal.is_blocked(far)
