"""Phase 1 acceptance tests — data loading, validation, hashing, calendars."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.calendars import CME_EQUITY_RTH, ExchangeCalendar, get_calendar, register_calendar
from data.loader import load_ohlcv
from data.resampler import resample_ohlcv
from data.validator import (
    DataValidationError,
    compute_data_hash,
    convert_timezone,
    detect_missing_bars,
    expected_session_index,
    localize_timestamps,
    validate_market_data,
)


def _aware_frame(
    *,
    tz: str = "America/Chicago",
    start: str = "2024-01-02 08:30",
    periods: int = 6,
    freq: str = "5min",
    base: float = 4800.0,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz=ZoneInfo(tz))
    rows = []
    for i, ts in enumerate(idx):
        c = base + i * 0.25
        rows.append(
            {
                "timestamp": ts,
                "open": c - 0.25,
                "high": c + 0.5,
                "low": c - 0.5,
                "close": c,
                "volume": 1000.0 + i,
            }
        )
    return pd.DataFrame(rows)


class TestTimezoneEnforcement:
    def test_naive_timestamps_rejected(self) -> None:
        df = _aware_frame()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        with pytest.raises(DataValidationError, match="Naive timestamps"):
            validate_market_data(df, require_timezone=True, detect_missing=False)

    def test_naive_timestamps_explicitly_localized(self) -> None:
        df = _aware_frame()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        result = validate_market_data(
            df,
            require_timezone=True,
            default_timezone="America/Chicago",
            detect_missing=False,
        )
        assert result.frame["timestamp"].dt.tz is not None
        assert str(result.frame["timestamp"].dt.tz) == "America/Chicago"

    def test_timezone_conversion_correct(self) -> None:
        series = pd.Series(pd.date_range("2024-01-02 08:30", periods=2, freq="5min", tz="America/Chicago"))
        converted = convert_timezone(series, "UTC")
        assert str(converted.dt.tz) == "UTC"
        # 08:30 CST (UTC-6 in January) => 14:30 UTC
        assert converted.iloc[0].hour == 14
        assert converted.iloc[0].minute == 30

    def test_convert_naive_fails(self) -> None:
        series = pd.Series(pd.date_range("2024-01-02 08:30", periods=2, freq="5min"))
        with pytest.raises(DataValidationError, match="naive"):
            convert_timezone(series, "UTC")

    def test_localize_helper_rejects_without_default(self) -> None:
        series = pd.Series(["2024-01-02 08:30:00", "2024-01-02 08:35:00"])
        with pytest.raises(DataValidationError, match="Naive"):
            localize_timestamps(series, require_timezone=True, default_timezone=None)


class TestDuplicateAndOrdering:
    def test_duplicate_timestamps_fail(self) -> None:
        df = _aware_frame(periods=4)
        dup = pd.concat([df, df.iloc[[1]]], ignore_index=True)
        with pytest.raises(DataValidationError, match="Duplicate"):
            validate_market_data(dup, detect_missing=False)

    def test_non_monotonic_is_sorted(self) -> None:
        df = _aware_frame(periods=4)
        shuffled = df.iloc[::-1].reset_index(drop=True)
        result = validate_market_data(shuffled, detect_missing=False)
        assert result.frame["timestamp"].is_monotonic_increasing
        assert any("sorted" in w.lower() for w in result.warnings)


class TestOHLCValidation:
    def test_high_less_than_low_fails(self) -> None:
        df = _aware_frame(periods=3)
        df.loc[1, "high"] = 100.0
        df.loc[1, "low"] = 200.0
        with pytest.raises(DataValidationError, match="high < low"):
            validate_market_data(df, detect_missing=False)

    def test_open_outside_range_fails(self) -> None:
        df = _aware_frame(periods=3)
        df.loc[0, "open"] = float(df.loc[0, "high"]) + 10.0
        with pytest.raises(DataValidationError, match="open outside"):
            validate_market_data(df, detect_missing=False)

    def test_close_outside_range_fails(self) -> None:
        df = _aware_frame(periods=3)
        df.loc[0, "close"] = float(df.loc[0, "low"]) - 10.0
        with pytest.raises(DataValidationError, match="close outside"):
            validate_market_data(df, detect_missing=False)

    def test_negative_volume_fails(self) -> None:
        df = _aware_frame(periods=3)
        df.loc[2, "volume"] = -1.0
        with pytest.raises(DataValidationError, match="Negative volume"):
            validate_market_data(df, detect_missing=False)

    def test_valid_relationships_pass(self) -> None:
        df = _aware_frame(periods=5)
        result = validate_market_data(df, detect_missing=False)
        assert len(result.frame) == 5


class TestMissingBarsAndCalendars:
    def test_missing_in_session_bar_detected(self) -> None:
        df = _aware_frame(periods=6)  # 08:30 .. 08:55
        # Drop 08:40 bar
        df = df.drop(index=2).reset_index(drop=True)
        with pytest.raises(DataValidationError, match="Missing in-session"):
            validate_market_data(
                df,
                detect_missing=True,
                missing_bar_tolerance=0,
                freq="5min",
                calendar="CME",
            )

    def test_missing_bar_within_tolerance_warns(self) -> None:
        df = _aware_frame(periods=6)
        df = df.drop(index=2).reset_index(drop=True)
        result = validate_market_data(
            df,
            detect_missing=True,
            missing_bar_tolerance=1,
            freq="5min",
            calendar="CME",
        )
        assert result.missing_bars is not None
        assert result.missing_bars.unexpected_gap_count == 1
        assert any("missing" in w.lower() for w in result.warnings)

    def test_weekend_not_treated_as_missing(self) -> None:
        """Friday session followed by Monday — weekend is an expected closure."""
        tz = ZoneInfo("America/Chicago")
        friday = pd.date_range("2024-01-05 08:30", periods=3, freq="5min", tz=tz)
        monday = pd.date_range("2024-01-08 08:30", periods=3, freq="5min", tz=tz)
        idx = friday.append(monday)
        rows = []
        for i, ts in enumerate(idx):
            c = 4800.0 + i
            rows.append(
                {
                    "timestamp": ts,
                    "open": c - 0.25,
                    "high": c + 0.5,
                    "low": c - 0.5,
                    "close": c,
                    "volume": 1000.0,
                }
            )
        df = pd.DataFrame(rows)
        result = validate_market_data(
            df,
            detect_missing=True,
            missing_bar_tolerance=0,
            freq="5min",
            calendar="CME",
        )
        assert result.missing_bars is not None
        # Cross-weekend continuity is not an unexpected gap
        assert result.missing_bars.unexpected_gap_count == 0
        missing_dates = {pd.Timestamp(ts).date() for ts in result.missing_bars.missing_timestamps}
        assert date(2024, 1, 6) not in missing_dates  # Saturday
        assert date(2024, 1, 7) not in missing_dates  # Sunday

    def test_holiday_not_treated_as_missing(self) -> None:
        """New Year's Day 2024 is in CME holiday list — not a missing session."""
        tz = ZoneInfo("America/Chicago")
        # 2023-12-29 Friday and 2024-01-02 Tuesday (2024-01-01 holiday)
        pre = pd.date_range("2023-12-29 08:30", periods=2, freq="5min", tz=tz)
        post = pd.date_range("2024-01-02 08:30", periods=2, freq="5min", tz=tz)
        idx = pre.append(post)
        rows = []
        for i, ts in enumerate(idx):
            c = 4800.0 + i
            rows.append(
                {
                    "timestamp": ts,
                    "open": c,
                    "high": c + 0.5,
                    "low": c - 0.5,
                    "close": c,
                    "volume": 100.0,
                }
            )
        df = pd.DataFrame(rows)
        result = validate_market_data(
            df,
            detect_missing=True,
            missing_bar_tolerance=0,
            freq="5min",
            calendar="CME",
        )
        assert result.missing_bars is not None
        assert result.missing_bars.unexpected_gap_count == 0
        missing_dates = {pd.Timestamp(ts).date() for ts in result.missing_bars.missing_timestamps}
        assert date(2024, 1, 1) not in missing_dates

    def test_expected_session_index_skips_weekend(self) -> None:
        tz = ZoneInfo("America/Chicago")
        start = pd.Timestamp("2024-01-05 08:30", tz=tz)  # Friday
        end = pd.Timestamp("2024-01-08 08:40", tz=tz)  # Monday
        expected = expected_session_index(CME_EQUITY_RTH, start, end, "5min")
        dates = {ts.date() for ts in expected}
        assert date(2024, 1, 6) not in dates  # Saturday
        assert date(2024, 1, 7) not in dates  # Sunday
        assert date(2024, 1, 5) in dates
        assert date(2024, 1, 8) in dates

    def test_get_calendar(self) -> None:
        assert get_calendar("CME").name == "CME"
        with pytest.raises(KeyError):
            get_calendar("NO_SUCH_EXCHANGE")

    def test_register_custom_calendar(self) -> None:
        cal = ExchangeCalendar(
            name="TEST_CAL",
            timezone="UTC",
            session_open=time(9, 0),
            session_close=time(17, 0),
            holidays=frozenset({date(2024, 7, 4)}),
        )
        register_calendar(cal, overwrite=True)
        assert get_calendar("TEST_CAL").is_trading_day(date(2024, 7, 4)) is False


class TestDataHashing:
    def test_identical_datasets_identical_hashes(self) -> None:
        df1 = _aware_frame(periods=10)
        df2 = df1.copy()
        h1 = compute_data_hash(df1)
        h2 = compute_data_hash(df2)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_hash_stable_across_column_order(self) -> None:
        df = _aware_frame(periods=5)
        h1 = compute_data_hash(df)
        reordered = df[["volume", "close", "low", "high", "open", "timestamp"]]
        h2 = compute_data_hash(reordered)
        assert h1 == h2

    def test_hash_changes_when_data_changes(self) -> None:
        df = _aware_frame(periods=5)
        h1 = compute_data_hash(df)
        df2 = df.copy()
        df2.loc[0, "close"] = float(df2.loc[0, "close"]) + 1.0
        # Keep OHLC valid
        df2.loc[0, "high"] = max(float(df2.loc[0, "high"]), float(df2.loc[0, "close"]))
        h2 = compute_data_hash(df2)
        assert h1 != h2

    def test_validation_result_includes_hash(self) -> None:
        result = validate_market_data(_aware_frame(periods=4), detect_missing=False)
        assert result.data_hash == compute_data_hash(result.frame)
        assert result.frame.attrs["data_hash"] == result.data_hash


class TestLoaderAndResampler:
    def test_load_csv_and_parquet(self, tmp_path: Path) -> None:
        df = _aware_frame(periods=8)
        csv_path = tmp_path / "sample.csv"
        pq_path = tmp_path / "sample.parquet"
        # Store timestamps as ISO strings with offset for CSV round-trip
        csv_df = df.copy()
        csv_df["timestamp"] = csv_df["timestamp"].astype(str)
        csv_df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, index=False)

        csv_result = load_ohlcv(csv_path, detect_missing=False)
        pq_result = load_ohlcv(pq_path, detect_missing=False)
        assert len(csv_result.frame) == 8
        assert len(pq_result.frame) == 8
        # Same economic content → same hash
        assert csv_result.data_hash == pq_result.data_hash

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ohlcv(tmp_path / "nope.csv")

    def test_resample_1m_to_5m_no_ffill(self) -> None:
        df = _aware_frame(periods=10, freq="1min", start="2024-01-02 08:30")
        result = validate_market_data(df, detect_missing=False)
        resampled = resample_ohlcv(result.frame, "5min", calendar="CME", validate=True)
        assert len(resampled) >= 1
        assert resampled["timestamp"].dt.tz is not None
        # No NaNs introduced via forward-fill
        assert not resampled[["open", "high", "low", "close", "volume"]].isna().any().any()

    def test_detect_missing_bars_api(self) -> None:
        df = _aware_frame(periods=4)
        report = detect_missing_bars(df["timestamp"], freq="5min", calendar=CME_EQUITY_RTH)
        assert report.observed_count == 4
        assert report.unexpected_gap_count == 0
