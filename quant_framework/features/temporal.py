"""Temporal features using exchange/broker local time — never a fixed UTC open."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.calendars import ExchangeCalendar, get_calendar


def _local_index(index: pd.DatetimeIndex, calendar: ExchangeCalendar) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("bar index must be timezone-aware for temporal features")
    return index.tz_convert(calendar.zoneinfo())


def minutes_since_session_open(index: pd.DatetimeIndex, calendar: ExchangeCalendar) -> pd.Series:
    local = _local_index(index, calendar)
    out: list[float] = []
    for ts in local:
        bounds = calendar.session_bounds(ts.date())
        if bounds is None:
            out.append(float("nan"))
            continue
        open_ts, _ = bounds
        out.append((ts.to_pydatetime() - open_ts).total_seconds() / 60.0)
    return pd.Series(out, index=index, name="minutes_since_open")


def minutes_until_session_close(index: pd.DatetimeIndex, calendar: ExchangeCalendar) -> pd.Series:
    local = _local_index(index, calendar)
    out: list[float] = []
    for ts in local:
        bounds = calendar.session_bounds(ts.date())
        if bounds is None:
            out.append(float("nan"))
            continue
        _, close_ts = bounds
        out.append((close_ts - ts.to_pydatetime()).total_seconds() / 60.0)
    return pd.Series(out, index=index, name="minutes_to_close")


def day_of_week(index: pd.DatetimeIndex, calendar: ExchangeCalendar) -> pd.Series:
    local = _local_index(index, calendar)
    return pd.Series(local.weekday, index=index, name="dow", dtype=float)


def cyclical_time(index: pd.DatetimeIndex, calendar: ExchangeCalendar) -> pd.DataFrame:
    local = _local_index(index, calendar)
    minutes = local.hour * 60 + local.minute + local.second / 60.0
    angle = 2.0 * np.pi * minutes / 1440.0
    return pd.DataFrame(
        {
            "temp.sin_tod": np.sin(angle),
            "temp.cos_tod": np.cos(angle),
        },
        index=index,
    )


def compute_temporal_features(
    bars: pd.DataFrame,
    *,
    calendar: ExchangeCalendar | str = "CME",
) -> pd.DataFrame:
    cal = get_calendar(calendar) if isinstance(calendar, str) else calendar
    idx = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.DatetimeIndex(bars.index)
    cyc = cyclical_time(idx, cal)
    return pd.DataFrame(
        {
            "temp.minutes_since_open": minutes_since_session_open(idx, cal),
            "temp.minutes_to_close": minutes_until_session_close(idx, cal),
            "temp.dow": day_of_week(idx, cal),
            "temp.sin_tod": cyc["temp.sin_tod"],
            "temp.cos_tod": cyc["temp.cos_tod"],
        },
        index=bars.index,
    )
