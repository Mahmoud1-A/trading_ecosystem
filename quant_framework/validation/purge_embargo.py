"""Purge and embargo gaps between train and validation windows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowSlice:
    """Inclusive-exclusive integer bar indices into a sorted frame."""

    train_start: int
    train_end: int  # exclusive
    validation_start: int
    validation_end: int  # exclusive
    purge_start: int
    purge_end: int
    embargo_start: int
    embargo_end: int

    def train_index(self) -> slice:
        return slice(self.train_start, self.train_end)

    def validation_index(self) -> slice:
        return slice(self.validation_start, self.validation_end)


def apply_purge_embargo(
    train_end: int,
    validation_start: int,
    validation_end: int,
    *,
    purge_gap_bars: int,
    embargo_gap_bars: int,
    n_bars: int,
) -> WindowSlice:
    """
    Insert purge after train and embargo after validation to prevent overlap leakage.

    Train uses [train_start, purged_train_end).
    Validation uses [embargoed_val_start, validation_end).
    Bars in purge/embargo are unused.
    """
    if train_end < 0 or validation_end > n_bars:
        raise ValueError("Window indices out of bounds")
    if validation_start < train_end:
        raise ValueError("Validation must start at or after train_end before purge")

    purged_train_end = max(0, train_end - purge_gap_bars)
    purge_start = purged_train_end
    purge_end = train_end

    # Embargo sits between original train_end and validation (or after train if contiguous)
    emb_start = train_end
    emb_end = min(n_bars, train_end + embargo_gap_bars)
    val_start = max(validation_start, emb_end)
    if val_start >= validation_end:
        raise ValueError(
            f"Embargo ({embargo_gap_bars} bars) consumes entire validation window "
            f"[{validation_start}, {validation_end})"
        )

    return WindowSlice(
        train_start=0,  # filled by caller for absolute windows
        train_end=purged_train_end,
        validation_start=val_start,
        validation_end=validation_end,
        purge_start=purge_start,
        purge_end=purge_end,
        embargo_start=emb_start,
        embargo_end=emb_end,
    )


def assert_no_overlap(slice_: WindowSlice) -> None:
    """Hard check: train and validation ranges must not overlap."""
    train = set(range(slice_.train_start, slice_.train_end))
    val = set(range(slice_.validation_start, slice_.validation_end))
    if train & val:
        raise ValueError(f"Train/validation overlap: {sorted(train & val)[:10]}")
    purge = set(range(slice_.purge_start, slice_.purge_end))
    if train & purge:
        raise ValueError("Purge overlaps train")
    if val & purge:
        raise ValueError("Purge overlaps validation")


def timestamps_ordered(ts: pd.Series | pd.DatetimeIndex) -> bool:
    idx = pd.DatetimeIndex(ts)
    return bool(idx.is_monotonic_increasing)
