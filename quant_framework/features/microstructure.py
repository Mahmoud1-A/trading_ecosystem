"""Microstructure features — only when quote/book source data exists."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.events.enums import DataCapability


class UnsupportedMicrostructureError(ValueError):
    """Raised when microstructure features are requested without source data."""


def quoted_spread(quotes: pd.DataFrame) -> pd.Series:
    if not {"bid_price", "ask_price"}.issubset(quotes.columns):
        raise UnsupportedMicrostructureError("quoted_spread requires bid_price and ask_price")
    return (quotes["ask_price"] - quotes["bid_price"]).rename("quoted_spread")


def order_book_imbalance(book: pd.DataFrame) -> pd.Series:
    if not {"bid_size", "ask_size"}.issubset(book.columns):
        raise UnsupportedMicrostructureError("order_book_imbalance requires bid_size and ask_size")
    denom = (book["bid_size"] + book["ask_size"]).replace(0, np.nan)
    return (book["bid_size"] / denom).rename("book_imbalance")


def compute_microstructure_features(
    *,
    quotes: pd.DataFrame | None = None,
    book: pd.DataFrame | None = None,
    available_capabilities: set[DataCapability] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute microstructure features only when capabilities and frames exist.

    Returns (frame, disabled_feature_ids). Never synthesizes quote/book data.
    Never computes true volume delta from OHLCV.
    """
    caps = available_capabilities or set()
    disabled: list[str] = []
    cols: dict[str, pd.Series] = {}

    if DataCapability.BID_ASK_QUOTES in caps and quotes is not None and not quotes.empty:
        cols["micro.quoted_spread"] = quoted_spread(quotes)
    else:
        disabled.append("micro.quoted_spread")

    if DataCapability.ORDER_BOOK_SNAPSHOTS in caps and book is not None and not book.empty:
        cols["micro.book_imbalance"] = order_book_imbalance(book)
    else:
        disabled.append("micro.book_imbalance")

    # Forbidden OHLCV volume-delta is always disabled
    disabled.append("micro.volume_delta_ohlcv_forbidden")

    if not cols:
        return pd.DataFrame(), disabled
    return pd.DataFrame(cols), disabled
