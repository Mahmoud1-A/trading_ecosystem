from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from control_plane.direct_bid_ask_dataset import (
    DirectBidAskResolutionError,
    direct_bid_ask_path,
    resolve_direct_bid_ask_bars,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_quotes(path: Path) -> None:
    timestamps = pd.date_range("2020-01-02", periods=12, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid": [3200.0 + i for i in range(len(timestamps))],
            "ask": [3200.5 + i for i in range(len(timestamps))],
        }
    )
    frame.to_parquet(path, index=False)


def test_direct_bid_ask_resolution_verifies_hash_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "quotes.parquet"
    _write_quotes(path)
    entry = SimpleNamespace(
        schema={
            "direct_bid_ask_parquet": str(path),
            "direct_bid_ask_sha256": _sha(path),
        },
        lake_root=None,
    )

    resolved = resolve_direct_bid_ask_bars(entry, timeframe="5m", max_rows=5)

    assert direct_bid_ask_path(entry) == path.resolve()
    assert resolved.timeframe == "5min"
    assert len(resolved.frame) == 5
    assert list(resolved.frame.columns) == ["timestamp", "bid", "ask"]
    assert resolved.as_dict()["source"] == "direct_multiyear_bid_ask_parquet"
    assert resolved.as_dict()["bid_normalized_hash"] == _sha(path)


def test_direct_bid_ask_resolution_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "quotes.parquet"
    _write_quotes(path)
    entry = SimpleNamespace(
        schema={
            "direct_bid_ask_parquet": str(path),
            "direct_bid_ask_sha256": "0" * 64,
        },
        lake_root=None,
    )

    with pytest.raises(DirectBidAskResolutionError, match="sha256 mismatch"):
        resolve_direct_bid_ask_bars(entry, timeframe="5m")


def test_direct_bid_ask_resolution_is_5m_only(tmp_path: Path) -> None:
    path = tmp_path / "quotes.parquet"
    _write_quotes(path)
    entry = SimpleNamespace(
        schema={"direct_bid_ask_parquet": str(path)},
        lake_root=None,
    )

    with pytest.raises(DirectBidAskResolutionError, match="supports 5m only"):
        resolve_direct_bid_ask_bars(entry, timeframe="1m")
