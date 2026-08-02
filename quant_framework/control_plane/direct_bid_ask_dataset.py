"""Resolve immutable direct BID/ASK parquet datasets for research runs.

This module supports locally materialized multi-year datasets without coupling the
control plane to the ignored Dukascopy lake implementation. A catalog entry opts
in by placing ``direct_bid_ask_parquet`` and ``direct_bid_ask_sha256`` in its
``schema`` object.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DIRECT_BID_ASK_UNAVAILABLE = "DIRECT_BID_ASK_UNAVAILABLE"
_SUPPORTED_5M = {"5m", "5min", "5minute", "5minutes"}


class DirectBidAskResolutionError(ValueError):
    """Raised when a direct BID/ASK catalog entry cannot be resolved safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_5m(value: str) -> str:
    token = str(value or "").strip().lower().replace(" ", "")
    if token not in _SUPPORTED_5M:
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: direct multi-year dataset supports 5m only; "
            f"requested={value!r}"
        )
    return "5min"


def direct_bid_ask_path(entry: Any) -> Path | None:
    schema = dict(getattr(entry, "schema", None) or {})
    raw = schema.get("direct_bid_ask_parquet")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        lake_root = getattr(entry, "lake_root", None)
        if lake_root:
            path = Path(str(lake_root)).expanduser() / path
    return path.resolve()


@dataclass(frozen=True)
class DirectBidAskResolution:
    frame: pd.DataFrame
    timeframe: str
    data_path: str
    data_hash: str
    row_count: int
    first_timestamp: str
    last_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        """Expose the metadata keys consumed by the existing event-WFO path."""
        version = f"direct_{self.data_hash[:16]}"
        return {
            "timeframe": self.timeframe,
            "bid_silver_version": version,
            "ask_silver_version": version,
            "bid_normalized_hash": self.data_hash,
            "ask_normalized_hash": self.data_hash,
            "bid_path": self.data_path,
            "ask_path": self.data_path,
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "source": "direct_multiyear_bid_ask_parquet",
        }


def resolve_direct_bid_ask_bars(
    entry: Any,
    *,
    timeframe: str,
    max_rows: int | None = None,
) -> DirectBidAskResolution:
    """Load and verify a direct immutable 5-minute BID/ASK parquet dataset."""
    normalised_tf = _normalise_5m(timeframe)
    path = direct_bid_ask_path(entry)
    if path is None:
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: catalog entry has no direct_bid_ask_parquet"
        )
    if not path.is_file():
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: parquet not found: {path}"
        )

    schema = dict(getattr(entry, "schema", None) or {})
    expected_hash = str(schema.get("direct_bid_ask_sha256") or "").strip().lower()
    actual_hash = _sha256_file(path)
    if expected_hash and actual_hash != expected_hash:
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: sha256 mismatch expected={expected_hash} "
            f"actual={actual_hash}"
        )

    try:
        frame = pd.read_parquet(path, columns=["timestamp", "bid", "ask"])
    except Exception as exc:  # noqa: BLE001
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: cannot read {path}: {exc}"
        ) from exc

    if frame.empty:
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: parquet contains no rows"
        )

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "bid", "ask"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame = frame.loc[(frame["bid"] > 0.0) & (frame["ask"] >= frame["bid"])]
    if frame.empty:
        raise DirectBidAskResolutionError(
            f"{DIRECT_BID_ASK_UNAVAILABLE}: no valid positive BID/ASK rows"
        )

    deltas = frame["timestamp"].diff().dropna()
    if not deltas.empty:
        median_seconds = float(deltas.dt.total_seconds().median())
        if not 240.0 <= median_seconds <= 360.0:
            raise DirectBidAskResolutionError(
                f"{DIRECT_BID_ASK_UNAVAILABLE}: median spacing is "
                f"{median_seconds:.3f}s, expected approximately 300s"
            )

    if max_rows is not None:
        limit = max(1, int(max_rows))
        frame = frame.head(limit).copy()

    first = frame["timestamp"].iloc[0].isoformat()
    last = frame["timestamp"].iloc[-1].isoformat()
    frame = frame.set_index("timestamp", drop=False)
    return DirectBidAskResolution(
        frame=frame,
        timeframe=normalised_tf,
        data_path=str(path),
        data_hash=actual_hash,
        row_count=int(len(frame)),
        first_timestamp=first,
        last_timestamp=last,
    )
