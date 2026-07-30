"""
Dataset preview read model (Phase 12.2).

Reads a registered dataset's Silver events (falling back to Bronze) so the
dashboard can show real sample rows, spread statistics, and storage estimates
without loading a whole dataset into the browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.catalog.catalog import CatalogEntry, DatasetCatalog

MAX_PREVIEW_ROWS = 50


@dataclass
class DatasetPreviewAPI:
    catalog: DatasetCatalog

    def preview(self, dataset_id: str, *, n: int = 10) -> dict[str, Any]:
        entry = self.catalog.get(dataset_id)
        if entry is None:
            raise KeyError(dataset_id)
        rows = max(1, min(int(n), MAX_PREVIEW_ROWS))
        frame, layer, path = self._read_frame(entry)

        payload: dict[str, Any] = {
            "dataset_id": entry.dataset_id,
            "display_name": entry.display_name,
            "asset_class": entry.asset_class,
            "event_type": entry.event_type,
            "timeframe": entry.timeframe,
            "provider_id": entry.provider_id,
            "exchange_or_broker": entry.exchange_or_broker,
            "volume_type": entry.volume_type,
            "centralized_exchange_volume": False
            if entry.asset_class.lower() == "cfd"
            else None,
            "row_count": entry.row_count,
            "research_eligible": entry.research_eligible,
            "quality_status": entry.quality_status,
            "cost_model_status": entry.cost_model_status,
            "paper_eligible": entry.paper_eligible,
            "paper_eligibility_reason": entry.paper_eligibility_reason,
            "rejection_reasons": entry.rejection_reasons,
            "layer": layer,
            "source_path": path,
        }
        if frame is None:
            payload["rows"] = []
            payload["columns"] = []
            payload["note"] = "no readable lake artifact for this dataset"
            return payload

        head = frame.head(rows).copy()
        for column in head.columns:
            if pd.api.types.is_datetime64_any_dtype(head[column]):
                head[column] = head[column].astype(str)
        payload["columns"] = [str(c) for c in frame.columns]
        payload["rows"] = head.to_dict(orient="records")
        payload["estimated_rows"] = int(len(frame))
        payload["estimated_storage_bytes"] = _storage_bytes(path)
        payload["spread_summary"] = _spread_summary(frame)
        if entry.asset_class.lower() == "cfd":
            payload["banner"] = (
                "DUKASCOPY USA500 DATA IS BROKER CFD DATA. IT IS NOT CME ES FUTURES DATA."
            )
        return payload

    def _read_frame(
        self, entry: CatalogEntry
    ) -> tuple[pd.DataFrame | None, str, str | None]:
        candidates: list[tuple[str, Path]] = []
        if entry.silver_manifest_path:
            candidates.append(("silver", Path(entry.silver_manifest_path) / "events.parquet"))
        if entry.bronze_manifest_path:
            candidates.append(("bronze", Path(entry.bronze_manifest_path) / "raw.parquet"))
        for layer, path in candidates:
            if path.is_file():
                try:
                    return pd.read_parquet(path), layer, str(path)
                except (OSError, ValueError):
                    continue
        return None, "none", None


def _storage_bytes(path: str | None) -> int:
    if not path:
        return 0
    directory = Path(path).parent
    if not directory.is_dir():
        return 0
    return int(sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()))


def _spread_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if {"bid_price", "ask_price"}.issubset(frame.columns):
        spread = frame["ask_price"].astype(float) - frame["bid_price"].astype(float)
    elif "spread" in frame.columns:
        spread = pd.to_numeric(frame["spread"], errors="coerce")
    else:
        return {"available": False}
    clean = spread.dropna()
    if clean.empty:
        return {"available": False}
    return {
        "available": True,
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }
