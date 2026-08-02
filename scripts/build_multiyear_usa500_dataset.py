"""Materialize and register the official USA500 2020-2022 discovery dataset.

Run from the repository root after the yearly Dukascopy acquisitions complete::

    python scripts/build_multiyear_usa500_dataset.py

The tool prefers existing catalog entries and the project's normal Silver
resolver. If the yearly acquisition entries live only below the multiyear job,
it discovers their manifests/parquets and normalizes them to an immutable
5-minute ``timestamp,bid,ask`` parquet. It never includes 2023, 2024, or 2025.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = REPO_ROOT / "quant_framework"
ARCHIVE_ROOT = FRAMEWORK_ROOT / "data_import" / "dukascopy_archive"
MAIN_CATALOG_ROOT = ARCHIVE_ROOT / "dataset_catalog"
MAIN_CATALOG_PATH = MAIN_CATALOG_ROOT / "dataset_catalog.json"

TARGET_DATASET_ID = "dukascopy_us500_cfd_5m_2020_2022"
TEMPLATE_DATASET_ID = "dukascopy_us500_cfd_ticks_2024"
SOURCE_IDS = {
    2020: "ds_4a73821e70a78da2d36ed823",
    2021: "ds_0422b1a6f5c60e5f08b05619",
    2022: "ds_a095377bcb1aa8fc59af6b78",
}
TARGET_YEARS = (2020, 2021, 2022)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocatedEntry:
    catalog_root: Path
    entry: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _find_entry_slot(
    value: Any,
    dataset_id: str,
) -> tuple[Any, Any, dict[str, Any]] | None:
    if isinstance(value, dict):
        if str(value.get("dataset_id") or value.get("id") or "") == dataset_id:
            return None, None, value
        if dataset_id in value and isinstance(value[dataset_id], dict):
            return value, dataset_id, value[dataset_id]
        for key, child in value.items():
            found = _find_entry_slot(child, dataset_id)
            if found is not None:
                parent, slot, node = found
                if parent is None:
                    return value, key, node
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_entry_slot(child, dataset_id)
            if found is not None:
                parent, slot, node = found
                if parent is None:
                    return value, index, node
                return found
    return None


def _remove_existing_entry(value: Any, dataset_id: str) -> None:
    while True:
        found = _find_entry_slot(value, dataset_id)
        if found is None:
            return
        parent, slot, _node = found
        if parent is None:
            raise BuildError("catalog root itself cannot be the target entry")
        if isinstance(parent, list):
            parent.pop(int(slot))
        else:
            parent.pop(slot, None)


def _set_existing(entry: dict[str, Any], names: Iterable[str], value: Any) -> None:
    for name in names:
        if name in entry:
            entry[name] = value


def _rewrite_nested_dates(value: Any, *, start: str, end: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            token = str(key).lower()
            if token in {"start", "start_date", "date_start", "date_range_start", "from"}:
                value[key] = start
            elif token in {"end", "end_date", "date_end", "date_range_end", "to"}:
                value[key] = end
            elif token in {"first_timestamp", "first_ts"}:
                value[key] = f"{start}T00:00:00+00:00"
            elif token in {"last_timestamp", "last_ts"}:
                value[key] = f"{end}T23:59:59+00:00"
            else:
                _rewrite_nested_dates(child, start=start, end=end)
    elif isinstance(value, list):
        for child in value:
            _rewrite_nested_dates(child, start=start, end=end)


def _build_catalog_entry(
    template: dict[str, Any],
    *,
    output_path: Path,
    output_hash: str,
    rows: int,
    first_timestamp: str,
    last_timestamp: str,
) -> dict[str, Any]:
    entry = copy.deepcopy(template)
    start = "2020-01-01"
    end = "2022-12-31"

    _set_existing(entry, ("dataset_id", "id"), TARGET_DATASET_ID)
    _set_existing(
        entry,
        ("name", "title", "display_name", "label"),
        "Dukascopy USA500 CFD 5m BID/ASK 2020-2022",
    )
    _set_existing(entry, ("description",), "Immutable discovery dataset: 2020-2022 only")
    _set_existing(entry, ("timeframe", "frequency", "bar_size"), "5m")
    _set_existing(entry, ("row_count", "rows", "record_count"), int(rows))
    _set_existing(entry, ("data_hash", "dataset_hash", "normalized_hash"), output_hash)
    _set_existing(entry, ("research_eligible",), True)
    _set_existing(entry, ("smoke_test_only",), False)
    _set_existing(entry, ("vault_locked",), False)
    _set_existing(entry, ("alpha_miner_access",), "ALLOWED")
    _set_existing(entry, ("rejection_reasons",), [])
    _set_existing(entry, ("validation_status", "quality_status"), "PASS")
    _set_existing(entry, ("compatible_timeframes",), ["5m"])
    _set_existing(entry, ("first_timestamp",), first_timestamp)
    _set_existing(entry, ("last_timestamp",), last_timestamp)
    _rewrite_nested_dates(entry, start=start, end=end)

    schema = dict(entry.get("schema") or {})
    schema.update(
        {
            "direct_bid_ask_parquet": str(output_path.resolve()),
            "direct_bid_ask_sha256": output_hash,
            "direct_bid_ask_timeframes": ["5m"],
            "derived_bar_rules": ["5m"],
            "columns": ["timestamp", "bid", "ask"],
            "source_dataset_ids": [SOURCE_IDS[y] for y in TARGET_YEARS],
            "source_years": list(TARGET_YEARS),
            "excluded_years": [2023, 2024, 2025],
            "materialization": "direct_multiyear_bid_ask_parquet_v1",
        }
    )
    entry["schema"] = schema

    # Add canonical fields when the catalog schema is permissive/dict based.
    entry["dataset_id"] = TARGET_DATASET_ID
    entry["timeframe"] = "5m"
    entry["row_count"] = int(rows)
    entry["research_eligible"] = True
    entry["smoke_test_only"] = False
    entry["data_hash"] = output_hash
    entry["first_timestamp"] = first_timestamp
    entry["last_timestamp"] = last_timestamp
    entry["date_range_start"] = start
    entry["date_range_end"] = end
    return entry


def _catalog_paths() -> list[Path]:
    paths = sorted(ARCHIVE_ROOT.rglob("dataset_catalog.json"))
    if MAIN_CATALOG_PATH.is_file() and MAIN_CATALOG_PATH not in paths:
        paths.insert(0, MAIN_CATALOG_PATH)
    return paths


def _locate_catalog_entry(dataset_id: str) -> LocatedEntry | None:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
    try:
        from data.catalog.catalog import DatasetCatalog
    except Exception:
        return None

    for catalog_path in _catalog_paths():
        try:
            catalog = DatasetCatalog(catalog_path.parent)
            entry = catalog.get(dataset_id)
        except Exception:
            continue
        if entry is not None:
            return LocatedEntry(catalog_root=catalog_path.parent, entry=entry)
    return None


def _resolve_from_catalog(dataset_id: str) -> pd.DataFrame | None:
    located = _locate_catalog_entry(dataset_id)
    if located is None:
        return None
    sys.path.insert(0, str(FRAMEWORK_ROOT))
    try:
        from data.catalog.silver_bar_resolution import resolve_bid_ask_bars

        resolved = resolve_bid_ask_bars(located.entry, timeframe="5m")
    except Exception as exc:  # noqa: BLE001
        print(f"catalog entry {dataset_id} found but Silver resolution failed: {exc}")
        return None
    frame = resolved.frame.reset_index(drop=True).copy()
    return _normalise_quote_frame(frame)


def _manifest_roots(dataset_id: str, year: int) -> list[Path]:
    roots: set[Path] = set()
    for path in ARCHIVE_ROOT.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if dataset_id in text or f'"year": {year}' in text or f'"year":{year}' in text:
            roots.add(path.parent)
    for path in ARCHIVE_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        low = path.name.lower()
        if dataset_id.lower() in low or low == str(year):
            roots.add(path)
    return sorted(roots, key=lambda p: (len(p.parts), str(p)))


def _timestamp_column(columns: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in ("timestamp", "time", "datetime", "date_time", "ts", "event_time"):
        if candidate in lookup:
            return lookup[candidate]
    return None


def _pick_column(columns: list[str], names: Iterable[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def _read_parquet_candidate(path: Path) -> tuple[pd.DataFrame | None, pd.Series | None, pd.Series | None]:
    try:
        import pyarrow.parquet as pq

        columns = list(pq.ParquetFile(path).schema.names)
    except Exception:
        return None, None, None
    ts_col = _timestamp_column(columns)
    if ts_col is None:
        return None, None, None

    bid_col = _pick_column(columns, ("bid", "bid_price", "best_bid"))
    ask_col = _pick_column(columns, ("ask", "ask_price", "best_ask"))
    if bid_col and ask_col:
        raw = pd.read_parquet(path, columns=[ts_col, bid_col, ask_col])
        raw = raw.rename(columns={ts_col: "timestamp", bid_col: "bid", ask_col: "ask"})
        return _normalise_quote_frame(raw), None, None

    side_col = _pick_column(columns, ("side", "quote_side", "event", "event_type"))
    price_col = _pick_column(columns, ("price", "close", "value", "mid"))
    if side_col and price_col:
        raw = pd.read_parquet(path, columns=[ts_col, side_col, price_col])
        raw[ts_col] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
        raw[side_col] = raw[side_col].astype(str).str.upper()
        raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
        raw = raw.dropna(subset=[ts_col, price_col])
        bid = raw.loc[raw[side_col].str.contains("BID"), [ts_col, price_col]].set_index(ts_col)[price_col]
        ask = raw.loc[raw[side_col].str.contains("ASK"), [ts_col, price_col]].set_index(ts_col)[price_col]
        return None, bid, ask

    low_path = str(path).lower()
    side = "bid" if "bid" in low_path else ("ask" if "ask" in low_path else None)
    price_col = price_col or bid_col or ask_col
    if side and price_col:
        raw = pd.read_parquet(path, columns=[ts_col, price_col])
        raw[ts_col] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
        raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
        series = raw.dropna().set_index(ts_col)[price_col]
        return None, series if side == "bid" else None, series if side == "ask" else None
    return None, None, None


def _resample_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalise_quote_frame(frame)
    indexed = frame.set_index("timestamp")[["bid", "ask"]]
    deltas = indexed.index.to_series().diff().dropna().dt.total_seconds()
    if not deltas.empty and 240.0 <= float(deltas.median()) <= 360.0:
        bars = indexed
    else:
        bars = indexed.resample("5min", label="right", closed="right").last()
    bars = bars.dropna(subset=["bid", "ask"])
    return bars.reset_index()


def _normalise_quote_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timestamp" not in out.columns and out.index.name == "timestamp":
        out = out.reset_index()
    required = {"timestamp", "bid", "ask"}
    if not required.issubset(out.columns):
        raise BuildError(f"quote frame missing columns {sorted(required - set(out.columns))}")
    out = out[["timestamp", "bid", "ask"]]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out = out.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    out = out.loc[(out["bid"] > 0.0) & (out["ask"] >= out["bid"])]
    return out


def _resolve_from_files(dataset_id: str, year: int) -> pd.DataFrame:
    roots = _manifest_roots(dataset_id, year)
    files: list[Path] = []
    for root in roots:
        files.extend(root.rglob("*.parquet"))
    files = sorted(set(files))
    preferred = [
        path
        for path in files
        if any(token in str(path).lower() for token in ("silver", "5m", "5min", "bar"))
    ]
    ordered = preferred + [path for path in files if path not in set(preferred)]
    if not ordered:
        raise BuildError(
            f"no parquet files found for {year} ({dataset_id}); roots={roots[:8]}"
        )

    quote_frames: list[pd.DataFrame] = []
    bids: list[pd.Series] = []
    asks: list[pd.Series] = []
    for path in ordered:
        try:
            quote, bid, ask = _read_parquet_candidate(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip unreadable candidate {path}: {exc}")
            continue
        if quote is not None and not quote.empty:
            quote_frames.append(_resample_quotes(quote))
        if bid is not None and not bid.empty:
            bids.append(bid)
        if ask is not None and not ask.empty:
            asks.append(ask)

    if bids and asks:
        bid_all = pd.concat(bids).sort_index()
        ask_all = pd.concat(asks).sort_index()
        bid_5m = bid_all[~bid_all.index.duplicated(keep="last")].resample("5min").last()
        ask_5m = ask_all[~ask_all.index.duplicated(keep="last")].resample("5min").last()
        paired = pd.concat({"bid": bid_5m, "ask": ask_5m}, axis=1).dropna().reset_index()
        paired = paired.rename(columns={paired.columns[0]: "timestamp"})
        quote_frames.append(_normalise_quote_frame(paired))

    if not quote_frames:
        raise BuildError(
            f"found parquet files for {year}, but none exposed timestamp/BID/ASK data"
        )
    result = _normalise_quote_frame(pd.concat(quote_frames, ignore_index=True))
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    result = result.loc[(result["timestamp"] >= start) & (result["timestamp"] < end)]
    if len(result) < 1000:
        raise BuildError(f"{year} produced only {len(result)} valid 5m bars")
    return result


def _resolve_year(year: int, dataset_id: str) -> pd.DataFrame:
    print(f"Resolving {year}: {dataset_id}")
    frame = _resolve_from_catalog(dataset_id)
    if frame is None:
        frame = _resolve_from_files(dataset_id, year)
    frame = _resample_quotes(frame)
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    frame = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    if frame.empty:
        raise BuildError(f"{year} has no rows after exact year filtering")
    print(
        f"  {len(frame):,} rows | {frame['timestamp'].iloc[0]} -> "
        f"{frame['timestamp'].iloc[-1]}"
    )
    return frame


def build(*, overwrite: bool = False, dry_run: bool = False) -> Path:
    if not MAIN_CATALOG_PATH.is_file():
        raise BuildError(f"main catalog missing: {MAIN_CATALOG_PATH}")
    catalog_doc = _load_json(MAIN_CATALOG_PATH)
    template_slot = _find_entry_slot(catalog_doc, TEMPLATE_DATASET_ID)
    if template_slot is None:
        raise BuildError(f"template catalog entry missing: {TEMPLATE_DATASET_ID}")
    _parent, _slot, template = template_slot

    yearly = [_resolve_year(year, SOURCE_IDS[year]) for year in TARGET_YEARS]
    combined = _normalise_quote_frame(pd.concat(yearly, ignore_index=True))
    observed_years = sorted(set(combined["timestamp"].dt.year.astype(int)))
    if observed_years != list(TARGET_YEARS):
        raise BuildError(
            f"year isolation failed: expected={list(TARGET_YEARS)} observed={observed_years}"
        )
    if combined["timestamp"].duplicated().any():
        raise BuildError("duplicate timestamps remain after normalization")
    if (combined["ask"] < combined["bid"]).any():
        raise BuildError("crossed quotes remain after normalization")

    out_dir = (
        MAIN_CATALOG_ROOT
        / "lake"
        / "silver"
        / "event=bar"
        / "symbol=USA500IDXUSD"
        / "range=2020-2022"
    )
    out_path = out_dir / f"{TARGET_DATASET_ID}.parquet"
    manifest_path = out_dir / f"{TARGET_DATASET_ID}.manifest.json"
    if out_path.exists() and not overwrite:
        raise BuildError(f"output already exists; pass --overwrite: {out_path}")

    first = combined["timestamp"].iloc[0].isoformat()
    last = combined["timestamp"].iloc[-1].isoformat()
    print(f"Combined {len(combined):,} rows | {first} -> {last}")
    if dry_run:
        print("Dry run complete; no files changed.")
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_parquet = out_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_parquet, index=False, compression="zstd")
    tmp_parquet.replace(out_path)
    output_hash = _sha256_file(out_path)

    manifest = {
        "dataset_id": TARGET_DATASET_ID,
        "source_dataset_ids": {str(year): SOURCE_IDS[year] for year in TARGET_YEARS},
        "source_years": list(TARGET_YEARS),
        "excluded_years": [2023, 2024, 2025],
        "timeframe": "5m",
        "row_count": int(len(combined)),
        "first_timestamp": first,
        "last_timestamp": last,
        "sha256": output_hash,
        "path": str(out_path.resolve()),
    }
    _write_json_atomic(manifest_path, manifest)

    new_entry = _build_catalog_entry(
        template,
        output_path=out_path,
        output_hash=output_hash,
        rows=len(combined),
        first_timestamp=first,
        last_timestamp=last,
    )
    _remove_existing_entry(catalog_doc, TARGET_DATASET_ID)
    template_slot = _find_entry_slot(catalog_doc, TEMPLATE_DATASET_ID)
    if template_slot is None:
        raise BuildError("template entry disappeared while updating catalog")
    parent, slot, _template = template_slot
    if isinstance(parent, list):
        parent.append(new_entry)
    elif isinstance(parent, dict) and slot == TEMPLATE_DATASET_ID:
        parent[TARGET_DATASET_ID] = new_entry
    elif isinstance(parent, dict):
        # Template is stored as a normal node inside a list-like dict field.
        container = parent.get(slot)
        if isinstance(container, list):
            container.append(new_entry)
        else:
            # Most catalog layouts use a list or a dataset-id keyed map. Keep a
            # deterministic side collection only as a final compatible fallback.
            parent.setdefault("multiyear_entries", []).append(new_entry)
    else:
        raise BuildError("unsupported catalog container")
    _write_json_atomic(MAIN_CATALOG_PATH, catalog_doc)

    print(f"Registered {TARGET_DATASET_ID}")
    print(f"Parquet: {out_path}")
    print(f"SHA256: {output_hash}")
    print("Restart the control-plane server, then select the 2020-2022 dataset in New Run.")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        build(overwrite=args.overwrite, dry_run=args.dry_run)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
