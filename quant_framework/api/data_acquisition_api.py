"""
Data acquisition read/command model for the control plane (Phase 12.2).

Two acquisition modes are supported. MANUAL_EXPORT_IMPORT reads a bid/ask CSV
or Parquet export from the existing safe import directory and always works.
AUTOMATED_PUBLIC_DOWNLOAD reads Dukascopy's public ``.bi5`` archive and is only
enabled when the operator has opted in explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from cfds.execution_profile import ExecutionTargetProfile, generic_prop_target_profile
from cfds.instrument_spec import CFDResearchInstrument, default_us500_cfd_instrument
from cfds.source_profile import HistoricalSourceProfile, dukascopy_us500_source_profile
from control_plane.security import ConfigValidationError
from data.acquisition.download_job import (
    AcquisitionMode,
    AcquisitionStage,
    DownloadJob,
    DownloadJobError,
    DownloadJobState,
    DownloadJobStore,
    new_download_job,
)
from data.acquisition.rate_limit import RateLimiter
from data.catalog.catalog import DatasetCatalog
from data.catalog.cfd_import import CFDIngestResult, ingest_cfd_quotes, probe_quote_frame
from data.catalog.safe_paths import default_import_root, resolve_safe_import_path
from data.events.enums import EventSchemaType, PriceConvention
from data.providers.dukascopy_download import (
    DukascopyDownloader,
    HttpArchiveFetcher,
    LocalArchiveFetcher,
    automated_download_available,
)
from data.providers.dukascopy_parser import parse_tick_bi5
from data.providers.dukascopy_symbols import get_dukascopy_symbol, list_dukascopy_symbols

ARCHIVE_ROOT_ENV = "QUANT_DUKASCOPY_ARCHIVE_ROOT"

CFD_BANNER = "DUKASCOPY USA500 DATA IS BROKER CFD DATA. IT IS NOT CME ES FUTURES DATA."

QUOTE_COLUMN_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time", "gmt_time"),
    "bid": ("bid", "bid_price", "bidprice"),
    "ask": ("ask", "ask_price", "askprice"),
    "bid_volume": ("bid_volume", "bidvolume", "bid_size"),
    "ask_volume": ("ask_volume", "askvolume", "ask_size"),
}


def _normalize_quote_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map a manual export's column names onto the canonical quote schema."""
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    renames: dict[str, str] = {}
    for canonical, aliases in QUOTE_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                renames[lowered[alias]] = canonical
                break
    out = frame.rename(columns=renames)
    missing = {"timestamp", "bid", "ask"} - set(out.columns)
    if missing:
        raise ConfigValidationError(
            f"export is missing bid/ask quote columns: {sorted(missing)}. "
            f"Recognized names: {QUOTE_COLUMN_ALIASES}"
        )
    return out


@dataclass
class DataAcquisitionAPI:
    """Commands and read models backing the Data Acquisition dashboard page."""

    catalog: DatasetCatalog
    job_store: DownloadJobStore
    storage_root: Path
    import_root: Path = field(default_factory=default_import_root)
    archive_root: Path | None = None
    source: HistoricalSourceProfile = field(default_factory=dukascopy_us500_source_profile)
    instrument: CFDResearchInstrument = field(default_factory=default_us500_cfd_instrument)
    target: ExecutionTargetProfile = field(default_factory=generic_prop_target_profile)

    def __post_init__(self) -> None:
        self.storage_root = Path(self.storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.import_root = Path(self.import_root)
        if self.archive_root is None:
            env_root = os.environ.get(ARCHIVE_ROOT_ENV)
            self.archive_root = Path(env_root) if env_root else None
        elif self.archive_root:
            self.archive_root = Path(self.archive_root)

    # ---------- read models ----------

    def source_summary(self) -> dict[str, Any]:
        return {
            "source": self.source.as_dict(),
            "instrument": self.instrument.as_dict(),
            "verified_symbols": [s.as_dict() for s in list_dukascopy_symbols()],
            "acquisition_modes": [m.value for m in AcquisitionMode],
            "stages": [s.value for s in AcquisitionStage],
            "job_states": [s.value for s in DownloadJobState],
            "automated_download": automated_download_available(
                enable_network=self.network_enabled
            ),
            "archive_root_configured": bool(
                self.archive_root and self.archive_root.is_dir()
            ),
            "archive_root_env": ARCHIVE_ROOT_ENV,
            "import_root": str(self.import_root),
            "banner": CFD_BANNER,
        }

    @property
    def network_enabled(self) -> bool:
        return bool(os.environ.get("QUANT_DUKASCOPY_ENABLE_DOWNLOAD"))

    def list_jobs(self) -> list[dict[str, Any]]:
        return [j.as_dict() for j in self.job_store.list_jobs()]

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.job_store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.as_dict()

    # ---------- commands ----------

    def create_job(
        self,
        *,
        symbol: str,
        start: str,
        end: str,
        granularity: str = "tick",
        acquisition_mode: str = AcquisitionMode.MANUAL_EXPORT_IMPORT.value,
        stage: str | None = None,
        rate_limit_rps: float = 2.0,
        retry_limit: int = 3,
    ) -> dict[str, Any]:
        resolved = get_dukascopy_symbol(symbol)
        mode = AcquisitionMode(acquisition_mode)
        if mode is AcquisitionMode.AUTOMATED_PUBLIC_DOWNLOAD and not self.network_enabled:
            raise ConfigValidationError(
                "automated public download is disabled; set "
                "QUANT_DUKASCOPY_ENABLE_DOWNLOAD=1 or use MANUAL_EXPORT_IMPORT"
            )
        stage_enum = (
            AcquisitionStage(stage)
            if stage
            else _infer_stage(date.fromisoformat(start), date.fromisoformat(end))
        )
        try:
            job = new_download_job(
                provider_id=self.source.provider_id,
                source_symbol=resolved.source_symbol,
                start_date=start,
                end_date=end,
                requested_event_type=EventSchemaType.QUOTE,
                requested_granularity=granularity,
                source_timezone=self.source.source_timezone,
                price_convention=PriceConvention.UNKNOWN,
                acquisition_mode=mode,
                stage=stage_enum,
                rate_limit_rps=rate_limit_rps,
                retry_limit=retry_limit,
            )
        except DownloadJobError as exc:
            raise ConfigValidationError(str(exc)) from exc
        self.job_store.save(job)
        return job.as_dict()

    def probe_one_day(
        self,
        *,
        symbol: str,
        day: str,
        relative_path: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Stage 1: parse a single day and report quality without storing anything."""
        resolved = get_dukascopy_symbol(symbol)
        quotes = self._load_day(resolved.source_symbol, day, relative_path=relative_path)
        report = probe_quote_frame(quotes, source=self.source)
        report["day"] = day
        report["source_of_bytes"] = (
            "manual_export" if relative_path else "local_bi5_archive"
        )
        if job_id:
            job = self.job_store.get(job_id)
            if job is not None:
                job.state = DownloadJobState.PROBING
                job.probe_report = {
                    k: v for k, v in report.items() if k not in {"sample_rows", "quality"}
                }
                job.quality_report = report["quality"]
                self.job_store.save(job)
                report["download_job_id"] = job_id
        return report

    def run_job(self, job_id: str) -> dict[str, Any]:
        """Acquire every chunk for a job, reusing anything already complete."""
        job = self.job_store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.acquisition_mode is AcquisitionMode.AUTOMATED_PUBLIC_DOWNLOAD:
            if not self.network_enabled:
                raise ConfigValidationError(
                    "automated public download is disabled; "
                    "set QUANT_DUKASCOPY_ENABLE_DOWNLOAD=1"
                )
            fetcher: Any = HttpArchiveFetcher(enable_network=True)
        else:
            if not (self.archive_root and self.archive_root.is_dir()):
                raise ConfigValidationError(
                    "no local .bi5 archive mirror configured; set "
                    f"{ARCHIVE_ROOT_ENV} or use the manual CSV/Parquet import path"
                )
            fetcher = LocalArchiveFetcher(self.archive_root)
        downloader = self._downloader(job)
        downloader.run(job, fetcher=fetcher, store=self.job_store)
        return self.get_job(job_id)

    def pause_job(self, job_id: str) -> dict[str, Any]:
        return self.job_store.transition(job_id, DownloadJobState.PAUSED).as_dict()

    def resume_job(self, job_id: str) -> dict[str, Any]:
        job = self.job_store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state is not DownloadJobState.PAUSED:
            raise ConfigValidationError(
                f"only PAUSED jobs can resume; job is {job.state.value}"
            )
        self.job_store.transition(job_id, DownloadJobState.DOWNLOADING)
        return self.run_job(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        self.job_store.transition(job_id, DownloadJobState.CANCEL_REQUESTED)
        return self.job_store.transition(
            job_id, DownloadJobState.CANCELLED, terminal_reason="operator_cancelled"
        ).as_dict()

    def register_dataset(
        self,
        *,
        job_id: str | None = None,
        symbol: str | None = None,
        relative_path: str | None = None,
        display_name: str | None = None,
    ) -> CFDIngestResult:
        """Persist to Bronze/Silver and register in the Dataset Catalog."""
        job: DownloadJob | None = self.job_store.get(job_id) if job_id else None
        resolved = get_dukascopy_symbol(
            symbol or (job.source_symbol if job else self.source.source_instrument_identifier)
        )
        expected_chunks: list[str] | None = None
        observed_chunks: list[str] | None = None

        if relative_path:
            quotes = self._load_manual_export(relative_path)
        elif job is not None:
            downloader = self._downloader(job)
            chunk_dir = downloader.storage_root / job.download_job_id
            quotes, observed_chunks = self._load_chunk_dir(
                chunk_dir, point_divisor=resolved.point_divisor
            )
            expected_chunks = [cid for cid, _ in downloader.plan_chunks(job)]
        else:
            raise ConfigValidationError(
                "register_dataset needs either job_id or relative_path"
            )

        result = ingest_cfd_quotes(
            quotes,
            catalog=self.catalog,
            source=self.source,
            instrument=self.instrument,
            target=self.target,
            display_name=display_name,
            expected_chunk_ids=expected_chunks,
            observed_chunk_ids=observed_chunks,
            download_job_id=job.download_job_id if job else None,
            acquisition_mode=(
                job.acquisition_mode.value
                if job
                else AcquisitionMode.MANUAL_EXPORT_IMPORT.value
            ),
        )
        if job is not None:
            job.registered_dataset_id = result.dataset_id
            job.quality_report = result.quality.as_dict()
            self.job_store.save(job)
        return result

    # ---------- loading helpers ----------

    def _downloader(self, job: DownloadJob) -> DukascopyDownloader:
        return DukascopyDownloader(
            storage_root=self.storage_root / "chunks",
            rate_limiter=RateLimiter(requests_per_second=job.rate_limit_rps),
        )

    def _load_day(
        self, source_symbol: str, day: str, *, relative_path: str | None
    ) -> pd.DataFrame:
        if relative_path:
            frame = self._load_manual_export(relative_path)
            target_day = pd.Timestamp(day, tz="UTC").date()
            same_day = frame["timestamp"].dt.date == target_day
            selected = frame.loc[same_day].reset_index(drop=True)
            if selected.empty:
                raise ConfigValidationError(f"export contains no rows for {day}")
            return selected

        if not (self.archive_root and self.archive_root.is_dir()):
            raise ConfigValidationError(
                f"no local .bi5 archive mirror configured (set {ARCHIVE_ROOT_ENV}) and "
                "no manual export selected"
            )
        resolved = get_dukascopy_symbol(source_symbol)
        moment = pd.Timestamp(day, tz="UTC")
        frames: list[pd.DataFrame] = []
        for hour in range(24):
            path = (
                self.archive_root
                / resolved.source_symbol
                / f"{moment.year:04d}"
                / f"{moment.month - 1:02d}"
                / f"{moment.day:02d}"
                / f"{hour:02d}h_ticks.bi5"
            )
            if not path.is_file():
                continue
            decoded = parse_tick_bi5(
                path.read_bytes(),
                hour_start=moment + pd.Timedelta(hours=hour),
                point_divisor=resolved.point_divisor,
            )
            if not decoded.empty:
                frames.append(decoded)
        if not frames:
            raise ConfigValidationError(
                f"no local .bi5 archive data found for {resolved.source_symbol} on {day}"
            )
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    def _load_manual_export(self, relative_path: str) -> pd.DataFrame:
        path = resolve_safe_import_path(self.import_root, relative_path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frame = _normalize_quote_columns(frame)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if frame["timestamp"].isna().all():
            raise ConfigValidationError("no parseable timestamps in the export")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        return frame.reset_index(drop=True)

    def _load_chunk_dir(
        self, chunk_dir: Path, *, point_divisor: int
    ) -> tuple[pd.DataFrame, list[str]]:
        frames: list[pd.DataFrame] = []
        observed: list[str] = []
        for path in sorted(Path(chunk_dir).glob("*.bi5")):
            chunk_id = path.stem
            hour = pd.Timestamp(
                f"{chunk_id[0:4]}-{chunk_id[4:6]}-{chunk_id[6:8]} {chunk_id[9:11]}:00",
                tz="UTC",
            )
            observed.append(chunk_id)
            decoded = parse_tick_bi5(
                path.read_bytes(), hour_start=hour, point_divisor=point_divisor
            )
            if not decoded.empty:
                frames.append(decoded)
        if not frames:
            raise ConfigValidationError("no decoded quotes in the acquired chunks")
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        return frame.reset_index(drop=True), observed


def _infer_stage(start: date, end: date) -> AcquisitionStage:
    span = (end - start).days + 1
    if span <= 1:
        return AcquisitionStage.STAGE_1_ONE_DAY_PROBE
    if span <= 31:
        return AcquisitionStage.STAGE_2_ONE_MONTH_VALIDATION
    return AcquisitionStage.STAGE_3_HISTORICAL_EXPANSION
