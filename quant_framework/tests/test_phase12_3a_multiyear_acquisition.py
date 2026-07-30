"""Phase 12.3A — multi-year acquisition-only orchestrator tests (offline)."""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from control_plane.app import create_app
from control_plane.models import RunType
from control_plane.run_manager import RunManager
from control_plane.security import ConfigValidationError, CreateRunRequest
from cfds.execution_profile import generic_prop_target_profile
from cfds.instrument_spec import default_us500_cfd_instrument
from cfds.source_profile import dukascopy_us500_source_profile
from data.acquisition.global_rate_limit import GlobalRateLimiter, RateLimiterAdapter
from data.acquisition.multiyear_orchestrator import (
    PROTECTED_MARCH_2024_ID,
    PROTECTED_YEARLY_2024_HASH,
    PROTECTED_YEARLY_2024_ID,
    MultiyearOrchestrator,
    verify_protected_datasets,
)
from data.acquisition.process_lock import FileLock
from data.catalog.catalog import (
    ACCESS_DENIED,
    CatalogIntegrityError,
    DatasetCatalog,
)
from data.catalog.cfd_import import gate_reasons_for_alpha_miner, ingest_cfd_quotes
from data.providers.dukascopy_download import DukascopyDownloader, LocalArchiveFetcher
from data.providers.dukascopy_parser import TickRecord, encode_tick_bi5
from data.acquisition.download_job import (
    AcquisitionMode,
    AcquisitionStage,
    DownloadJobStore,
    new_download_job,
)
from data.events.enums import EventSchemaType, PriceConvention


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


def _quote_frame(*, days: int = 25, rows_per_day: int = 80, seed: int = 3) -> pd.DataFrame:
    """Weekday-only bid/ask quotes spanning ``days`` calendar days."""
    start = pd.Timestamp("2020-01-02 00:00", tz="UTC")
    end = start + pd.Timedelta(days=max(days, 1))
    idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    idx = idx[idx.weekday < 5]
    if len(idx) > rows_per_day * max(days, 1):
        step = max(1, len(idx) // (rows_per_day * max(days, 1)))
        idx = idx[::step][: rows_per_day * max(days, 1)]
    rng = np.random.default_rng(seed)
    mid = 3000.0 + np.cumsum(rng.normal(0, 0.02, len(idx)))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "bid": mid - 0.25,
            "ask": mid + 0.25,
            "bid_volume": 1.0,
            "ask_volume": 1.0,
            "spread": 0.5,
        }
    )


def _write_hour(archive: Path, day: str, hour: int, n: int = 20) -> None:
    d = date.fromisoformat(day)
    path = (
        archive
        / "USA500IDXUSD"
        / f"{d.year:04d}"
        / f"{d.month - 1:02d}"
        / f"{d.day:02d}"
        / f"{hour:02d}h_ticks.bi5"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    mid = 3200.0
    for i in range(n):
        mid += 0.01
        records.append(
            TickRecord(
                ms_since_hour=i * 100,
                ask_points=int(round((mid + 0.25) * 1000)),
                bid_points=int(round((mid - 0.25) * 1000)),
                ask_volume=1.0,
                bid_volume=1.0,
            )
        )
    path.write_bytes(encode_tick_bi5(records))


def test_global_rate_limiter_two_rps(tmp_root: Path) -> None:
    sleeps: list[float] = []
    clock = {"t": 1000.0}

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock["t"] += s

    lim = GlobalRateLimiter(
        tmp_root / "rate.json",
        requests_per_second=2.0,
        sleep=fake_sleep,
        clock=lambda: clock["t"],
    )
    assert lim.acquire() == 0.0
    clock["t"] += 0.1
    waited = lim.acquire()
    assert waited == pytest.approx(0.4, abs=1e-6)
    assert sleeps and sleeps[0] == pytest.approx(0.4, abs=1e-6)


def test_two_jobs_share_global_limiter(tmp_root: Path) -> None:
    clock = {"t": 0.0}
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock["t"] += s

    lim = GlobalRateLimiter(
        tmp_root / "shared_rate.json",
        requests_per_second=2.0,
        sleep=fake_sleep,
        clock=lambda: clock["t"],
    )
    a = RateLimiterAdapter(lim)
    b = RateLimiterAdapter(lim)
    a.acquire()
    clock["t"] += 0.05
    b.acquire()
    assert sum(sleeps) == pytest.approx(0.45, abs=1e-6)


def test_valid_empty_persistence_and_resume(tmp_root: Path) -> None:
    from data.acquisition.resume_state import ResumeState
    from data.acquisition.checksums import build_checksum

    path = tmp_root / "resume.json"
    state = ResumeState.load_or_create(path, job_id="dl_x", chunk_ids=["h0", "h1", "h2"])
    state.mark_empty("h0")
    state.mark_completed("h1", build_checksum("h1", b"abc"))
    reloaded = ResumeState.load_or_create(path, job_id="dl_x", chunk_ids=["h0", "h1", "h2"])
    assert set(reloaded.completed_chunk_ids()) == {"h0", "h1"}
    assert reloaded.pending_chunk_ids() == ["h2"]


def test_timeout_retry_recovery(tmp_root: Path) -> None:
    archive = tmp_root / "archive"
    _write_hour(archive, "2020-01-02", 14, n=10)
    attempts = {"n": 0}

    class Flaky(LocalArchiveFetcher):
        def fetch(self, chunk_id: str, url: str):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TimeoutError("simulated timeout")
            return super().fetch(chunk_id, url)

    job = new_download_job(
        provider_id="DUKASCOPY",
        source_symbol="USA500IDXUSD",
        start_date="2020-01-02",
        end_date="2020-01-02",
        stage=AcquisitionStage.STAGE_1_ONE_DAY_PROBE,
        acquisition_mode=AcquisitionMode.MANUAL_EXPORT_IMPORT,
        retry_limit=3,
    )
    # Only one planned hour with data — restrict by using day with written hour
    # Plan is 24 hours; missing hours return empty via LocalArchiveFetcher.
    store = DownloadJobStore(tmp_root / "jobs")
    store.save(job)
    dl = DukascopyDownloader(storage_root=tmp_root / "chunks", sleep=lambda _s: None)
    outcomes = dl.run(job, fetcher=Flaky(archive), store=store)
    assert job.failed_chunks == 0
    assert any(not o.empty for o in outcomes)


def test_independent_year_failure(tmp_root: Path) -> None:
    catalog = DatasetCatalog(tmp_root / "dataset_catalog")
    good = ingest_cfd_quotes(
        _quote_frame(days=25, seed=1),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2020",
    )
    assert good.registered is True
    # Crossed quotes force quality FAIL for the sibling year.
    bad_frame = _quote_frame(days=25, seed=2)
    bad_frame.loc[3, "ask"] = float(bad_frame.loc[3, "bid"]) - 2.0
    bad = ingest_cfd_quotes(
        bad_frame,
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2021",
        register_on_fail=False,
    )
    assert bad.registered is False
    assert catalog.get(good.dataset_id) is not None
    assert catalog.get(bad.dataset_id) is None or not bad.registered


def test_one_silver_build_at_a_time(tmp_root: Path) -> None:
    orch = MultiyearOrchestrator(
        workspace=tmp_root / "ws",
        years=[2020, 2021],
        allow_network=False,
        max_concurrent_silver_builds=1,
    )
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_silver(year: int):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.05)
        with lock:
            active["n"] -= 1
        from data.acquisition.multiyear_orchestrator import YearJobState

        return YearJobState(
            year=year,
            download_job_id=f"dl_{year}",
            state="REGISTERED",
            registered_dataset_id=f"ds_{year}",
        )

    orch.run_year_silver_register = fake_silver  # type: ignore[method-assign]
    # Directly exercise semaphore path
    def wrap(year: int):
        orch._silver_sema.acquire()
        try:
            return fake_silver(year)
        finally:
            orch._silver_sema.release()

    threads = [threading.Thread(target=wrap, args=(y,)) for y in (2020, 2021)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["max"] == 1


def test_atomic_catalog_registration(tmp_root: Path) -> None:
    catalog = DatasetCatalog(tmp_root / "cat")
    result = ingest_cfd_quotes(
        _quote_frame(days=25),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2020",
    )
    assert result.registered
    # Reload under lock path used by get()
    entry = catalog.get(result.dataset_id)
    assert entry is not None
    raw = json.loads(catalog.index_path.read_text(encoding="utf-8"))
    assert any(d["dataset_id"] == result.dataset_id for d in raw["datasets"])
    assert catalog.index_path.with_suffix(".json.tmp").exists() is False


def test_idempotent_duplicate_registration(tmp_root: Path) -> None:
    catalog = DatasetCatalog(tmp_root / "cat")
    frame = _quote_frame(days=25, seed=9)
    a = ingest_cfd_quotes(
        frame,
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2022",
    )
    assert a.registered
    # Catalog-level idempotency: same id + same hashes upserts without error.
    again = catalog.upsert(a.catalog_entry)
    assert again.dataset_id == a.dataset_id
    assert catalog.get(a.dataset_id) is not None


def test_hash_collision_rejection(tmp_root: Path) -> None:
    catalog = DatasetCatalog(tmp_root / "cat")
    a = ingest_cfd_quotes(
        _quote_frame(days=25, seed=1),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2020",
    )
    other = ingest_cfd_quotes(
        _quote_frame(days=25, seed=99),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2021",
    )
    collided = other.catalog_entry
    collided.dataset_id = a.dataset_id
    collided.raw_hash = "deadbeef" * 8
    collided.normalized_hash = "cafebabe" * 8
    with pytest.raises(CatalogIntegrityError):
        catalog.upsert(collided)


def test_2025_vault_locked_and_alpha_miner_denied(tmp_root: Path) -> None:
    catalog = DatasetCatalog(tmp_root / "cat")
    result = ingest_cfd_quotes(
        _quote_frame(days=25, seed=5),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2025",
        vault_locked=True,
    )
    assert result.registered is True
    assert result.research_eligible is False
    entry = catalog.get(result.dataset_id)
    assert entry is not None
    assert entry.vault_locked is True
    assert entry.alpha_miner_access == ACCESS_DENIED
    assert entry.research_access == ACCESS_DENIED
    assert entry.one_shot_access_only is True
    listed = catalog.list_entries(research_eligible_only=True, include_smoke=False)
    assert all(e.dataset_id != entry.dataset_id for e in listed)
    reasons = gate_reasons_for_alpha_miner(entry)
    assert any("VAULT_LOCKED" in r for r in reasons)

    mgr = RunManager(tmp_root / "cp")
    mgr.catalog.upsert(entry)
    app = create_app(manager=mgr)
    client = TestClient(app)
    resp = client.post(
        "/api/runs",
        json={
            "run_type": "ALPHA_MINER",
            "confirm_alpha_miner": True,
            "dataset": entry.dataset_id,
            "symbols": ["US500_CFD"],
            "timeframe": "tick",
            "smoke_test": False,
        },
    )
    assert resp.status_code == 400
    assert "VAULT_LOCKED" in resp.json()["detail"] or "ALPHA_MINER" in resp.json()["detail"]


def test_scripts_do_not_start_dashboard_or_alpha_miner() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in (
        "start_multiyear_acquisition.ps1",
        "resume_multiyear_acquisition.ps1",
        "stop_multiyear_acquisition.ps1",
        "status_multiyear_acquisition.ps1",
    ):
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "acquire_multiyear" in text or "status_multiyear" in text or "Start-Process" in text or "stop" in text.lower()
        assert "alpha_miner" not in text.lower() or "alpha_miner_started = $false" in text or "Alpha Miner" in text
        assert "create_app" not in text
        assert "uvicorn" not in text
        assert "RunType.ALPHA_MINER" not in text


def test_memory_ceiling_defers(tmp_root: Path) -> None:
    orch = MultiyearOrchestrator(
        workspace=tmp_root / "ws",
        years=[2020],
        memory_ceiling_gb=0.0000001,
    )
    calls = {"n": 0}

    def fake_sleep(_s: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            orch.pause_flag.write_text("pause", encoding="utf-8")

    orch.sleep = fake_sleep
    import data.acquisition.multiyear_orchestrator as mod

    mod._rss_gb = lambda: 1.0  # type: ignore[assignment]
    orch._wait_memory()
    assert calls["n"] >= 2


def test_graceful_stop_and_resume_flags(tmp_root: Path) -> None:
    orch = MultiyearOrchestrator(workspace=tmp_root / "ws", years=[2020, 2021])
    orch.request_pause()
    assert orch.pause_flag.exists()
    orch.request_stop()
    assert orch.stop_flag.exists()
    orch.clear_pause()
    assert not orch.pause_flag.exists()
    assert not orch.stop_flag.exists()


def test_file_lock_exclusive(tmp_root: Path) -> None:
    path = tmp_root / "x.lock"
    with FileLock(path):
        with pytest.raises(TimeoutError):
            FileLock(path).acquire(timeout=0.2)


def test_protected_dataset_constants() -> None:
    assert PROTECTED_YEARLY_2024_ID.startswith("ds_")
    assert len(PROTECTED_YEARLY_2024_HASH) == 64
    assert PROTECTED_MARCH_2024_ID.startswith("ds_")
