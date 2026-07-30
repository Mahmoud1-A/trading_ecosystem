"""Phase 12.2 — Free real CFD data acquisition for prop research.

All tests use local fixtures. No internet access is required.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfds.cost_policy import ModeledCostSpec, assess_cost_model
from cfds.execution_profile import (
    compare_source_to_target,
    evaluate_paper_eligibility,
    generic_prop_target_profile,
)
from cfds.instrument_spec import default_us500_cfd_instrument
from cfds.source_profile import dukascopy_us500_source_profile
from cfds.symbol_mapping import SymbolMappingError, assert_not_futures_identifier
from config.asset_spec import default_es_futures
from control_plane.app import create_app
from control_plane.models import RunType
from control_plane.run_manager import RunManager
from control_plane.security import ConfigValidationError, CreateRunRequest
from data.acquisition.checksums import ChecksumMismatch, build_checksum, verify_chunk
from data.acquisition.download_job import (
    AcquisitionMode,
    AcquisitionStage,
    DownloadJobStore,
    new_download_job,
)
from data.acquisition.resume_state import ResumeState
from data.catalog.catalog import SMOKE_DATASET_ID, DatasetCatalog
from data.catalog.cfd_import import (
    gate_reasons_for_alpha_miner,
    ingest_cfd_quotes,
    probe_quote_frame,
)
from data.catalog.safe_paths import resolve_safe_import_path
from data.events.enums import VolumeType
from data.providers.dukascopy_download import (
    DukascopyDownloader,
    LocalArchiveFetcher,
)
from data.providers.dukascopy_parser import TickRecord, encode_tick_bi5, parse_tick_bi5
from data.providers.dukascopy_quality import assess_quote_quality
from data.providers.dukascopy_symbols import get_dukascopy_symbol
from engine.financing import CFDFinancingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _one_day_tick_records(*, n: int = 200, base_mid: float = 5000.0, crossed: bool = False):
    """Synthetic USA500-scale ticks. Prices are encoded as points (÷1000)."""
    records: list[TickRecord] = []
    mid = base_mid
    for i in range(n):
        mid += 0.01
        bid = mid - 0.125
        ask = mid + 0.125
        if crossed and i == n // 2:
            bid, ask = ask + 1.0, bid  # bid > ask
        records.append(
            TickRecord(
                ms_since_hour=i * 100,
                ask_points=int(round(ask * 1000)),
                bid_points=int(round(bid * 1000)),
                ask_volume=1.5,
                bid_volume=2.0,
            )
        )
    return records


def _write_hour_archive(archive: Path, *, day: str = "2024-03-04", hour: int = 13, n: int = 200) -> Path:
    """Layout mirrors Dukascopy: SYMBOL/YYYY/MM0/DD/HHh_ticks.bi5 (month 0-indexed)."""
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
    path.write_bytes(encode_tick_bi5(_one_day_tick_records(n=n)))
    return path


def _quote_frame(*, days: int = 1, rows_per_day: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Weekday-only bid/ask quotes spanning ``days`` calendar days."""
    start = pd.Timestamp("2024-03-04 00:00", tz="UTC")
    end = start + pd.Timedelta(days=max(days, 1))
    idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    idx = idx[idx.weekday < 5]
    if len(idx) > rows_per_day * max(days, 1):
        step = max(1, len(idx) // (rows_per_day * max(days, 1)))
        idx = idx[::step][: rows_per_day * max(days, 1)]
    rng = np.random.default_rng(seed)
    mid = 5000.0 + np.cumsum(rng.normal(0, 0.02, len(idx)))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "bid": mid - 0.125,
            "ask": mid + 0.125,
            "bid_volume": 1.0,
            "ask_volume": 1.0,
        }
    )


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def catalog(tmp_root: Path) -> DatasetCatalog:
    return DatasetCatalog(tmp_root / "dataset_catalog")


@pytest.fixture
def client(tmp_root: Path) -> TestClient:
    mgr = RunManager(tmp_root / "control_plane")
    return TestClient(create_app(manager=mgr, root=tmp_root / "control_plane"))


# ---------------------------------------------------------------------------
# 1. One-day Dukascopy fixture can be parsed
# ---------------------------------------------------------------------------


def test_one_day_dukascopy_fixture_parsed(tmp_root: Path) -> None:
    archive = tmp_root / "archive"
    path = _write_hour_archive(archive, hour=13, n=50)
    hour = pd.Timestamp("2024-03-04 13:00", tz="UTC")
    frame = parse_tick_bi5(path.read_bytes(), hour_start=hour, point_divisor=1000)
    assert len(frame) == 50
    assert list(frame.columns) == ["timestamp", "bid", "ask", "bid_volume", "ask_volume", "spread"]
    assert frame["timestamp"].dt.tz is not None
    sym = get_dukascopy_symbol("USA500IDXUSD")
    assert sym.source_symbol == "USA500IDXUSD"
    assert sym.verified is True


# ---------------------------------------------------------------------------
# 2. Bid and ask values are preserved
# ---------------------------------------------------------------------------


def test_bid_and_ask_preserved() -> None:
    records = [TickRecord(0, 5001250, 5001000, 1.5, 2.0)]
    payload = encode_tick_bi5(records)
    frame = parse_tick_bi5(
        payload, hour_start=pd.Timestamp("2024-03-04 13:00", tz="UTC"), point_divisor=1000
    )
    assert float(frame.iloc[0]["bid"]) == pytest.approx(5001.0)
    assert float(frame.iloc[0]["ask"]) == pytest.approx(5001.25)
    assert float(frame.iloc[0]["spread"]) == pytest.approx(0.25)
    assert float(frame.iloc[0]["bid_volume"]) == pytest.approx(2.0)
    assert float(frame.iloc[0]["ask_volume"]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 3. Broker data is never labeled exchange data
# ---------------------------------------------------------------------------


def test_broker_data_never_labeled_exchange(catalog: DatasetCatalog) -> None:
    source = dukascopy_us500_source_profile()
    instrument = default_us500_cfd_instrument()
    assert source.centralized_exchange_volume is False
    assert source.volume_type is VolumeType.TICK_ACTIVITY_PROXY
    result = ingest_cfd_quotes(
        _quote_frame(days=25, rows_per_day=120),
        catalog=catalog,
        source=source,
        instrument=instrument,
        target=generic_prop_target_profile(),
    )
    entry = result.catalog_entry
    assert entry.volume_type == VolumeType.TICK_ACTIVITY_PROXY.value
    assert entry.asset_class.lower() == "cfd"
    assert entry.futures_meta["applicable"] is False
    assert "EXCHANGE" not in entry.volume_type
    banner = entry.cfd_meta.get("instrument_class_banner", "")
    assert "NOT CME ES FUTURES" in banner


# ---------------------------------------------------------------------------
# 4. Tick activity is never labeled exchange volume
# ---------------------------------------------------------------------------


def test_tick_activity_never_exchange_volume() -> None:
    frame = _quote_frame(days=1, rows_per_day=100)
    report = assess_quote_quality(
        frame,
        volume_type=VolumeType.EXCHANGE_EXECUTED_VOLUME,
        centralized_exchange_volume=True,
    )
    assert report.research_eligible is False
    assert any("EXCHANGE_EXECUTED_VOLUME" in r for r in report.rejection_reasons)

    source = dukascopy_us500_source_profile()
    assert source.volume_type is not VolumeType.EXCHANGE_EXECUTED_VOLUME
    with pytest.raises(Exception):
        # HistoricalSourceProfile rejects exchange volume on CFD construction
        from cfds.source_profile import HistoricalSourceProfile, RetrievalMethod
        from data.events.enums import EventSchemaType, PriceConvention

        HistoricalSourceProfile(
            provider_id="X",
            provider_version="1",
            source_instrument_identifier="USA500IDXUSD",
            source_display_identifier="USA500.IDX/USD",
            internal_symbol="US500_CFD",
            asset_class="CFD",
            underlying_reference="S&P 500",
            source_timezone="UTC",
            broker_or_venue="X",
            available_event_types=(EventSchemaType.QUOTE,),
            price_convention=PriceConvention.BID,
            spread_available=True,
            volume_type=VolumeType.EXCHANGE_EXECUTED_VOLUME,
            centralized_exchange_volume=False,
            futures_contract_identity="NOT_APPLICABLE",
            futures_rollover="NOT_APPLICABLE",
            historical_availability={},
            retrieval_method=RetrievalMethod(),
            licensing_metadata={},
        )


# ---------------------------------------------------------------------------
# 5. USA500 CFD does not receive Futures rollover accounting
# ---------------------------------------------------------------------------


def test_cfd_no_futures_rollover() -> None:
    source = dukascopy_us500_source_profile()
    instrument = default_us500_cfd_instrument()
    assert source.futures_rollover == "NOT_APPLICABLE"
    assert source.futures_contract_identity == "NOT_APPLICABLE"
    assert instrument.as_dict()["futures_rollover"] == "NOT_APPLICABLE"
    assert instrument.as_dict()["futures_expiration"] is None
    # CFD financing engine exists; futures rollover rules do not apply
    engine = CFDFinancingEngine(asset=instrument.to_asset_spec())
    assert hasattr(engine, "asset")
    assert not hasattr(instrument, "rollover_rules")
    assert not hasattr(instrument, "continuous_series")


# ---------------------------------------------------------------------------
# 6. USA500 CFD does not inherit ES multiplier or tick value
# ---------------------------------------------------------------------------


def test_cfd_does_not_inherit_es_multiplier() -> None:
    es = default_es_futures()
    cfd = default_us500_cfd_instrument()
    assert es.multiplier == 50.0
    assert es.tick_value == 12.50
    assert cfd.value_per_point == 1.0
    assert cfd.value_per_point != es.multiplier
    payload = cfd.as_dict()
    assert payload["futures_multiplier"] is None
    assert payload["futures_tick_value"] is None
    with pytest.raises(SymbolMappingError):
        assert_not_futures_identifier("ES", field_name="provider_symbol")
    with pytest.raises(SymbolMappingError):
        assert_not_futures_identifier("CME:ES", field_name="provider_symbol")


# ---------------------------------------------------------------------------
# 7. Download jobs resume without duplicate chunks
# ---------------------------------------------------------------------------


def test_download_resume_no_duplicate_chunks(tmp_root: Path) -> None:
    archive = tmp_root / "archive"
    for hour in (13, 14):
        _write_hour_archive(archive, hour=hour, n=20)
    job = new_download_job(
        provider_id="DUKASCOPY",
        source_symbol="USA500IDXUSD",
        start_date="2024-03-04",
        end_date="2024-03-04",
        acquisition_mode=AcquisitionMode.MANUAL_EXPORT_IMPORT,
        stage=AcquisitionStage.STAGE_1_ONE_DAY_PROBE,
    )
    store = DownloadJobStore(tmp_root / "jobs")
    store.save(job)
    downloader = DukascopyDownloader(storage_root=tmp_root / "chunks")
    fetcher = LocalArchiveFetcher(archive)

    first = downloader.run(job, fetcher=fetcher, store=store)
    assert job.state.value == "COMPLETED"
    nonempty = [o for o in first if not o.empty and not o.reused]
    assert len(nonempty) == 2

    # Resume: every previously completed chunk must be reused, none re-fetched
    second = downloader.run(job, fetcher=fetcher, store=store)
    assert all(o.reused for o in second)
    assert sum(1 for o in second if not o.reused) == 0
    assert len(second) == 24  # one-day = 24 hour chunks


# ---------------------------------------------------------------------------
# 8. Invalid checksums reject corrupted chunks
# ---------------------------------------------------------------------------


def test_invalid_checksum_rejects_corrupted_chunk() -> None:
    checksum = build_checksum("h13", b"good-bytes")
    verify_chunk(checksum, b"good-bytes")
    with pytest.raises(ChecksumMismatch):
        verify_chunk(checksum, b"corrupted!!!")


# ---------------------------------------------------------------------------
# 9. Duplicate timestamps are reported
# ---------------------------------------------------------------------------


def test_duplicate_timestamps_reported() -> None:
    frame = _quote_frame(days=1, rows_per_day=50)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = assess_quote_quality(frame, volume_type=VolumeType.TICK_ACTIVITY_PROXY)
    assert report.duplicate_count >= 1
    assert report.research_eligible is False
    assert any("duplicate" in r.lower() for r in report.rejection_reasons)


# ---------------------------------------------------------------------------
# 10. Invalid bid > ask rows are rejected
# ---------------------------------------------------------------------------


def test_crossed_quotes_rejected() -> None:
    frame = _quote_frame(days=1, rows_per_day=40)
    frame.loc[5, "ask"] = float(frame.loc[5, "bid"]) - 1.0
    report = assess_quote_quality(frame, volume_type=VolumeType.TICK_ACTIVITY_PROXY)
    assert report.crossed_quote_count >= 1
    assert report.research_eligible is False
    assert any("bid > ask" in r or "crossed" in r.lower() for r in report.rejection_reasons)


# ---------------------------------------------------------------------------
# 11. Missing spread data requires a cost model
# ---------------------------------------------------------------------------


def test_missing_spread_requires_cost_model() -> None:
    # No quotes, no modeled cost → not eligible
    missing = assess_cost_model(spread_available=False, quotes=None, modeled_cost=None)
    assert missing.eligibility.value == "NOT_ELIGIBLE_MISSING_COST_MODEL"
    assert missing.is_eligible is False

    # Zero spread is never accepted as a default model
    bad_model = ModeledCostSpec(
        base_spread_points=0.0,
        stressed_spread_points=0.0,
        commission_per_lot=0.0,
        slippage_points=0.1,
        financing_configured=True,
    )
    zero = assess_cost_model(spread_available=False, modeled_cost=bad_model)
    assert zero.is_eligible is False

    # Valid modeled cost works when bid/ask is absent
    good_model = ModeledCostSpec(
        base_spread_points=0.5,
        stressed_spread_points=3.0,
        commission_per_lot=0.0,
        slippage_points=0.1,
        financing_configured=True,
    )
    ok = assess_cost_model(spread_available=False, modeled_cost=good_model)
    assert ok.eligibility.value == "ELIGIBLE_WITH_MODELED_SPREAD"

    # Observed spread from bid/ask
    quotes = _quote_frame(days=1, rows_per_day=30).rename(
        columns={"bid": "bid_price", "ask": "ask_price"}
    )
    observed = assess_cost_model(spread_available=True, quotes=quotes)
    assert observed.eligibility.value == "ELIGIBLE_WITH_OBSERVED_SPREAD"


# ---------------------------------------------------------------------------
# 12. Path traversal is rejected
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(tmp_root: Path) -> None:
    root = tmp_root / "data_import"
    root.mkdir()
    with pytest.raises(ConfigValidationError, match="traversal|escapes|absolute"):
        resolve_safe_import_path(root, "../secrets.csv")
    with pytest.raises(ConfigValidationError):
        resolve_safe_import_path(root, "..\\..\\etc\\passwd.csv")


# ---------------------------------------------------------------------------
# 13. Unsafe absolute paths are rejected
# ---------------------------------------------------------------------------


def test_absolute_paths_rejected(tmp_root: Path) -> None:
    root = tmp_root / "data_import"
    root.mkdir()
    with pytest.raises(ConfigValidationError, match="absolute"):
        resolve_safe_import_path(root, "C:/Users/DEMO/secret.csv")
    with pytest.raises(ConfigValidationError):
        resolve_safe_import_path(root, "/etc/passwd.csv")
    with pytest.raises(ConfigValidationError, match="URL|absolute"):
        resolve_safe_import_path(root, "https://evil.example/data.csv")


# ---------------------------------------------------------------------------
# 14. Quality-approved dataset appears in Dataset Catalog
# ---------------------------------------------------------------------------


def test_quality_approved_dataset_in_catalog(catalog: DatasetCatalog) -> None:
    # Need enough history + observations to pass Alpha Miner gates
    frame = _quote_frame(days=25, rows_per_day=200)
    result = ingest_cfd_quotes(
        frame,
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
    )
    assert result.research_eligible is True
    entry = catalog.get(result.dataset_id)
    assert entry is not None
    assert entry.research_eligible is True
    assert entry.cost_model_assigned is True
    assert entry.instrument_spec_assigned is True
    assert entry.source_profile_assigned is True
    assert entry.volume_type == VolumeType.TICK_ACTIVITY_PROXY.value
    listed = catalog.list_entries(research_eligible_only=True, include_smoke=False)
    assert any(e.dataset_id == result.dataset_id for e in listed)


# ---------------------------------------------------------------------------
# 15. Invalid datasets cannot start Alpha Miner
# ---------------------------------------------------------------------------


def test_invalid_dataset_cannot_start_alpha_miner(
    catalog: DatasetCatalog, client: TestClient
) -> None:
    # Tiny probe that fails duration/observation gates
    frame = _quote_frame(days=1, rows_per_day=50)
    # Inject crossed quote so quality fails
    frame.loc[3, "ask"] = float(frame.loc[3, "bid"]) - 2.0
    result = ingest_cfd_quotes(
        frame,
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
    )
    assert result.research_eligible is False
    # Sync into the control-plane catalog used by the client
    client.app.state.manager.catalog.upsert(result.catalog_entry)

    resp = client.post(
        "/api/runs",
        json={
            "run_type": "ALPHA_MINER",
            "confirm_alpha_miner": True,
            "dataset": result.dataset_id,
            "symbols": ["US500_CFD"],
            "timeframe": "tick",
            "smoke_test": False,
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "research_eligible" in detail or "not research" in detail or "cannot start" in detail


# ---------------------------------------------------------------------------
# 16. Historical source and target broker profiles remain separate
# ---------------------------------------------------------------------------


def test_source_and_target_profiles_remain_separate() -> None:
    source = dukascopy_us500_source_profile()
    instrument = default_us500_cfd_instrument()
    target = generic_prop_target_profile(broker_id="OTHER_PROP_BROKER", broker_symbol="US500")
    assert source.broker_or_venue != target.broker_id
    assert source.provider_id == "DUKASCOPY"
    warnings = compare_source_to_target(source=source, instrument=instrument, target=target)
    codes = {w.code for w in warnings}
    assert "SOURCE_TARGET_BROKER_MISMATCH" in codes
    assert "TARGET_FEED_NOT_IMPORTED" in codes
    # Profiles are distinct objects / dicts
    assert source.as_dict()["broker_or_venue"] != target.as_dict()["broker_id"]


# ---------------------------------------------------------------------------
# 17. Dukascopy cannot become Paper eligible for unrelated broker without
#     target-feed validation
# ---------------------------------------------------------------------------


def test_dukascopy_not_paper_eligible_without_target_validation(catalog: DatasetCatalog) -> None:
    result = ingest_cfd_quotes(
        _quote_frame(days=25, rows_per_day=200),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(broker_id="OTHER_PROP_BROKER"),
    )
    entry = result.catalog_entry
    # Research may pass; Paper must not
    assert entry.paper_eligible is False
    assert "target" in entry.paper_eligibility_reason.lower() or "broker" in (
        entry.paper_eligibility_reason.lower()
    )
    decision = evaluate_paper_eligibility(
        compare_source_to_target(
            source=dukascopy_us500_source_profile(),
            instrument=default_us500_cfd_instrument(),
            target=generic_prop_target_profile(broker_id="OTHER_PROP_BROKER"),
        )
    )
    assert decision.paper_eligible is False
    assert "TARGET_FEED_NOT_IMPORTED" in decision.blocking_codes


# ---------------------------------------------------------------------------
# 18. synthetic_demo remains smoke-test-only
# ---------------------------------------------------------------------------


def test_synthetic_demo_remains_smoke_only(catalog: DatasetCatalog, client: TestClient) -> None:
    entry = catalog.get(SMOKE_DATASET_ID)
    assert entry is not None
    assert entry.smoke_test_only is True
    assert entry.research_eligible is False

    resp = client.post(
        "/api/runs",
        json={
            "run_type": "ALPHA_MINER",
            "confirm_alpha_miner": True,
            "dataset": "synthetic_demo",
            "smoke_test": False,
        },
    )
    assert resp.status_code == 400
    assert "smoke" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Extra coverage: probe, staged span guard, API, resume ledger
# ---------------------------------------------------------------------------


def test_probe_requires_approval_and_stores_nothing(catalog: DatasetCatalog) -> None:
    frame = _quote_frame(days=1, rows_per_day=100)
    before = len(catalog.list_entries(include_smoke=False))
    report = probe_quote_frame(frame, source=dukascopy_us500_source_profile())
    assert report["approval_required"] is True
    assert report["source_symbol"] == "USA500IDXUSD"
    assert "NOT CME ES FUTURES" in report["instrument_class_banner"]
    assert len(catalog.list_entries(include_smoke=False)) == before


def test_stage_guard_blocks_multiyear_probe() -> None:
    with pytest.raises(Exception, match="at most 1 day"):
        new_download_job(
            provider_id="DUKASCOPY",
            source_symbol="USA500IDXUSD",
            start_date="2020-01-01",
            end_date="2024-01-01",
            stage=AcquisitionStage.STAGE_1_ONE_DAY_PROBE,
        )


def test_resume_state_skips_completed_chunks(tmp_root: Path) -> None:
    path = tmp_root / "resume.json"
    state = ResumeState.load_or_create(
        path, job_id="dl_test", chunk_ids=["h00", "h01", "h02"]
    )
    state.mark_completed("h00", build_checksum("h00", b"abc"))
    state.mark_empty("h01")
    reloaded = ResumeState.load_or_create(
        path, job_id="dl_test", chunk_ids=["h00", "h01", "h02"]
    )
    assert reloaded.pending_chunk_ids() == ["h02"]
    assert set(reloaded.completed_chunk_ids()) == {"h00", "h01"}


def test_acquisition_api_source_and_source_target(client: TestClient) -> None:
    src = client.get("/api/acquisition/source")
    assert src.status_code == 200
    body = src.json()
    assert "NOT CME ES FUTURES" in body["banner"]
    assert body["source"]["exact_source_symbol"] == "USA500IDXUSD"
    assert body["source"]["centralized_exchange_volume"] is False
    # Reflects process env — multi-year acquisition may enable download globally
    expected = os.environ.get("QUANT_DUKASCOPY_ENABLE_DOWNLOAD", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
    }
    assert body["automated_download"]["enabled"] is expected

    st = client.get("/api/acquisition/source_target")
    assert st.status_code == 200
    payload = st.json()
    assert payload["paper_eligibility"]["paper_eligible"] is False
    assert payload["critical_count"] >= 1


def test_acquisition_rejects_futures_symbol(client: TestClient) -> None:
    resp = client.post(
        "/api/acquisition/jobs",
        json={"symbol": "ES", "start": "2024-03-04", "end": "2024-03-04"},
    )
    assert resp.status_code in {400, 404}


def test_gate_reasons_for_eligible_dataset(catalog: DatasetCatalog) -> None:
    result = ingest_cfd_quotes(
        _quote_frame(days=25, rows_per_day=200),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
    )
    assert gate_reasons_for_alpha_miner(result.catalog_entry) == []
    # Mark derived bar timeframes as available (immutable 2024-style schema)
    entry = catalog.get(result.dataset_id)
    assert entry is not None
    entry.schema = {
        **(entry.schema or {}),
        "derived_bar_rules": ["1min", "5min"],
        "derived_bar_sides": ["bid", "ask"],
    }
    catalog.upsert(entry)
    req = CreateRunRequest(
        run_type=RunType.ALPHA_MINER,
        confirm_alpha_miner=True,
        dataset=result.dataset_id,
        symbols=["US500_CFD"],
        timeframe="1m",
        smoke_test=False,
        cost_model_version="cfd_cost_v1",
    )
    req.validate_against_catalog(catalog)


def test_dashboard_pages_served(client: TestClient) -> None:
    assert client.get("/data-acquisition").status_code == 200
    assert client.get("/source-target").status_code == 200
    js = client.get("/assets/dashboard.js").text
    assert "renderDataAcquisition" in js
    assert "renderSourceTarget" in js
    assert "IT IS NOT CME ES FUTURES DATA" in js


def test_yearly_and_monthly_dataset_ids_differ(catalog: DatasetCatalog) -> None:
    """Full-year registration must not collide with / overwrite a monthly dataset."""
    from data.acquire_dukascopy import dataset_display_name

    assert dataset_display_name("2024-03-01", "2024-03-31") == (
        "dukascopy_us500_cfd_ticks_2024_03"
    )
    assert dataset_display_name("2024-01-01", "2024-12-31") == (
        "dukascopy_us500_cfd_ticks_2024"
    )

    monthly = ingest_cfd_quotes(
        _quote_frame(days=25, rows_per_day=80),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2024_03",
    )
    yearly = ingest_cfd_quotes(
        _quote_frame(days=30, rows_per_day=80, seed=11),
        catalog=catalog,
        source=dukascopy_us500_source_profile(),
        instrument=default_us500_cfd_instrument(),
        target=generic_prop_target_profile(),
        display_name="dukascopy_us500_cfd_ticks_2024",
    )
    assert monthly.registered and yearly.registered
    assert monthly.dataset_id != yearly.dataset_id
    assert catalog.get(monthly.dataset_id) is not None
    assert catalog.get(yearly.dataset_id) is not None
    assert catalog.get(monthly.dataset_id).display_name != (
        catalog.get(yearly.dataset_id).display_name
    )


def test_matching_download_job_reused_on_resume(tmp_root: Path) -> None:
    from data.acquisition.download_job import find_matching_download_job

    store = DownloadJobStore(tmp_root / "jobs")
    first = new_download_job(
        provider_id="DUKASCOPY",
        source_symbol="USA500IDXUSD",
        start_date="2024-01-01",
        end_date="2024-12-31",
        stage=AcquisitionStage.STAGE_3_HISTORICAL_EXPANSION,
        acquisition_mode=AcquisitionMode.AUTOMATED_PUBLIC_DOWNLOAD,
    )
    store.save(first)
    found = find_matching_download_job(
        store,
        source_symbol="USA500IDXUSD",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    assert found is not None
    assert found.download_job_id == first.download_job_id


def test_monthly_quality_table_covers_all_months() -> None:
    from data.providers.dukascopy_quality import build_monthly_quality_table

    frame = _quote_frame(days=40, rows_per_day=50)
    # Force timestamps into Jan–Feb 2024 window.
    start = pd.Timestamp("2024-01-02", tz="UTC")
    frame = frame.copy()
    frame["timestamp"] = [
        start + pd.Timedelta(minutes=i) for i in range(len(frame))
    ]
    table = build_monthly_quality_table(
        frame,
        chunk_outcomes=[
            {"chunk_id": "20240102T13h", "empty": False, "reused": True, "bytes": 10},
            {"chunk_id": "20240201T10h", "empty": True, "reused": False, "bytes": 0},
        ],
        expected_chunk_ids=[f"20240101T{h:02d}h" for h in range(24)],
        year=2024,
    )
    assert len(table) == 12
    assert table[0]["month_label"] == "2024-01"
    assert table[0]["unique_ticks"] > 0
    assert table[11]["month_label"] == "2024-12"
