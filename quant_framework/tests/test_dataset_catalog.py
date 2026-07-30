"""Phase 12.1 — Dataset Catalog, safe import, New Run constraints."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from control_plane.app import create_app
from control_plane.run_manager import RunManager
from control_plane.security import ConfigValidationError, CreateRunRequest
from data.catalog.catalog import SMOKE_DATASET_ID, DatasetCatalog
from data.catalog.import_pipeline import (
    ColumnMapping,
    ImportRequest,
    commit_import,
    validate_import,
)
from data.catalog.safe_paths import list_importable_files, resolve_safe_import_path
from data.events.enums import VolumeType
from fastapi.testclient import TestClient
from control_plane.models import RunType

CHI = ZoneInfo("America/Chicago")


def _write_bars_csv(path: Path, *, n: int = 12, bad_ohlc: bool = False, dup: bool = False) -> Path:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz=CHI)
    high = 101.0
    low = 99.0
    if bad_ohlc:
        high, low = 99.0, 101.0  # inverted
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": high,
            "low": low,
            "close": 100.5,
            "volume": 1_000.0,
        }
    )
    if dup:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_bars_parquet(path: Path, *, n: int = 12) -> Path:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz=CHI)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _futures_req(relative: str, **extra: object) -> ImportRequest:
    base = dict(
        relative_path=relative,
        asset_class="FUTURES",
        column_mapping=ColumnMapping(),
        provider_id="local_csv",
        provider_version="1.0.0",
        source_timezone="America/Chicago",
        exchange_or_broker="CME",
        timeframe="5min",
        volume_type=VolumeType.EXCHANGE_EXECUTED_VOLUME.value,
        root_symbol="ES",
        tradable_contract="ESH24",
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.5,
        contract_expiration="2024-03-15",
        confirm=True,
        display_name="ES ESH24 5min local",
    )
    base.update(extra)
    return ImportRequest(**base)  # type: ignore[arg-type]


@pytest.fixture()
def import_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data_import"
    root.mkdir()
    monkeypatch.setenv("QUANT_DATA_IMPORT_ROOT", str(root))
    return root


@pytest.fixture()
def client(tmp_path: Path, import_root: Path) -> TestClient:
    mgr = RunManager(tmp_path / "cp", max_workers=1)
    return TestClient(create_app(mgr, root=tmp_path / "cp"))


class TestSafePaths:
    def test_path_traversal_rejected(self, import_root: Path) -> None:
        with pytest.raises(ConfigValidationError, match="traversal|escapes"):
            resolve_safe_import_path(import_root, "../secrets.csv")
        with pytest.raises(ConfigValidationError, match="traversal|escapes|absolute"):
            resolve_safe_import_path(import_root, "..\\..\\Windows\\System32\\x.csv")

    def test_absolute_unsafe_paths_rejected(self, import_root: Path) -> None:
        with pytest.raises(ConfigValidationError, match="absolute"):
            resolve_safe_import_path(import_root, "C:/temp/evil.csv")
        with pytest.raises(ConfigValidationError, match="absolute|/"):
            resolve_safe_import_path(import_root, "/etc/passwd.csv")

    def test_pickle_rejected(self, import_root: Path) -> None:
        p = import_root / "x.pkl"
        p.write_bytes(b"not-a-pickle")
        with pytest.raises(ConfigValidationError, match="forbidden|unsupported"):
            resolve_safe_import_path(import_root, "x.pkl")

    def test_url_rejected(self, import_root: Path) -> None:
        with pytest.raises(ConfigValidationError, match="URL"):
            resolve_safe_import_path(import_root, "https://example.com/x.csv")


class TestImportPipeline:
    def test_csv_import_bronze_silver(self, import_root: Path, tmp_path: Path) -> None:
        _write_bars_csv(import_root / "esh24_5min.csv", n=12)
        catalog = DatasetCatalog(tmp_path / "catalog")
        entry = commit_import(
            import_root=import_root,
            catalog=catalog,
            req=_futures_req("esh24_5min.csv"),
        )
        assert entry.immutable
        assert entry.raw_hash
        assert entry.normalized_hash
        assert Path(entry.bronze_manifest_path or "").exists() or entry.bronze_manifest_path
        assert entry.row_count >= 12
        assert entry.asset_class in {"futures", "FUTURES"} or entry.asset_class == "futures"
        assert "ESH24" in entry.tradable_contracts

    def test_parquet_import(self, import_root: Path, tmp_path: Path) -> None:
        _write_bars_parquet(import_root / "esh24_5min.parquet", n=10)
        catalog = DatasetCatalog(tmp_path / "catalog")
        entry = commit_import(
            import_root=import_root,
            catalog=catalog,
            req=_futures_req("esh24_5min.parquet", display_name="ES parquet"),
        )
        assert entry.row_count >= 10
        assert entry.provider_id == "local_csv"

    def test_manifest_deterministic(self, import_root: Path, tmp_path: Path) -> None:
        _write_bars_csv(import_root / "a.csv", n=8)
        c1 = DatasetCatalog(tmp_path / "c1")
        c2 = DatasetCatalog(tmp_path / "c2")
        e1 = commit_import(import_root=import_root, catalog=c1, req=_futures_req("a.csv"))
        e2 = commit_import(import_root=import_root, catalog=c2, req=_futures_req("a.csv"))
        assert e1.raw_hash == e2.raw_hash
        assert e1.normalized_hash == e2.normalized_hash

    def test_invalid_ohlc_not_eligible(self, import_root: Path, tmp_path: Path) -> None:
        _write_bars_csv(import_root / "bad.csv", n=8, bad_ohlc=True)
        catalog = DatasetCatalog(tmp_path / "catalog")
        entry = commit_import(
            import_root=import_root,
            catalog=catalog,
            req=_futures_req("bad.csv", display_name="bad ohlc"),
        )
        assert entry.research_eligible is False
        assert entry.quality_status == "FAIL"
        assert entry.rejection_reasons
        listed = catalog.get(entry.dataset_id)
        assert listed is not None
        assert listed.research_eligible is False

    def test_duplicate_timestamps_reported(self, import_root: Path) -> None:
        _write_bars_csv(import_root / "dup.csv", n=6, dup=True)
        report = validate_import(import_root=import_root, req=_futures_req("dup.csv", confirm=False))
        assert report["duplicate_count"] >= 1
        assert report["research_eligible"] is False
        assert any("duplicate" in r.lower() for r in report["rejection_reasons"])

    def test_futures_without_contract_rejected(self, import_root: Path) -> None:
        _write_bars_csv(import_root / "root_only.csv", n=6)
        report = validate_import(
            import_root=import_root,
            req=_futures_req("root_only.csv", tradable_contract=None, confirm=False),
        )
        assert report["research_eligible"] is False
        assert any("tradable_contract" in r for r in report["rejection_reasons"])

    def test_futures_continuous_root_as_contract_rejected(self, import_root: Path) -> None:
        _write_bars_csv(import_root / "cont.csv", n=6)
        report = validate_import(
            import_root=import_root,
            req=_futures_req("cont.csv", tradable_contract="ES", confirm=False),
        )
        assert report["research_eligible"] is False
        assert any("continuous root" in r.lower() or "tradable_contract" in r for r in report["rejection_reasons"])

    def test_broker_volume_not_exchange(self, import_root: Path) -> None:
        _write_bars_csv(import_root / "cfd.csv", n=6)
        req = ImportRequest(
            relative_path="cfd.csv",
            asset_class="CFD",
            column_mapping=ColumnMapping(),
            provider_id="broker_x",
            provider_version="1.0.0",
            source_timezone="UTC",
            exchange_or_broker="DemoBroker",
            timeframe="5min",
            volume_type=VolumeType.EXCHANGE_EXECUTED_VOLUME.value,
            root_symbol="EURUSD",
            broker_symbol="EURUSD.r",
            spread_available=True,
            financing_rate_available=False,
            confirm=False,
        )
        report = validate_import(import_root=import_root, req=req)
        assert report["research_eligible"] is False
        blob = " ".join(report["rejection_reasons"]).lower()
        assert "exchange" in blob or "broker volume" in blob


class TestCatalogAndNewRun:
    def test_synthetic_demo_smoke_label(self, tmp_path: Path) -> None:
        cat = DatasetCatalog(tmp_path / "catalog")
        entry = cat.get(SMOKE_DATASET_ID)
        assert entry is not None
        assert entry.smoke_test_only
        assert entry.research_eligible is False
        assert "SMOKE TEST ONLY" in entry.display_name

    def test_new_run_lists_registered_datasets(self, client: TestClient, import_root: Path) -> None:
        _write_bars_csv(import_root / "listed.csv", n=10)
        mgr = client.app.state.manager
        entry = commit_import(
            import_root=import_root,
            catalog=mgr.catalog,
            req=_futures_req("listed.csv", display_name="listed ES"),
        )
        listed = client.get("/api/datasets").json()
        ids = {d["dataset_id"] for d in listed["datasets"]}
        assert SMOKE_DATASET_ID in ids
        assert entry.dataset_id in ids
        assert any(d["display_name"] == "listed ES" for d in listed["datasets"])

    def test_unsupported_symbol_timeframe_rejected(self, client: TestClient, import_root: Path) -> None:
        _write_bars_csv(import_root / "gate.csv", n=10)
        mgr = client.app.state.manager
        entry = commit_import(
            import_root=import_root,
            catalog=mgr.catalog,
            req=_futures_req("gate.csv"),
        )
        if not entry.research_eligible:
            entry.research_eligible = True
            entry.quality_status = "PASS"
            entry.rejection_reasons = []
            mgr.catalog.upsert(entry)
        bad_sym = client.post(
            "/api/runs",
            json={
                "run_type": "RESEARCH_DEMO",
                "dataset": entry.dataset_id,
                "symbols": ["NOT_IN_DATASET"],
                "timeframe": entry.timeframe,
            },
        )
        assert bad_sym.status_code == 400
        assert "not in dataset" in bad_sym.json()["detail"].lower() or "symbol" in bad_sym.json()["detail"].lower()
        bad_tf = client.post(
            "/api/runs",
            json={
                "run_type": "RESEARCH_DEMO",
                "dataset": entry.dataset_id,
                "symbols": entry.tradable_contracts or entry.symbols,
                "timeframe": "1d",
            },
        )
        assert bad_tf.status_code == 400
        assert "timeframe" in bad_tf.json()["detail"].lower()

    def test_alpha_miner_rejects_ineligible_and_smoke_without_flag(
        self, client: TestClient, import_root: Path
    ) -> None:
        # synthetic without smoke_test
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
        detail = resp.json()["detail"].lower()
        assert "smoke" in detail

        _write_bars_csv(import_root / "inel.csv", n=6, bad_ohlc=True)
        mgr = client.app.state.manager
        entry = commit_import(
            import_root=import_root,
            catalog=mgr.catalog,
            req=_futures_req("inel.csv", display_name="ineligible"),
        )
        assert entry.research_eligible is False
        resp2 = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "dataset": entry.dataset_id,
                "symbols": entry.tradable_contracts or entry.symbols,
                "timeframe": entry.timeframe,
                "smoke_test": False,
            },
        )
        assert resp2.status_code == 400
        assert "research_eligible" in resp2.json()["detail"].lower() or "not research" in resp2.json()["detail"].lower()

    def test_smoke_alpha_miner_allowed_with_flag(self, client: TestClient) -> None:
        resp = client.post(
            "/api/runs",
            json={
                "run_type": "ALPHA_MINER",
                "confirm_alpha_miner": True,
                "dataset": "synthetic_demo",
                "smoke_test": True,
                "search_budget": {
                    "max_generated_candidates": 4,
                    "max_evaluated_candidates": 3,
                    "population_size": 2,
                    "max_runtime_seconds": 20,
                },
            },
        )
        assert resp.status_code == 200, resp.text

    def test_import_files_api_and_list(self, client: TestClient, import_root: Path) -> None:
        _write_bars_csv(import_root / "visible.csv", n=4)
        files = client.get("/api/import/files").json()
        assert any(f["relative_path"] == "visible.csv" for f in files["files"])
        assert list_importable_files(import_root)


class TestCreateRunRequestCatalog:
    def test_validate_against_catalog_direct(self, tmp_path: Path) -> None:
        cat = DatasetCatalog(tmp_path / "catalog")
        req = CreateRunRequest(
            run_type=RunType.ALPHA_MINER,
            confirm_alpha_miner=True,
            dataset=SMOKE_DATASET_ID,
            smoke_test=False,
        )
        with pytest.raises(ConfigValidationError, match="SMOKE|smoke"):
            req.validate_against_catalog(cat)
