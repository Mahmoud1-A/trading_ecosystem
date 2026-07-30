"""
Phase 6A acceptance tests — providers, event schemas, and data lake.

No paid external API is required. Credentials never appear in logs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.models import Timeframe
from data.events import LakeBarEvent, LakeQuoteEvent, VolumeType
from data.events.enums import DataCapability, EventSchemaType
from data.lake import BronzeStore, DatasetImmutableError, GoldStore, SilverStore
from data.providers import (
    CapabilityUnavailable,
    DatabentoProvider,
    FileProvider,
    FirstRateProvider,
    MT5Provider,
    MappingSecretProvider,
    RetrievalRequest,
    load_required_credentials,
    redact_secrets,
)
from data.synchronization import asof_join, measure_staleness, reject_stale, StaleDataError


CHI = ZoneInfo("America/Chicago")


def _bars_csv(path: Path, *, n: int = 5) -> Path:
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="5min", tz=CHI)
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
    out = path / "ES.csv"
    df.to_csv(out, index=False)
    return out


class TestFileProviderAndBronzeImmutability:
    def test_bronze_is_immutable_and_corrections_create_new_versions(self, tmp_path: Path) -> None:
        _bars_csv(tmp_path)
        provider = FileProvider(tmp_path, volume_semantics=VolumeType.UNKNOWN)
        assert provider.is_available()
        assert provider.capabilities.provider_id == "file"
        assert not provider.capabilities.credential_env_vars

        req = RetrievalRequest(
            symbol="ES",
            tradable_contract="ESH24",
            event_type=EventSchemaType.BAR,
            start=pd.Timestamp("2024-01-02 09:00", tz=CHI),
            end=pd.Timestamp("2024-01-02 10:00", tz=CHI),
            timeframe="5min",
        )
        batch = provider.fetch(req)
        assert batch.provider_id == "file"
        assert batch.meta["volume_semantics"] == VolumeType.UNKNOWN.value

        lake = BronzeStore(tmp_path / "lake")
        m1 = lake.write(
            batch,
            asset_class="futures",
            licensing=provider.capabilities.licensing.as_dict(),
            volume_type=provider.capabilities.volume_semantics.value,
        )
        # Overwrite attempt fails
        with pytest.raises(DatasetImmutableError):
            lake.write(
                batch,
                asset_class="futures",
                licensing=provider.capabilities.licensing.as_dict(),
                volume_type=provider.capabilities.volume_semantics.value,
            )

        # Correction creates a new version with parent linkage
        corrected = batch.frame.copy()
        corrected.loc[corrected.index[0], "close"] = 100.25
        from data.providers.protocol import ProviderBatch

        batch2 = ProviderBatch(
            provider_id=batch.provider_id,
            provider_version=batch.provider_version,
            request=batch.request,
            frame=corrected,
            source_timezone=batch.source_timezone,
            retrieval_timestamp=datetime.now(tz=timezone.utc),
            raw_fields=batch.raw_fields,
            meta=dict(batch.meta),
        )
        m2 = lake.write_correction(m1, batch2)
        assert m2.dataset_version != m1.dataset_version
        assert m2.correction_version == "1"
        assert m1.dataset_version in m2.parent_dataset_ids[0]
        assert m1.raw_hash != m2.raw_hash

        # Identical content hashes identically
        again = BronzeStore(tmp_path / "lake2").write(
            batch,
            asset_class="futures",
            licensing=provider.capabilities.licensing.as_dict(),
            volume_type=VolumeType.UNKNOWN.value,
        )
        assert again.raw_hash == m1.raw_hash

    def test_provider_identity_preserved_through_silver(self, tmp_path: Path) -> None:
        _bars_csv(tmp_path)
        provider = FileProvider(
            tmp_path,
            broker_or_exchange="local_research",
            volume_semantics=VolumeType.EXCHANGE_EXECUTED_VOLUME,
        )
        req = RetrievalRequest(
            symbol="ES",
            tradable_contract="ESH24",
            event_type=EventSchemaType.BAR,
            start=pd.Timestamp("2024-01-02 09:00", tz=CHI),
            end=pd.Timestamp("2024-01-02 10:00", tz=CHI),
        )
        batch = provider.fetch(req)
        bronze = BronzeStore(tmp_path / "lake")
        bm = bronze.write(batch, asset_class="futures", volume_type=VolumeType.EXCHANGE_EXECUTED_VOLUME.value)
        silver = SilverStore(tmp_path / "lake")
        events, sm = silver.normalize_bars(bm, volume_type=VolumeType.EXCHANGE_EXECUTED_VOLUME)
        assert events
        assert all(e.provider_id == "file" for e in events)
        assert all(e.exchange_or_broker == "local_research" for e in events)
        assert all(e.volume_type is VolumeType.EXCHANGE_EXECUTED_VOLUME for e in events)
        assert sm.quality_report["provider_id"] == "file"
        assert sm.normalized_hash


class TestVolumeAndCapabilitySemantics:
    def test_mt5_cannot_claim_exchange_volume(self) -> None:
        with pytest.raises(ValueError, match="EXCHANGE_EXECUTED_VOLUME"):
            MT5Provider(volume_semantics=VolumeType.EXCHANGE_EXECUTED_VOLUME)

    def test_mt5_declares_tick_proxy(self) -> None:
        p = MT5Provider(broker="DemoBroker", server="Demo-Server")
        assert p.capabilities.volume_semantics is VolumeType.TICK_ACTIVITY_PROXY
        assert "DemoBroker" in p.capabilities.broker_or_exchange
        assert DataCapability.OHLCV_BARS in p.capabilities.capabilities
        assert p.capabilities.supports(DataCapability.ORDER_BOOK_SNAPSHOTS) is False
        with pytest.raises(Exception):
            p.capabilities.require(DataCapability.ORDER_BOOK_SNAPSHOTS)

    def test_databento_preserves_venue_dataset_schema(self) -> None:
        p = DatabentoProvider(dataset="GLBX.MDP3", schema="ohlcv-1m", venue="CME")
        assert p.capabilities.volume_semantics is VolumeType.EXCHANGE_EXECUTED_VOLUME
        assert "GLBX.MDP3" in p.capabilities.notes
        assert p.capabilities.broker_or_exchange == "CME"
        assert p.is_available() is False  # no paid API / optional dep in tests

    def test_firstrate_requires_explicit_schema(self) -> None:
        with pytest.raises(ValueError, match="column_map"):
            FirstRateProvider(column_map={}, volume_semantics=VolumeType.EXCHANGE_EXECUTED_VOLUME)
        with pytest.raises(ValueError, match="volume_semantics"):
            FirstRateProvider(column_map={"Date": "timestamp"}, volume_semantics=VolumeType.UNKNOWN)

    def test_absent_capabilities_are_explicit(self) -> None:
        p = FileProvider(".")
        assert EventSchemaType.BOOK not in p.capabilities.supported_event_types
        req = RetrievalRequest(
            symbol="ES",
            tradable_contract="ESH24",
            event_type=EventSchemaType.BOOK,
            start=pd.Timestamp("2024-01-02 09:00", tz="UTC"),
            end=pd.Timestamp("2024-01-02 10:00", tz="UTC"),
        )
        with pytest.raises(ValueError, match="does not support"):
            p.fetch(req)


class TestEventSchemas:
    def test_invalid_bar_schema_fails_clearly(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            LakeBarEvent(
                event_timestamp=pd.Timestamp("2024-01-02 09:00"),
                availability_timestamp=pd.Timestamp("2024-01-02 09:00"),
                provider_id="file",
                symbol="ES",
                tradable_contract="ESH24",
                timeframe=Timeframe.M5,
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                volume=10,
                volume_type=VolumeType.UNKNOWN,
                currency="USD",
                exchange_or_broker="x",
                source_timezone="UTC",
            )

    def test_invalid_quote_spread_fails(self) -> None:
        ts = pd.Timestamp("2024-01-02 09:00", tz="UTC")
        with pytest.raises(ValueError, match="ask_price"):
            LakeQuoteEvent(
                event_timestamp=ts,
                availability_timestamp=ts,
                provider_id="file",
                symbol="ES",
                tradable_contract="ESH24",
                bid_price=10.0,
                ask_price=9.0,
                bid_size=1,
                ask_size=1,
                exchange_or_broker="x",
            )


class TestCredentialsNeverLogged:
    def test_redact_secrets(self) -> None:
        payload = {"api_key": "SUPERSECRET", "symbol": "ES", "nested": {"password": "x"}}
        redacted = redact_secrets(payload)
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["symbol"] == "ES"
        assert redacted["nested"]["password"] == "***REDACTED***"

    def test_missing_credentials_log_names_only(self, caplog: pytest.LogCaptureFixture) -> None:
        secrets = MappingSecretProvider({"OTHER": "1"})
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                load_required_credentials(("DATABENTO_API_KEY",), secret_provider=secrets)
        joined = " ".join(r.message for r in caplog.records)
        assert "DATABENTO_API_KEY" in joined
        assert "SUPERSECRET" not in joined


class TestSynchronization:
    def test_asof_never_selects_future(self) -> None:
        left = pd.DataFrame(
            {
                "decision_timestamp": pd.to_datetime(
                    ["2024-01-02 10:00", "2024-01-02 11:00"], utc=True
                ),
                "id": [1, 2],
            }
        )
        right = pd.DataFrame(
            {
                "availability_timestamp": pd.to_datetime(
                    ["2024-01-02 09:00", "2024-01-02 10:30", "2024-01-02 12:00"], utc=True
                ),
                "feat": [10.0, 20.0, 99.0],
            }
        )
        out = asof_join(left, right)
        assert list(out["feat"]) == [10.0, 20.0]

    def test_staleness_rejection(self) -> None:
        src = pd.Timestamp("2024-01-02 09:00", tz="UTC")
        dec = pd.Timestamp("2024-01-02 10:00", tz="UTC")
        result = measure_staleness(source_timestamp=src, decision_timestamp=dec, max_age=pd.Timedelta("30min"))
        assert result.is_stale
        with pytest.raises(StaleDataError):
            reject_stale(result)


class TestGoldScaffold:
    def test_gold_requires_availability_and_is_immutable(self, tmp_path: Path) -> None:
        gold = GoldStore(tmp_path / "lake")
        frame = pd.DataFrame(
            {
                "availability_timestamp": pd.to_datetime(["2024-01-02 09:05"], utc=True),
                "ret_1": [0.01],
            }
        )
        m = gold.write_feature_frame(
            frame,
            feature_set_version="fs_test_v1",
            symbol="ES",
            parent_dataset_ids=["ds:v1"],
            feature_set_hash="abc",
            tradable_contract="ESH24",
        )
        assert m.layer == "gold"
        with pytest.raises(DatasetImmutableError):
            gold.write_feature_frame(
                frame,
                feature_set_version="fs_test_v1",
                symbol="ES",
                parent_dataset_ids=["ds:v1"],
                feature_set_hash="abc",
                tradable_contract="ESH24",
            )
        with pytest.raises(ValueError, match="availability_timestamp"):
            gold.write_feature_frame(
                pd.DataFrame({"ret_1": [0.01]}),
                feature_set_version="fs_x",
                symbol="ES",
                parent_dataset_ids=[],
                feature_set_hash="x",
            )
