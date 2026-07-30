"""
HistoricalSourceProfile — where research data came from (Phase 12.2).

This describes the *data origin* only. It never describes where a strategy
may eventually execute; see ``cfds.execution_profile.ExecutionTargetProfile``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cfds.symbol_mapping import (
    DUKASCOPY_US500,
    SymbolMapping,
    assert_not_futures_identifier,
)
from data.events.enums import EventSchemaType, PriceConvention, VolumeType

DUKASCOPY_PROVIDER_ID = "DUKASCOPY"


class SourceProfileError(ValueError):
    """Raised when a source profile would misrepresent the data origin."""


@dataclass(frozen=True)
class RetrievalMethod:
    """How the bytes are obtained. Manual export is always available."""

    manual_export_import: bool = True
    automated_public_download: bool = False
    public_endpoint_template: str | None = None
    requires_credentials: bool = False
    rate_limit_requests_per_second: float = 2.0
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalSourceProfile:
    """Typed description of a historical market-data source."""

    provider_id: str
    provider_version: str
    source_instrument_identifier: str
    source_display_identifier: str
    internal_symbol: str
    asset_class: str
    underlying_reference: str
    source_timezone: str
    broker_or_venue: str
    available_event_types: tuple[EventSchemaType, ...]
    price_convention: PriceConvention
    spread_available: bool
    volume_type: VolumeType
    centralized_exchange_volume: bool
    futures_contract_identity: str
    futures_rollover: str
    historical_availability: dict[str, str]
    retrieval_method: RetrievalMethod
    licensing_metadata: dict[str, Any]
    financing_rate_available: bool = False
    point_divisor: int = 1
    symbol_mapping: SymbolMapping | None = None
    notes: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.asset_class.upper() != "CFD":
            raise SourceProfileError(
                f"HistoricalSourceProfile for CFD research requires asset_class=CFD, "
                f"got {self.asset_class!r}"
            )
        assert_not_futures_identifier(
            self.source_instrument_identifier, field_name="source_instrument_identifier"
        )
        assert_not_futures_identifier(self.internal_symbol, field_name="internal_symbol")
        if self.centralized_exchange_volume:
            raise SourceProfileError(
                "A broker CFD feed cannot report centralized exchange volume"
            )
        if self.volume_type is VolumeType.EXCHANGE_EXECUTED_VOLUME:
            raise SourceProfileError(
                "CFD volume must not be labeled EXCHANGE_EXECUTED_VOLUME; "
                "use TICK_ACTIVITY_PROXY, BROKER_REPORTED_VOLUME, or UNKNOWN"
            )

    @property
    def is_broker_cfd(self) -> bool:
        return True

    @property
    def provider_key(self) -> str:
        """Lower-case provider id as used for lake and catalog storage keys."""
        return self.provider_id.lower()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available_event_types"] = [e.value for e in self.available_event_types]
        payload["price_convention"] = self.price_convention.value
        payload["volume_type"] = self.volume_type.value
        payload["retrieval_method"] = self.retrieval_method.as_dict()
        payload["symbol_mapping"] = (
            self.symbol_mapping.as_dict() if self.symbol_mapping else None
        )
        payload["warnings"] = list(self.warnings)
        payload["instrument_class_banner"] = (
            "DUKASCOPY USA500 DATA IS BROKER CFD DATA. IT IS NOT CME ES FUTURES DATA."
        )
        return payload


def dukascopy_us500_source_profile(
    *,
    provider_version: str = "1.0.0",
    automated_public_download: bool = False,
) -> HistoricalSourceProfile:
    """
    Verified Dukascopy USA500.IDX/USD CFD source profile.

    The datafeed identifier ``USA500IDXUSD`` is the value used in Dukascopy's
    public historical archive paths; ``USA500.IDX/USD`` is the display name.
    """
    return HistoricalSourceProfile(
        provider_id=DUKASCOPY_PROVIDER_ID,
        provider_version=provider_version,
        source_instrument_identifier="USA500IDXUSD",
        source_display_identifier="USA500.IDX/USD",
        internal_symbol=DUKASCOPY_US500.internal_symbol,
        asset_class="CFD",
        underlying_reference="S&P 500 index",
        source_timezone="UTC",
        broker_or_venue="Dukascopy Bank SA",
        available_event_types=(EventSchemaType.QUOTE, EventSchemaType.BAR),
        price_convention=PriceConvention.BID,
        spread_available=True,
        volume_type=VolumeType.TICK_ACTIVITY_PROXY,
        centralized_exchange_volume=False,
        futures_contract_identity="NOT_APPLICABLE",
        futures_rollover="NOT_APPLICABLE",
        historical_availability={
            "tick": "2012-01-16",
            "m1": "2011-09-19",
            "h1": "2011-09-18",
            "d1": "1980-01-02",
        },
        retrieval_method=RetrievalMethod(
            manual_export_import=True,
            automated_public_download=automated_public_download,
            public_endpoint_template=(
                "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month0:02d}/"
                "{day:02d}/{hour:02d}h_ticks.bi5"
            ),
            requires_credentials=False,
            rate_limit_requests_per_second=2.0,
            notes=(
                "Public LZMA-compressed .bi5 archive. Month is zero-indexed (0-11). "
                "Automated download is opt-in and disabled by default."
            ),
        ),
        licensing_metadata={
            "license_name": "dukascopy_terms",
            "redistributable": False,
            "commercial_use": False,
            "notes": "Historical data subject to Dukascopy Bank SA terms; do not redistribute.",
        },
        financing_rate_available=False,
        point_divisor=1000,
        symbol_mapping=DUKASCOPY_US500,
        notes=(
            "Broker OTC CFD referencing the S&P 500 index. Quote volumes are tick "
            "activity proxies, not exchange-executed contracts."
        ),
        warnings=(
            "Dukascopy pricing, spread, and session behaviour will differ from any "
            "other CFD broker; validate the execution target before paper trading.",
        ),
    )
