"""
Internal <-> provider <-> execution-target CFD symbol mapping (Phase 12.2).

A broker CFD on the S&P 500 index is a different instrument from the CME
E-mini S&P 500 future. This module makes that distinction explicit and
refuses to map a CFD internal symbol onto a futures identifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

US500_CFD = "US500_CFD"

# Identifiers that denote CME-listed E-mini S&P 500 futures. A CFD instrument
# may never adopt one of these as its provider or broker symbol.
FUTURES_IDENTIFIERS = frozenset(
    {
        "ES",
        "ESH24",
        "ESM24",
        "ESU24",
        "ESZ24",
        "ES=F",
        "/ES",
        "E-MINI S&P 500",
        "EMINI S&P 500",
        "CME:ES",
        "XCME:ES",
    }
)

FUTURES_SUBSTRINGS = ("E-MINI", "EMINI", "CME", "GLOBEX", "XCME")


class SymbolMappingError(ValueError):
    """Raised when a symbol mapping would conflate distinct instruments."""


def assert_not_futures_identifier(symbol: str, *, field_name: str) -> None:
    """Reject any attempt to label a CFD with a CME futures identity."""
    token = (symbol or "").strip().upper()
    if not token:
        return
    if token in FUTURES_IDENTIFIERS:
        raise SymbolMappingError(
            f"{field_name}={symbol!r} is a CME E-mini S&P 500 futures identifier; "
            "a broker CFD must not be labeled as exchange-traded futures"
        )
    for needle in FUTURES_SUBSTRINGS:
        if needle in token:
            raise SymbolMappingError(
                f"{field_name}={symbol!r} references {needle}; "
                "CFD instruments must not carry futures-exchange identity"
            )


@dataclass(frozen=True)
class SymbolMapping:
    """
    One internal research symbol and how it is named by each external party.

    ``verified`` is only true once a human has confirmed that the provider
    symbol really is the instrument we think it is. Unverified mappings are
    still usable for research but always raise a comparison warning.
    """

    internal_symbol: str
    asset_class: str
    underlying_reference: str
    provider_id: str
    provider_symbol: str
    provider_display_symbol: str
    verified: bool = False
    target_broker_id: str | None = None
    target_broker_symbol: str | None = None
    target_symbol_verified: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.asset_class.upper() != "CFD":
            raise SymbolMappingError(
                f"SymbolMapping is CFD-only; got asset_class={self.asset_class!r}"
            )
        assert_not_futures_identifier(self.provider_symbol, field_name="provider_symbol")
        assert_not_futures_identifier(
            self.provider_display_symbol, field_name="provider_display_symbol"
        )
        if self.target_broker_symbol:
            assert_not_futures_identifier(
                self.target_broker_symbol, field_name="target_broker_symbol"
            )
        for alias in self.aliases:
            assert_not_futures_identifier(alias, field_name="alias")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DUKASCOPY_US500 = SymbolMapping(
    internal_symbol=US500_CFD,
    asset_class="CFD",
    underlying_reference="S&P 500 index",
    provider_id="DUKASCOPY",
    provider_symbol="USA500IDXUSD",
    provider_display_symbol="USA500.IDX/USD",
    verified=True,
    aliases=("US500", "SPX500", "USA500"),
    notes=(
        "Dukascopy Bank SA OTC index CFD priced against the S&P 500. "
        "Not CME ES futures; no contract expiration and no futures rollover."
    ),
)

_MAPPINGS: dict[str, SymbolMapping] = {DUKASCOPY_US500.internal_symbol: DUKASCOPY_US500}


def get_symbol_mapping(internal_symbol: str) -> SymbolMapping:
    key = (internal_symbol or "").strip().upper()
    if key not in _MAPPINGS:
        known = ", ".join(sorted(_MAPPINGS)) or "(none)"
        raise KeyError(f"Unknown internal symbol {internal_symbol!r}. Known: {known}")
    return _MAPPINGS[key]


def list_symbol_mappings() -> list[SymbolMapping]:
    return sorted(_MAPPINGS.values(), key=lambda m: m.internal_symbol)


def register_symbol_mapping(mapping: SymbolMapping, *, overwrite: bool = False) -> SymbolMapping:
    key = mapping.internal_symbol.upper()
    if key in _MAPPINGS and not overwrite:
        raise SymbolMappingError(f"Symbol mapping {key} already registered")
    _MAPPINGS[key] = mapping
    return mapping


def with_target_broker(
    mapping: SymbolMapping,
    *,
    target_broker_id: str,
    target_broker_symbol: str,
    verified: bool = False,
) -> SymbolMapping:
    """Attach an execution-target broker symbol without mutating the source mapping."""
    assert_not_futures_identifier(target_broker_symbol, field_name="target_broker_symbol")
    return SymbolMapping(
        internal_symbol=mapping.internal_symbol,
        asset_class=mapping.asset_class,
        underlying_reference=mapping.underlying_reference,
        provider_id=mapping.provider_id,
        provider_symbol=mapping.provider_symbol,
        provider_display_symbol=mapping.provider_display_symbol,
        verified=mapping.verified,
        target_broker_id=target_broker_id,
        target_broker_symbol=target_broker_symbol,
        target_symbol_verified=verified,
        aliases=mapping.aliases,
        notes=mapping.notes,
    )
