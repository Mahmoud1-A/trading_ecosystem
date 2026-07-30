"""Broker capability cards — do not assume Futures/CFD/equity support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AssetClassSupport(str, Enum):
    FUTURES = "FUTURES"
    CFD = "CFD"
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    FOREX = "FOREX"


class OrderTypeSupport(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass(frozen=True)
class BrokerCapabilities:
    broker_id: str
    asset_classes: frozenset[AssetClassSupport]
    market_data: bool
    paper_support: bool
    live_support: bool
    order_types: frozenset[OrderTypeSupport]
    partial_fills: bool
    positions: bool
    account_equity: bool
    margin: bool
    order_history: bool
    transaction_history: bool
    financing: bool
    contract_metadata: bool
    notes: str = ""

    def supports_asset(self, asset: AssetClassSupport) -> bool:
        return asset in self.asset_classes

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["asset_classes"] = sorted(a.value for a in self.asset_classes)
        d["order_types"] = sorted(o.value for o in self.order_types)
        return d


def simulated_capabilities() -> BrokerCapabilities:
    return BrokerCapabilities(
        broker_id="simulated",
        asset_classes=frozenset(
            {AssetClassSupport.FUTURES, AssetClassSupport.CFD, AssetClassSupport.EQUITY}
        ),
        market_data=True,
        paper_support=True,
        live_support=False,
        order_types=frozenset(OrderTypeSupport),
        partial_fills=True,
        positions=True,
        account_equity=True,
        margin=True,
        order_history=True,
        transaction_history=True,
        financing=True,
        contract_metadata=True,
        notes="in-process simulator for shadow/paper research",
    )


def paper_adapter_capabilities() -> BrokerCapabilities:
    return BrokerCapabilities(
        broker_id="paper_adapter",
        asset_classes=frozenset({AssetClassSupport.EQUITY, AssetClassSupport.FUTURES}),
        market_data=True,
        paper_support=True,
        live_support=False,
        order_types=frozenset(
            {OrderTypeSupport.MARKET, OrderTypeSupport.LIMIT, OrderTypeSupport.STOP}
        ),
        partial_fills=True,
        positions=True,
        account_equity=True,
        margin=True,
        order_history=True,
        transaction_history=True,
        financing=False,
        contract_metadata=False,
        notes="paper-only adapter; live credentials rejected",
    )


def legacy_alpaca_capabilities() -> BrokerCapabilities:
    return BrokerCapabilities(
        broker_id="legacy_alpaca",
        asset_classes=frozenset({AssetClassSupport.EQUITY}),
        market_data=True,
        paper_support=True,
        live_support=False,  # framework refuses live mode
        order_types=frozenset({OrderTypeSupport.MARKET, OrderTypeSupport.LIMIT}),
        partial_fills=True,
        positions=True,
        account_equity=True,
        margin=False,
        order_history=True,
        transaction_history=True,
        financing=False,
        contract_metadata=False,
        notes="legacy Alpaca paper bridge — live trading disabled",
    )
