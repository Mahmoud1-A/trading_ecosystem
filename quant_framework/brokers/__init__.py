"""Broker adapters for shadow and paper trading (Phase 9) — research only."""

from brokers.capabilities import (
    AssetClassSupport,
    BrokerCapabilities,
    OrderTypeSupport,
    legacy_alpaca_capabilities,
    paper_adapter_capabilities,
    simulated_capabilities,
)
from brokers.legacy_alpaca_adapter import LegacyAlpacaAdapter
from brokers.paper_adapter import PaperBrokerAdapter
from brokers.protocol import (
    AccountSnapshot,
    BrokerError,
    BrokerFill,
    BrokerMode,
    BrokerOrder,
    BrokerPosition,
    BrokerProtocol,
    LiveCredentialsError,
    OrderSide,
    OrderStatus,
    ShadowSubmitError,
)
from brokers.simulated import SimulatedBroker

__all__ = [
    "AccountSnapshot",
    "AssetClassSupport",
    "BrokerCapabilities",
    "BrokerError",
    "BrokerFill",
    "BrokerMode",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerProtocol",
    "LegacyAlpacaAdapter",
    "LiveCredentialsError",
    "OrderSide",
    "OrderStatus",
    "OrderTypeSupport",
    "PaperBrokerAdapter",
    "ShadowSubmitError",
    "SimulatedBroker",
    "legacy_alpaca_capabilities",
    "paper_adapter_capabilities",
    "simulated_capabilities",
]
