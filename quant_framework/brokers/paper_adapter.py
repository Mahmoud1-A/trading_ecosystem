"""Paper broker adapter — rejects live credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brokers.capabilities import BrokerCapabilities, paper_adapter_capabilities
from brokers.protocol import (
    AccountSnapshot,
    BrokerMode,
    BrokerOrder,
    BrokerPosition,
    LiveCredentialsError,
)
from brokers.simulated import SimulatedBroker

LIVE_CREDENTIAL_MARKERS = frozenset(
    {
        "live",
        "prod",
        "production",
        "real_trading",
        "LIVE_API_KEY",
        "alpaca_live",
    }
)


def _looks_live(credentials: dict[str, Any] | None) -> bool:
    if not credentials:
        return False
    blob = " ".join(f"{k}={v}" for k, v in credentials.items()).lower()
    for marker in LIVE_CREDENTIAL_MARKERS:
        if marker.lower() in blob:
            return True
    # Explicit live endpoint / flag
    if credentials.get("live") is True:
        return True
    if credentials.get("trading_mode", "").lower() == "live":
        return True
    endpoint = str(credentials.get("base_url", "")).lower()
    if "paper" not in endpoint and ("api.alpaca" in endpoint or "live" in endpoint):
        return True
    return False


@dataclass
class PaperBrokerAdapter:
    """
    Paper-only broker façade.

    Refuses live credentials. Delegates execution to an in-process paper simulator
    (or an injected paper backend).
    """

    credentials: dict[str, Any] | None = None
    backend: SimulatedBroker | None = None
    _mode: BrokerMode = field(default=BrokerMode.PAPER, init=False)

    def __post_init__(self) -> None:
        if _looks_live(self.credentials):
            raise LiveCredentialsError(
                "PaperBrokerAdapter cannot access live credentials / live endpoints"
            )
        if self.backend is None:
            self.backend = SimulatedBroker(mode=BrokerMode.PAPER)
        elif self.backend.mode is not BrokerMode.PAPER:
            self.backend.mode = BrokerMode.PAPER

    @property
    def capabilities(self) -> BrokerCapabilities:
        return paper_adapter_capabilities()

    @property
    def mode(self) -> BrokerMode:
        return BrokerMode.PAPER

    def connected(self) -> bool:
        assert self.backend is not None
        return self.backend.connected()

    def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        assert self.backend is not None
        return self.backend.submit_order(order)

    def cancel_order(self, client_order_id: str) -> BrokerOrder:
        assert self.backend is not None
        return self.backend.cancel_order(client_order_id)

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        assert self.backend is not None
        return self.backend.get_order(client_order_id)

    def positions(self) -> list[BrokerPosition]:
        assert self.backend is not None
        return self.backend.positions()

    def account(self) -> AccountSnapshot:
        assert self.backend is not None
        return self.backend.account()

    def flatten_all(self) -> list[BrokerOrder]:
        assert self.backend is not None
        return self.backend.flatten_all()
