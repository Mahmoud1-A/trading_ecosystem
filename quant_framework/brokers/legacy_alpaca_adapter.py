"""Legacy Alpaca adapter — paper bridge only; Sim performance cannot promote."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brokers.capabilities import BrokerCapabilities, legacy_alpaca_capabilities
from brokers.paper_adapter import PaperBrokerAdapter, _looks_live
from brokers.protocol import (
    AccountSnapshot,
    BrokerMode,
    BrokerOrder,
    BrokerPosition,
    LiveCredentialsError,
)


@dataclass
class LegacyAlpacaAdapter:
    """
    Read/write paper bridge for the legacy Alpaca integration.

    Live mode is hard-disabled. Historical Sim performance metrics are archival
    only and must never promote a portfolio.
    """

    credentials: dict[str, Any] | None = None
    legacy_sim_sharpe: float | None = None  # archival — not a promotion input
    _paper: PaperBrokerAdapter = field(init=False)

    def __post_init__(self) -> None:
        creds = dict(self.credentials or {})
        # Force paper endpoint semantics
        if _looks_live(creds):
            raise LiveCredentialsError("legacy Alpaca adapter refuses live credentials")
        creds.setdefault("trading_mode", "paper")
        creds.setdefault("base_url", "https://paper-api.alpaca.markets")
        self._paper = PaperBrokerAdapter(credentials=creds)

    @property
    def capabilities(self) -> BrokerCapabilities:
        return legacy_alpaca_capabilities()

    @property
    def mode(self) -> BrokerMode:
        return BrokerMode.PAPER

    def connected(self) -> bool:
        return self._paper.connected()

    def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        return self._paper.submit_order(order)

    def cancel_order(self, client_order_id: str) -> BrokerOrder:
        return self._paper.cancel_order(client_order_id)

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        return self._paper.get_order(client_order_id)

    def positions(self) -> list[BrokerPosition]:
        return self._paper.positions()

    def account(self) -> AccountSnapshot:
        return self._paper.account()

    def flatten_all(self) -> list[BrokerOrder]:
        return self._paper.flatten_all()

    def legacy_sim_performance(self) -> dict[str, Any]:
        """Archival Sim metrics — explicitly not valid for promotion."""
        return {
            "legacy_sim_sharpe": self.legacy_sim_sharpe,
            "promotable": False,
            "reason": "legacy_sim_performance_cannot_promote",
        }
