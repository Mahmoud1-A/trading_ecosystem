"""Deterministic research identifiers for orders and fills."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _digest(*parts: object) -> str:
    material = "|".join(str(p) for p in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class DeterministicIdFactory:
    """
    Stable ID generator for research/backtest runs.

    Does not use wall-clock time or uuid4. Live adapters may map broker IDs separately.
    """

    run_id: str
    candidate_id: str = "default"
    fold_id: str | int = "0"
    _order_seq: int = 0
    _fill_seq: int = 0

    def next_order_id(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        decision_timestamp: str,
        contract: str,
    ) -> str:
        self._order_seq += 1
        digest = _digest(
            self.run_id,
            self.candidate_id,
            self.fold_id,
            "ORD",
            self._order_seq,
            symbol,
            side,
            f"{quantity:.10g}",
            order_type,
            decision_timestamp,
            contract,
        )
        return f"ord_{digest[:24]}"

    def next_fill_id(
        self,
        *,
        order_id: str,
        fill_timestamp: str,
        quantity: float,
        price: float,
    ) -> str:
        self._fill_seq += 1
        digest = _digest(
            self.run_id,
            self.candidate_id,
            self.fold_id,
            "FIL",
            self._fill_seq,
            order_id,
            fill_timestamp,
            f"{quantity:.10g}",
            f"{price:.10g}",
        )
        return f"fil_{digest[:24]}"

    def next_signal_id(self, *, decision_timestamp: str, symbol: str, side: str) -> str:
        digest = _digest(
            self.run_id,
            self.candidate_id,
            self.fold_id,
            "SIG",
            decision_timestamp,
            symbol,
            side,
            self._order_seq,
        )
        return f"sig_{digest[:24]}"
