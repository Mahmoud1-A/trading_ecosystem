"""
CFD overnight financing profile (Phase 12.2).

Financing is a CFD-specific carry cost. It is deliberately kept separate from
futures rollover accounting: a CFD has no expiring contract to roll, so a
financing profile must never be populated from futures rollover rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class FinancingProfileError(ValueError):
    """Raised when a financing profile is internally inconsistent."""


@dataclass(frozen=True)
class CFDFinancingProfile:
    """
    Overnight financing terms as charged by a specific broker.

    Rates are expressed as a fraction of notional per night. A positive rate
    means the holder of that side pays.
    """

    broker_id: str
    overnight_financing_long: float
    overnight_financing_short: float
    financing_rollover_time: time = time(22, 0)
    broker_timezone: str = "UTC"
    triple_swap_weekday: int | None = 2  # Wednesday (Mon=0); None disables
    financing_rate_source: str = "BROKER_PUBLISHED"
    verified: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.broker_timezone)
        except ZoneInfoNotFoundError as exc:
            raise FinancingProfileError(
                f"Unknown IANA timezone: {self.broker_timezone}"
            ) from exc
        if self.triple_swap_weekday is not None and not 0 <= self.triple_swap_weekday <= 6:
            raise FinancingProfileError(
                f"triple_swap_weekday must be 0..6 or None, got {self.triple_swap_weekday}"
            )

    @property
    def financing_known(self) -> bool:
        """True when the profile carries usable, verified financing terms."""
        return self.verified

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["financing_rollover_time"] = self.financing_rollover_time.isoformat()
        payload["financing_known"] = self.financing_known
        return payload


def unknown_financing_profile(broker_id: str) -> CFDFinancingProfile:
    """
    Placeholder used when a broker has not published its financing terms.

    Rates are zero *and* ``verified`` is False so that downstream eligibility
    checks can distinguish "genuinely zero" from "not yet known".
    """
    return CFDFinancingProfile(
        broker_id=broker_id,
        overnight_financing_long=0.0,
        overnight_financing_short=0.0,
        verified=False,
        financing_rate_source="UNKNOWN",
        notes="Financing terms not yet captured for this broker.",
    )


def dukascopy_us500_financing_profile() -> CFDFinancingProfile:
    """
    Dukascopy financing placeholder for the USA500 CFD.

    Dukascopy publishes overnight financing separately from the historical
    price archive, so the archive alone cannot establish these rates. The
    profile is intentionally left unverified until the user supplies them.
    """
    return CFDFinancingProfile(
        broker_id="Dukascopy Bank SA",
        overnight_financing_long=0.0,
        overnight_financing_short=0.0,
        financing_rollover_time=time(22, 0),
        broker_timezone="UTC",
        triple_swap_weekday=2,
        financing_rate_source="NOT_IN_HISTORICAL_ARCHIVE",
        verified=False,
        notes=(
            "The public .bi5 price archive contains no financing rates. Supply "
            "broker-published swap rates before relying on overnight carry."
        ),
    )
