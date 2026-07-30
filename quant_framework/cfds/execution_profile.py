"""
ExecutionTargetProfile and source-vs-target comparison (Phase 12.2).

Research data comes from a historical source broker. A strategy may eventually
execute at a different (prop-firm) broker. Those are separate profiles, and a
dataset sourced from one broker never becomes paper-eligible for another until
target-feed data has been imported and compared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import time
from typing import Any

from cfds.financing_profile import CFDFinancingProfile, unknown_financing_profile
from cfds.instrument_spec import CFDCommissionModel, CFDResearchInstrument
from cfds.source_profile import HistoricalSourceProfile
from cfds.symbol_mapping import assert_not_futures_identifier

# Relative spread divergence above which the difference is material for research.
SPREAD_MATERIAL_RATIO = 0.25


class Severity(str):
    """String severities keep the payload JSON-friendly for the dashboard."""


CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"


@dataclass(frozen=True)
class ComparisonWarning:
    code: str
    severity: str
    message: str
    source_value: Any = None
    target_value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "source_value": self.source_value,
            "target_value": self.target_value,
        }


@dataclass(frozen=True)
class ExecutionTargetProfile:
    """The broker/prop environment a strategy may eventually run on."""

    broker_id: str
    broker_symbol: str
    asset_class: str
    quote_currency: str
    value_per_point: float
    minimum_price_increment: float
    typical_spread_points: float
    stressed_spread_points: float
    commission_model: CFDCommissionModel
    leverage: float
    margin_requirement: float
    financing: CFDFinancingProfile
    broker_timezone: str
    session_open: time = time(0, 0)
    session_close: time = time(23, 59, 59)
    price_scale_decimals: int = 2
    execution_rules: dict[str, Any] = field(default_factory=dict)
    target_feed_imported: bool = False
    target_feed_dataset_id: str | None = None
    symbol_mapping_verified: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        assert_not_futures_identifier(self.broker_symbol, field_name="broker_symbol")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commission_model"] = self.commission_model.as_dict()
        payload["financing"] = self.financing.as_dict()
        payload["session_open"] = self.session_open.isoformat()
        payload["session_close"] = self.session_close.isoformat()
        return payload


def generic_prop_target_profile(
    *,
    broker_id: str = "UNSPECIFIED_PROP_BROKER",
    broker_symbol: str = "US500",
) -> ExecutionTargetProfile:
    """
    Placeholder execution target.

    Values are intentionally generic; a real target profile must be created
    from the actual purchased account before paper trading.
    """
    return ExecutionTargetProfile(
        broker_id=broker_id,
        broker_symbol=broker_symbol,
        asset_class="CFD",
        quote_currency="USD",
        value_per_point=1.0,
        minimum_price_increment=0.01,
        typical_spread_points=0.8,
        stressed_spread_points=4.0,
        commission_model=CFDCommissionModel(),
        leverage=20.0,
        margin_requirement=0.05,
        financing=unknown_financing_profile(broker_id),
        broker_timezone="UTC",
        price_scale_decimals=2,
        execution_rules={
            "market_orders": True,
            "stop_orders": True,
            "hedging_allowed": None,
            "min_stop_distance_points": None,
        },
        target_feed_imported=False,
        symbol_mapping_verified=False,
        notes="Placeholder target. Replace with the real prop account configuration.",
    )


def _relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def compare_source_to_target(
    *,
    source: HistoricalSourceProfile,
    instrument: CFDResearchInstrument,
    target: ExecutionTargetProfile,
) -> list[ComparisonWarning]:
    """Enumerate every way the research source diverges from the execution target."""
    warnings: list[ComparisonWarning] = []

    if source.broker_or_venue.strip().lower() != target.broker_id.strip().lower():
        warnings.append(
            ComparisonWarning(
                code="SOURCE_TARGET_BROKER_MISMATCH",
                severity=CRITICAL,
                message=(
                    "Historical data source broker differs from the execution target "
                    "broker; prices, spreads, and sessions will not match."
                ),
                source_value=source.broker_or_venue,
                target_value=target.broker_id,
            )
        )

    if not target.symbol_mapping_verified:
        warnings.append(
            ComparisonWarning(
                code="SYMBOL_MAPPING_UNVERIFIED",
                severity=CRITICAL,
                message=(
                    f"Mapping {instrument.provider_symbol} -> {target.broker_symbol} "
                    "has not been verified against the target broker."
                ),
                source_value=instrument.provider_symbol,
                target_value=target.broker_symbol,
            )
        )

    if not target.target_feed_imported:
        warnings.append(
            ComparisonWarning(
                code="TARGET_FEED_NOT_IMPORTED",
                severity=CRITICAL,
                message=(
                    "No target-broker feed has been imported, so source-vs-target "
                    "price behaviour is unvalidated."
                ),
                source_value=source.provider_id,
                target_value=target.broker_id,
            )
        )

    source_open = instrument.trading_sessions[0].open_time
    source_close = instrument.trading_sessions[0].close_time
    if (source_open, source_close) != (target.session_open, target.session_close) or (
        instrument.broker_timezone != target.broker_timezone
    ):
        warnings.append(
            ComparisonWarning(
                code="SESSION_HOURS_DIFFER",
                severity=WARNING,
                message="Source and target session windows or timezones differ.",
                source_value=(
                    f"{source_open.isoformat()}-{source_close.isoformat()} "
                    f"{instrument.broker_timezone}"
                ),
                target_value=(
                    f"{target.session_open.isoformat()}-{target.session_close.isoformat()} "
                    f"{target.broker_timezone}"
                ),
            )
        )

    if _relative_difference(
        instrument.minimum_price_increment, target.minimum_price_increment
    ) > 1e-9:
        warnings.append(
            ComparisonWarning(
                code="PRICE_SCALE_DIFFERS",
                severity=WARNING,
                message=(
                    "Minimum price increment differs; signals tuned on the source "
                    "grid may not be representable at the target."
                ),
                source_value=instrument.minimum_price_increment,
                target_value=target.minimum_price_increment,
            )
        )

    if _relative_difference(instrument.value_per_point, target.value_per_point) > 1e-9:
        warnings.append(
            ComparisonWarning(
                code="CONTRACT_SIZE_DIFFERS",
                severity=CRITICAL,
                message="Value per point differs; position sizing will not transfer.",
                source_value=instrument.value_per_point,
                target_value=target.value_per_point,
            )
        )

    if (
        _relative_difference(
            instrument.typical_spread_points, target.typical_spread_points
        )
        > SPREAD_MATERIAL_RATIO
    ):
        warnings.append(
            ComparisonWarning(
                code="SPREAD_DIFFERS_MATERIALLY",
                severity=CRITICAL,
                message=(
                    "Typical spread differs materially; edge measured on source "
                    "spreads may not survive at the target."
                ),
                source_value=instrument.typical_spread_points,
                target_value=target.typical_spread_points,
            )
        )

    src_fin = instrument.financing
    tgt_fin = target.financing
    if not (src_fin.financing_known and tgt_fin.financing_known) or (
        src_fin.overnight_financing_long != tgt_fin.overnight_financing_long
        or src_fin.overnight_financing_short != tgt_fin.overnight_financing_short
    ):
        warnings.append(
            ComparisonWarning(
                code="FINANCING_DIFFERS_OR_UNKNOWN",
                severity=WARNING,
                message="Overnight financing terms differ or are not yet verified.",
                source_value=src_fin.as_dict(),
                target_value=tgt_fin.as_dict(),
            )
        )

    if instrument.commission_model.as_dict() != target.commission_model.as_dict():
        warnings.append(
            ComparisonWarning(
                code="COMMISSION_DIFFERS",
                severity=WARNING,
                message="Commission models differ between source research and target.",
                source_value=instrument.commission_model.as_dict(),
                target_value=target.commission_model.as_dict(),
            )
        )

    unresolved_rules = sorted(
        k for k, v in (target.execution_rules or {}).items() if v is None
    )
    if unresolved_rules:
        warnings.append(
            ComparisonWarning(
                code="EXECUTION_RULES_UNRESOLVED",
                severity=WARNING,
                message=f"Target execution rules not established: {unresolved_rules}",
                source_value=None,
                target_value=unresolved_rules,
            )
        )

    return warnings


@dataclass(frozen=True)
class PaperEligibilityDecision:
    paper_eligible: bool
    reason: str
    blocking_codes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "paper_eligible": self.paper_eligible,
            "reason": self.reason,
            "blocking_codes": list(self.blocking_codes),
        }


def evaluate_paper_eligibility(
    warnings: list[ComparisonWarning],
) -> PaperEligibilityDecision:
    """
    Research eligibility never implies paper eligibility.

    Any CRITICAL source-vs-target divergence blocks paper eligibility.
    """
    blocking = tuple(w.code for w in warnings if w.severity == CRITICAL)
    if blocking:
        return PaperEligibilityDecision(
            paper_eligible=False,
            reason=(
                "Target-broker validation incomplete; import and compare the target "
                "feed before paper trading."
            ),
            blocking_codes=blocking,
        )
    return PaperEligibilityDecision(
        paper_eligible=True,
        reason="Source and target profiles reconciled with no critical divergence.",
    )
