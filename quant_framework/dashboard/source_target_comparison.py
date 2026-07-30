"""Historical source vs execution target comparison panel (Phase 12.2)."""

from __future__ import annotations

from typing import Any

from cfds.execution_profile import (
    CRITICAL,
    ExecutionTargetProfile,
    compare_source_to_target,
    evaluate_paper_eligibility,
)
from cfds.instrument_spec import CFDResearchInstrument
from cfds.source_profile import HistoricalSourceProfile

CFD_BANNER = "DUKASCOPY USA500 DATA IS BROKER CFD DATA. IT IS NOT CME ES FUTURES DATA."


def source_target_comparison_view(
    *,
    source: HistoricalSourceProfile,
    instrument: CFDResearchInstrument,
    target: ExecutionTargetProfile,
) -> dict[str, Any]:
    """Side-by-side source/target fields plus every unresolved warning."""
    warnings = compare_source_to_target(
        source=source, instrument=instrument, target=target
    )
    decision = evaluate_paper_eligibility(warnings)
    source_session = instrument.trading_sessions[0]

    rows = [
        _row(
            "symbol_mapping",
            f"{source.source_instrument_identifier} ({source.source_display_identifier})",
            target.broker_symbol,
            verified=target.symbol_mapping_verified,
        ),
        _row("broker", source.broker_or_venue, target.broker_id),
        _row(
            "price_scale",
            instrument.minimum_price_increment,
            target.minimum_price_increment,
        ),
        _row("contract_size", instrument.value_per_point, target.value_per_point),
        _row(
            "session",
            f"{source_session.open_time.isoformat()}–"
            f"{source_session.close_time.isoformat()} {instrument.broker_timezone}",
            f"{target.session_open.isoformat()}–"
            f"{target.session_close.isoformat()} {target.broker_timezone}",
        ),
        _row(
            "typical_spread",
            instrument.typical_spread_points,
            target.typical_spread_points,
        ),
        _row(
            "stressed_spread",
            instrument.stressed_spread_points,
            target.stressed_spread_points,
        ),
        _row(
            "financing_long",
            instrument.financing.overnight_financing_long,
            target.financing.overnight_financing_long,
            verified=instrument.financing.financing_known
            and target.financing.financing_known,
        ),
        _row(
            "financing_short",
            instrument.financing.overnight_financing_short,
            target.financing.overnight_financing_short,
            verified=instrument.financing.financing_known
            and target.financing.financing_known,
        ),
        _row(
            "commission",
            instrument.commission_model.as_dict(),
            target.commission_model.as_dict(),
        ),
        _row("leverage", instrument.leverage, target.leverage),
        _row("margin_requirement", instrument.margin_requirement, target.margin_requirement),
    ]

    payload = [w.as_dict() for w in warnings]
    return {
        "panel": "source_target_comparison",
        "banner": CFD_BANNER,
        "historical_source_broker": source.broker_or_venue,
        "target_prop_broker": target.broker_id,
        "target_feed_imported": target.target_feed_imported,
        "target_feed_dataset_id": target.target_feed_dataset_id,
        "comparison_rows": rows,
        "warnings": payload,
        "unresolved_warnings": [w for w in payload if w["severity"] == CRITICAL],
        "warning_count": len(payload),
        "critical_count": sum(1 for w in payload if w["severity"] == CRITICAL),
        "research_eligible_possible": True,
        "paper_eligibility": decision.as_dict(),
        "note": (
            "Dukascopy data may become RESEARCH_ELIGIBLE. It cannot become "
            "PAPER_ELIGIBLE for a different broker until that broker's feed has "
            "been imported and compared."
        ),
    }


def _row(
    field: str, source_value: Any, target_value: Any, *, verified: bool | None = None
) -> dict[str, Any]:
    return {
        "field": field,
        "source": source_value,
        "target": target_value,
        "matches": source_value == target_value,
        "verified": verified,
    }
