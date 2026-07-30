"""Data acquisition dashboard panel (Phase 12.2)."""

from __future__ import annotations

from typing import Any

from api.data_acquisition_api import CFD_BANNER, DataAcquisitionAPI


def data_acquisition_view(api: DataAcquisitionAPI) -> dict[str, Any]:
    """Everything the Data Acquisition page renders in one payload."""
    summary = api.source_summary()
    source = summary["source"]
    instrument = summary["instrument"]
    jobs = api.list_jobs()
    active = [j for j in jobs if not j["is_terminal"]]

    return {
        "panel": "data_acquisition",
        "banner": CFD_BANNER,
        "source": {
            "provider_id": source["provider_id"],
            "broker_or_venue": source["broker_or_venue"],
            "instrument": source["underlying_reference"],
            "internal_symbol": source["internal_symbol"],
            "exact_source_symbol": source["source_instrument_identifier"],
            "source_display_symbol": source["source_display_identifier"],
            "asset_class": source["asset_class"],
            "source_timezone": source["source_timezone"],
            "price_convention": source["price_convention"],
            "spread_available": source["spread_available"],
            "volume_type": source["volume_type"],
            "centralized_exchange_volume": source["centralized_exchange_volume"],
            "futures_contract_identity": source["futures_contract_identity"],
            "futures_rollover": source["futures_rollover"],
            "historical_availability": source["historical_availability"],
            "licensing": source["licensing_metadata"],
        },
        "event_types": source["available_event_types"],
        "granularities": ["tick", "1m", "5m"],
        "acquisition_modes": summary["acquisition_modes"],
        "stages": summary["stages"],
        "job_states": summary["job_states"],
        "automated_download": summary["automated_download"],
        "archive_root_configured": summary["archive_root_configured"],
        "archive_root_env": summary["archive_root_env"],
        "import_root": summary["import_root"],
        "verified_symbols": summary["verified_symbols"],
        "instrument_spec": instrument,
        "jobs": jobs,
        "active_jobs": active,
        "job_count": len(jobs),
        "actions": {
            "probe_one_day": "/api/acquisition/probe",
            "create_job": "/api/acquisition/jobs",
            "run_job": "/api/acquisition/jobs/{job_id}/run",
            "pause": "/api/acquisition/jobs/{job_id}/pause",
            "resume": "/api/acquisition/jobs/{job_id}/resume",
            "cancel": "/api/acquisition/jobs/{job_id}/cancel",
            "register_dataset": "/api/acquisition/register",
        },
    }


def job_progress_view(job: dict[str, Any]) -> dict[str, Any]:
    """Compact progress payload for one download job."""
    return {
        "download_job_id": job["download_job_id"],
        "state": job["state"],
        "stage": job["stage"],
        "acquisition_mode": job["acquisition_mode"],
        "source_symbol": job["source_symbol"],
        "start_date": job["start_date"],
        "end_date": job["end_date"],
        "requested_event_type": job["requested_event_type"],
        "requested_granularity": job["requested_granularity"],
        "current_chunk": job["current_chunk"],
        "completed_chunks": job["completed_chunks"],
        "failed_chunks": job["failed_chunks"],
        "total_chunks": job["total_chunks"],
        "retries": job["retries"],
        "downloaded_bytes": job["downloaded_bytes"],
        "progress_pct": job["progress_pct"],
        "terminal_reason": job["terminal_reason"],
        "registered_dataset_id": job["registered_dataset_id"],
    }
