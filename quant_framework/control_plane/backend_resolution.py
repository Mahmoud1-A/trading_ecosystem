"""Resolve Alpha Miner evaluation backend — no silent synthetic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data.catalog.catalog import SMOKE_DATASET_ID
from data.catalog.silver_bar_resolution import (
    REAL_DATA_BACKEND_UNAVAILABLE,
    display_timeframe,
    strategy_allows_timeframe,
)

SYNTHETIC_BACKEND = "synthetic_oos_probe"
EVENT_WFO_BACKEND = "event_driven_wfo"


class BackendResolutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class BackendResolution:
    evaluation_backend: str
    is_full_event_wfo: bool
    requires_silver_bars: bool
    smoke_path: bool
    banners: tuple[str, ...]
    block_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_backend": self.evaluation_backend,
            "is_full_event_wfo": self.is_full_event_wfo,
            "requires_silver_bars": self.requires_silver_bars,
            "smoke_path": self.smoke_path,
            "banners": list(self.banners),
            "block_reason": self.block_reason,
        }


def resolve_alpha_miner_backend(
    *,
    dataset_id: str,
    smoke_test: bool,
    research_eligible: bool,
    smoke_test_only: bool,
    budget_cfg: dict[str, Any] | None = None,
    strategy_family: str = "mean_reversion_vwap_bb",
    timeframe: str = "1m",
) -> BackendResolution:
    """
    Decide evaluation backend.

    synthetic_oos_probe is allowed only for synthetic_demo or an explicit smoke test.
    Genuine RESEARCH_ELIGIBLE + smoke_test=false must use event_driven_wfo.
    Never silently falls back from real WFO to synthetic.
    """
    budget_cfg = budget_cfg or {}
    requested = str(budget_cfg.get("evaluation_backend") or "").strip().lower()
    is_synthetic_dataset = dataset_id == SMOKE_DATASET_ID or smoke_test_only
    explicit_smoke = bool(smoke_test)
    genuine_real = (
        bool(research_eligible)
        and not is_synthetic_dataset
        and not explicit_smoke
    )

    real_banners = (
        "REAL EVENT-DRIVEN WFO",
        "GENUINE DUKASCOPY DATA",
        "OBSERVED BID/ASK SPREAD",
        "INTRADAY ONLY",
        "VAULT / PAPER / LIVE DISABLED",
    )
    smoke_banners = (
        "SYNTHETIC OOS PROBE",
        "SMOKE TEST ONLY",
        "VAULT / PAPER / LIVE DISABLED",
    )

    if genuine_real:
        if not strategy_allows_timeframe(strategy_family, timeframe):
            raise BackendResolutionError(
                REAL_DATA_BACKEND_UNAVAILABLE,
                f"strategy_family {strategy_family!r} does not support timeframe "
                f"{display_timeframe(timeframe)!r}",
            )
        if requested in {SYNTHETIC_BACKEND, "synthetic", "probe"}:
            raise BackendResolutionError(
                REAL_DATA_BACKEND_UNAVAILABLE,
                "synthetic_oos_probe is forbidden for genuine RESEARCH_ELIGIBLE "
                "datasets when smoke_test=false (no silent fallback)",
            )
        return BackendResolution(
            evaluation_backend=EVENT_WFO_BACKEND,
            is_full_event_wfo=True,
            requires_silver_bars=True,
            smoke_path=False,
            banners=real_banners,
        )

    # Explicit smoke path may use synthetic probe
    if explicit_smoke or is_synthetic_dataset:
        if requested in {"event_driven", "event_driven_wfo", "institutional", EVENT_WFO_BACKEND}:
            # Allowed but still smoke-labeled unless genuine
            if is_synthetic_dataset:
                # synthetic dataset cannot load real silver — stay on probe unless bars injected
                return BackendResolution(
                    evaluation_backend=SYNTHETIC_BACKEND,
                    is_full_event_wfo=False,
                    requires_silver_bars=False,
                    smoke_path=True,
                    banners=smoke_banners,
                )
        return BackendResolution(
            evaluation_backend=SYNTHETIC_BACKEND,
            is_full_event_wfo=False,
            requires_silver_bars=False,
            smoke_path=True,
            banners=smoke_banners,
        )

    # Non-eligible, non-smoke — reject rather than silent synthetic
    raise BackendResolutionError(
        REAL_DATA_BACKEND_UNAVAILABLE,
        f"dataset {dataset_id!r} is not RESEARCH_ELIGIBLE and smoke_test is false",
    )


def preview_backend_for_form(
    *,
    dataset_id: str,
    smoke_test: bool,
    research_eligible: bool,
    smoke_test_only: bool,
    strategy_family: str,
    timeframe: str,
    budget_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        res = resolve_alpha_miner_backend(
            dataset_id=dataset_id,
            smoke_test=smoke_test,
            research_eligible=research_eligible,
            smoke_test_only=smoke_test_only,
            budget_cfg=budget_cfg,
            strategy_family=strategy_family,
            timeframe=timeframe,
        )
        return {**res.as_dict(), "ok": True}
    except BackendResolutionError as exc:
        return {
            "ok": False,
            "evaluation_backend": SYNTHETIC_BACKEND if smoke_test or smoke_test_only else None,
            "is_full_event_wfo": False,
            "requires_silver_bars": False,
            "smoke_path": bool(smoke_test or smoke_test_only),
            "banners": [],
            "block_reason": str(exc),
            "error_code": exc.code,
        }
