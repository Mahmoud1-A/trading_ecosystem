"""Causality and Future Invariance validation (Phase 6B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from features.catalog import FeatureCatalog
from features.contracts import FeatureApprovalStatus, FeatureContract


@dataclass(frozen=True)
class FutureInvarianceResult:
    passed: bool
    cutoff: str
    compared_rows: int
    max_abs_diff: float
    failing_columns: tuple[str, ...]
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "cutoff": self.cutoff,
            "compared_rows": self.compared_rows,
            "max_abs_diff": self.max_abs_diff,
            "failing_columns": list(self.failing_columns),
            "reason": self.reason,
        }


def assert_no_vault_columns(frame: pd.DataFrame) -> None:
    forbidden = [c for c in frame.columns if "vault" in str(c).lower()]
    if forbidden:
        raise ValueError(f"Feature frame must not access Vault columns: {forbidden}")


def assert_warmup_enforced(frame: pd.DataFrame, contract: FeatureContract) -> None:
    if contract.feature_id not in frame.columns:
        return
    series = frame[contract.feature_id]
    warm = int(contract.warm_up_bars)
    if warm <= 0:
        return
    head = series.iloc[:warm]
    if head.notna().any():
        # Allow features with warm_up 0; for warm_up>0 first warm rows should be NaN
        raise AssertionError(
            f"{contract.feature_id}: warm-up of {warm} bars not enforced "
            f"({int(head.notna().sum())} non-null values in warm-up window)"
        )


def future_invariance_test(
    bars: pd.DataFrame,
    *,
    generate_fn: Callable[[pd.DataFrame], pd.DataFrame],
    cutoff_index: int,
    atol: float = 1e-10,
    feature_columns: list[str] | None = None,
) -> FutureInvarianceResult:
    """
    1. Compute features through timestamp t (cutoff).
    2. Change all data after t.
    3. Recompute.
    4. Assert features at and before t remain identical.
    """
    if cutoff_index < 0 or cutoff_index >= len(bars) - 1:
        return FutureInvarianceResult(
            False,
            cutoff="",
            compared_rows=0,
            max_abs_diff=float("nan"),
            failing_columns=(),
            reason="cutoff_index must leave at least one future bar",
        )

    baseline = generate_fn(bars)
    assert_no_vault_columns(baseline)

    mutated = bars.copy()
    future = mutated.iloc[cutoff_index + 1 :]
    for col in ("open", "high", "low", "close", "volume"):
        if col in mutated.columns:
            mutated.loc[future.index, col] = mutated.loc[future.index, col] * 1.5 + 3.14159

    recomputed = generate_fn(mutated)
    cols = feature_columns or [
        c
        for c in baseline.columns
        if c not in {"availability_timestamp", "source_timestamp", "staleness_age"}
    ]
    cols = [c for c in cols if c in baseline.columns and c in recomputed.columns]
    if not cols:
        return FutureInvarianceResult(
            False, str(bars.index[cutoff_index]), 0, float("nan"), (), "no comparable columns"
        )

    a = baseline.iloc[: cutoff_index + 1][cols]
    b = recomputed.iloc[: cutoff_index + 1][cols]
    diff = (a - b).abs()
    # Ignore both-NaN
    mask = a.notna() | b.notna()
    max_diff = float(diff.where(mask, 0.0).max().max()) if len(diff) else 0.0
    failing = tuple(c for c in cols if float(diff[c].where(mask[c], 0.0).max()) > atol)
    return FutureInvarianceResult(
        passed=len(failing) == 0,
        cutoff=str(bars.index[cutoff_index]),
        compared_rows=int(len(a)),
        max_abs_diff=max_diff,
        failing_columns=failing,
        reason="" if not failing else f"future leakage in {failing}",
    )


def miner_feature_ids(
    catalog: FeatureCatalog,
    *,
    available_capabilities: set,
) -> list[str]:
    """IDs the Alpha Miner may consume: APPROVED + causal + capabilities present."""
    return [f.feature_id for f in catalog.approved_causal(available_capabilities=available_capabilities)]


def assert_tick_volume_not_mislabeled(volume_type: str | None) -> None:
    if volume_type is None:
        return
    from data.events.enums import VolumeType

    # Guardrail for feature metadata consumers
    if volume_type == VolumeType.TICK_ACTIVITY_PROXY.value:
        return
    if volume_type == VolumeType.EXCHANGE_EXECUTED_VOLUME.value:
        return
    if volume_type in {v.value for v in VolumeType}:
        return
    raise ValueError(f"Unknown volume_type label: {volume_type!r}")
