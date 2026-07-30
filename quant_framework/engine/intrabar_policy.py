"""Intrabar stop/target ambiguity resolution policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from config.models import IntrabarAmbiguityPolicy, Side


class IntrabarOutcome(str, Enum):
    NONE = "NONE"
    STOP = "STOP"
    TARGET = "TARGET"
    BOTH_AMBIGUOUS = "BOTH_AMBIGUOUS"


@dataclass(frozen=True)
class IntrabarResolution:
    outcome: IntrabarOutcome
    fill_price: float | None
    policy_applied: IntrabarAmbiguityPolicy
    gap_through: bool = False
    notes: str = ""


def _stop_hit(side: Side, low: float, high: float, stop: float) -> bool:
    # Long position stop is sell-stop below; short stop is buy-stop above.
    if side == Side.BUY:  # long
        return low <= stop
    return high >= stop


def _target_hit(side: Side, low: float, high: float, target: float) -> bool:
    if side == Side.BUY:  # long take-profit
        return high >= target
    return low <= target


def _gap_through_stop(
    *,
    side: Side,
    open_: float,
    stop: float,
) -> tuple[bool, float]:
    """
    Conservative gap-through: if the bar opens beyond the stop, fill at the open
    (worse than stop for the position), never at a better retroactive price.
    """
    if side == Side.BUY:  # long stopped out with sell
        if open_ < stop:
            return True, open_
        return False, stop
    # short stopped out with buy
    if open_ > stop:
        return True, open_
    return False, stop


def resolve_intrabar(
    *,
    position_side: Side,
    open_: float,
    high: float,
    low: float,
    close: float,
    stop: float | None,
    target: float | None,
    policy: IntrabarAmbiguityPolicy = IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE,
    seed: int | None = None,
    lower_tf_path: list[float] | None = None,
) -> IntrabarResolution:
    """
    Resolve whether stop and/or target are touched within a single OHLC bar.

    Default CONSERVATIVE_WORST_CASE always favors the worse outcome for the strategy
    when both levels are touched in the same bar.
    """
    if stop is None and target is None:
        return IntrabarResolution(IntrabarOutcome.NONE, None, policy)

    stop_touched = stop is not None and _stop_hit(position_side, low, high, stop)
    target_touched = target is not None and _target_hit(position_side, low, high, target)

    gap = False
    stop_fill = stop
    if stop is not None and stop_touched:
        gap, stop_fill = _gap_through_stop(side=position_side, open_=open_, stop=stop)

    if stop_touched and not target_touched:
        return IntrabarResolution(
            IntrabarOutcome.STOP, float(stop_fill), policy, gap_through=gap, notes="stop_only"
        )
    if target_touched and not stop_touched:
        return IntrabarResolution(
            IntrabarOutcome.TARGET, float(target), policy, notes="target_only"
        )
    if not stop_touched and not target_touched:
        return IntrabarResolution(IntrabarOutcome.NONE, None, policy)

    # Both touched — ambiguous path inside the bar
    assert stop is not None and target is not None and stop_fill is not None

    if policy == IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE:
        # Worst for the strategy: assume stop is hit first.
        return IntrabarResolution(
            IntrabarOutcome.STOP,
            float(stop_fill),
            policy,
            gap_through=gap,
            notes="both_touched:conservative_stop_first",
        )

    if policy == IntrabarAmbiguityPolicy.OPTIMISTIC:
        return IntrabarResolution(
            IntrabarOutcome.TARGET,
            float(target),
            policy,
            notes="both_touched:optimistic_target_first",
        )

    if policy == IntrabarAmbiguityPolicy.RANDOMIZED_WITH_SEED:
        rng = np.random.default_rng(seed if seed is not None else 0)
        choose_stop = bool(rng.random() < 0.5)
        if choose_stop:
            return IntrabarResolution(
                IntrabarOutcome.STOP,
                float(stop_fill),
                policy,
                gap_through=gap,
                notes="both_touched:random_stop",
            )
        return IntrabarResolution(
            IntrabarOutcome.TARGET,
            float(target),
            policy,
            notes="both_touched:random_target",
        )

    if policy == IntrabarAmbiguityPolicy.LOWER_TIMEFRAME_REPLAY:
        if not lower_tf_path:
            # Fall back to conservative if no LTF path provided
            return IntrabarResolution(
                IntrabarOutcome.STOP,
                float(stop_fill),
                policy,
                gap_through=gap,
                notes="both_touched:ltf_missing_fallback_conservative",
            )
        for px in lower_tf_path:
            if _stop_hit(position_side, px, px, stop):
                g, sf = _gap_through_stop(side=position_side, open_=open_, stop=stop)
                # Path touch uses path price but never better than conservative gap rule at open
                fill = min(sf, px) if position_side == Side.BUY else max(sf, px)
                return IntrabarResolution(
                    IntrabarOutcome.STOP, float(fill), policy, gap_through=g, notes="ltf_stop_first"
                )
            if _target_hit(position_side, px, px, target):
                return IntrabarResolution(
                    IntrabarOutcome.TARGET, float(target), policy, notes="ltf_target_first"
                )
        return IntrabarResolution(IntrabarOutcome.NONE, None, policy, notes="ltf_path_no_touch")

    raise ValueError(f"Unsupported intrabar policy: {policy}")
