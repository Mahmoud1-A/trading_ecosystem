"""Deterministic blueprints for hypothesis-driven strategy families."""

from __future__ import annotations

from typing import Any

from discovery.operators import OperatorId


# Shared utility features always injected so ATR stops remain legal.
_ATR = "vol.atr_14"

_CMP = (
    OperatorId.GREATER_THAN.value,
    OperatorId.LESS_THAN.value,
    OperatorId.GREATER_EQUAL.value,
    OperatorId.LESS_EQUAL.value,
)
_LOGIC = (OperatorId.AND.value, OperatorId.OR.value, OperatorId.NOT.value)
_ROLL = (
    OperatorId.ROLLING_MEAN.value,
    OperatorId.ROLLING_STD.value,
    OperatorId.ROLLING_ZSCORE.value,
    OperatorId.EMA.value,
)
_CROSS = (OperatorId.CROSS_ABOVE.value, OperatorId.CROSS_BELOW.value)


def _blueprint(
    family_id: str,
    *,
    hypothesis: str,
    features: tuple[str, ...],
    operators: tuple[str, ...],
    entry_patterns: tuple[str, ...],
    exit_patterns: tuple[str, ...],
    regime_constraints: tuple[str, ...] = (),
    parameter_ranges: dict[str, tuple[float, float]] | None = None,
    complexity_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    feats = tuple(dict.fromkeys([*features, _ATR]))  # stable unique + ATR
    return {
        "family_id": family_id,
        "hypothesis": hypothesis,
        "allowed_features": feats,
        "allowed_operators": tuple(dict.fromkeys(operators)),
        "entry_patterns": entry_patterns,
        "exit_patterns": exit_patterns,
        "regime_constraints": regime_constraints,
        "parameter_ranges": parameter_ranges
        or {
            "entry_threshold": (-2.5, 2.5),
            "exit_threshold": (-1.0, 1.0),
            "lookback": (5.0, 40.0),
        },
        "complexity_limits": complexity_limits
        or {
            "max_tree_depth": 4,
            "max_nodes": 18,
            "max_distinct_features": 4,
            "max_free_parameters": 4,
            "max_entry_conditions": 3,
            "max_exit_conditions": 3,
            "max_regime_gates": 1,
            "max_rolling_lookback": 60,
        },
    }


FAMILY_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "mean_reversion": _blueprint(
        "mean_reversion",
        hypothesis="Prices revert toward a rolling mean after z-score extremes.",
        features=(
            "price.rolling_z_20",
            "price.dist_rolling_mean_20",
            "vol.norm_atr_14",
            "liq.dist_session_vwap",
        ),
        operators=(*_CMP, *_LOGIC, *_ROLL, OperatorId.BETWEEN.value, OperatorId.ABS.value),
        entry_patterns=("zscore_extreme", "dist_mean_threshold"),
        exit_patterns=("zscore_normalize", "mean_reentry"),
        regime_constraints=("prefer_range_regime",),
        parameter_ranges={
            "entry_threshold": (-3.0, -1.0),
            "exit_threshold": (-0.5, 0.5),
            "lookback": (10.0, 40.0),
        },
    ),
    "momentum": _blueprint(
        "momentum",
        hypothesis="Short-horizon return persistence continues after confirmation.",
        features=(
            "price.simple_return_1",
            "price.return_5",
            "price.log_return_1",
            "vol.realized_20",
            "regime.trend_state",
        ),
        operators=(*_CMP, *_LOGIC, *_CROSS, *_ROLL, OperatorId.PERCENT_CHANGE.value),
        entry_patterns=("return_persistence", "cross_momentum"),
        exit_patterns=("momentum_fade", "time_decay_exit"),
        regime_constraints=("require_trend_regime",),
        parameter_ranges={
            "entry_threshold": (0.0, 0.02),
            "exit_threshold": (-0.01, 0.01),
            "lookback": (5.0, 30.0),
        },
    ),
    "breakout": _blueprint(
        "breakout",
        hypothesis="Price breakouts beyond recent range expand with volume confirmation.",
        features=(
            "price.breakout_distance_20",
            "liq.volume_pct_20",
            "vol.range_compression_20",
            "price.return_5",
        ),
        operators=(
            *_CMP,
            *_LOGIC,
            *_CROSS,
            OperatorId.ROLLING_MAX.value,
            OperatorId.ROLLING_MIN.value,
            OperatorId.ROLLING_MEAN.value,
        ),
        entry_patterns=("breakout_distance", "compression_release"),
        exit_patterns=("breakout_failure", "trailing_range_exit"),
        parameter_ranges={
            "entry_threshold": (0.0, 0.05),
            "exit_threshold": (-0.02, 0.02),
            "lookback": (10.0, 40.0),
        },
    ),
    "trend_pullback": _blueprint(
        "trend_pullback",
        hypothesis="Pullbacks toward EMA in an established trend offer continuation entries.",
        features=(
            "price.dist_rolling_mean_20",
            "price.return_5",
            "price.rolling_z_20",
            "regime.trend_state",
            "vol.norm_atr_14",
        ),
        operators=(*_CMP, *_LOGIC, *_CROSS, OperatorId.EMA.value, OperatorId.ROLLING_MEAN.value),
        entry_patterns=("pullback_to_ema", "trend_resume_cross"),
        exit_patterns=("trend_exhaustion", "ema_loss"),
        regime_constraints=("require_trend_regime",),
        parameter_ranges={
            "entry_threshold": (-1.5, 0.0),
            "exit_threshold": (-0.5, 1.0),
            "lookback": (8.0, 40.0),
        },
    ),
    "volatility_expansion": _blueprint(
        "volatility_expansion",
        hypothesis="Volatility expansion after compression predicts directional moves.",
        features=(
            "vol.range_compression_20",
            "vol.realized_20",
            "vol.norm_atr_14",
            "price.simple_return_1",
            "regime.volatility_state",
        ),
        operators=(
            *_CMP,
            *_LOGIC,
            OperatorId.ROLLING_STD.value,
            OperatorId.ROLLING_MEAN.value,
            OperatorId.ABS.value,
            OperatorId.BETWEEN.value,
        ),
        entry_patterns=("vol_expansion_break", "compression_then_move"),
        exit_patterns=("vol_mean_revert", "realized_collapse"),
        regime_constraints=("prefer_vol_expansion",),
        parameter_ranges={
            "entry_threshold": (0.0, 2.0),
            "exit_threshold": (0.0, 1.0),
            "lookback": (10.0, 40.0),
        },
    ),
    "session_reversal": _blueprint(
        "session_reversal",
        hypothesis="Intraday extremes near session open reverse within the session.",
        features=(
            "temp.minutes_since_open",
            "price.rolling_z_20",
            "liq.dist_session_vwap",
            "price.simple_return_1",
        ),
        operators=(
            *_CMP,
            *_LOGIC,
            OperatorId.BETWEEN.value,
            OperatorId.SESSION_EXIT.value,
            OperatorId.TIME_EXIT.value,
            OperatorId.ROLLING_ZSCORE.value,
        ),
        entry_patterns=("session_extreme", "open_fade"),
        exit_patterns=("session_flatten", "vwap_touch"),
        regime_constraints=("intraday_session_only",),
        parameter_ranges={
            "entry_threshold": (-2.5, -1.0),
            "exit_threshold": (-0.25, 0.25),
            "lookback": (5.0, 30.0),
            "minutes_open_max": (5.0, 120.0),
        },
        complexity_limits={
            "max_tree_depth": 4,
            "max_nodes": 16,
            "max_distinct_features": 4,
            "max_free_parameters": 5,
            "max_entry_conditions": 3,
            "max_exit_conditions": 3,
            "max_regime_gates": 1,
            "max_rolling_lookback": 40,
        },
    ),
    "gap_fade": _blueprint(
        "gap_fade",
        hypothesis="Opening gaps fade back toward prior close / VWAP.",
        features=(
            "price.simple_return_1",
            "liq.dist_session_vwap",
            "temp.minutes_since_open",
            "price.dist_rolling_mean_20",
        ),
        operators=(*_CMP, *_LOGIC, OperatorId.ABS.value, OperatorId.BETWEEN.value, *_ROLL),
        entry_patterns=("gap_fade_entry", "vwap_gap_reversion"),
        exit_patterns=("gap_filled", "session_flatten"),
        parameter_ranges={
            "entry_threshold": (0.002, 0.03),
            "exit_threshold": (-0.005, 0.005),
            "lookback": (5.0, 20.0),
            "minutes_open_max": (1.0, 90.0),
        },
    ),
    "VWAP_reversion": _blueprint(
        "VWAP_reversion",
        hypothesis="Deviations from session VWAP mean-revert under normal liquidity.",
        features=(
            "liq.dist_session_vwap",
            "liq.volume_pct_20",
            "price.rolling_z_20",
            "vol.norm_atr_14",
        ),
        operators=(
            *_CMP,
            *_LOGIC,
            OperatorId.ABS.value,
            OperatorId.BETWEEN.value,
            OperatorId.ROLLING_ZSCORE.value,
            OperatorId.ROLLING_MEAN.value,
            OperatorId.LIQUIDITY_GATE.value,
        ),
        entry_patterns=("vwap_deviation", "vwap_zscore"),
        exit_patterns=("vwap_reentry", "liquidity_exit"),
        regime_constraints=("prefer_liquid_session",),
        parameter_ranges={
            "entry_threshold": (-0.01, -0.001),
            "exit_threshold": (-0.002, 0.002),
            "lookback": (10.0, 40.0),
        },
    ),
}


DEFAULT_FAMILY_ORDER: tuple[str, ...] = (
    "mean_reversion",
    "momentum",
    "breakout",
    "trend_pullback",
    "volatility_expansion",
    "session_reversal",
    "gap_fade",
    "VWAP_reversion",
)
