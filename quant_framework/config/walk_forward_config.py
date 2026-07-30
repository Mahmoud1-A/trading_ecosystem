"""Walk-forward configuration (rolling windows — not a single 70/30 split)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WalkForwardConfig(BaseModel):
    """Rolling train / validation / step configuration."""

    model_config = {"extra": "forbid", "frozen": True}

    train_window_days: int = Field(10, ge=1)
    validation_window_days: int = Field(3, ge=1)
    step_forward_days: int = Field(3, ge=1)
    purge_gap_bars: int = Field(0, ge=0)
    embargo_gap_bars: int = Field(0, ge=0)
    optimize_metric: Literal["sharpe", "sortino", "profit_factor", "expectancy"] = "sharpe"
    param_grid: dict[str, list[float | int]] = Field(
        default_factory=lambda: {
            "lookback": [15, 20],
            "z_entry": [1.5, 2.0],
            "z_exit": [0.15, 0.25],
        }
    )
    bars_per_day: int = Field(78, ge=1, description="Approx RTH bars per day for day→bar conversion")
    max_folds: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on generated rolling folds (None = unlimited).",
    )
