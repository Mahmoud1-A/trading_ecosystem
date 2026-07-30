"""Top-level system configuration aggregating all Phase-1 config objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from config.asset_spec import (
    CFDAssetSpec,
    FuturesAssetSpec,
    default_cfd_index,
    default_es_futures,
)
from config.cost_model import CostModel, default_cfd_cost_model, default_futures_cost_model
from config.models import IntrabarAmbiguityPolicy, Timeframe
from config.prop_profile import PropProfile, default_prop_profile
from config.strategy_config import RegimeParams, SizingParams, StrategyParams
from config.walk_forward_config import WalkForwardConfig


class DataConfig(BaseModel):
    """Data ingestion / validation settings."""

    model_config = {"extra": "forbid", "frozen": True}

    path: Path | None = None
    timeframe: Timeframe = Timeframe.M5
    require_timezone: bool = True
    default_timezone: str | None = Field(
        None,
        description=(
            "If set, naive timestamps are localized to this zone (then convertible). "
            "If None and require_timezone=True, naive timestamps are rejected."
        ),
    )
    allow_duplicates: bool = False
    detect_missing_bars: bool = True
    missing_bar_tolerance: int = Field(
        0,
        ge=0,
        description="Number of unexpected gaps allowed before validation fails (0 = strict).",
    )
    hash_algorithm: Literal["sha256"] = "sha256"

    @field_validator("path", mode="before")
    @classmethod
    def _path(cls, value: object) -> Path | None:
        if value is None:
            return None
        return Path(str(value))


class SystemConfig(BaseModel):
    """
    Root research/backtest system configuration.

    Research and backtesting only — not live-trading ready.
    """

    model_config = {"extra": "forbid", "frozen": True}

    system_version: str = "0.12.0-phase12"
    random_seed: int = 42
    asset: FuturesAssetSpec | CFDAssetSpec
    prop_profile: PropProfile
    cost_model: CostModel
    data: DataConfig = Field(default_factory=DataConfig)
    intrabar_policy: IntrabarAmbiguityPolicy = (
        IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE
    )
    starting_equity: float = Field(100_000.0, gt=0)
    output_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    strategy: StrategyParams = Field(default_factory=StrategyParams)
    regime: RegimeParams = Field(default_factory=RegimeParams)
    sizing: SizingParams = Field(default_factory=SizingParams)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _outdir(cls, value: object) -> Path:
        return Path(str(value))

    def to_snapshot(self) -> dict[str, Any]:
        """JSON-serializable config snapshot for experiment registry / reproducibility."""
        return self.model_dump(mode="json")


def default_system_config(
    asset_class: Literal["futures", "cfd"] = "futures",
) -> SystemConfig:
    if asset_class == "futures":
        return SystemConfig(
            asset=default_es_futures(),
            prop_profile=default_prop_profile("challenge"),
            cost_model=default_futures_cost_model(),
            data=DataConfig(timeframe=Timeframe.M5, require_timezone=True),
        )
    return SystemConfig(
        asset=default_cfd_index(),
        prop_profile=default_prop_profile("challenge"),
        cost_model=default_cfd_cost_model(),
        data=DataConfig(timeframe=Timeframe.M5, require_timezone=True),
    )
