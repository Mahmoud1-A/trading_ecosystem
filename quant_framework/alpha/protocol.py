"""Strategy protocol — strategies are not hardcoded into the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from config.models import Timeframe
from config.strategy_config import StrategyParams
from engine.events import SignalEvent
from engine.fills import Fill
from engine.portfolio import Portfolio


@dataclass
class StrategyState:
    """Mutable per-run strategy state."""

    bars_in_position: int = 0
    entry_side: str | None = None
    entry_price: float | None = None
    entry_bar_index: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Contract for pluggable intraday strategies.

    Feature computation is vectorized; signal generation respects information timing.
    """

    @property
    def family(self) -> str: ...

    @property
    def params(self) -> StrategyParams: ...

    def required_timeframes(self) -> list[Timeframe]: ...

    def required_features(self) -> list[str]: ...

    def compute_features(
        self,
        bars: pd.DataFrame,
        *,
        regime_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame: ...

    def generate_signals(
        self,
        row: pd.Series,
        *,
        bar_index: int,
        portfolio: Portfolio,
        state: StrategyState,
    ) -> list[SignalEvent]: ...

    def on_fill(self, fill: Fill, state: StrategyState) -> None: ...

    def on_session_end(self, ts: pd.Timestamp, portfolio: Portfolio, state: StrategyState) -> list[SignalEvent]: ...
