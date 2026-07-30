"""Strategy adapter + runner bridging vectorized alpha to event engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alpha.protocol import StrategyProtocol, StrategyState
from config.asset_spec import CFDAssetSpec, FuturesAssetSpec
from config.strategy_config import SizingParams
from engine.event_execution import EventExecutionEngine, ExecutionResult, bars_from_frame
from engine.events import BarEvent, SignalEvent
from engine.portfolio import Portfolio
from risk.position_sizing import volatility_target_size


@dataclass
class StrategyRunResult:
    execution: ExecutionResult
    features: pd.DataFrame
    state: StrategyState = field(default_factory=StrategyState)


def build_signal_fn(
    strategy: StrategyProtocol,
    features: pd.DataFrame,
    *,
    sizing_params: SizingParams | None = None,
    asset: FuturesAssetSpec | CFDAssetSpec | None = None,
    state: StrategyState | None = None,
) -> Any:
    """Return a closure compatible with EventExecutionEngine.run(signal_fn=...)."""
    st = state or StrategyState()
    feat_by_ts = features.copy()
    if feat_by_ts.index.name != "timestamp" and "timestamp" not in feat_by_ts.columns:
        feat_by_ts = feat_by_ts.reset_index()

    if "timestamp" in feat_by_ts.columns:
        index_map = {pd.Timestamp(t): i for i, t in enumerate(feat_by_ts["timestamp"])}
        rows = {pd.Timestamp(r["timestamp"]): r for _, r in feat_by_ts.iterrows()}
    else:
        index_map = {pd.Timestamp(t): i for i, t in enumerate(feat_by_ts.index)}
        rows = {pd.Timestamp(t): feat_by_ts.loc[t] for t in feat_by_ts.index}

    def _fn(bar: BarEvent, portfolio: Portfolio, bar_index: int) -> list[SignalEvent] | None:
        row = rows.get(pd.Timestamp(bar.timestamp))
        if row is None:
            return None
        idx = index_map.get(pd.Timestamp(bar.timestamp), bar_index)
        raw = strategy.generate_signals(row, bar_index=idx, portfolio=portfolio, state=st)
        if not raw:
            return None
        sized: list[SignalEvent] = []
        for sig in raw:
            if sig.side.upper() != "FLAT" and sizing_params is not None and asset is not None:
                atr = float(row.get("atr", 0.0) or 0.0)
                sd = volatility_target_size(
                    portfolio=portfolio,
                    sizing=sizing_params,
                    asset=asset,
                    atr_points=atr,
                    symbol=sig.symbol,
                )
                if sd.rejected or sd.allowed_quantity <= 0:
                    continue
                sig = SignalEvent(
                    timing=sig.timing,
                    symbol=sig.symbol,
                    side=sig.side,
                    quantity=sd.allowed_quantity,
                    stop_price=sig.stop_price,
                    target_price=sig.target_price,
                    order_type=sig.order_type,
                    reason=sig.reason,
                    meta={**sig.meta, "sized_qty": sd.allowed_quantity},
                )
            sized.append(sig)
        return sized or None

    return _fn


def run_strategy_backtest(
    engine: EventExecutionEngine,
    strategy: StrategyProtocol,
    bars: pd.DataFrame,
    *,
    sizing_params: SizingParams | None = None,
) -> StrategyRunResult:
    """End-to-end: features -> signals -> event execution."""
    features = strategy.compute_features(bars)
    state = StrategyState()
    signal_fn = build_signal_fn(
        strategy,
        features,
        sizing_params=sizing_params,
        asset=engine.asset,
        state=state,
    )
    events = bars_from_frame(
        features.reset_index() if features.index.name else features,
        symbol=engine.symbol,
        contract=(
            engine.asset.active_contract
            if isinstance(engine.asset, FuturesAssetSpec)
            else engine.asset.broker_symbol
        ),
    )
    result = engine.run(events, signal_fn)
    return StrategyRunResult(execution=result, features=features, state=state)
