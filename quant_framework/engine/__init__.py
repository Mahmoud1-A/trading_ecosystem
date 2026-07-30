"""Event-driven execution engine — Phase 2."""

from engine.event_execution import (
    EventExecutionEngine,
    ExecutionConfig,
    ExecutionResult,
    bars_from_frame,
)
from risk.fsm import PropRiskFSM, RiskState
from engine.events import BarEvent, EngineEvent, EventType, InformationTiming, SignalEvent
from engine.financing import CFDFinancingEngine, FinancingEvent
from engine.fills import CostBreakdown, Fill
from engine.friction import FrictionContext, FrictionEngine
from engine.ids import DeterministicIdFactory
from engine.intrabar_policy import IntrabarOutcome, IntrabarResolution, resolve_intrabar
from engine.orders import TERMINAL_STATUSES, Order, OrderLifecycleError
from engine.portfolio import Portfolio, TradeReport
from engine.positions import Position
from engine.rollover import RolloverDecision, execute_futures_rollover
from engine.vectorized_features import (
    assert_no_lookahead,
    compute_feature_frame,
    timing_for_row,
)

__all__ = [
    "BarEvent",
    "CFDFinancingEngine",
    "CostBreakdown",
    "DeterministicIdFactory",
    "EngineEvent",
    "EventExecutionEngine",
    "EventType",
    "ExecutionConfig",
    "ExecutionResult",
    "Fill",
    "FinancingEvent",
    "FrictionContext",
    "FrictionEngine",
    "InformationTiming",
    "IntrabarOutcome",
    "IntrabarResolution",
    "Order",
    "OrderLifecycleError",
    "Portfolio",
    "Position",
    "PropRiskFSM",
    "RiskState",
    "RolloverDecision",
    "SignalEvent",
    "TERMINAL_STATUSES",
    "TradeReport",
    "assert_no_lookahead",
    "bars_from_frame",
    "compute_feature_frame",
    "execute_futures_rollover",
    "resolve_intrabar",
    "timing_for_row",
]
