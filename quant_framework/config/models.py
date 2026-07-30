"""Shared enums and typed primitives used across configuration models."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import Field

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]


class AssetClass(str, Enum):
    FUTURES = "futures"
    CFD = "cfd"


class DrawdownType(str, Enum):
    STATIC = "STATIC"
    TRAILING = "TRAILING"


class SettlementBehavior(str, Enum):
    MARK_TO_MARKET = "mark_to_market"
    CASH_SETTLED = "cash_settled"
    PHYSICAL = "physical"


class ContinuousAdjustment(str, Enum):
    NONE = "none"
    BACKWARD_RATIO = "backward_ratio"
    BACKWARD_PANAMA = "backward_panama"
    FORWARD_PANAMA = "forward_panama"


class RolloverMethod(str, Enum):
    VOLUME = "volume"
    OPEN_INTEREST = "open_interest"
    CALENDAR = "calendar"
    MANUAL = "manual"


class IntrabarAmbiguityPolicy(str, Enum):
    CONSERVATIVE_WORST_CASE = "CONSERVATIVE_WORST_CASE"
    OPTIMISTIC = "OPTIMISTIC"
    RANDOMIZED_WITH_SEED = "RANDOMIZED_WITH_SEED"
    LOWER_TIMEFRAME_REPLAY = "LOWER_TIMEFRAME_REPLAY"


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    REDUCE_ONLY = "REDUCE_ONLY"
    FLATTEN = "FLATTEN"
    HALTED = "HALTED"
    MANUAL_LOCK = "MANUAL_LOCK"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Timeframe(str, Enum):
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    H1 = "1h"
    D1 = "1D"


class RegimeLabel(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"
