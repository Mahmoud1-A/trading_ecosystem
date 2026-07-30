"""Prop-firm risk finite state machine (Phase 3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any

import pandas as pd

from config.models import DrawdownType, RiskState, Side
from config.prop_profile import PropProfile
from engine.orders import Order
from engine.portfolio import Portfolio
from risk.exposure import is_risk_increasing, is_risk_reducing, snapshot_exposure
from risk.position_sizing import SizingDecision, size_by_risk_budget

logger = logging.getLogger(__name__)


class RiskTransitionReason(str, Enum):
    SOFT_DAILY = "soft_daily_breach"
    SOFT_ESCALATION = "soft_escalation_to_reduce_only"
    HARD_DAILY = "hard_daily_breach"
    HARD_TOTAL = "hard_total_drawdown_breach"
    FLATTEN_COMPLETE = "flatten_complete"
    MANUAL_LOCK = "manual_lock"
    AUTHORIZED_RESET = "authorized_reset"
    DAILY_RESET = "daily_reset"
    RECOVERED = "metrics_recovered"


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    adjusted_quantity: float | None = None
    reason: str = ""
    state: RiskState = RiskState.NORMAL


@dataclass
class RiskEvent:
    timestamp: pd.Timestamp
    from_state: RiskState
    to_state: RiskState
    reason: RiskTransitionReason
    daily_pnl_pct: float
    total_drawdown_pct: float
    equity: float
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": str(self.timestamp),
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason.value,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_drawdown_pct": self.total_drawdown_pct,
            "equity": self.equity,
            "meta": dict(self.meta),
        }


@dataclass
class RiskAccountState:
    """Mutable account metrics tracked by the FSM."""

    starting_equity: float
    peak_equity: float
    daily_start_equity: float
    daily_reset_date: datetime | None = None
    authorized_halt_cleared: bool = False


class PropRiskFSM:
    """
    Configurable prop-firm risk FSM.

    States:
        NORMAL -> CAUTION -> REDUCE_ONLY   (soft breach path)
        any active -> FLATTEN -> HALTED    (hard breach path)
        any -> MANUAL_LOCK                 (operator lock; no auto-resume)

    Daily PnL resets at PropProfile.daily_reset_time in daily_reset_timezone.
    """

    _ACTIVE_STATES = frozenset(
        {RiskState.NORMAL, RiskState.CAUTION, RiskState.REDUCE_ONLY, RiskState.FLATTEN}
    )

    def __init__(self, profile: PropProfile, *, starting_equity: float) -> None:
        self.profile = profile
        self.state = RiskState.NORMAL
        self.account = RiskAccountState(
            starting_equity=float(starting_equity),
            peak_equity=float(starting_equity),
            daily_start_equity=float(starting_equity),
        )
        self.events: list[RiskEvent] = []
        self._flatten_pending: bool = False
        self._last_mark_ts: pd.Timestamp | None = None
        self._last_reset_boundary: pd.Timestamp | None = None

    def _reset_boundary(self, local_ts: pd.Timestamp) -> pd.Timestamp:
        """Risk-day boundary containing ``local_ts`` (resets at daily_reset_time)."""
        tz = self.profile.zoneinfo()
        reset_time = self.profile.daily_reset_time
        boundary = pd.Timestamp(datetime.combine(local_ts.date(), reset_time, tzinfo=tz))
        if local_ts < boundary:
            boundary -= pd.Timedelta(days=1)
        return boundary

    @property
    def tick_value(self) -> float:
        return 1.0  # overridden by engine via sizing context when needed

    def daily_pnl_pct(self, equity: float) -> float:
        base = self.account.daily_start_equity
        if base <= 0:
            return 0.0
        return (equity - base) / base

    def mark_equity(self, portfolio: Portfolio) -> float:
        """Equity used for risk checks honoring include_unrealized_pnl."""
        if self.profile.include_unrealized_pnl:
            return portfolio.equity()
        return float(portfolio.cash or 0.0)

    def remaining_hard_daily_budget(self, portfolio: Portfolio) -> float:
        """Remaining cash buffer before hard daily loss (positive = room left)."""
        equity = self.mark_equity(portfolio)
        daily = self.daily_pnl_pct(equity)
        hard = float(self.profile.prop_hard_daily_loss_limit)
        # room in fraction terms
        return max(0.0, daily - hard) * self.account.daily_start_equity

    def remaining_soft_daily_budget(self, portfolio: Portfolio) -> float:
        equity = self.mark_equity(portfolio)
        daily = self.daily_pnl_pct(equity)
        soft = float(self.profile.internal_soft_daily_limit)
        return max(0.0, daily - soft) * self.account.daily_start_equity

    def total_drawdown_pct(self, equity: float) -> float:
        if self.profile.drawdown_type == DrawdownType.TRAILING:
            peak = max(self.account.peak_equity, 1e-12)
            return (equity - peak) / peak
        # STATIC: drawdown from starting equity
        base = max(self.account.starting_equity, 1e-12)
        return (equity - base) / base

    def _soft_progress_to_hard(self, daily_pnl_pct: float) -> float:
        """
        Fraction of the soft→hard daily loss band consumed.

        0 at soft limit, 1 at hard limit.
        """
        soft = float(self.profile.internal_soft_daily_limit)
        hard = float(self.profile.prop_hard_daily_loss_limit)
        if daily_pnl_pct >= soft:
            return 0.0
        if daily_pnl_pct <= hard:
            return 1.0
        return (soft - daily_pnl_pct) / (soft - hard)

    def _record_transition(
        self,
        ts: pd.Timestamp,
        new_state: RiskState,
        reason: RiskTransitionReason,
        equity: float,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if new_state == self.state:
            return
        evt = RiskEvent(
            timestamp=ts,
            from_state=self.state,
            to_state=new_state,
            reason=reason,
            daily_pnl_pct=self.daily_pnl_pct(equity),
            total_drawdown_pct=self.total_drawdown_pct(equity),
            equity=equity,
            meta=meta or {},
        )
        logger.info(
            "Risk transition %s -> %s (%s) daily=%.4f total_dd=%.4f",
            evt.from_state.value,
            evt.to_state.value,
            reason.value,
            evt.daily_pnl_pct,
            evt.total_drawdown_pct,
        )
        self.events.append(evt)
        self.state = new_state

    def _maybe_daily_reset(self, ts: pd.Timestamp, equity: float) -> None:
        tz = self.profile.zoneinfo()
        local = ts.tz_convert(tz)
        boundary = self._reset_boundary(local)

        if self._last_reset_boundary is None:
            self._last_reset_boundary = boundary
            self.account.daily_reset_date = boundary.to_pydatetime()
            return

        if boundary <= self._last_reset_boundary:
            return

        self.account.daily_start_equity = equity
        self.account.daily_reset_date = boundary.to_pydatetime()
        self._last_reset_boundary = boundary
        meta = {"reset_boundary": str(boundary), "equity": equity}

        if self.state in {RiskState.CAUTION, RiskState.REDUCE_ONLY}:
            self._record_transition(ts, RiskState.NORMAL, RiskTransitionReason.DAILY_RESET, equity, meta=meta)
        else:
            self.events.append(
                RiskEvent(
                    timestamp=ts,
                    from_state=self.state,
                    to_state=self.state,
                    reason=RiskTransitionReason.DAILY_RESET,
                    daily_pnl_pct=0.0,
                    total_drawdown_pct=self.total_drawdown_pct(equity),
                    equity=equity,
                    meta=meta,
                )
            )

    def on_mark(self, portfolio: Portfolio, ts: pd.Timestamp) -> RiskState:
        """
        Evaluate drawdown metrics and apply FSM transitions.

        Call after mark-to-market on each bar.
        """
        ts = pd.Timestamp(ts)
        equity = self.mark_equity(portfolio)
        if equity > self.account.peak_equity:
            self.account.peak_equity = equity

        self._maybe_daily_reset(ts, equity)
        self._last_mark_ts = ts

        if self.state == RiskState.MANUAL_LOCK:
            return self.state

        daily = self.daily_pnl_pct(equity)
        total_dd = self.total_drawdown_pct(equity)

        hard_daily = daily <= float(self.profile.prop_hard_daily_loss_limit)
        hard_total = total_dd <= float(self.profile.prop_total_drawdown_limit)

        if hard_daily or hard_total:
            reason = (
                RiskTransitionReason.HARD_DAILY
                if hard_daily
                else RiskTransitionReason.HARD_TOTAL
            )
            if self.state != RiskState.FLATTEN:
                self._flatten_pending = True
                self._record_transition(ts, RiskState.FLATTEN, reason, equity)
            return self.state

        if self.state == RiskState.HALTED:
            return self.state

        if self.state == RiskState.FLATTEN:
            # Remain in FLATTEN until flat; engine calls complete_flatten()
            return self.state

        soft = daily <= float(self.profile.internal_soft_daily_limit)
        progress = self._soft_progress_to_hard(daily)
        reduce_only = progress >= float(self.profile.soft_to_reduce_only_ratio)

        if reduce_only and self.state != RiskState.REDUCE_ONLY:
            self._record_transition(
                ts, RiskState.REDUCE_ONLY, RiskTransitionReason.SOFT_ESCALATION, equity
            )
        elif soft and self.state == RiskState.NORMAL:
            self._record_transition(ts, RiskState.CAUTION, RiskTransitionReason.SOFT_DAILY, equity)
        elif not soft and self.state in {RiskState.CAUTION, RiskState.REDUCE_ONLY}:
            self._record_transition(ts, RiskState.NORMAL, RiskTransitionReason.RECOVERED, equity)

        return self.state

    def set_manual_lock(self, ts: pd.Timestamp, portfolio: Portfolio) -> None:
        equity = portfolio.equity()
        self._record_transition(ts, RiskState.MANUAL_LOCK, RiskTransitionReason.MANUAL_LOCK, equity)

    def authorized_reset(self, ts: pd.Timestamp, portfolio: Portfolio) -> bool:
        """
        Operator-authorized reset from HALTED or MANUAL_LOCK only.

        Does not resume trading if hard limits are still breached.
        """
        if self.state not in {RiskState.HALTED, RiskState.MANUAL_LOCK}:
            return False
        equity = portfolio.equity()
        daily = self.daily_pnl_pct(equity)
        total_dd = self.total_drawdown_pct(equity)
        if daily <= float(self.profile.prop_hard_daily_loss_limit):
            return False
        if total_dd <= float(self.profile.prop_total_drawdown_limit):
            return False
        self.account.authorized_halt_cleared = True
        self._flatten_pending = False
        self._record_transition(ts, RiskState.NORMAL, RiskTransitionReason.AUTHORIZED_RESET, equity)
        return True

    def complete_flatten(self, ts: pd.Timestamp, portfolio: Portfolio) -> None:
        """Transition FLATTEN -> HALTED once flat and pending orders cleared."""
        if self.state != RiskState.FLATTEN:
            return
        exp = snapshot_exposure(portfolio)
        if exp.open_positions == 0:
            equity = portfolio.equity()
            self._flatten_pending = False
            self._record_transition(ts, RiskState.HALTED, RiskTransitionReason.FLATTEN_COMPLETE, equity)

    def evaluate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        *,
        stop_distance_points: float | None = None,
        tick_value: float = 1.0,
    ) -> RiskDecision:
        """Gate or resize an order based on current FSM state."""
        equity = portfolio.equity()
        state = self.state

        if state == RiskState.MANUAL_LOCK:
            return RiskDecision(False, reason="manual_lock", state=state)

        if state == RiskState.HALTED:
            return RiskDecision(False, reason="halted", state=state)

        if state == RiskState.FLATTEN:
            if is_risk_reducing(order, portfolio) or order.meta.get("flatten"):
                return RiskDecision(True, order.quantity, reason="flatten_exit", state=state)
            return RiskDecision(False, reason="flatten_mode", state=state)

        if state == RiskState.REDUCE_ONLY:
            if is_risk_increasing(order, portfolio):
                return RiskDecision(False, reason="reduce_only", state=state)
            return RiskDecision(True, order.quantity, reason="reduce_allowed", state=state)

        exp = snapshot_exposure(portfolio, order.symbol)
        if not order.reduce_only:
            if exp.open_positions >= int(self.profile.max_positions) and exp.symbol_exposure <= 0:
                return RiskDecision(False, reason="max_positions", state=state)
            if exp.symbol_exposure + order.quantity > float(self.profile.max_symbol_exposure) + 1e-12:
                return RiskDecision(False, reason="max_symbol_exposure", state=state)
            if exp.portfolio_exposure + order.quantity > float(self.profile.max_portfolio_exposure) + 1e-12:
                return RiskDecision(False, reason="max_portfolio_exposure", state=state)

        sizing = size_by_risk_budget(
            requested_qty=order.quantity,
            risk_state=state,
            prop_profile=self.profile,
            portfolio=portfolio,
            symbol=order.symbol,
            stop_distance_points=stop_distance_points,
            tick_value=tick_value,
        )
        if sizing.rejected:
            return RiskDecision(False, reason=sizing.reason, state=state)

        if state == RiskState.CAUTION and sizing.allowed_quantity + 1e-12 < order.quantity:
            return RiskDecision(
                True,
                sizing.allowed_quantity,
                reason="caution_size_reduction",
                state=state,
            )

        return RiskDecision(True, order.quantity, reason="ok", state=state)

    def should_cancel_pending(self) -> bool:
        return (
            self.state == RiskState.FLATTEN
            and bool(self.profile.cancel_pending_orders_on_hard_breach)
        )

    def should_flatten_positions(self) -> bool:
        return (
            self.state == RiskState.FLATTEN
            and bool(self.profile.close_positions_on_hard_breach)
        )

    def risk_budget_multiplier(self) -> float:
        if self.state == RiskState.CAUTION:
            return float(self.profile.internal_risk_budget)
        if self.state == RiskState.REDUCE_ONLY:
            return 0.0
        return 1.0
