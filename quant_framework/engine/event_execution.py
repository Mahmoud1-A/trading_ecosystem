"""
Event-driven execution engine with explicit information timing.

Portfolio execution is sequential and deterministic — not DataFrame arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

import pandas as pd

from config.asset_spec import CFDAssetSpec, FuturesAssetSpec
from config.cost_model import CostModel
from config.models import IntrabarAmbiguityPolicy, OrderStatus, OrderType, Side
from engine.events import BarEvent, InformationTiming, SignalEvent, require_aware
from engine.financing import CFDFinancingEngine, FinancingEvent
from engine.fills import Fill
from engine.friction import FrictionContext, FrictionEngine
from engine.ids import DeterministicIdFactory
from engine.intrabar_policy import IntrabarOutcome, resolve_intrabar
from engine.orders import Order
from engine.portfolio import Portfolio
from engine.rollover import RolloverDecision, execute_futures_rollover
from risk.fsm import PropRiskFSM, RiskState

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Runtime execution assumptions (research/backtest only)."""

    latency: timedelta = field(default_factory=lambda: timedelta(0))
    intrabar_policy: IntrabarAmbiguityPolicy = IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE
    stale_rejects_fills: bool = True
    expire_day_orders_at_session_close: bool = True
    allow_partial_fills: bool = True
    random_seed: int = 42
    # Maximum age of a bar for it to be considered fresh relative to wall/event time
    max_bar_staleness: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    # Accrue CFD overnight financing on crossed broker rollover boundaries
    accrue_cfd_financing: bool = True
    # Roll futures positions when the bar's tradable contract changes
    auto_roll_futures: bool = True
    # Research/WFO: close residual open lots after the final bar so window-end
    # FLAT signals are not stranded by no_next_bar_for_execution.
    # Default False so unit execution tests can leave intentional open lots.
    flatten_at_window_end: bool = False


@dataclass
class ExecutionResult:
    portfolio: Portfolio
    orders: list[Order]
    fills: list[Fill]
    rejected_signals: list[dict[str, Any]]
    risk_events: list[dict[str, Any]]
    financing_events: list[FinancingEvent] = field(default_factory=list)
    rollover_decisions: list[RolloverDecision] = field(default_factory=list)


SignalFn = Callable[[BarEvent, Portfolio, int], SignalEvent | list[SignalEvent] | None]


class EventExecutionEngine:
    """
    Bar-by-bar event loop:

    1. Advance clock to bar open
    2. Activate eligible orders whose activation_timestamp <= now
    3. Process working orders against this bar (market / limit / stop)
    4. Mark portfolio
    5. At bar close: accept new signals (decision_timestamp = bar_end)
       and submit orders that can activate no earlier than next bar open
    """

    def __init__(
        self,
        asset: FuturesAssetSpec | CFDAssetSpec,
        cost_model: CostModel,
        *,
        starting_equity: float = 100_000.0,
        exec_config: ExecutionConfig | None = None,
        risk_manager: PropRiskFSM | None = None,
        id_factory: DeterministicIdFactory | None = None,
        financing_engine: CFDFinancingEngine | None = None,
    ) -> None:
        self.asset = asset
        self.cost_model = cost_model
        self.exec_config = exec_config or ExecutionConfig()
        self.risk = risk_manager
        self.id_factory = id_factory or DeterministicIdFactory(run_id="default")
        self.friction = FrictionEngine(cost_model, asset, seed=self.exec_config.random_seed)
        self.portfolio = Portfolio(starting_equity=starting_equity)
        symbol = asset.symbol if isinstance(asset, FuturesAssetSpec) else asset.broker_symbol
        mult = asset.multiplier if isinstance(asset, FuturesAssetSpec) else asset.contract_size
        self.symbol = symbol
        self.portfolio.set_multiplier(symbol, float(mult))
        self._working: list[Order] = []
        self._all_orders: list[Order] = []
        self._rejected_signals: list[dict[str, Any]] = []
        self._risk_events: list[dict[str, Any]] = []
        self._pending_brackets: dict[str, dict[str, float | None]] = {}
        self._clock: pd.Timestamp | None = None

        # P5.5-C: remaining participation capacity for the bar currently being processed
        self._bar_capacity: float = 0.0

        # P5.5-E: CFD overnight financing
        self.financing: CFDFinancingEngine | None = financing_engine
        if (
            self.financing is None
            and isinstance(asset, CFDAssetSpec)
            and self.exec_config.accrue_cfd_financing
            and bool(cost_model.overnight_swap_enabled)
        ):
            self.financing = CFDFinancingEngine(asset=asset)
        self.financing_events: list[FinancingEvent] = []
        self._prev_mark_ts: pd.Timestamp | None = None

        # P5.5-F: futures rollover on active contract change
        self.rollover_decisions: list[RolloverDecision] = []
        self._active_contract: str = (
            asset.active_contract if isinstance(asset, FuturesAssetSpec) else asset.broker_symbol
        )
        self._prev_bar_close: float | None = None

    def _tradable_contract(self, bar: BarEvent) -> str:
        if isinstance(self.asset, FuturesAssetSpec):
            # Executable fills always on explicit tradable contract
            return bar.contract or self.asset.active_contract
        return bar.contract or self.asset.broker_symbol

    def _new_order_id(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType,
        decision_timestamp: pd.Timestamp,
        contract: str,
    ) -> str:
        return self.id_factory.next_order_id(
            symbol=symbol,
            side=side.value,
            quantity=float(quantity),
            order_type=order_type.value,
            decision_timestamp=str(decision_timestamp),
            contract=contract,
        )

    def _new_fill_id(self, *, order_id: str, fill_ts: pd.Timestamp, quantity: float, price: float) -> str:
        return self.id_factory.next_fill_id(
            order_id=order_id,
            fill_timestamp=str(fill_ts),
            quantity=float(quantity),
            price=float(price),
        )

    def _friction_ctx(self, bar: BarEvent, mid: float) -> FrictionContext:
        return FrictionContext(
            mid_price=mid,
            atr=bar.atr,
            bar_volume=bar.volume,
            is_near_session_open=bar.is_session_open,
            is_near_session_close=bar.is_session_close,
            bid=bar.bid,
            ask=bar.ask,
        )

    def _reject_stale(self, bar: BarEvent) -> bool:
        if not self.exec_config.stale_rejects_fills:
            return False
        if bar.is_stale:
            return True
        return False

    def _tick_value(self) -> float:
        if isinstance(self.asset, FuturesAssetSpec):
            return float(self.asset.tick_value)
        return float(self.asset.contract_size) * float(self.asset.tick_size)

    def _record_risk_event(self, payload: dict[str, Any]) -> None:
        self._risk_events.append(payload)

    def _cancel_all_working(self, ts: pd.Timestamp, *, reason: str) -> None:
        for order in self._working:
            if order.is_terminal:
                continue
            order.cancel(ts, reason)
            self._record_risk_event(
                {
                    "type": "order_cancelled",
                    "order_id": order.order_id,
                    "reason": reason,
                    "timestamp": str(ts),
                }
            )
        self._working = []

    def _submit_flatten_orders(self, bar: BarEvent) -> list[Order]:
        submitted: list[Order] = []
        for sym, pos in self.portfolio.positions.items():
            if pos.is_flat:
                continue
            sig = SignalEvent(
                timing=InformationTiming(
                    source_timestamp=bar.timestamp,
                    availability_timestamp=bar.bar_end,
                    decision_timestamp=bar.bar_end,
                ),
                symbol=sym,
                side="FLAT",
                quantity=abs(pos.quantity),
                signal_id=self.id_factory.next_signal_id(
                    decision_timestamp=str(bar.bar_end), symbol=sym, side="FLAT"
                ),
                reason="risk_flatten",
                meta={"flatten": True},
            )
            order = self._create_order_from_signal(
                sig,
                next_bar_open=bar.bar_end,
                skip_risk=True,
            )
            if order is not None:
                order.meta["flatten"] = True
                submitted.append(order)
        return submitted

    def _create_order_from_signal(
        self,
        signal: SignalEvent,
        *,
        next_bar_open: pd.Timestamp,
        skip_risk: bool = False,
    ) -> Order | None:
        """
        Convert a close-of-bar signal into a submitted order.

        Earliest activation is max(decision + latency, next_bar_open).
        Never grants a fill before next bar open for a bar-close decision.
        """
        decision_ts = signal.timing.decision_timestamp
        submission = decision_ts
        activation = max(
            submission + self.exec_config.latency,
            require_aware(next_bar_open, name="next_bar_open"),
        )
        timing = InformationTiming(
            source_timestamp=signal.timing.source_timestamp,
            availability_timestamp=signal.timing.availability_timestamp,
            decision_timestamp=decision_ts,
            order_submission_timestamp=submission,
            order_activation_timestamp=activation,
        )

        if signal.side.upper() == "FLAT":
            pos = self.portfolio.get_position(signal.symbol)
            if pos.is_flat:
                return None
            side = Side.SELL if pos.quantity > 0 else Side.BUY
            qty = abs(pos.quantity)
            reduce_only = True
        else:
            side = Side.BUY if signal.side.upper() == "BUY" else Side.SELL
            qty = float(signal.quantity)
            reduce_only = False

        if qty <= 0:
            self._rejected_signals.append({"reason": "non_positive_qty", "signal": signal.signal_id})
            return None

        otype = OrderType[signal.order_type] if isinstance(signal.order_type, str) else signal.order_type
        contract = self._active_contract
        order = Order(
            symbol=signal.symbol,
            contract=contract,
            side=side,
            quantity=qty,
            order_type=otype,
            timing=timing,
            order_id=self._new_order_id(
                symbol=signal.symbol,
                side=side,
                quantity=qty,
                order_type=otype,
                decision_timestamp=decision_ts,
                contract=contract,
            ),
            limit_price=signal.limit_price,
            stop_price=signal.stop_trigger,
            parent_signal_id=signal.signal_id,
            reduce_only=reduce_only,
            latency=self.exec_config.latency,
            linked_stop=signal.stop_price,
            linked_target=signal.target_price,
            meta=dict(signal.meta),
        )
        order.status = OrderStatus.CREATED
        order.status_history = [(OrderStatus.CREATED, decision_ts)]
        order.transition(OrderStatus.SUBMITTED, submission)

        if self.risk is not None and not skip_risk:
            risk_decision = self.risk.evaluate_order(
                order,
                self.portfolio,
                tick_value=self._tick_value(),
            )
            if not risk_decision.allowed:
                order.reject(submission, risk_decision.reason)
                self._record_risk_event(
                    {
                        "type": "order_rejected",
                        "order_id": order.order_id,
                        "reason": risk_decision.reason,
                        "state": risk_decision.state.value,
                        "timestamp": str(submission),
                    }
                )
                self._all_orders.append(order)
                return None
            if (
                risk_decision.adjusted_quantity is not None
                and risk_decision.adjusted_quantity + 1e-12 < order.quantity
            ):
                order.quantity = float(risk_decision.adjusted_quantity)
                order.remaining_quantity = float(risk_decision.adjusted_quantity)

        self.portfolio.register_order(order)
        self._working.append(order)
        self._all_orders.append(order)
        if signal.stop_price is not None or signal.target_price is not None:
            self._pending_brackets[order.order_id] = {
                "stop": signal.stop_price,
                "target": signal.target_price,
            }
        logger.debug(
            "Submitted %s %s qty=%s activate=%s decision=%s",
            order.side.value,
            order.order_type.value,
            order.quantity,
            activation,
            decision_ts,
        )
        return order

    def submit_signal(self, signal: SignalEvent, *, next_bar_open: pd.Timestamp) -> Order | None:
        return self._create_order_from_signal(signal, next_bar_open=next_bar_open)

    def _reference_price_for_activation(
        self,
        order: Order,
        bar: BarEvent,
    ) -> tuple[float, pd.Timestamp, str]:
        """
        First legally available reference price at/after order_activation_timestamp.

        If activation <= bar open → may use bar open.
        If activation > bar open → must NOT use retroactive open; use a
        conservative proxy: bar close if activation <= bar_end, else no fill on this bar.
        For market orders activating mid-bar without LTF data, use bar close
        (never the already-elapsed open).
        """
        assert order.activation_timestamp is not None
        act = order.activation_timestamp
        if act <= bar.timestamp:
            return bar.open, bar.timestamp, "bar_open"
        if act <= bar.bar_end:
            # Latency missed the open — no retroactive open pricing
            return bar.close, max(act, bar.timestamp), "post_open_no_retroactive"
        # Activation after this bar ends — cannot fill on this bar
        raise LookupError("activation_after_bar")

    def _allowed_fill_qty(self, order: Order, bar: BarEvent) -> tuple[float, str]:
        """
        Quantity that may fill on this bar for this order.

        Bounded by (a) the order's own remaining quantity, (b) the per-bar
        participation cap, and (c) the capacity not already consumed by earlier
        orders in the same bar. Returns ``(quantity, reason)``.
        """
        want = float(order.remaining_quantity or 0.0)
        if want <= 0:
            return 0.0, "no_remaining_quantity"
        bar_cap = self.friction.max_fill_quantity(bar.volume, want)
        if bar_cap <= 0:
            return 0.0, "insufficient_liquidity"
        qty = min(bar_cap, self._bar_capacity)
        if qty <= 1e-12:
            return 0.0, "bar_capacity_exhausted"
        return float(qty), "ok"

    def _consume_capacity(self, quantity: float) -> None:
        self._bar_capacity = max(0.0, self._bar_capacity - float(quantity))

    def _try_fill_market(self, order: Order, bar: BarEvent) -> Fill | None:
        if self._reject_stale(bar):
            order.reject(bar.timestamp, "stale_bar")
            return None
        try:
            ref, fill_ts, how = self._reference_price_for_activation(order, bar)
        except LookupError:
            return None

        if order.status == OrderStatus.SUBMITTED:
            order.activate(max(order.activation_timestamp or bar.timestamp, fill_ts))

        qty = float(order.remaining_quantity or 0.0)
        max_qty, reason = self._allowed_fill_qty(order, bar)
        if reason == "insufficient_liquidity":
            order.reject(fill_ts, "insufficient_liquidity")
            return None
        if max_qty <= 0:
            # Bar participation already consumed — order stays working for the next bar
            return None
        if not self.exec_config.allow_partial_fills and max_qty + 1e-12 < qty:
            # Cannot fully fill under participation cap — leave working
            return None
        fill_qty = max_qty

        ctx = self._friction_ctx(bar, mid=ref)
        fill_px, costs = self.friction.build_costs(
            side=order.side, quantity=fill_qty, reference_price=ref, ctx=ctx
        )
        timing = order.timing.with_fill(fill_ts)
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            contract=self._tradable_contract(bar),
            side=order.side,
            quantity=fill_qty,
            price=fill_px,
            timing=timing,
            costs=costs,
            fill_id=self._new_fill_id(
                order_id=order.order_id, fill_ts=fill_ts, quantity=fill_qty, price=fill_px
            ),
            reference_price=ref,
            mid_price=ref,
            liquidity_used=fill_qty,
            partial=fill_qty + 1e-12 < qty,
            rollover_decision=None,
            meta={"pricing_mode": how},
        )
        order.apply_fill_qty(fill_qty, fill_ts)
        order.timing = timing
        self.portfolio.apply_fill(fill)
        self._consume_capacity(fill_qty)
        return fill

    def _try_fill_limit(self, order: Order, bar: BarEvent) -> Fill | None:
        if order.limit_price is None:
            order.reject(bar.timestamp, "missing_limit_price")
            return None
        if self._reject_stale(bar):
            order.reject(bar.timestamp, "stale_bar")
            return None
        assert order.activation_timestamp is not None
        if order.activation_timestamp > bar.bar_end:
            return None
        if order.status == OrderStatus.SUBMITTED and order.activation_timestamp <= bar.timestamp:
            order.activate(bar.timestamp)
        elif order.status == OrderStatus.SUBMITTED and order.activation_timestamp <= bar.bar_end:
            order.activate(order.activation_timestamp)
        if order.status not in {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}:
            return None

        # No fill before activation; if activation after open, ignore open touch
        usable_low, usable_high = bar.low, bar.high
        touched = (
            usable_low <= order.limit_price
            if order.side == Side.BUY
            else usable_high >= order.limit_price
        )
        if not touched:
            return None

        # Conservative limit fill: buy fills at min(limit, open-if-gapped), never better than limit
        if order.side == Side.BUY:
            if order.activation_timestamp <= bar.timestamp and bar.open < order.limit_price:
                ref = bar.open  # gap through — still pay open (worse for limit buy? actually better)
                # For limit buys, gap below limit fills at open (price improvement). Keep it.
            else:
                ref = order.limit_price
            # But if activation missed open, do not grant open improvement
            if order.activation_timestamp > bar.timestamp:
                ref = order.limit_price
        else:
            if order.activation_timestamp <= bar.timestamp and bar.open > order.limit_price:
                ref = bar.open
            else:
                ref = order.limit_price
            if order.activation_timestamp > bar.timestamp:
                ref = order.limit_price

        fill_ts = max(order.activation_timestamp, bar.timestamp)
        qty = float(order.remaining_quantity or 0.0)
        fill_qty, _reason = self._allowed_fill_qty(order, bar)
        if fill_qty <= 0:
            return None
        ctx = self._friction_ctx(bar, mid=ref)
        # Limits: apply commission + residual slip only on spread already embedded
        fill_px, costs = self.friction.build_costs(
            side=order.side, quantity=fill_qty, reference_price=ref, ctx=ctx
        )
        # Cap buy fills at limit (don't pay through limit due to friction model on mid)
        if order.side == Side.BUY:
            fill_px = min(fill_px, order.limit_price + (fill_px - ref))
        timing = order.timing.with_fill(fill_ts)
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            contract=self._tradable_contract(bar),
            side=order.side,
            quantity=fill_qty,
            price=fill_px,
            timing=timing,
            costs=costs,
            fill_id=self._new_fill_id(
                order_id=order.order_id, fill_ts=fill_ts, quantity=fill_qty, price=fill_px
            ),
            reference_price=ref,
            liquidity_used=fill_qty,
            partial=fill_qty + 1e-12 < qty,
            meta={"pricing_mode": "limit"},
        )
        order.apply_fill_qty(fill_qty, fill_ts)
        order.timing = timing
        self.portfolio.apply_fill(fill)
        self._consume_capacity(fill_qty)
        return fill

    def _try_fill_stop(self, order: Order, bar: BarEvent) -> Fill | None:
        if order.stop_price is None:
            order.reject(bar.timestamp, "missing_stop_price")
            return None
        if self._reject_stale(bar):
            order.reject(bar.timestamp, "stale_bar")
            return None
        assert order.activation_timestamp is not None
        if order.activation_timestamp > bar.bar_end:
            return None
        if order.status == OrderStatus.SUBMITTED:
            if order.activation_timestamp <= bar.bar_end:
                order.activate(max(order.activation_timestamp, bar.timestamp))
        if order.status not in {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}:
            return None

        stop = order.stop_price
        # Trigger check — buy stop if high>=stop; sell stop if low<=stop
        triggered = bar.high >= stop if order.side == Side.BUY else bar.low <= stop
        if not triggered:
            return None

        # Conservative gap-through
        gap = False
        if order.side == Side.BUY:
            if order.activation_timestamp <= bar.timestamp and bar.open > stop:
                ref = bar.open
                gap = True
            elif order.activation_timestamp > bar.timestamp:
                # Missed open — cannot use open; fill at worse of stop and close proxy
                ref = max(stop, bar.close)
                gap = bar.close > stop
            else:
                ref = stop
        else:
            if order.activation_timestamp <= bar.timestamp and bar.open < stop:
                ref = bar.open
                gap = True
            elif order.activation_timestamp > bar.timestamp:
                ref = min(stop, bar.close)
                gap = bar.close < stop
            else:
                ref = stop

        fill_ts = max(order.activation_timestamp, bar.timestamp)
        qty = float(order.remaining_quantity or 0.0)
        fill_qty, _reason = self._allowed_fill_qty(order, bar)
        if fill_qty <= 0:
            return None
        ctx = self._friction_ctx(bar, mid=ref)
        fill_px, costs = self.friction.build_costs(
            side=order.side, quantity=fill_qty, reference_price=ref, ctx=ctx
        )
        timing = order.timing.with_fill(fill_ts)
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            contract=self._tradable_contract(bar),
            side=order.side,
            quantity=fill_qty,
            price=fill_px,
            timing=timing,
            costs=costs,
            fill_id=self._new_fill_id(
                order_id=order.order_id, fill_ts=fill_ts, quantity=fill_qty, price=fill_px
            ),
            reference_price=ref,
            gap_through=gap,
            liquidity_used=fill_qty,
            partial=fill_qty + 1e-12 < qty,
            meta={"pricing_mode": "stop"},
        )
        order.apply_fill_qty(fill_qty, fill_ts)
        order.timing = timing
        self.portfolio.apply_fill(fill)
        self._consume_capacity(fill_qty)
        return fill

    def _try_fill_stop_limit(self, order: Order, bar: BarEvent) -> Fill | None:
        """
        True stop-limit: the stop only *activates* a limit order — it never marketizes.

        Lifecycle:
        1. The stop triggers when the bar trades through ``stop_price``. The order is
           marked ``meta["stop_triggered"]`` and becomes a live limit at ``limit_price``.
        2. A fill happens only when ``limit_price`` is legally available.
        3. A gap beyond the limit leaves the order triggered but unfilled; it may fill
           on a later bar, be cancelled, or expire.
        """
        if order.stop_price is None:
            order.reject(bar.timestamp, "missing_stop_price")
            return None
        if order.limit_price is None:
            order.reject(bar.timestamp, "missing_limit_price")
            return None
        if self._reject_stale(bar):
            order.reject(bar.timestamp, "stale_bar")
            return None
        assert order.activation_timestamp is not None
        if order.activation_timestamp > bar.bar_end:
            return None
        if order.status == OrderStatus.SUBMITTED:
            order.activate(max(order.activation_timestamp, bar.timestamp))
        if order.status not in {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}:
            return None

        stop = float(order.stop_price)
        limit = float(order.limit_price)
        is_buy = order.side == Side.BUY
        fill_ts = max(order.activation_timestamp, bar.timestamp)
        open_visible = order.activation_timestamp <= bar.timestamp

        if not order.meta.get("stop_triggered"):
            triggered = bar.high >= stop if is_buy else bar.low <= stop
            if not triggered:
                return None
            # Price at which the stop was hit (gap-aware, never retroactive)
            if is_buy:
                trigger_px = max(stop, bar.open) if (open_visible and bar.open >= stop) else stop
            else:
                trigger_px = min(stop, bar.open) if (open_visible and bar.open <= stop) else stop
            order.meta["stop_triggered"] = True
            order.meta["stop_triggered_at"] = str(fill_ts)
            order.meta["stop_trigger_price"] = float(trigger_px)
            self._record_risk_event(
                {
                    "type": "stop_limit_triggered",
                    "order_id": order.order_id,
                    "trigger_price": float(trigger_px),
                    "limit_price": limit,
                    "timestamp": str(fill_ts),
                }
            )
            available = trigger_px <= limit + 1e-12 if is_buy else trigger_px >= limit - 1e-12
            if not available:
                # Gapped past the limit — activated limit stays unfilled
                order.meta["stop_limit_gapped_beyond_limit"] = True
                return None
            ref = float(trigger_px)
            mode = "stop_limit_trigger"
        else:
            # Already triggered: behave as a resting limit order at limit_price
            touched = bar.low <= limit + 1e-12 if is_buy else bar.high >= limit - 1e-12
            if not touched:
                return None
            ref = limit
            mode = "stop_limit_resting"

        qty = float(order.remaining_quantity or 0.0)
        fill_qty, _reason = self._allowed_fill_qty(order, bar)
        if fill_qty <= 0:
            return None
        ctx = self._friction_ctx(bar, mid=ref)
        fill_px, costs = self.friction.build_costs(
            side=order.side, quantity=fill_qty, reference_price=ref, ctx=ctx
        )
        timing = order.timing.with_fill(fill_ts)
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            contract=self._tradable_contract(bar),
            side=order.side,
            quantity=fill_qty,
            price=fill_px,
            timing=timing,
            costs=costs,
            fill_id=self._new_fill_id(
                order_id=order.order_id, fill_ts=fill_ts, quantity=fill_qty, price=fill_px
            ),
            reference_price=ref,
            liquidity_used=fill_qty,
            partial=fill_qty + 1e-12 < qty,
            meta={"pricing_mode": mode, "stop_price": stop, "limit_price": limit},
        )
        order.apply_fill_qty(fill_qty, fill_ts)
        order.timing = timing
        self.portfolio.apply_fill(fill)
        self._consume_capacity(fill_qty)
        return fill

    def _expires_at_session_close(self, order: Order) -> bool:
        if not self.exec_config.expire_day_orders_at_session_close:
            return False
        if order.meta.get("time_in_force", "DAY") != "DAY":
            return False
        return order.status in {
            OrderStatus.SUBMITTED,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        }

    def _process_working_orders(self, bar: BarEvent) -> list[Fill]:
        fills: list[Fill] = []
        still: list[Order] = []
        # P5.5-C: one shared participation budget per bar across all working orders
        self._bar_capacity = float(self.cost_model.participation_rate_cap) * max(
            float(bar.volume), 0.0
        )
        for order in self._working:
            if order.is_terminal:
                continue
            # Orders that cannot activate before this session closes never trade
            if (
                bar.is_session_close
                and self._expires_at_session_close(order)
                and (order.order_type != OrderType.MARKET or order.status == OrderStatus.SUBMITTED)
                and order.activation_timestamp is not None
                and order.activation_timestamp > bar.bar_end
            ):
                order.expire(bar.bar_end, "session_close")
                continue

            fill: Fill | None = None
            if order.order_type == OrderType.MARKET:
                # Activate when clock reaches activation
                if order.activation_timestamp and order.activation_timestamp <= bar.bar_end:
                    fill = self._try_fill_market(order, bar)
            elif order.order_type == OrderType.LIMIT:
                fill = self._try_fill_limit(order, bar)
            elif order.order_type == OrderType.STOP:
                fill = self._try_fill_stop(order, bar)
            elif order.order_type == OrderType.STOP_LIMIT:
                fill = self._try_fill_stop_limit(order, bar)

            if fill is not None:
                fills.append(fill)
            # Working resting orders die at session close after this bar's fill attempt
            if (
                not order.is_terminal
                and bar.is_session_close
                and order.order_type != OrderType.MARKET
                and self._expires_at_session_close(order)
            ):
                order.expire(bar.bar_end, "session_close")
            if not order.is_terminal:
                still.append(order)
        self._working = still
        return fills

    def _manage_open_brackets(self, bar: BarEvent) -> list[Fill]:
        """Evaluate stop/target on open positions using intrabar policy."""
        fills: list[Fill] = []
        pos = self.portfolio.get_position(self.symbol)
        if pos.is_flat or pos.side is None:
            return fills
        stop = pos.meta.get("stop_price")
        target = pos.meta.get("target_price")
        if stop is None and target is None:
            return fills

        resolution = resolve_intrabar(
            position_side=pos.side,
            open_=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            stop=float(stop) if stop is not None else None,
            target=float(target) if target is not None else None,
            policy=self.exec_config.intrabar_policy,
            seed=self.exec_config.random_seed,
        )
        if resolution.outcome == IntrabarOutcome.NONE or resolution.fill_price is None:
            return fills

        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
        qty = abs(pos.quantity)
        # Synthetic reduce-only market at resolved price with friction
        decision = bar.timestamp  # protective exit evaluated during bar
        timing = InformationTiming(
            source_timestamp=bar.timestamp,
            availability_timestamp=bar.timestamp,
            decision_timestamp=decision,
            order_submission_timestamp=decision,
            order_activation_timestamp=decision,
            fill_timestamp=decision,
        )
        contract = self._tradable_contract(bar)
        order = Order(
            symbol=self.symbol,
            contract=contract,
            side=exit_side,
            quantity=qty,
            order_type=OrderType.MARKET,
            timing=timing,
            order_id=self._new_order_id(
                symbol=self.symbol,
                side=exit_side,
                quantity=qty,
                order_type=OrderType.MARKET,
                decision_timestamp=decision,
                contract=contract,
            ),
            reduce_only=True,
            meta={"bracket_exit": resolution.outcome.value},
        )
        order.transition(OrderStatus.SUBMITTED, decision)
        order.activate(decision)
        ctx = self._friction_ctx(bar, mid=resolution.fill_price)
        fill_px, costs = self.friction.build_costs(
            side=exit_side, quantity=qty, reference_price=resolution.fill_price, ctx=ctx
        )
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            contract=order.contract,
            side=exit_side,
            quantity=qty,
            price=fill_px,
            timing=timing,
            costs=costs,
            fill_id=self._new_fill_id(
                order_id=order.order_id, fill_ts=decision, quantity=qty, price=fill_px
            ),
            reference_price=resolution.fill_price,
            gap_through=resolution.gap_through,
            liquidity_used=qty,
            meta={"intrabar": resolution.notes, "outcome": resolution.outcome.value},
        )
        order.apply_fill_qty(qty, decision)
        self._all_orders.append(order)
        self.portfolio.register_order(order)
        self.portfolio.apply_fill(fill)
        pos.meta.pop("stop_price", None)
        pos.meta.pop("target_price", None)
        fills.append(fill)
        return fills

    def cancel_order(self, order_id: str, at: pd.Timestamp, *, reason: str = "client_cancel") -> bool:
        """
        Cancel a working order (including the unfilled remainder of a partial fill).

        Returns False when the order is unknown or already terminal.
        """
        at = require_aware(at, name="cancel_at")
        for order in self._working:
            if order.order_id != order_id:
                continue
            if order.is_terminal:
                return False
            order.cancel(at, reason)
            self._record_risk_event(
                {
                    "type": "order_cancelled",
                    "order_id": order.order_id,
                    "reason": reason,
                    "timestamp": str(at),
                }
            )
            self._working = [o for o in self._working if not o.is_terminal]
            return True
        return False

    def _accrue_financing(self, bar: BarEvent) -> list[FinancingEvent]:
        """P5.5-E: debit CFD overnight swap for rollover boundaries crossed since last mark."""
        if self.financing is None:
            return []
        events = self.financing.accrue(
            self.portfolio,
            prev_ts=self._prev_mark_ts,
            curr_ts=bar.bar_end,
            mark_price=float(bar.close),
            symbol=self.symbol,
        )
        for evt in events:
            self.financing_events.append(evt)
            self._record_risk_event({"type": "financing_accrual", **evt.as_dict()})
        return events

    def _maybe_roll_futures(self, bar: BarEvent) -> None:
        """
        P5.5-F: roll an open futures position when the tradable contract changes.

        The continuous back-adjusted series is only ever a signal input; the roll is
        booked as an explicit close on the front contract and open on the back
        contract, so a synthetic adjustment jump cannot become tradable PnL.
        """
        if not isinstance(self.asset, FuturesAssetSpec) or not self.exec_config.auto_roll_futures:
            return
        bar_contract = bar.contract or self.asset.active_contract
        if bar_contract == self._active_contract:
            return

        from_contract = self._active_contract
        self._active_contract = bar_contract
        pos = self.portfolio.get_position(self.symbol)
        if pos.is_flat:
            return

        front_price = float(bar.meta.get("roll_front_price", self._prev_bar_close or bar.open))
        back_price = float(bar.meta.get("roll_back_price", bar.open))
        fills, decision = execute_futures_rollover(
            self.portfolio,
            asset=self.asset,
            symbol=self.symbol,
            from_contract=from_contract,
            to_contract=bar_contract,
            timestamp=bar.timestamp,
            front_price=front_price,
            back_price=back_price,
            friction=self.friction,
            id_factory=self.id_factory,
        )
        if not fills:
            return
        self.rollover_decisions.append(decision)
        self._record_risk_event({"type": "futures_rollover", **decision.as_dict()})
        logger.info(
            "Rolled %s %s -> %s front=%.4f back=%.4f",
            self.symbol,
            from_contract,
            bar_contract,
            front_price,
            back_price,
        )

    def _attach_brackets_after_entry(self, fill: Fill) -> None:
        brackets = self._pending_brackets.pop(fill.order_id, None)
        if not brackets:
            return
        pos = self.portfolio.get_position(fill.symbol)
        if pos.is_flat:
            return
        pos.meta["stop_price"] = brackets.get("stop")
        pos.meta["target_price"] = brackets.get("target")

    def run(self, bars: list[BarEvent], signal_fn: SignalFn | None = None) -> ExecutionResult:
        if not bars:
            return ExecutionResult(self.portfolio, self._all_orders, [], [], [])

        for i, bar in enumerate(bars):
            self._clock = bar.timestamp

            # 0) Roll an open futures position before this bar can trade the new contract
            self._maybe_roll_futures(bar)

            # 1) Working orders at/after activation against this bar
            new_fills = self._process_working_orders(bar)
            for f in new_fills:
                self._attach_brackets_after_entry(f)

            # 2) Protective brackets
            self._manage_open_brackets(bar)

            # 2b) CFD overnight financing before marking, so equity reflects the debit
            self._accrue_financing(bar)
            self._prev_mark_ts = bar.bar_end

            # 3) Mark to market at bar close path (use close)
            self.portfolio.mark_to_market(bar.bar_end, {bar.symbol: bar.close})
            self._prev_bar_close = float(bar.close)

            # 3b) Prop risk FSM evaluation
            if self.risk is not None:
                prev_state = self.risk.state
                self.risk.on_mark(self.portfolio, bar.bar_end)
                if self.risk.should_cancel_pending():
                    self._cancel_all_working(bar.bar_end, reason="hard_breach_cancel")
                if self.risk.should_flatten_positions() and self.risk.state == RiskState.FLATTEN:
                    self._submit_flatten_orders(bar)
                if prev_state != self.risk.state:
                    self._record_risk_event(
                        {
                            "type": "state_transition",
                            "event": self.risk.events[-1].as_dict() if self.risk.events else {},
                        }
                    )
                self.risk.complete_flatten(bar.bar_end, self.portfolio)

            # 4) Signals at bar close — executable only from next bar open
            if signal_fn is not None and i + 1 < len(bars):
                next_open = bars[i + 1].timestamp
                raw = signal_fn(bar, self.portfolio, i)
                signals: list[SignalEvent] = []
                if raw is None:
                    signals = []
                elif isinstance(raw, list):
                    signals = raw
                else:
                    signals = [raw]
                for sig in signals:
                    # Enforce: decision at/after availability; fill not before next open
                    if sig.timing.decision_timestamp < sig.timing.availability_timestamp:
                        self._rejected_signals.append(
                            {"reason": "decision_before_availability", "signal": sig.signal_id}
                        )
                        continue
                    if self.risk is not None and self.risk.state in {
                        RiskState.HALTED,
                        RiskState.MANUAL_LOCK,
                    }:
                        self._rejected_signals.append(
                            {
                                "reason": f"risk_{self.risk.state.value.lower()}",
                                "signal": sig.signal_id,
                            }
                        )
                        continue
                    # Decision should be at this bar's close for close-based signals
                    self.submit_signal(sig, next_bar_open=next_open)
            elif signal_fn is not None and i + 1 >= len(bars):
                # Last bar: entry signals cannot fill (no next bar). Record and
                # defer residual flatten to _force_flat_open_positions below.
                raw = signal_fn(bar, self.portfolio, i)
                if raw is not None:
                    self._rejected_signals.append(
                        {"reason": "no_next_bar_for_execution", "bar": str(bar.timestamp)}
                    )

        if bars and self.exec_config.flatten_at_window_end:
            before_fills = len(self.portfolio.fills)
            self._force_flat_open_positions(bars[-1], reason="window_end_residual_flatten")
            forced = len(self.portfolio.fills) - before_fills
            if forced > 0:
                self._rejected_signals.append(
                    {"reason": "forced_window_closes", "count": int(forced)}
                )

        return ExecutionResult(
            portfolio=self.portfolio,
            orders=list(self._all_orders),
            fills=list(self.portfolio.fills),
            rejected_signals=list(self._rejected_signals),
            risk_events=list(self._risk_events),
            financing_events=list(self.financing_events),
            rollover_decisions=list(self.rollover_decisions),
        )

    def _force_flat_open_positions(self, bar: BarEvent, *, reason: str) -> None:
        """Close any residual open lots at window end (research/WFO integrity)."""
        for sym, pos in list(self.portfolio.positions.items()):
            if pos.is_flat:
                continue
            sig = SignalEvent(
                timing=InformationTiming(
                    source_timestamp=bar.timestamp,
                    availability_timestamp=bar.bar_end,
                    decision_timestamp=bar.bar_end,
                ),
                symbol=sym,
                side="FLAT",
                quantity=abs(float(pos.quantity)),
                signal_id=self.id_factory.next_signal_id(
                    decision_timestamp=str(bar.bar_end), symbol=sym, side="FLAT"
                ),
                reason=reason,
                meta={"window_end_flatten": True},
            )
            # Activate on this bar so the market flatten can fill at close.
            order = self._create_order_from_signal(
                sig,
                next_bar_open=bar.timestamp,
                skip_risk=False,
            )
            if order is None:
                continue
            fill = self._try_fill_market(order, bar)
            if fill is not None:
                self._attach_brackets_after_entry(fill)
            if order in self._working:
                self._working = [o for o in self._working if o.order_id != order.order_id]


def bars_from_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    contract: str,
    freq: str = "5min",
) -> list[BarEvent]:
    """Build BarEvent list from a validated OHLCV frame with timezone-aware timestamps."""
    work = df.copy()
    if "timestamp" in work.columns:
        work = work.set_index("timestamp")
    if work.index.tz is None:
        raise ValueError("bars_from_frame requires timezone-aware timestamps")
    delta = pd.tseries.frequencies.to_offset(freq)
    if delta is None:
        raise ValueError(f"bad freq {freq}")
    events: list[BarEvent] = []
    for ts, row in work.iterrows():
        bar_end = pd.Timestamp(ts) + delta
        events.append(
            BarEvent(
                timestamp=pd.Timestamp(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                symbol=symbol,
                contract=str(row["contract"]) if "contract" in work.columns else contract,
                bar_end=bar_end,
                atr=float(row["atr"]) if "atr" in work.columns and pd.notna(row["atr"]) else None,
                is_stale=bool(row["is_stale"]) if "is_stale" in work.columns else False,
                bid=float(row["bid"]) if "bid" in work.columns and pd.notna(row.get("bid")) else None,
                ask=float(row["ask"]) if "ask" in work.columns and pd.notna(row.get("ask")) else None,
                is_session_open=bool(row["is_session_open"])
                if "is_session_open" in work.columns
                else False,
                is_session_close=bool(row["is_session_close"])
                if "is_session_close" in work.columns
                else False,
            )
        )
    return events
