from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, load_yaml
from trading_ecosystem.common.contracts import AccountState, Fill, OrderIntent, OrderSide, RiskDecision
from trading_ecosystem.data_pipeline.news_calendar import NewsCalendar

logger = logging.getLogger(__name__)


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    REDUCE_ONLY = "REDUCE_ONLY"
    LIQUIDATE = "LIQUIDATE"
    HALTED = "HALTED"
    MANUAL_LOCK = "MANUAL_LOCK"


class MasterRiskManager:
    """Central risk layer with explicit state machine + soft/hard limits."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        news_calendar: NewsCalendar | None = None,
    ) -> None:
        path = Path(rules_path) if rules_path else CONFIG_DIR / "prop_rules.yaml"
        self.rules: dict[str, Any] = load_yaml(path)
        self.mode = str(self.rules.get("mode", "HARD")).upper()
        before = int(self.rules.get("news_block_minutes_before", 30))
        after = int(self.rules.get("news_block_minutes_after", 30))
        self.news = news_calendar or NewsCalendar(
            block_before_minutes=before,
            block_after_minutes=after,
        )
        self.audit: list[RiskDecision] = []
        self.state = RiskState.NORMAL
        self.manual_lock = False
        self.pending_notional_reserve = 0.0
        self.cluster_exposure: dict[str, float] = {}
        self._slippage_reserve_bps = float(self.rules.get("slippage_reserve_bps", 5.0))
        self._soft_daily_frac = float(self.rules.get("soft_daily_dd_frac", 0.70))
        self._soft_total_frac = float(self.rules.get("soft_total_dd_frac", 0.70))

    def set_manual_lock(self, locked: bool = True) -> None:
        self.manual_lock = bool(locked)
        self.state = RiskState.MANUAL_LOCK if locked else RiskState.NORMAL

    def cluster_id(self, intent: OrderIntent) -> str:
        meta = intent.meta or {}
        return str(meta.get("cluster_id") or meta.get("family") or intent.strategy_id.split("_")[0])

    def _transition_from_dd(self, account: AccountState) -> RiskState:
        if self.manual_lock:
            return RiskState.MANUAL_LOCK
        max_daily = float(self.rules["max_daily_drawdown_pct"])
        max_total = float(self.rules["max_total_drawdown_pct"])
        daily_dd = account.daily_drawdown_pct()
        total_dd = account.total_drawdown_pct()
        if account.halted or total_dd >= max_total:
            return RiskState.HALTED
        if account.daily_halted or daily_dd >= max_daily:
            return RiskState.LIQUIDATE if bool(self.rules.get("flatten_on_daily_breach", True)) else RiskState.REDUCE_ONLY
        if daily_dd >= max_daily * self._soft_daily_frac or total_dd >= max_total * self._soft_total_frac:
            return RiskState.CAUTION
        return RiskState.NORMAL

    def on_mark(self, account: AccountState) -> RiskState:
        self.state = self._transition_from_dd(account)
        return self.state

    def on_fill(self, account: AccountState, fill: Fill) -> None:
        # Release approximate reserve after fill
        notional = abs(float(fill.qty) * float(fill.price))
        self.pending_notional_reserve = max(0.0, self.pending_notional_reserve - notional)
        self.state = self._transition_from_dd(account)

    def end_of_day(self, account: AccountState) -> list[str]:
        actions = self.update_halt_flags(account)
        self.state = self._transition_from_dd(account)
        self.pending_notional_reserve = 0.0
        return actions

    def evaluate(
        self,
        intent: OrderIntent,
        account: AccountState,
        mark_price: float,
    ) -> RiskDecision:
        ts = intent.ts if intent.ts.tzinfo else intent.ts.replace(tzinfo=timezone.utc)
        self.state = self._transition_from_dd(account)
        decision = self._evaluate(intent, account, mark_price, ts)
        self.audit.append(decision)
        level = logging.INFO if decision.approved else logging.WARNING
        logger.log(
            level,
            "risk %s state=%s intent=%s strategy=%s symbol=%s reason=%s",
            "APPROVE" if decision.approved else "REJECT",
            self.state.value,
            intent.intent_id,
            intent.strategy_id,
            intent.symbol,
            decision.reason,
        )
        return decision

    def _evaluate(
        self,
        intent: OrderIntent,
        account: AccountState,
        mark_price: float,
        ts: datetime,
    ) -> RiskDecision:
        def reject(reason: str) -> RiskDecision:
            return RiskDecision(
                approved=False,
                intent_id=intent.intent_id,
                reason=reason,
                mode=self.mode,
                ts=ts,
            )

        def approve(reason: str, qty: float | None = None) -> RiskDecision:
            return RiskDecision(
                approved=True,
                intent_id=intent.intent_id,
                reason=reason,
                mode=self.mode,
                adjusted_qty=qty,
                ts=ts,
            )

        if self.state == RiskState.MANUAL_LOCK:
            return reject("manual_lock")
        if self.state == RiskState.HALTED or account.halted:
            return reject("account_halted_total_dd")
        if self.state in {RiskState.REDUCE_ONLY, RiskState.LIQUIDATE} and not intent.reduce_only:
            return reject(f"state_{self.state.value.lower()}_entries_blocked")
        if account.daily_halted and not intent.reduce_only:
            return reject("daily_halt_active")

        max_daily = float(self.rules["max_daily_drawdown_pct"])
        max_total = float(self.rules["max_total_drawdown_pct"])
        daily_dd = account.daily_drawdown_pct()
        total_dd = account.total_drawdown_pct()

        if total_dd >= max_total:
            account.halted = True
            self.state = RiskState.HALTED
            return reject(f"total_dd_breach:{total_dd:.2f}>={max_total}")
        if daily_dd >= max_daily and not intent.reduce_only:
            account.daily_halted = True
            self.state = RiskState.LIQUIDATE
            return reject(f"daily_dd_breach:{daily_dd:.2f}>={max_daily}")

        # Soft proximity: CAUTION state (size cut) — do not invent a second hard reject gate
        if not intent.reduce_only:
            event = self.news.blocking_event(ts)
            if event is not None:
                return reject(f"news_window:{event['name']}")

        if intent.qty <= 0:
            return reject("non_positive_qty")
        if mark_price <= 0:
            return reject("invalid_mark_price")

        adjusted_qty = intent.qty
        if not intent.reduce_only and intent.side == OrderSide.BUY:
            risk_pct = float(self.rules["risk_per_trade_pct"])
            max_symbol_pct = float(self.rules["max_symbol_exposure_pct"])
            max_gross_pct = float(self.rules["max_gross_exposure_pct"])
            max_net_pct = float(self.rules.get("max_net_exposure_pct", max_gross_pct))
            max_positions = int(self.rules["max_concurrent_positions"])
            max_cluster_pct = float(self.rules.get("max_cluster_exposure_pct", 40.0))

            open_positions = [p for p in account.positions.values() if not p.is_flat]
            key = account.position_key(intent.strategy_id, intent.symbol)
            already_open = key in account.positions and not account.positions[key].is_flat
            if not already_open and len(open_positions) >= max_positions:
                return reject("max_concurrent_positions")

            risk_budget = account.equity * (risk_pct / 100.0)
            stop = intent.stop_price
            if stop is not None and stop > 0 and stop < mark_price:
                per_unit_risk = mark_price - stop
                if per_unit_risk > 0:
                    adjusted_qty = min(adjusted_qty, risk_budget / per_unit_risk)

            max_symbol_notional = account.equity * (max_symbol_pct / 100.0)
            existing_symbol = sum(
                abs(p.qty * p.avg_price) for p in account.positions.values() if p.symbol == intent.symbol
            )
            room_symbol = max(0.0, max_symbol_notional - existing_symbol)
            adjusted_qty = min(adjusted_qty, room_symbol / mark_price if mark_price else 0.0)

            # Gross + pending reserve + slippage buffer
            slip_buf = 1.0 + self._slippage_reserve_bps / 10_000.0
            max_gross = account.equity * (max_gross_pct / 100.0)
            room_gross = max(0.0, max_gross - account.gross_exposure() - self.pending_notional_reserve)
            adjusted_qty = min(adjusted_qty, (room_gross / slip_buf) / mark_price if mark_price else 0.0)

            # Net exposure (long-only book ≈ gross)
            max_net = account.equity * (max_net_pct / 100.0)
            room_net = max(0.0, max_net - account.gross_exposure() - self.pending_notional_reserve)
            adjusted_qty = min(adjusted_qty, room_net / mark_price if mark_price else 0.0)

            # Cluster cap
            cid = self.cluster_id(intent)
            cluster_used = float(self.cluster_exposure.get(cid, 0.0))
            for p in account.positions.values():
                if p.is_flat:
                    continue
                # Approximate cluster by strategy prefix
                if str(p.strategy_id).split("_")[0] == cid or cid in str(p.strategy_id):
                    cluster_used += abs(p.qty * p.avg_price)
            max_cluster = account.equity * (max_cluster_pct / 100.0)
            room_cluster = max(0.0, max_cluster - cluster_used)
            adjusted_qty = min(adjusted_qty, room_cluster / mark_price if mark_price else 0.0)

            raw_pos_pct = self.rules.get("max_position_exposure_pct", "auto")
            if raw_pos_pct is None or str(raw_pos_pct).strip().lower() in {"", "auto"}:
                pos_pct = max_gross_pct / max(1, max_positions)
            else:
                pos_pct = float(raw_pos_pct)
            max_pos_notional = account.equity * (pos_pct / 100.0)
            existing_this = 0.0
            if already_open:
                existing_this = abs(account.positions[key].qty * account.positions[key].avg_price)
            room_pos = max(0.0, max_pos_notional - existing_this)
            adjusted_qty = min(adjusted_qty, room_pos / mark_price if mark_price else 0.0)

            if account.buying_power is not None:
                bp = float(account.buying_power)
                if bp <= 0:
                    return reject("insufficient_buying_power")
                adjusted_qty = min(adjusted_qty, (bp * 0.98) / mark_price)

            # CAUTION: halve new risk
            if self.state == RiskState.CAUTION:
                adjusted_qty *= 0.5

            adjusted_qty = float(int(adjusted_qty))
            if adjusted_qty <= 0:
                return reject("qty_clipped_to_zero_by_exposure_or_risk")

            # Reserve pending notional until fill/EOD
            self.pending_notional_reserve += adjusted_qty * mark_price * slip_buf

        if self.mode != "HARD":
            return approve("soft_pass", adjusted_qty)

        return approve("ok", adjusted_qty if adjusted_qty != intent.qty else None)

    def update_halt_flags(self, account: AccountState) -> list[str]:
        actions: list[str] = []
        max_daily = float(self.rules["max_daily_drawdown_pct"])
        max_total = float(self.rules["max_total_drawdown_pct"])
        if account.total_drawdown_pct() >= max_total:
            account.halted = True
            self.state = RiskState.HALTED
            actions.append("halt_total")
        if account.daily_drawdown_pct() >= max_daily:
            account.daily_halted = True
            self.state = RiskState.LIQUIDATE
            actions.append("halt_daily")
        elif not account.halted and not self.manual_lock:
            self.state = self._transition_from_dd(account)
        return actions

    def liquidation_plan(self, account: AccountState) -> list[dict[str, Any]]:
        """Order exits by liquidity proxy (larger notional first) when LIQUIDATE."""
        rows = []
        for p in account.positions.values():
            if p.is_flat:
                continue
            notional = abs(p.qty * p.avg_price)
            rows.append(
                {
                    "strategy_id": p.strategy_id,
                    "symbol": p.symbol,
                    "qty": abs(p.qty),
                    "notional": notional,
                    "priority": notional,
                }
            )
        rows.sort(key=lambda r: r["priority"], reverse=True)
        return rows
