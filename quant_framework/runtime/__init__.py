"""Shadow and paper trading runtime (Phase 9) — research only, not live."""

from runtime.clock import RuntimeClock
from runtime.kill_switch import KillSwitch
from runtime.market_stream import MarketBar, MarketStream, StaleDataError
from runtime.order_router import OrderRouter, OrderRouterError
from runtime.reconciliation import ReconMismatch, ReconciliationReport, Reconciler
from runtime.recovery import RecoveryManager
from runtime.scheduler import Scheduler
from runtime.signal_service import Signal, SignalService
from runtime.state_store import RuntimeState, StateStore

# TradingRuntime defined below to avoid circular imports in type checkers
from dataclasses import dataclass, field
from typing import Any

from brokers.protocol import BrokerMode, BrokerProtocol
from brokers.simulated import SimulatedBroker
from engine.ids import DeterministicIdFactory


class RuntimeHalt(RuntimeError):
    pass


@dataclass
class TradingRuntime:
    broker: BrokerProtocol
    state: RuntimeState
    store: StateStore
    clock: RuntimeClock = field(default_factory=RuntimeClock)
    stream: MarketStream = field(default_factory=MarketStream)
    signals: SignalService = field(default_factory=SignalService)
    scheduler: Scheduler = field(default_factory=Scheduler)
    safe_state: bool = False
    safe_reason: str | None = None
    _router: OrderRouter | None = None
    _kill: KillSwitch | None = None

    def __post_init__(self) -> None:
        factory = DeterministicIdFactory(run_id=self.state.run_id, candidate_id="runtime")
        self._router = OrderRouter(
            broker=self.broker,
            id_factory=factory,
            state=self.state,
            store=self.store,
        )
        self._kill = KillSwitch(
            broker=self.broker,
            state=self.state,
            store=self.store,
            flatten_on_engage=True,
        )

    @property
    def router(self) -> OrderRouter:
        assert self._router is not None
        return self._router

    @property
    def kill_switch(self) -> KillSwitch:
        assert self._kill is not None
        return self._kill

    def enter_safe_state(self, reason: str) -> None:
        self.safe_state = True
        self.safe_reason = reason
        self.store.append_audit(self.state, {"event": "safe_state", "reason": reason})

    def check_broker(self) -> None:
        if not self.broker.connected():
            self.enter_safe_state("broker_disconnect")
            raise RuntimeHalt("broker disconnected — entered safe state")

    def on_bar(self, bar: MarketBar) -> None:
        if self.state.kill_switch_engaged or self.safe_state:
            raise RuntimeHalt(self.safe_reason or "halted")
        self.check_broker()
        try:
            self.stream.ingest(bar, now=self.clock.now())
        except StaleDataError as exc:
            self.enter_safe_state(str(exc))
            raise RuntimeHalt(str(exc)) from exc
        if isinstance(self.broker, SimulatedBroker):
            self.broker.set_price(bar.symbol, bar.mid)
        self.scheduler.dispatch(bar)

    def process_signal(self, signal: Signal) -> Any:
        if self.state.kill_switch_engaged or self.safe_state:
            raise RuntimeHalt(self.safe_reason or "halted")
        self.check_broker()
        if self.stream.halted:
            raise RuntimeHalt(self.stream.halt_reason or "stale")
        emitted = self.signals.emit(signal)
        if emitted is None:
            return None
        order = self.router.route(emitted)
        if self.broker.mode is BrokerMode.PAPER:
            self.state.positions = {
                p.symbol: {"quantity": p.quantity, "avg_price": p.avg_price}
                for p in self.broker.positions()
            }
            self.state.cash = self.broker.account().cash
            self.store.save(self.state)
        return order

    def reconcile(self) -> ReconciliationReport:
        return Reconciler(self.broker, self.state).run()

    def shadow_real_submits(self) -> int:
        if isinstance(self.broker, SimulatedBroker):
            return self.broker.real_submit_count()
        return -1


__all__ = [
    "KillSwitch",
    "MarketBar",
    "MarketStream",
    "OrderRouter",
    "OrderRouterError",
    "ReconMismatch",
    "ReconciliationReport",
    "Reconciler",
    "RecoveryManager",
    "RuntimeClock",
    "RuntimeHalt",
    "RuntimeState",
    "Scheduler",
    "Signal",
    "SignalService",
    "StaleDataError",
    "StateStore",
    "TradingRuntime",
]
