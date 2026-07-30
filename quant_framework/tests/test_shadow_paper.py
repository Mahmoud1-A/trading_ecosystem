"""
Phase 9 acceptance tests — Shadow and paper trading.

Prove: shadow cannot send orders, paper rejects live credentials, duplicate
prevention, restart recovery, reconciliation, stale halt, broker disconnect,
kill switch, Vault-gated paper promotion, legacy Sim cannot promote.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from brokers import (
    BrokerMode,
    LegacyAlpacaAdapter,
    LiveCredentialsError,
    OrderSide,
    PaperBrokerAdapter,
    SimulatedBroker,
    paper_adapter_capabilities,
)
from monitoring import PaperPromotionError, PaperPromotionGate, reject_legacy_sim_promotion
from portfolio import (
    ConstructionMethod,
    PortfolioConstraints,
    PortfolioPipelineStage,
    build_portfolio_version,
    freeze_portfolio,
)
from runtime import (
    MarketBar,
    RecoveryManager,
    RuntimeHalt,
    RuntimeState,
    Signal,
    StateStore,
    StaleDataError,
    TradingRuntime,
)


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "runtime_state.json")


def _shadow_runtime(tmp_path: Path) -> TradingRuntime:
    broker = SimulatedBroker(mode=BrokerMode.SHADOW)
    broker.set_price("ES", 5000.0)
    state = RuntimeState(run_id="shadow_run", mode="SHADOW", cash=100_000.0)
    store = _store(tmp_path)
    store.save(state)
    return TradingRuntime(broker=broker, state=state, store=store)


def _paper_runtime(tmp_path: Path) -> TradingRuntime:
    broker = SimulatedBroker(mode=BrokerMode.PAPER)
    broker.set_price("ES", 5000.0)
    state = RuntimeState(run_id="paper_run", mode="PAPER", cash=100_000.0)
    store = _store(tmp_path)
    store.save(state)
    return TradingRuntime(broker=broker, state=state, store=store)


def _fresh_bar(symbol: str = "ES", age_s: float = 1.0) -> MarketBar:
    ts = datetime.now(tz=timezone.utc) - timedelta(seconds=age_s)
    return MarketBar(symbol=symbol, timestamp=ts, bid=4999.0, ask=5001.0, last=5000.0)


class TestShadowAndPaperSafety:
    def test_shadow_mode_cannot_send_orders(self, tmp_path) -> None:
        rt = _shadow_runtime(tmp_path)
        rt.on_bar(_fresh_bar())
        sig = Signal(
            signal_id="s1",
            symbol="ES",
            side=OrderSide.BUY,
            quantity=1.0,
            bar_timestamp=_fresh_bar().timestamp.isoformat(),
        )
        order = rt.process_signal(sig)
        assert order is not None
        assert order.meta.get("submitted") is False
        assert order.status.value == "THEORETICAL"
        assert rt.shadow_real_submits() == 0
        assert isinstance(rt.broker, SimulatedBroker)
        assert rt.broker.real_submit_count() == 0

    def test_paper_mode_cannot_access_live_credentials(self) -> None:
        with pytest.raises(LiveCredentialsError):
            PaperBrokerAdapter(credentials={"api_key": "X", "trading_mode": "live"})
        with pytest.raises(LiveCredentialsError):
            PaperBrokerAdapter(credentials={"base_url": "https://api.alpaca.markets", "key": "k"})
        with pytest.raises(LiveCredentialsError):
            LegacyAlpacaAdapter(credentials={"live": True, "key": "k"})
        # Paper credentials accepted
        paper = PaperBrokerAdapter(
            credentials={"trading_mode": "paper", "base_url": "https://paper-api.alpaca.markets"}
        )
        assert paper.mode is BrokerMode.PAPER


class TestIdempotencyAndRecovery:
    def test_duplicate_orders_are_prevented(self, tmp_path) -> None:
        rt = _paper_runtime(tmp_path)
        rt.on_bar(_fresh_bar())
        ts = _fresh_bar().timestamp.isoformat()
        sig = Signal("s1", "ES", OrderSide.BUY, 1.0, ts)
        o1 = rt.process_signal(sig)
        o2 = rt.process_signal(sig)  # duplicate signal suppressed
        assert o1 is not None
        assert o2 is None
        assert rt.signals.duplicate_count == 1
        # Router idempotency: same signal routed twice returns same order
        again = rt.router.route(sig)
        assert again.client_order_id == o1.client_order_id
        assert len([x for x in rt.state.client_order_ids if x == o1.client_order_id]) == 1

    def test_restart_recovery_does_not_duplicate_positions(self, tmp_path) -> None:
        rt = _paper_runtime(tmp_path)
        rt.on_bar(_fresh_bar())
        rt.process_signal(Signal("s1", "ES", OrderSide.BUY, 2.0, "2024-01-01T00:00:00+00:00"))
        qty_before = sum(abs(p.quantity) for p in rt.broker.positions())
        assert qty_before == 2.0
        # Crash + recover without re-submitting
        recovered, report = RecoveryManager(rt.store, rt.broker).recover()
        assert report.ok
        qty_after = sum(abs(p.quantity) for p in rt.broker.positions())
        assert qty_after == qty_before
        # Re-routing same signal is idempotent
        rt2 = TradingRuntime(broker=rt.broker, state=recovered, store=rt.store)
        rt2.router.route(Signal("s1", "ES", OrderSide.BUY, 2.0, "2024-01-01T00:00:00+00:00"))
        assert sum(abs(p.quantity) for p in rt.broker.positions()) == qty_before


class TestReconStaleDisconnectKill:
    def test_reconciliation_identifies_mismatches(self, tmp_path) -> None:
        rt = _paper_runtime(tmp_path)
        rt.on_bar(_fresh_bar())
        rt.process_signal(Signal("s1", "ES", OrderSide.BUY, 1.0, "t0"))
        # Corrupt local state
        rt.state.positions["ES"] = {"quantity": 99.0, "avg_price": 1.0}
        report = rt.reconcile()
        assert report.ok is False
        assert any(m.kind == "position_qty" for m in report.mismatches)

    def test_stale_data_halts_trading(self, tmp_path) -> None:
        rt = _shadow_runtime(tmp_path)
        rt.stream.max_staleness_seconds = 5.0
        with pytest.raises(RuntimeHalt, match="stale"):
            rt.on_bar(_fresh_bar(age_s=60.0))
        assert rt.safe_state is True

    def test_broker_disconnect_triggers_safe_state(self, tmp_path) -> None:
        rt = _paper_runtime(tmp_path)
        assert isinstance(rt.broker, SimulatedBroker)
        rt.broker.disconnect()
        with pytest.raises(RuntimeHalt, match="disconnect"):
            rt.check_broker()
        assert rt.safe_state is True
        assert rt.safe_reason == "broker_disconnect"

    def test_kill_switch_cancels_and_flattens(self, tmp_path) -> None:
        rt = _paper_runtime(tmp_path)
        rt.on_bar(_fresh_bar())
        rt.process_signal(Signal("s1", "ES", OrderSide.BUY, 3.0, "t0"))
        assert rt.broker.positions()
        event = rt.kill_switch.engage(reason="test")
        assert rt.kill_switch.engaged
        assert event["reason"] == "test"
        # Flattened — no residual position
        assert all(abs(p.quantity) < 1e-9 for p in rt.broker.positions()) or not rt.broker.positions()


class TestPaperPromotion:
    def test_paper_promotion_requires_vault_pass(self, tmp_path) -> None:
        version = build_portfolio_version(
            weights={"a": 1.0},
            lineage_ids={"a": "la"},
            construction_method=ConstructionMethod.GREEDY_MARGINAL.value,
            constraints=PortfolioConstraints().as_dict(),
            feature_set_version="fs1",
            cost_model_version="cost_v1",
        )
        frozen = freeze_portfolio(version)
        gate = PaperPromotionGate()
        # Frozen but not vault-passed
        result = gate.evaluate(
            frozen,
            capabilities=paper_adapter_capabilities(),
            kill_switch_tested=True,
        )
        assert result["accepted"] is False
        assert "vault_not_passed" in result["reasons"]

        passed = frozen.with_stage(PortfolioPipelineStage.VAULT_PASSED)
        ok = gate.evaluate(
            passed,
            capabilities=paper_adapter_capabilities(),
            kill_switch_tested=True,
        )
        assert ok["accepted"] is True

    def test_legacy_sim_performance_cannot_promote(self) -> None:
        adapter = LegacyAlpacaAdapter(
            credentials={"trading_mode": "paper", "base_url": "https://paper-api.alpaca.markets"},
            legacy_sim_sharpe=3.5,
        )
        with pytest.raises(PaperPromotionError, match="legacy_sim"):
            reject_legacy_sim_promotion(adapter)
        version = build_portfolio_version(
            weights={"a": 1.0},
            lineage_ids={"a": "la"},
            construction_method="equal_risk_contribution",
            constraints={},
            feature_set_version="fs1",
            cost_model_version="cost_v1",
        )
        frozen = freeze_portfolio(version).with_stage(PortfolioPipelineStage.VAULT_PASSED)
        result = PaperPromotionGate().evaluate(
            frozen,
            capabilities=adapter.capabilities,
            kill_switch_tested=True,
            legacy_sim_sharpe=adapter.legacy_sim_sharpe,
        )
        assert result["accepted"] is False
        assert "legacy_sim_performance_cannot_promote" in result["reasons"]
