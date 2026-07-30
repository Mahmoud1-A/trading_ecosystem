"""
Phase 10 acceptance tests — Controlled live-readiness.

Prove: no live orders without authorization, paper/test credentials blocked,
capital limits enforced, independent kill switches, broker-failure safe state,
state preservation, rollback disables new risk, audited transitions,
unapproved commits cannot start production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brokers import BrokerMode, OrderSide, SimulatedBroker
from live import (
    ApprovalRegistry,
    CapitalRamp,
    CapitalStage,
    Environment,
    IncidentResponse,
    LiveEnableError,
    LiveOrderGate,
    RampError,
    RollbackController,
    build_live_config,
    evaluate_readiness,
)
from runtime import KillSwitch, RuntimeState, StateStore


def _live_config(*, commit_approved: bool = True, env: Environment = Environment.LIVE) -> object:
    return build_live_config(
        environment=env,
        git_commit="abc123approved",
        commit_approved=commit_approved,
        credential_scope="live" if env is Environment.LIVE else "paper",
        broker_risk_limits={"max_notional": 10_000.0},
        local_risk_limits={"max_notional": 5_000.0},
    )


def _ready_stack(
    tmp_path: Path,
    *,
    commit_approved: bool = True,
    stage: CapitalStage = CapitalStage.CANARY_LIVE,
):
    cfg = _live_config(commit_approved=commit_approved)
    approvals = ApprovalRegistry()
    approvals.grant(
        approver="human.operator",
        config_hash=cfg.frozen_hash,
        git_commit=cfg.git_commit,
        stage=stage.value,
    )
    # Approvals for intermediate stages if needed
    for s in CapitalStage:
        if not approvals.has_valid(
            config_hash=cfg.frozen_hash, git_commit=cfg.git_commit, stage=s.value
        ):
            approvals.grant(
                approver="human.operator",
                config_hash=cfg.frozen_hash,
                git_commit=cfg.git_commit,
                stage=s.value,
            )
    readiness = evaluate_readiness(
        cfg,
        approvals=approvals,
        vault_passed=True,
        paper_gates_passed=True,
        reconciliation_clean=True,
        kill_switch_tested=True,
        audit_log_present=True,
        stage_for_auth=stage.value,
    )
    assert readiness.complete
    ramp = CapitalRamp(config=cfg, approvals=approvals, stage=stage)
    ramp.live_orders_armed = False
    rollback = RollbackController(ramp=ramp)
    broker = SimulatedBroker(mode=BrokerMode.PAPER)
    state = RuntimeState(run_id="live_ready", mode="LIVE_READY", cash=100_000.0)
    store = StateStore(tmp_path / "state.json")
    store.save(state)
    kill = KillSwitch(broker=broker, state=state, store=store)
    incidents = IncidentResponse(
        ramp=ramp, rollback=rollback, local_kill=kill, state=state, store=store
    )
    gate = LiveOrderGate(
        config=cfg,
        ramp=ramp,
        approvals=approvals,
        readiness=readiness,
        rollback=rollback,
        incidents=incidents,
    )
    return cfg, ramp, approvals, readiness, rollback, incidents, gate, state, store


class TestLiveAuthorization:
    def test_live_orders_impossible_without_manual_authorization(self, tmp_path) -> None:
        cfg, ramp, approvals, readiness, rollback, incidents, gate, _, _ = _ready_stack(tmp_path)
        # Not armed yet
        with pytest.raises(LiveEnableError, match="manual authorization|without manual"):
            gate.assert_live_order_allowed(notional=100.0, credentials_scope="live")
        ramp.arm_live_orders(readiness=readiness, manual_stage=CapitalStage.CANARY_LIVE.value)
        # Armed — gate allows evaluation but still does not broker-submit live
        result = gate.try_submit_live(notional=100.0, credentials_scope="live")
        assert result["accepted_by_gate"] is True
        assert result["live_broker_submit"] is False

    def test_test_and_paper_credentials_cannot_send_live_orders(self, tmp_path) -> None:
        # Test environment
        cfg = build_live_config(
            environment=Environment.TEST,
            git_commit="c1",
            commit_approved=True,
            credential_scope="live",
        )
        approvals = ApprovalRegistry()
        ramp = CapitalRamp(config=cfg, approvals=approvals, stage=CapitalStage.CANARY_LIVE)
        ramp.live_orders_armed = True
        readiness = evaluate_readiness(
            cfg,
            approvals=approvals,
            vault_passed=True,
            paper_gates_passed=True,
            reconciliation_clean=True,
            kill_switch_tested=True,
            audit_log_present=True,
        )
        gate = LiveOrderGate(
            config=cfg,
            ramp=ramp,
            approvals=approvals,
            readiness=readiness,
            rollback=RollbackController(ramp),
            incidents=IncidentResponse(ramp=ramp, rollback=RollbackController(ramp)),
        )
        with pytest.raises(LiveEnableError, match="test"):
            gate.assert_live_order_allowed(notional=10.0, credentials_scope="live")

        # Paper credentials on live env
        _, ramp2, _, readiness2, rollback2, incidents2, gate2, _, _ = _ready_stack(tmp_path / "p")
        ramp2.arm_live_orders(readiness=readiness2, manual_stage=CapitalStage.CANARY_LIVE.value)
        with pytest.raises(LiveEnableError, match="paper|credentials"):
            gate2.assert_live_order_allowed(notional=10.0, credentials_scope="paper")


class TestCapitalAndKill:
    def test_capital_limits_cannot_be_bypassed(self, tmp_path) -> None:
        _, ramp, _, readiness, _, _, gate, _, _ = _ready_stack(tmp_path)
        ramp.arm_live_orders(readiness=readiness, manual_stage=CapitalStage.CANARY_LIVE.value)
        # CANARY fraction 0.01 of 100_000 = 1000
        assert ramp.capital_limit() == 1000.0
        with pytest.raises(LiveEnableError, match="capital limit|risk limit"):
            gate.assert_live_order_allowed(notional=50_000.0, credentials_scope="live")
        with pytest.raises(RampError, match="capital limit"):
            ramp.assert_within_capital(50_000.0)

    def test_kill_switches_work_independently(self, tmp_path) -> None:
        _, ramp, _, readiness, rollback, incidents, gate, _, _ = _ready_stack(tmp_path)
        ramp.arm_live_orders(readiness=readiness, manual_stage=CapitalStage.CANARY_LIVE.value)
        # Remote kill alone blocks
        incidents.engage_remote_kill(reason="ops")
        with pytest.raises(LiveEnableError, match="safe state|kill"):
            gate.assert_live_order_allowed(notional=10.0, credentials_scope="live")

        # Fresh stack — local kill alone blocks
        _, ramp2, _, readiness2, _, incidents2, gate2, _, _ = _ready_stack(tmp_path / "k2")
        ramp2.arm_live_orders(readiness=readiness2, manual_stage=CapitalStage.CANARY_LIVE.value)
        incidents2.engage_local_kill(reason="local")
        with pytest.raises(LiveEnableError):
            gate2.assert_live_order_allowed(notional=10.0, credentials_scope="live")


class TestBrokerFailureRecoveryRollback:
    def test_broker_failure_triggers_safe_state(self, tmp_path) -> None:
        _, ramp, _, _, _, incidents, gate, _, _ = _ready_stack(tmp_path)
        ramp.live_orders_armed = True
        incidents.on_broker_failure("disconnect")
        assert incidents.safe_state is True
        assert ramp.live_orders_armed is False
        with pytest.raises(LiveEnableError):
            gate.assert_live_order_allowed(notional=10.0, credentials_scope="live")

    def test_restart_recovery_preserves_exact_state(self, tmp_path) -> None:
        _, _, _, _, _, incidents, _, state, store = _ready_stack(tmp_path)
        state.client_order_ids = ["coid_a", "coid_b"]
        state.positions = {"ES": {"quantity": 2.0, "avg_price": 5000.0}}
        state.cash = 88_000.0
        store.save(state)
        preserved = incidents.preserve_state_for_recovery()
        assert preserved["preserved"] is True
        assert preserved["state"]["client_order_ids"] == ["coid_a", "coid_b"]
        assert preserved["state"]["positions"]["ES"]["quantity"] == 2.0
        assert preserved["state"]["cash"] == 88_000.0

    def test_rollback_disables_new_risk(self, tmp_path) -> None:
        _, ramp, _, readiness, rollback, _, gate, _, _ = _ready_stack(
            tmp_path, stage=CapitalStage.SMALL_LIVE
        )
        ramp.arm_live_orders(readiness=readiness, manual_stage=CapitalStage.SMALL_LIVE.value)
        event = rollback.rollback(reason="drift", to_stage=CapitalStage.PAPER)
        assert event.new_risk_disabled is True
        assert ramp.live_orders_armed is False
        assert ramp.stage is CapitalStage.PAPER
        with pytest.raises(LiveEnableError, match="new risk"):
            rollback.assert_new_risk_allowed()
        with pytest.raises(LiveEnableError):
            gate.assert_live_order_allowed(notional=10.0, credentials_scope="live")


class TestAuditAndCommit:
    def test_every_stage_transition_is_audited(self, tmp_path) -> None:
        cfg = build_live_config(
            environment=Environment.PAPER,
            git_commit="papercommit",
            commit_approved=True,
            credential_scope="paper",
        )
        approvals = ApprovalRegistry()
        ramp = CapitalRamp(
            config=cfg,
            approvals=approvals,
            stage=CapitalStage.SHADOW,
            min_observation_hours=1.0,
        )
        ramp.execution_quality_ok = True
        ramp.risk_gate_ok = True
        ramp.drift_gate_ok = True
        ramp.rollback_ready = True
        t1 = ramp.advance(approver="alice", force_observation_met=True)
        assert t1.from_stage is CapitalStage.SHADOW
        assert t1.to_stage is CapitalStage.PAPER
        assert len(ramp.audit) == 1
        assert ramp.audit[0].as_dict()["approver"] == "alice"

        # Cannot jump paper → production; next stage CANARY requires LIVE env
        ramp.stage = CapitalStage.PAPER
        ramp.execution_quality_ok = True
        ramp.risk_gate_ok = True
        ramp.drift_gate_ok = True
        ramp.rollback_ready = True
        with pytest.raises(LiveEnableError, match="LIVE environment"):
            ramp.advance(approver="alice", force_observation_met=True)

    def test_production_mode_cannot_start_from_unapproved_commit(self, tmp_path) -> None:
        cfg, ramp, _, readiness, _, _, gate, _, _ = _ready_stack(
            tmp_path, commit_approved=False, stage=CapitalStage.PRODUCTION
        )
        with pytest.raises(LiveEnableError, match="unapproved commit"):
            ramp.arm_live_orders(readiness=readiness, manual_stage=CapitalStage.PRODUCTION.value)
        ramp.live_orders_armed = True  # even if falsely armed
        with pytest.raises(LiveEnableError, match="unapproved commit"):
            gate.assert_live_order_allowed(notional=10.0, credentials_scope="live")
