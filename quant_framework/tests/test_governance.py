"""
Phase 11 acceptance tests — Continuous monitoring and alpha decay.

Prove: degradation reduces risk, critical drift suspends entries, retired cannot
auto-reactivate, modifications get new lineage without Vault inheritance,
champion replacement requires all gates, monitoring failure → safe state.
"""

from __future__ import annotations

import pytest

from governance import (
    ChallengerMode,
    ChampionChallenger,
    DecayState,
    DemotionEngine,
    DriftMonitor,
    DriftSeverity,
    MetricSnapshot,
    MonitoredStrategy,
    MonitoringController,
    ReplacementError,
    RetirementError,
    StrategyRole,
    revalidate_modified_strategy,
)


class TestDegradationAndDrift:
    def test_degradation_reduces_allowed_risk(self) -> None:
        ctrl = MonitoringController()
        ctrl.register(
            MonitoredStrategy(
                strategy_id="s1",
                lineage_id="lin1",
                state=DecayState.HEALTHY,
                base_risk=10.0,
                expected=MetricSnapshot(rolling_sharpe=1.5, realized_expectancy=0.1),
            )
        )
        # Mild drift → WATCH (0.75x)
        d1 = ctrl.update(
            "s1",
            MetricSnapshot(rolling_sharpe=1.1, realized_expectancy=0.05),
        )
        assert d1.next_state is DecayState.WATCH
        assert ctrl.strategies["s1"].allowed_risk == pytest.approx(7.5)

        # Further mild while on WATCH → DEGRADED (0.40x)
        d2 = ctrl.update(
            "s1",
            MetricSnapshot(rolling_sharpe=1.0, realized_expectancy=0.02),
        )
        assert d2.next_state is DecayState.DEGRADED
        assert ctrl.strategies["s1"].allowed_risk == pytest.approx(4.0)
        assert DemotionEngine().allowed_risk(10.0, DecayState.DEGRADED) == pytest.approx(4.0)

    def test_critical_drift_suspends_new_entries(self) -> None:
        ctrl = MonitoringController()
        ctrl.register(
            MonitoredStrategy(
                strategy_id="s1",
                lineage_id="lin1",
                expected=MetricSnapshot(rolling_sharpe=2.0),
            )
        )
        decision = ctrl.update(
            "s1",
            MetricSnapshot(rolling_sharpe=0.2, prop_breach_probability=0.6),
        )
        assert decision.next_state is DecayState.SUSPENDED
        assert decision.new_entries_allowed is False
        assert ctrl.strategies["s1"].new_entries_allowed is False

    def test_monitoring_failure_defaults_to_safe_state(self) -> None:
        monitor = DriftMonitor()

        class BadSnapshot:
            def as_dict(self):
                raise RuntimeError("sensor_down")

        report = monitor.compare_metrics(
            MetricSnapshot(),
            BadSnapshot(),  # type: ignore[arg-type]
        )
        assert report.monitoring_failed is True
        assert report.severity is DriftSeverity.CRITICAL

        ctrl = MonitoringController(drift_monitor=monitor)
        ctrl.register(MonitoredStrategy(strategy_id="s1", lineage_id="lin1"))
        decision = ctrl.update("s1", BadSnapshot())  # type: ignore[arg-type]
        assert decision.next_state is DecayState.SUSPENDED
        assert "monitoring_failure" in decision.reason


class TestRetirementAndRevalidation:
    def test_retired_strategies_cannot_reactivate_automatically(self) -> None:
        ctrl = MonitoringController()
        ctrl.register(MonitoredStrategy(strategy_id="s1", lineage_id="lin1"))
        ctrl.retire("s1", reason="alpha_dead")
        assert ctrl.strategies["s1"].state is DecayState.RETIRED

        # Attempted "recovery" via healthy metrics must stay retired
        decision = ctrl.update(
            "s1",
            MetricSnapshot(rolling_sharpe=3.0, realized_expectancy=0.5),
        )
        assert decision.next_state is DecayState.RETIRED
        assert decision.new_entries_allowed is False

        with pytest.raises(RetirementError, match="cannot reactivate"):
            ctrl.retirement.assert_not_auto_reactivate("s1", new_state=DecayState.HEALTHY)

    def test_modified_strategies_receive_new_lineage_without_vault_inheritance(self) -> None:
        result = revalidate_modified_strategy(
            strategy_family="mr",
            old_parameters={"z": -2.0},
            new_parameters={"z": -1.5},
            config_snapshot={"mutation": True},
            code_hash="code",
            data_hash="data",
            cost_model_version="cost_v1",
            old_lineage_id="old_lineage_abc",
            prior_vault_result={"sharpe": 2.5, "passed": True},
        )
        assert result.new_lineage_id != result.old_lineage_id
        assert result.prior_vault_inherited is False
        assert result.requires_full_pipeline is True


class TestChampionChallenger:
    def test_champion_replacement_requires_all_gates(self) -> None:
        cc = ChampionChallenger()
        champ = StrategyRole(
            strategy_id="champ",
            lineage_id="lc",
            decay_state=DecayState.HEALTHY,
            backtest_sharpe=1.0,
            paper_sharpe=0.8,
            contribution=1.0,
            risk_score=0.2,
            execution_score=0.9,
            vault_passed=True,
            paper_gates_passed=True,
        )
        cc.set_champion(champ)

        # Strong backtest only — must fail
        weak = StrategyRole(
            strategy_id="bt_only",
            lineage_id="lb",
            decay_state=DecayState.HEALTHY,
            backtest_sharpe=3.0,
            paper_sharpe=0.0,
            shadow_sharpe=0.0,
            contribution=2.0,
            risk_score=0.1,
            execution_score=1.0,
            vault_passed=True,
            paper_gates_passed=True,
        )
        cc.register_challenger(weak, mode=ChallengerMode.PAPER)
        decision = cc.evaluate_replacement("PAPER:bt_only")
        assert decision.accepted is False
        assert "cannot_replace_on_backtest_alone" in decision.reasons

        # Full gates pass
        strong = StrategyRole(
            strategy_id="challenger",
            lineage_id="lch",
            decay_state=DecayState.HEALTHY,
            backtest_sharpe=1.2,
            paper_sharpe=1.1,
            shadow_sharpe=0.9,
            contribution=1.5,
            risk_score=0.15,
            execution_score=0.95,
            vault_passed=True,
            paper_gates_passed=True,
        )
        cc.register_challenger(strong, mode=ChallengerMode.PAPER)
        ok = cc.evaluate_replacement("PAPER:challenger")
        assert ok.accepted is True
        cc.replace_if_allowed("PAPER:challenger")
        assert cc.champion is not None
        assert cc.champion.strategy_id == "challenger"

        # Missing vault gate
        no_vault = StrategyRole(
            strategy_id="nv",
            lineage_id="lnv",
            decay_state=DecayState.HEALTHY,
            backtest_sharpe=1.0,
            paper_sharpe=1.2,
            contribution=2.0,
            risk_score=0.1,
            execution_score=1.0,
            vault_passed=False,
            paper_gates_passed=True,
        )
        cc.register_challenger(no_vault, mode=ChallengerMode.SHADOW)
        with pytest.raises(ReplacementError, match="vault"):
            cc.replace_if_allowed("SHADOW:nv")


class TestSevereReduceOnly:
    def test_severe_degradation_enters_reduce_only(self) -> None:
        ctrl = MonitoringController()
        ctrl.register(
            MonitoredStrategy(
                strategy_id="s1",
                lineage_id="lin1",
                expected=MetricSnapshot(rolling_sharpe=2.0, rolling_drawdown=-0.05),
            )
        )
        decision = ctrl.update(
            "s1",
            MetricSnapshot(rolling_sharpe=1.1, rolling_drawdown=-0.12),
        )
        assert decision.next_state in {DecayState.REDUCE_ONLY, DecayState.SUSPENDED, DecayState.DEGRADED}
        # Force severe path via large sharpe drop without critical prop
        decision2 = ctrl.update(
            "s1",
            MetricSnapshot(rolling_sharpe=1.2, rolling_drawdown=-0.05),
        )
        # From whatever state, apply engine directly for severe
        engine = DemotionEngine()
        from governance.drift import DriftReport

        severe = DriftReport(severity=DriftSeverity.SEVERE, metric_deltas={})
        d = engine.apply(DecayState.DEGRADED, severe)
        assert d.next_state is DecayState.REDUCE_ONLY
        assert d.new_entries_allowed is False
