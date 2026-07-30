"""
Phase 5.5-H acceptance — complete diagnostic metrics with explicit availability.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.models import Side
from engine.events import InformationTiming
from engine.fills import CostBreakdown, Fill
from engine.portfolio import Portfolio
from metrics.diagnostics import (
    MetricStatus,
    by_regime,
    by_session,
    cost_to_gross,
    daily_loss_distribution,
    exposure_time,
    monthly_consistency,
)


TZ = ZoneInfo("UTC")


class TestDailyLossDistribution:
    def test_hand_calculated_losing_days(self) -> None:
        # Daily fractional returns: +2%, -1%, -3%, +1%, -2%
        # Loss magnitudes (positive): 0.01, 0.03, 0.02
        rets = pd.Series([0.02, -0.01, -0.03, 0.01, -0.02])
        dist = daily_loss_distribution(rets)
        assert dist.status == MetricStatus.OK
        assert dist.sample_size == 5
        assert dist.n_losing_days == 3
        assert dist.mean_loss == pytest.approx((0.01 + 0.03 + 0.02) / 3, rel=1e-9)
        assert dist.worst_loss == pytest.approx(0.03, rel=1e-9)
        assert dist.assumptions

    def test_insufficient_sample(self) -> None:
        dist = daily_loss_distribution(pd.Series([0.01, -0.02]), min_observations=5)
        assert dist.status == MetricStatus.INSUFFICIENT_DATA
        assert dist.mean_loss is None


class TestMonthlyConsistency:
    def test_hand_calculated_months(self) -> None:
        idx = pd.date_range("2024-01-31", periods=4, freq="ME", tz=TZ)
        eq = pd.Series([100.0, 105.0, 103.0, 110.0], index=idx)
        # monthly returns: +5%, -1.905%, +6.796%
        m = monthly_consistency(eq)
        assert m.status == MetricStatus.OK
        assert m.n_months == 3
        assert m.positive_months == 2
        assert m.negative_months == 1
        assert m.positive_month_fraction == pytest.approx(2 / 3)
        assert m.sample_size == 3


class TestByRegimeAndSession:
    def test_missing_labels_are_explicitly_unavailable(self) -> None:
        rets = pd.Series([0.01, -0.005, 0.002], index=pd.RangeIndex(3))
        regime = by_regime(rets, None)
        session = by_session(rets, None)
        assert regime.status == MetricStatus.UNAVAILABLE_MISSING_LABELS
        assert session.status == MetricStatus.UNAVAILABLE_MISSING_LABELS
        assert regime.groups == {}
        assert session.groups == {}

    def test_labelled_breakdown_reports_sample_size(self) -> None:
        idx = pd.RangeIndex(6)
        rets = pd.Series([0.01, 0.02, -0.01, 0.005, -0.02, 0.01], index=idx)
        labels = pd.Series(["range", "range", "trend", "trend", "range", "trend"], index=idx)
        out = by_regime(rets, labels, min_observations=2)
        assert out.status == MetricStatus.OK
        assert "range" in out.groups
        assert "trend" in out.groups
        assert out.groups["range"].sample_size == 3
        assert out.groups["trend"].sample_size == 3


class TestExposureAndCost:
    def test_exposure_time_from_portfolio_state(self) -> None:
        p = Portfolio(100_000.0)
        p.cash = 100_000.0
        t0 = pd.Timestamp("2024-01-02 10:00", tz=TZ)
        t1 = pd.Timestamp("2024-01-02 10:05", tz=TZ)
        t2 = pd.Timestamp("2024-01-02 10:10", tz=TZ)
        t3 = pd.Timestamp("2024-01-02 10:15", tz=TZ)
        timing = InformationTiming(
            source_timestamp=t0,
            availability_timestamp=t0,
            decision_timestamp=t0,
            order_submission_timestamp=t0,
            order_activation_timestamp=t0,
            fill_timestamp=t1,
        )
        fill = Fill(
            order_id="o1",
            symbol="ES",
            contract="ESH24",
            side=Side.BUY,
            quantity=1.0,
            price=100.0,
            timing=timing,
            costs=CostBreakdown(),
            fill_id="f1",
        )
        p.set_multiplier("ES", 50.0)
        p.apply_fill(fill)
        p.mark_to_market(t0, {"ES": 100.0})
        p.mark_to_market(t1, {"ES": 100.0})
        p.mark_to_market(t2, {"ES": 101.0})
        p.mark_to_market(t3, {"ES": 101.0})
        exp = exposure_time(p)
        assert exp.status == MetricStatus.OK
        assert exp.sample_size == 4
        # Flat at t0 (fill at t1); exposed at t1,t2,t3 => 75%
        assert exp.bars_in_market == 3
        assert exp.exposure_time_pct == pytest.approx(75.0)

    def test_cost_to_gross_from_attribution(self) -> None:
        trades = pd.DataFrame(
            {
                "gross_pnl": [100.0, -40.0, 60.0],
                "commission": [5.0, 5.0, 5.0],
                "spread_cost": [2.0, 2.0, 2.0],
                "slippage_cost": [1.0, 1.0, 1.0],
                "financing_cost": [0.0, 0.0, 0.0],
                "rollover_cost": [0.0, 0.0, 0.0],
            }
        )
        # total cost = 24; gross profit = 160; ratio = 24/160 = 0.15
        ratio = cost_to_gross(trades)
        assert ratio.status == MetricStatus.OK
        assert ratio.sample_size == 3
        assert ratio.total_cost == pytest.approx(24.0)
        assert ratio.cost_to_gross_profit == pytest.approx(24.0 / 160.0)
