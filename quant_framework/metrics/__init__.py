"""Metrics package — Phase 4 core suite plus Phase 5.5 diagnostics."""

from metrics.diagnostics import (
    CostToGross,
    DailyLossDistribution,
    ExposureTime,
    GroupedBreakdown,
    GroupStat,
    MetricStatus,
    MonthlyConsistency,
    by_regime,
    by_session,
    cost_to_gross,
    daily_loss_distribution,
    exposure_time,
    monthly_consistency,
)
from metrics.drawdown import drawdown_duration_bars, equity_drawdown, max_drawdown_pct
from metrics.dsr import (
    DSRResult,
    DSRStatus,
    compute_deflated_sharpe,
    deflated_sharpe_from_scores,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from metrics.performance import PerformanceMetrics, compute_metrics
from metrics.pbo import (
    PBOResult,
    PBOStatus,
    compute_pbo,
    pbo_from_is_oos,
    probability_backtest_overfitting,
)
from metrics.prop_breach import PropBreachEstimate, estimate_prop_breach_probability
from metrics.trial_population import (
    IndependenceEstimate,
    TopNOnlyInputError,
    TrialPopulation,
    TrialSelection,
    build_trial_population,
    estimate_effective_trials,
)

__all__ = [
    "CostToGross",
    "DSRResult",
    "DSRStatus",
    "DailyLossDistribution",
    "ExposureTime",
    "GroupStat",
    "GroupedBreakdown",
    "IndependenceEstimate",
    "MetricStatus",
    "MonthlyConsistency",
    "PBOResult",
    "PBOStatus",
    "PerformanceMetrics",
    "PropBreachEstimate",
    "TopNOnlyInputError",
    "TrialPopulation",
    "TrialSelection",
    "build_trial_population",
    "by_regime",
    "by_session",
    "compute_deflated_sharpe",
    "compute_metrics",
    "compute_pbo",
    "cost_to_gross",
    "daily_loss_distribution",
    "deflated_sharpe_from_scores",
    "deflated_sharpe_ratio",
    "drawdown_duration_bars",
    "equity_drawdown",
    "estimate_effective_trials",
    "estimate_prop_breach_probability",
    "expected_max_sharpe",
    "exposure_time",
    "max_drawdown_pct",
    "monthly_consistency",
    "pbo_from_is_oos",
    "probability_backtest_overfitting",
]
