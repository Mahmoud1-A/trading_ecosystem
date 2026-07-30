"""Stress tests for discovery finalists."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from discovery.candidate import StrategyCandidate
from discovery.evaluator import BacktestBackend, EvaluationRecord, SyntheticOOSBackend
from discovery.fitness import FoldOOSMetrics, RobustFitness
from discovery.search_budget import BudgetCounters, SearchBudget


STRESS_SCENARIOS: tuple[str, ...] = (
    "base_costs",
    "costs_2x",
    "costs_4x",
    "delayed_execution",
    "wider_spread",
    "worse_slippage",
    "conservative_intrabar",
    "reduced_participation",
    "removed_best_trades",
    "removed_best_day",
    "parameter_perturbation",
    "alt_wfo_alignment",
    "alt_start_dates",
    "regime_exclusion",
    "symbol_exclusion",
)


@dataclass
class StressResult:
    scenario: str
    fitness: float
    median_expectancy: float
    max_drawdown: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "fitness": self.fitness,
            "median_expectancy": self.median_expectancy,
            "max_drawdown": self.max_drawdown,
            "passed": self.passed,
            "details": dict(self.details),
        }


def _default_backend_factory(scenario: str) -> BacktestBackend:
    return SyntheticOOSBackend(seed_salt=abs(hash(scenario)) % 10_000)


@dataclass
class StressTester:
    budget: SearchBudget
    counters: BudgetCounters
    fitness_model: RobustFitness = field(default_factory=RobustFitness)
    backend_factory: Callable[[str], BacktestBackend] = field(default=_default_backend_factory)
    min_fitness_ratio: float = 0.35  # vs base

    def run(
        self,
        candidate: StrategyCandidate,
        *,
        base_fitness: float,
        scenarios: tuple[str, ...] | None = None,
    ) -> list[StressResult]:
        chosen = scenarios or STRESS_SCENARIOS
        results: list[StressResult] = []
        for scenario in chosen:
            if self.counters.stress >= self.budget.max_stress_evaluations:
                break
            backend = self.backend_factory(scenario)
            folds, _ = backend.evaluate(candidate)
            # Apply scenario-specific haircuts to OOS folds
            folds = [_apply_scenario(f, scenario) for f in folds]
            fit = self.fitness_model.score(folds, complexity=candidate.complexity_score)
            ratio = fit.fitness / base_fitness if abs(base_fitness) > 1e-12 else 0.0
            passed = fit.fitness >= base_fitness * self.min_fitness_ratio or (
                base_fitness <= 0 and fit.fitness >= base_fitness
            )
            results.append(
                StressResult(
                    scenario=scenario,
                    fitness=fit.fitness,
                    median_expectancy=float(np_median([f.expectancy for f in folds])),
                    max_drawdown=float(min(f.max_drawdown for f in folds)),
                    passed=passed,
                    details={"fitness_ratio_vs_base": ratio},
                )
            )
            self.counters.stress += 1
        return results


def np_median(xs: list[float]) -> float:
    import numpy as np

    return float(np.median(xs)) if xs else 0.0


def _apply_scenario(fold: FoldOOSMetrics, scenario: str) -> FoldOOSMetrics:
    exp = fold.expectancy
    sharpe = fold.sharpe
    dd = fold.max_drawdown
    pf = fold.profit_factor
    if scenario in {"costs_2x", "wider_spread", "worse_slippage"}:
        exp *= 0.7
        sharpe *= 0.75
        dd *= 1.15
    elif scenario == "costs_4x":
        exp *= 0.4
        sharpe *= 0.5
        dd *= 1.3
        pf = max(0.5, pf * 0.6)
    elif scenario in {"delayed_execution", "conservative_intrabar", "reduced_participation"}:
        exp *= 0.85
        sharpe *= 0.9
    elif scenario in {"removed_best_trades", "removed_best_day"}:
        exp *= 0.6
        sharpe *= 0.65
    elif scenario in {"regime_exclusion", "symbol_exclusion"}:
        exp *= 0.8
        sharpe *= 0.85
    elif scenario in {"parameter_perturbation", "alt_wfo_alignment", "alt_start_dates"}:
        exp *= 0.9
        sharpe *= 0.92
    # base_costs: unchanged
    return FoldOOSMetrics(
        fold_id=fold.fold_id,
        expectancy=exp,
        sharpe=sharpe,
        profit_factor=pf,
        calmar=fold.calmar * (exp / fold.expectancy if abs(fold.expectancy) > 1e-12 else 1.0),
        max_drawdown=dd,
        drawdown_duration=fold.drawdown_duration,
        worst_day=fold.worst_day,
        turnover=fold.turnover,
        prop_breach_prob=fold.prop_breach_prob,
        regime_entropy=fold.regime_entropy,
        n_trades=fold.n_trades,
    )


def attach_stress(record: EvaluationRecord, results: list[StressResult]) -> EvaluationRecord:
    record.stress_results = {r.scenario: r.as_dict() for r in results}
    return record
