"""Stress tests for discovery finalists — real event-driven by default on research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from discovery.candidate import StrategyCandidate
from discovery.evaluator import BacktestBackend, EvaluationRecord, SyntheticOOSBackend
from discovery.fitness import FoldOOSMetrics, RobustFitness
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.stable_hash import stable_int_hash
from discovery.stress_backend import (
    STRESS_BACKEND_KIND,
    SYNTHETIC_STRESS_FORBIDDEN,
    StressScenarioBackend,
    assert_not_synthetic_stress,
    make_stress_backend_factory,
)


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
    candidate_id: str
    fitness: float
    median_expectancy: float
    max_drawdown: float
    passed: bool
    failure_reason: str | None = None
    signal_source: str = "candidate_dsl_trees"
    backend_kind: str = STRESS_BACKEND_KIND
    cost_model_changes: dict[str, Any] = field(default_factory=dict)
    execution_changes: dict[str, Any] = field(default_factory=dict)
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    trade_counts: int = 0
    orders_count: int = 0
    fills_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "candidate_id": self.candidate_id,
            "signal_source": self.signal_source,
            "backend_kind": self.backend_kind,
            "cost_model_changes": dict(self.cost_model_changes),
            "execution_changes": dict(self.execution_changes),
            "fold_metrics": list(self.fold_metrics),
            "trade_counts": self.trade_counts,
            "orders_count": self.orders_count,
            "fills_count": self.fills_count,
            "fitness": self.fitness,
            "median_expectancy": self.median_expectancy,
            "max_drawdown": self.max_drawdown,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "details": dict(self.details),
        }


def _default_backend_factory(scenario: str) -> BacktestBackend:
    # Smoke / unit-test default only — research paths must inject real factory.
    return SyntheticOOSBackend(seed_salt=stable_int_hash(scenario, bits=32) % 10_000)


@dataclass
class StressTester:
    budget: SearchBudget
    counters: BudgetCounters
    fitness_model: RobustFitness = field(default_factory=RobustFitness)
    backend_factory: Callable[[str], BacktestBackend] = field(default=_default_backend_factory)
    research_eligible: bool = False
    synthetic_stress_forbidden: bool = False
    # Legacy field retained but NOT used as sole pass criterion on research paths.
    min_fitness_ratio: float = 0.35

    def run(
        self,
        candidate: StrategyCandidate,
        *,
        base_fitness: float,
        scenarios: tuple[str, ...] | None = None,
    ) -> list[StressResult]:
        chosen = scenarios or STRESS_SCENARIOS
        results: list[StressResult] = []
        forbid = bool(self.synthetic_stress_forbidden or self.research_eligible)

        # Seed baseline artifacts from a real base_costs run when factory supports it.
        baseline_arts: dict[str, Any] | None = None
        cache = getattr(self.backend_factory, "baseline_cache", None)
        base_event = getattr(self.backend_factory, "base_event_backend", None)
        if base_event is not None and hasattr(base_event, "evaluate"):
            try:
                base_event.evaluate(candidate)
                baseline_arts = dict(getattr(base_event, "last_run_artifacts", {}) or {})
                if cache is not None:
                    cache["arts"] = baseline_arts
            except Exception:  # noqa: BLE001
                baseline_arts = None

        for scenario in chosen:
            if self.counters.stress >= self.budget.max_stress_evaluations:
                break
            backend = self.backend_factory(scenario)
            assert_not_synthetic_stress(backend, research_eligible=forbid)
            if forbid and isinstance(backend, SyntheticOOSBackend):
                raise RuntimeError(SYNTHETIC_STRESS_FORBIDDEN)

            folds, _ = backend.evaluate(candidate)
            # Research integrity: never post-process / haircut fold metrics.
            fit = self.fitness_model.score(folds, complexity=candidate.complexity_score)
            arts: dict[str, Any] = {}
            cost_changes: dict[str, Any] = {}
            exec_changes: dict[str, Any] = {}
            backend_kind = str(getattr(backend, "backend_kind", type(backend).__name__))
            if isinstance(backend, StressScenarioBackend):
                cost_changes = dict(backend.cost_model_changes)
                exec_changes = dict(backend.execution_changes)
                arts = dict(getattr(backend.inner, "last_run_artifacts", {}) or {})
                backend_kind = backend.backend_kind
            elif hasattr(backend, "last_run_artifacts"):
                arts = dict(getattr(backend, "last_run_artifacts", {}) or {})

            trade_counts = int(arts.get("trades_count", sum(f.n_trades for f in folds)))
            orders_count = int(arts.get("orders_count", 0))
            fills_count = int(arts.get("fills_count", 0))

            # Pass only if real rerun satisfies economic SCORE gates (not fitness ratio alone).
            passed = not fit.rejected
            failure_reason = fit.rejection_reason if fit.rejected else None
            if not passed and failure_reason is None:
                failure_reason = "STRESS_ECONOMIC_GATE_FAILED"

            results.append(
                StressResult(
                    scenario=scenario,
                    candidate_id=candidate.candidate_id,
                    fitness=fit.fitness,
                    median_expectancy=float(np_median([f.expectancy for f in folds])),
                    max_drawdown=float(min((f.max_drawdown for f in folds), default=0.0)),
                    passed=passed,
                    failure_reason=failure_reason,
                    signal_source="candidate_dsl_trees",
                    backend_kind=backend_kind,
                    cost_model_changes=cost_changes,
                    execution_changes=exec_changes,
                    fold_metrics=[f.as_dict() for f in folds],
                    trade_counts=trade_counts,
                    orders_count=orders_count,
                    fills_count=fills_count,
                    details={
                        "base_fitness": base_fitness,
                        "fitness_ratio_vs_base": (
                            fit.fitness / base_fitness if abs(base_fitness) > 1e-12 else 0.0
                        ),
                        "economic_gate": "RobustFitness.score",
                        "baseline_seeded": baseline_arts is not None,
                    },
                )
            )
            self.counters.stress += 1
        return results


def np_median(xs: list[float]) -> float:
    import numpy as np

    return float(np.median(xs)) if xs else 0.0


def attach_stress(record: EvaluationRecord, results: list[StressResult]) -> EvaluationRecord:
    record.stress_results = {r.scenario: r.as_dict() for r in results}
    return record


# Re-export factory helper for control-plane wiring.
__all__ = [
    "STRESS_SCENARIOS",
    "StressResult",
    "StressTester",
    "attach_stress",
    "make_stress_backend_factory",
    "SYNTHETIC_STRESS_FORBIDDEN",
]
