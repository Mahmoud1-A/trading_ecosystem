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
    NOT_APPLICABLE_SINGLE_SYMBOL,
    STRESS_BACKEND_KIND,
    SYNTHETIC_STRESS_FORBIDDEN,
    UNSUPPORTED_STRESS_SCENARIO,
    StressScenarioBackend,
    assert_not_synthetic_stress,
    make_stress_backend_factory,
)

# Signal-source / backend-kind values that must be true for a research-eligible
# executed Stress scenario to be trusted (never hardcoded — always read from
# the scenario backend's own returned artifact).
_REQUIRED_RESEARCH_SIGNAL_SOURCE = "candidate_dsl_trees"
STRESS_SIGNAL_INTEGRITY_FAILED = "STRESS_SIGNAL_INTEGRITY_FAILED"
STRESS_INTEGRITY_FAILED = "STRESS_INTEGRITY_FAILED"
STRESS_BACKEND_KIND_INVALID = "STRESS_BACKEND_KIND_INVALID"
STRESS_SIGNAL_SOURCE_INVALID = "STRESS_SIGNAL_SOURCE_INVALID"
STRESS_WFO_INCOMPLETE = "STRESS_WFO_INCOMPLETE"

BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO = "original_qualifying_wfo"


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
    # "executed" (real rerun happened, normal pass/fail semantics) |
    # "not_applicable" (e.g. symbol_exclusion on a single-symbol campaign) |
    # "unsupported" (unknown scenario name) |
    # "baseline_unavailable" (real baseline closed trades could not be read).
    # Only "executed" scenarios belong in the Stress pass-rate denominator.
    status: str = "executed"
    completed_fold_count: int = 0
    integrity_ok: bool = True

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
            "status": self.status,
            "completed_fold_count": self.completed_fold_count,
            "integrity_ok": self.integrity_ok,
        }


def _default_backend_factory(scenario: str) -> BacktestBackend:
    # Smoke / unit-test default only — research paths must inject real factory.
    return SyntheticOOSBackend(seed_salt=stable_int_hash(scenario, bits=32) % 10_000)


def _integrity_failure_reason(
    *,
    signal_source: str,
    backend_kind: str,
    completed_fold_count: int,
) -> str:
    if signal_source != _REQUIRED_RESEARCH_SIGNAL_SOURCE:
        return STRESS_SIGNAL_SOURCE_INVALID
    if backend_kind != STRESS_BACKEND_KIND:
        return STRESS_BACKEND_KIND_INVALID
    if completed_fold_count <= 0:
        return STRESS_WFO_INCOMPLETE
    return STRESS_INTEGRITY_FAILED


@dataclass
class StressTester:
    budget: SearchBudget
    counters: BudgetCounters
    fitness_model: RobustFitness = field(default_factory=RobustFitness)
    backend_factory: Callable[[str], BacktestBackend] = field(default=_default_backend_factory)
    research_eligible: bool = False
    synthetic_stress_forbidden: bool = False
    progress_hook: Callable[[str, dict[str, Any]], None] | None = None
    progress_context: dict[str, Any] = field(default_factory=dict)
    # Legacy field retained but NOT used as sole pass criterion on research paths.
    min_fitness_ratio: float = 0.35
    # Accounting from the most recent run() call (honest Stress budget proof).
    last_run_accounting: dict[str, Any] = field(default_factory=dict)

    def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self.progress_hook is not None:
            # Candidate-level context is supplied by MultiFamilyCampaign so
            # scenario progress is monotonic across the full Stress queue.
            self.progress_hook(name, {**self.progress_context, **payload})

    def run(
        self,
        candidate: StrategyCandidate,
        *,
        base_fitness: float,
        scenarios: tuple[str, ...] | None = None,
        baseline_artifacts: dict[str, Any] | None = None,
    ) -> list[StressResult]:
        chosen = scenarios or STRESS_SCENARIOS
        results: list[StressResult] = []
        forbid = bool(self.synthetic_stress_forbidden or self.research_eligible)
        scenario_backend_calls = 0
        stress_before = int(self.counters.stress)

        # Seed removed_best_* from the candidate-specific original WFO snapshot.
        # Never re-evaluate the base event backend here — that would be a hidden,
        # unbudgeted Full WFO rerun outside Stress counters.
        baseline_arts: dict[str, Any] | None = (
            dict(baseline_artifacts) if baseline_artifacts else None
        )
        cache = getattr(self.backend_factory, "baseline_cache", None)
        if cache is not None:
            cache["arts"] = baseline_arts

        baseline_candidate_id = None
        if baseline_arts is not None:
            baseline_candidate_id = str(
                baseline_arts.get("candidate_id") or candidate.candidate_id
            )
        hidden_baseline_rerun = False

        single_symbol = len(candidate.asset_universe) <= 1

        for scenario_index, scenario in enumerate(chosen, start=1):
            if self.counters.stress >= self.budget.max_stress_evaluations:
                break

            self._emit(
                "STRESS_SCENARIO_STARTED",
                {
                    "candidate_id": candidate.candidate_id,
                    "scenario": scenario,
                    "scenario_index": scenario_index,
                    "total_scenarios": len(chosen),
                    "stress_consumed": int(self.counters.stress),
                    "max_stress_evaluations": int(self.budget.max_stress_evaluations),
                },
            )

            # Scenarios that can never be a genuine rerun must not consume a
            # stress-evaluation slot, must never be marked passed or failed,
            # and must never touch the real backend.
            if scenario == "symbol_exclusion" and single_symbol:
                results.append(
                    self._not_executed_result(
                        candidate, scenario, status="not_applicable",
                        reason=NOT_APPLICABLE_SINGLE_SYMBOL,
                    )
                )
                continue
            if scenario not in STRESS_SCENARIOS:
                results.append(
                    self._not_executed_result(
                        candidate, scenario, status="unsupported",
                        reason=UNSUPPORTED_STRESS_SCENARIO,
                    )
                )
                continue

            backend = self.backend_factory(scenario)
            assert_not_synthetic_stress(backend, research_eligible=forbid)
            if forbid and isinstance(backend, SyntheticOOSBackend):
                raise RuntimeError(SYNTHETIC_STRESS_FORBIDDEN)

            stress_status = getattr(backend, "stress_status", None)
            if stress_status is not None:
                # Real factory determined this scenario cannot be a genuine
                # rerun (e.g. baseline closed trades unavailable) — never
                # silently fall back to rerunning the unchanged baseline.
                status = (
                    "not_applicable"
                    if stress_status == NOT_APPLICABLE_SINGLE_SYMBOL
                    else "unsupported"
                    if stress_status == UNSUPPORTED_STRESS_SCENARIO
                    else "baseline_unavailable"
                )
                results.append(
                    self._not_executed_result(candidate, scenario, status=status, reason=stress_status)
                )
                continue

            folds, _ = backend.evaluate(candidate)
            scenario_backend_calls += 1
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
            completed_fold_count = int(arts.get("wfo_completed_folds", len(folds)))
            # Never hardcode — always read the actual signal source the
            # scenario backend claims to have used.
            signal_source = str(arts.get("signal_source") or "signal_source_missing")
            integrity_ok = (
                signal_source == _REQUIRED_RESEARCH_SIGNAL_SOURCE
                and backend_kind == STRESS_BACKEND_KIND
                and completed_fold_count > 0
            )

            # Pass only if real rerun satisfies economic SCORE gates (not fitness ratio alone).
            passed = not fit.rejected
            failure_reason = fit.rejection_reason if fit.rejected else None
            if not passed and failure_reason is None:
                failure_reason = "STRESS_ECONOMIC_GATE_FAILED"
            if self.research_eligible and not integrity_ok:
                # A research-eligible executed scenario whose own artifact does
                # not prove a real DSL-tree event-driven rerun is an integrity
                # failure, not a pass — regardless of the economic gate result.
                passed = False
                failure_reason = _integrity_failure_reason(
                    signal_source=signal_source,
                    backend_kind=backend_kind,
                    completed_fold_count=completed_fold_count,
                )

            results.append(
                StressResult(
                    scenario=scenario,
                    candidate_id=candidate.candidate_id,
                    fitness=fit.fitness,
                    median_expectancy=float(np_median([f.expectancy for f in folds])),
                    max_drawdown=float(min((f.max_drawdown for f in folds), default=0.0)),
                    passed=passed,
                    failure_reason=failure_reason,
                    signal_source=signal_source,
                    backend_kind=backend_kind,
                    cost_model_changes=cost_changes,
                    execution_changes=exec_changes,
                    fold_metrics=[f.as_dict() for f in folds],
                    trade_counts=trade_counts,
                    orders_count=orders_count,
                    fills_count=fills_count,
                    status="executed",
                    completed_fold_count=completed_fold_count,
                    integrity_ok=integrity_ok,
                    details={
                        "base_fitness": base_fitness,
                        "fitness_ratio_vs_base": (
                            fit.fitness / base_fitness if abs(base_fitness) > 1e-12 else 0.0
                        ),
                        "economic_gate": "RobustFitness.score",
                        "baseline_seeded": baseline_arts is not None,
                        "baseline_artifact_source": (
                            BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO
                            if baseline_arts is not None
                            else None
                        ),
                        "baseline_candidate_id": baseline_candidate_id,
                        "hidden_baseline_rerun": hidden_baseline_rerun,
                    },
                )
            )
            self.counters.stress += 1
            completed = results[-1]
            self._emit(
                "STRESS_SCENARIO_COMPLETED",
                {
                    "candidate_id": candidate.candidate_id,
                    "scenario": scenario,
                    "scenario_index": scenario_index,
                    "total_scenarios": len(chosen),
                    "status": completed.status,
                    "passed": completed.passed,
                    "failure_reason": completed.failure_reason,
                    "stress_consumed": int(self.counters.stress),
                    "max_stress_evaluations": int(self.budget.max_stress_evaluations),
                },
            )

        stress_counter_delta = int(self.counters.stress) - stress_before
        self.last_run_accounting = {
            "baseline_artifact_source": (
                BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO if baseline_arts is not None else None
            ),
            "baseline_candidate_id": baseline_candidate_id,
            "hidden_baseline_rerun": hidden_baseline_rerun,
            "scenario_backend_calls": scenario_backend_calls,
            "stress_counter_delta": stress_counter_delta,
        }
        return results

    @staticmethod
    def _not_executed_result(
        candidate: StrategyCandidate, scenario: str, *, status: str, reason: str
    ) -> "StressResult":
        """Build a StressResult for a scenario that was never genuinely rerun.

        Never passed, never failed, never executed, and excluded from the
        pass-rate denominator by callers filtering on ``status == "executed"``.
        """
        return StressResult(
            scenario=scenario,
            candidate_id=candidate.candidate_id,
            fitness=0.0,
            median_expectancy=0.0,
            max_drawdown=0.0,
            passed=False,
            failure_reason=reason,
            signal_source="not_executed",
            backend_kind="stress_status_no_rerun",
            status=status,
            completed_fold_count=0,
            integrity_ok=False,
            details={"executed": False, "reason": reason},
        )


def np_median(xs: list[float]) -> float:
    import numpy as np

    return float(np.median(xs)) if xs else 0.0


def attach_stress(record: EvaluationRecord, results: list[StressResult]) -> EvaluationRecord:
    record.stress_results = {r.scenario: r.as_dict() for r in results}
    return record


# Re-export factory helper for control-plane wiring.
__all__ = [
    "STRESS_SCENARIOS",
    "STRESS_INTEGRITY_FAILED",
    "STRESS_BACKEND_KIND_INVALID",
    "STRESS_SIGNAL_SOURCE_INVALID",
    "STRESS_WFO_INCOMPLETE",
    "BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO",
    "StressResult",
    "StressTester",
    "attach_stress",
    "make_stress_backend_factory",
    "SYNTHETIC_STRESS_FORBIDDEN",
]
