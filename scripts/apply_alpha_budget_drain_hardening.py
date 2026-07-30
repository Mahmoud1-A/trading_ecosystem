from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.write_text(text, encoding="utf-8")
    print(f"updated {rel}")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_search_budget() -> None:
    rel = "quant_framework/discovery/search_budget.py"
    text = read(rel)
    text = replace_once(
        text,
        "    max_stress_evaluations: int = 8\n",
        "    # Stress budget counts candidates, not individual scenarios.\n"
        "    max_stress_evaluations: int = 8\n"
        "    max_stress_scenarios_per_candidate: int = 8\n"
        "    min_oos_trades: int = 8\n"
        "    min_oos_trades_per_fold: int = 1\n",
        label="SearchBudget stress/trade fields",
    )
    text = replace_once(
        text,
        "    stress: int = 0\n",
        "    # stress = completed stress candidates; stress_scenarios = scenario runs.\n"
        "    stress: int = 0\n"
        "    stress_scenarios: int = 0\n",
        label="BudgetCounters stress counters",
    )
    write(rel, text)


def patch_fitness() -> None:
    rel = "quant_framework/discovery/fitness.py"
    text = read(rel)
    text = replace_once(
        text,
        "    min_folds: int = 1\n",
        "    min_folds: int = 1\n"
        "    min_total_oos_trades: int = 8\n"
        "    min_oos_trades_per_fold: int = 1\n",
        label="RobustFitness trade thresholds",
    )
    marker = """        expectancies = [f.expectancy for f in folds]\n"""
    insertion = """        total_oos_trades = sum(max(0, int(f.n_trades)) for f in folds)\n        fold_trade_counts = tuple(max(0, int(f.n_trades)) for f in folds)\n        if total_oos_trades == 0:\n            return FitnessResult(\n                fitness=float(\"-inf\"),\n                ranking_source=ranking_source,\n                components={\n                    \"total_oos_trades\": 0.0,\n                    \"minimum_required_oos_trades\": float(self.min_total_oos_trades),\n                },\n                fold_scores=(),\n                rejected=True,\n                rejection_reason=\"NO_OOS_TRADES\",\n            )\n        if (\n            total_oos_trades < self.min_total_oos_trades\n            or any(n < self.min_oos_trades_per_fold for n in fold_trade_counts)\n        ):\n            return FitnessResult(\n                fitness=float(\"-inf\"),\n                ranking_source=ranking_source,\n                components={\n                    \"total_oos_trades\": float(total_oos_trades),\n                    \"minimum_required_oos_trades\": float(self.min_total_oos_trades),\n                    \"minimum_required_oos_trades_per_fold\": float(\n                        self.min_oos_trades_per_fold\n                    ),\n                },\n                fold_scores=(),\n                rejected=True,\n                rejection_reason=\"INSUFFICIENT_OOS_TRADES\",\n            )\n\n        expectancies = [f.expectancy for f in folds]\n"""
    text = replace_once(text, marker, insertion, label="RobustFitness trade gate")
    write(rel, text)


def patch_stress() -> None:
    rel = "quant_framework/discovery/stress.py"
    content = '''"""Stress tests for discovery finalists.

Real-data candidates are re-evaluated through scenario-specific event-driven WFO
backends. Synthetic stress is retained only for explicit synthetic/smoke runs.
The stress budget counts candidates; scenario executions are tracked separately.
"""

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
    "removed_best_day",
    "parameter_perturbation",
)

# Same-process registry populated by EventDrivenDiscoveryBackend.evaluate().
_REAL_STRESS_BACKENDS: dict[str, BacktestBackend] = {}


def register_real_stress_backend(candidate_id: str, backend: BacktestBackend) -> None:
    if bool(getattr(backend, "is_full_event_wfo", False)):
        _REAL_STRESS_BACKENDS[candidate_id] = backend


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
    min_fitness_ratio: float = 0.35

    def _backend_for(self, candidate: StrategyCandidate, scenario: str) -> BacktestBackend:
        base = _REAL_STRESS_BACKENDS.get(candidate.candidate_id)
        if base is not None:
            factory = getattr(base, "for_stress_scenario", None)
            if not callable(factory):
                raise RuntimeError(
                    "REAL_STRESS_BACKEND_UNAVAILABLE: event-driven backend lacks "
                    "for_stress_scenario()"
                )
            backend = factory(scenario)
            if not bool(getattr(backend, "is_full_event_wfo", False)):
                raise RuntimeError(
                    "REAL_STRESS_BACKEND_UNAVAILABLE: scenario backend is not full event WFO"
                )
            return backend
        return self.backend_factory(scenario)

    def run(
        self,
        candidate: StrategyCandidate,
        *,
        base_fitness: float,
        scenarios: tuple[str, ...] | None = None,
    ) -> list[StressResult]:
        # Candidate budget, not scenario budget.
        if self.counters.stress >= self.budget.max_stress_evaluations:
            return []

        chosen = tuple(scenarios or STRESS_SCENARIOS)[
            : max(1, int(self.budget.max_stress_scenarios_per_candidate))
        ]
        results: list[StressResult] = []
        for scenario in chosen:
            backend = self._backend_for(candidate, scenario)
            folds, _ = backend.evaluate(candidate)
            is_real = bool(getattr(backend, "is_full_event_wfo", False))
            # Synthetic smoke tests keep the legacy metric haircuts. Real backends
            # already embody the scenario in prices/costs/latency and are not
            # cosmetically altered after execution.
            if not is_real:
                folds = [_apply_scenario(f, scenario) for f in folds]
            fit = self.fitness_model.score(folds, complexity=candidate.complexity_score)
            ratio = fit.fitness / base_fitness if abs(base_fitness) > 1e-12 else 0.0
            passed = (not fit.rejected) and (
                fit.fitness >= base_fitness * self.min_fitness_ratio
                or (base_fitness <= 0 and fit.fitness >= base_fitness)
            )
            results.append(
                StressResult(
                    scenario=scenario,
                    fitness=fit.fitness,
                    median_expectancy=float(np_median([f.expectancy for f in folds])),
                    max_drawdown=float(min((f.max_drawdown for f in folds), default=0.0)),
                    passed=passed,
                    details={
                        "fitness_ratio_vs_base": ratio,
                        "backend_kind": getattr(backend, "backend_kind", type(backend).__name__),
                        "is_full_event_wfo": is_real,
                        "rejection_reason": fit.rejection_reason,
                        "total_oos_trades": sum(int(f.n_trades) for f in folds),
                    },
                )
            )
            self.counters.stress_scenarios += 1

        if results:
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
    elif scenario == "delayed_execution":
        exp *= 0.85
        sharpe *= 0.9
    elif scenario == "removed_best_day":
        exp *= 0.6
        sharpe *= 0.65
    elif scenario == "parameter_perturbation":
        exp *= 0.9
        sharpe *= 0.92
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
'''
    write(rel, content)


def patch_event_backend() -> None:
    rel = "quant_framework/discovery/event_wfo_backend.py"
    text = read(rel)
    text = replace_once(
        text,
        "from dataclasses import dataclass, field\n",
        "from dataclasses import dataclass, field, replace\n",
        label="event backend dataclasses import",
    )
    text = replace_once(
        text,
        "    last_run_artifacts: dict[str, Any] = field(default_factory=dict)\n",
        "    last_run_artifacts: dict[str, Any] = field(default_factory=dict)\n"
        "    stress_scenario: str = \"base_costs\"\n",
        label="event backend stress field",
    )
    context_marker = """            tf = str(self.silver_resolution.get("timeframe") or "")\n"""
    context_insert = """            # Scenario-specific execution assumptions for real stress reruns.\n            if self.stress_scenario in {"costs_2x", "costs_4x"}:\n                mult = 2.0 if self.stress_scenario == "costs_2x" else 4.0\n                cost = replace(\n                    cost,\n                    commission_per_contract=cost.commission_per_contract * mult,\n                    minimum_commission=cost.minimum_commission * mult,\n                    fixed_spread_ticks=cost.fixed_spread_ticks * mult,\n                    slippage_ticks_mean=cost.slippage_ticks_mean * mult,\n                    slippage_ticks_std=cost.slippage_ticks_std * mult,\n                )\n            elif self.stress_scenario == "wider_spread":\n                cost = replace(cost, fixed_spread_ticks=max(1.0, cost.fixed_spread_ticks * 2.0))\n            elif self.stress_scenario == "worse_slippage":\n                cost = replace(\n                    cost,\n                    slippage_ticks_mean=max(0.5, cost.slippage_ticks_mean * 2.0),\n                    slippage_ticks_std=max(0.5, cost.slippage_ticks_std * 2.0),\n                )\n\n            tf = str(self.silver_resolution.get("timeframe") or "")\n"""
    text = replace_once(text, context_marker, context_insert, label="event stress costs")

    evaluate_marker = """    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:\n"""
    method = '''    def for_stress_scenario(self, scenario: str) -> "EventDrivenDiscoveryBackend":
        supported = {
            "base_costs",
            "costs_2x",
            "costs_4x",
            "wider_spread",
            "worse_slippage",
            "delayed_execution",
            "removed_best_day",
            "parameter_perturbation",
        }
        if scenario not in supported:
            raise ValueError(f"unsupported real stress scenario: {scenario}")
        assert self.bars is not None
        bars = self.bars.copy()
        if scenario == "removed_best_day" and len(bars):
            ts = pd.to_datetime(bars["timestamp"], utc=True) if "timestamp" in bars.columns else pd.DatetimeIndex(bars.index)
            close = bars["close"].astype(float)
            daily = close.groupby(ts.floor("D")).last().pct_change()
            if daily.notna().any():
                best_day = daily.idxmax()
                bars = bars.loc[ts.floor("D") != best_day].copy()
        return EventDrivenDiscoveryBackend(
            bars=bars,
            wfo_config=self.wfo_config,
            require_real_bars=True,
            asset_class=self.asset_class,
            intraday_only=self.intraday_only,
            artifact_dir=self.artifact_dir,
            silver_resolution=dict(self.silver_resolution),
            stress_scenario=scenario,
        )

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
'''
    text = replace_once(text, evaluate_marker, method, label="event stress clone method")

    params_marker = '''        params = {
            "lookback": int(candidate.parameters.get("lookback", 15)),
            "z_entry": float(candidate.parameters.get("z_entry", 2.0)),
        }
'''
    params_new = '''        params = {
            "lookback": int(candidate.parameters.get("lookback", 15)),
            "z_entry": float(candidate.parameters.get("z_entry", 2.0)),
        }
        if self.stress_scenario == "parameter_perturbation":
            params["lookback"] = max(2, params["lookback"] + 2)
            params["z_entry"] = params["z_entry"] * 1.10
        if self.stress_scenario == "delayed_execution":
            params["_delay_bars"] = 1
'''
    text = replace_once(text, params_marker, params_new, label="event stress params")

    signal_marker = '''    lookback = int(params.get("lookback", 20))
    z_entry = float(params.get("z_entry", 2.0))
'''
    signal_new = '''    lookback = int(params.get("lookback", 20))
    z_entry = float(params.get("z_entry", 2.0))
    delay_bars = max(0, int(params.get("_delay_bars", 0)))
'''
    text = replace_once(text, signal_marker, signal_new, label="signal delay parameter")
    old_signal_logic = '''        if i < lookback or i >= len(closes):
            return None
        slice_ = closes[max(0, i - lookback) : i]
        if len(slice_) < 2:
            return None
        mu = float(np.mean(slice_))
        sd = float(np.std(slice_)) or 1e-9
        z = (closes[i] - mu) / sd
'''
    new_signal_logic = '''        signal_i = i - delay_bars
        if signal_i < lookback or signal_i >= len(closes):
            return None
        slice_ = closes[max(0, signal_i - lookback) : signal_i]
        if len(slice_) < 2:
            return None
        mu = float(np.mean(slice_))
        sd = float(np.std(slice_)) or 1e-9
        z = (closes[signal_i] - mu) / sd
'''
    text = replace_once(text, old_signal_logic, new_signal_logic, label="signal delayed evaluation")

    return_marker = """        return folds, train_diag\n"""
    return_new = """        from discovery.stress import register_real_stress_backend\n\n        register_real_stress_backend(candidate.candidate_id, self)\n        return folds, train_diag\n"""
    text = replace_once(text, return_marker, return_new, label="register real stress backend")
    write(rel, text)


def patch_search_controller() -> None:
    rel = "quant_framework/discovery/search_controller.py"
    text = read(rel)
    old = """        for fin in finalists[: max(1, self.budget.max_vault_submissions + 2)]:\n"""
    new = """        # Drain downstream stress work for the best behaviorally unique\n        # representatives. Stress budget counts candidates, not scenarios.\n        stress_queue = finalists[: max(0, int(self.budget.max_stress_evaluations))]\n        for fin in stress_queue:\n"""
    text = replace_once(text, old, new, label="controller stress drain queue")
    write(rel, text)


def patch_jobs() -> None:
    rel = "quant_framework/control_plane/jobs.py"
    text = read(rel)
    old = '''    budget = SearchBudget(
        max_generated_candidates=int(budget_cfg.get("max_generated_candidates", 12)),
        max_evaluated_candidates=int(budget_cfg.get("max_evaluated_candidates", 10)),
        max_full_wfo_evaluations=int(budget_cfg.get("max_full_wfo_evaluations", 10)),
        max_stress_evaluations=int(budget_cfg.get("max_stress_evaluations", 4)),
'''
    new = '''    full_wfo_cap = int(budget_cfg.get("max_full_wfo_evaluations", 10))
    # When omitted, stress candidate capacity follows Full WFO capacity so the
    # downstream pipeline can drain instead of stopping at an unrelated default.
    stress_candidate_cap = int(budget_cfg.get("max_stress_evaluations", full_wfo_cap))
    budget = SearchBudget(
        max_generated_candidates=int(budget_cfg.get("max_generated_candidates", 12)),
        max_evaluated_candidates=int(budget_cfg.get("max_evaluated_candidates", 10)),
        max_full_wfo_evaluations=full_wfo_cap,
        max_stress_evaluations=stress_candidate_cap,
        max_stress_scenarios_per_candidate=int(
            budget_cfg.get("max_stress_scenarios_per_candidate", 8)
        ),
'''
    text = replace_once(text, old, new, label="control-plane effective stress budget")
    write(rel, text)


def patch_alpha_results() -> None:
    rel = "quant_framework/control_plane/alpha_results.py"
    text = read(rel)
    old = '''def _latest_by_candidate(trials: list[TrialRecord]) -> dict[str, TrialRecord]:
    """Keep one trial per candidate_id (last write wins) to avoid double-counting."""
    out: dict[str, TrialRecord] = {}
    for t in trials:
        out[t.candidate_id] = t
    return out
'''
    new = '''def _latest_by_candidate(trials: list[TrialRecord]) -> dict[str, TrialRecord]:
    """Keep the richest trial per candidate, not a later duplicate shell.

    A later duplicate registry entry must not erase completed WFO fold evidence
    from the original evaluation of the same deterministic candidate ID.
    """

    def richness(t: TrialRecord) -> tuple[int, int, int, int]:
        reason = (t.rejection_reason or "").lower()
        return (
            0 if "duplicate" in reason else 1,
            1 if t.fold_records else 0,
            1 if t.ranking_score is not None else 0,
            len(t.fold_records or []),
        )

    out: dict[str, TrialRecord] = {}
    for t in trials:
        prev = out.get(t.candidate_id)
        if prev is None or richness(t) >= richness(prev):
            out[t.candidate_id] = t
    return out
'''
    text = replace_once(text, old, new, label="richest candidate trial selection")

    funnel_marker = """    funnel = funnel_counters(rows, cluster_count=len(result.clusters))\n"""
    funnel_new = """    funnel = funnel_counters(rows, cluster_count=len(result.clusters))\n    # Keep separate meanings: consumed backend attempts versus distinct candidates\n    # whose completed fold evidence is visible in the registry-backed table.\n    funnel[\"full_wfo_completed_candidates\"] = funnel[\"full_wfo_evaluations\"]\n    funnel[\"full_wfo_budget_consumed\"] = int(counters.full_wfo)\n    funnel[\"full_wfo_evaluations\"] = int(counters.full_wfo)\n"""
    text = replace_once(text, funnel_marker, funnel_new, label="WFO reporting reconciliation")

    terminal_old = '''    elif all_precheck:
        discovery = "ALL_CANDIDATES_PRECHECK_REJECTED"
        terminal = "ALL_CANDIDATES_PRECHECK_REJECTED"
    else:
        discovery = "NO_QUALIFIED_CANDIDATE"
        terminal = map_terminal_reason(result.stop_reason, cancelled=False)
'''
    terminal_new = '''    elif all_precheck:
        discovery = "ALL_CANDIDATES_PRECHECK_REJECTED"
        terminal = "ALL_CANDIDATES_PRECHECK_REJECTED"
    else:
        unresolved = [
            r
            for r in rows
            if r.get("promotion_label")
            in {"SHORTLISTED", "TOP_RANKED_UNVALIDATED", "RESEARCH_SHORTLISTED"}
            and r.get("next_missing_gate") not in {None, "", "NONE"}
        ]
        if unresolved:
            discovery = "CAMPAIGN_INCOMPLETE"
            terminal = "PIPELINE_NOT_DRAINED"
        else:
            discovery = "NO_QUALIFIED_CANDIDATE"
            terminal = map_terminal_reason(result.stop_reason, cancelled=False)
'''
    text = replace_once(text, terminal_old, terminal_new, label="NQC pipeline-drain semantics")

    stress_summary_old = '''        "stress_summary": {
            "stress_evaluations": counters.stress,
            "stress_passed": funnel["stress_passed"],
            "status": "EVALUATED" if counters.stress else STATUS_NOT_EVALUATED,
        },
'''
    stress_summary_new = '''        "stress_summary": {
            "stress_candidates_completed": counters.stress,
            "stress_scenarios_completed": getattr(counters, "stress_scenarios", 0),
            "stress_evaluations": counters.stress,
            "stress_passed": funnel["stress_passed"],
            "status": "EVALUATED" if counters.stress else STATUS_NOT_EVALUATED,
        },
'''
    text = replace_once(text, stress_summary_old, stress_summary_new, label="stress report counters")
    write(rel, text)


def add_tests() -> None:
    rel = "quant_framework/tests/test_alpha_budget_drain_hardening.py"
    path = ROOT / rel
    if path.exists():
        print(f"exists {rel}")
        return
    path.write_text(
        '''from __future__ import annotations

from dataclasses import dataclass

from discovery.candidate import StrategyCandidate
from discovery.fitness import FoldOOSMetrics, RobustFitness
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.stress import StressTester, register_real_stress_backend


def _fold(n_trades: int) -> FoldOOSMetrics:
    return FoldOOSMetrics(
        fold_id=0,
        expectancy=0.1,
        sharpe=0.5,
        profit_factor=1.2,
        calmar=0.3,
        max_drawdown=-0.05,
        n_trades=n_trades,
    )


def test_zero_trade_candidate_is_rejected() -> None:
    result = RobustFitness().score([_fold(0)])
    assert result.rejected is True
    assert result.rejection_reason == "NO_OOS_TRADES"


def test_insufficient_trade_candidate_is_rejected() -> None:
    result = RobustFitness(min_total_oos_trades=8).score([_fold(3), _fold(3)])
    assert result.rejected is True
    assert result.rejection_reason == "INSUFFICIENT_OOS_TRADES"


@dataclass
class _ScenarioBackend:
    backend_kind: str = "event_driven_wfo"
    is_full_event_wfo: bool = True

    def for_stress_scenario(self, scenario: str) -> "_ScenarioBackend":
        return self

    def evaluate(self, candidate: StrategyCandidate):
        return [_fold(12)], {"scenario": "real"}


def test_stress_budget_counts_candidates_not_scenarios() -> None:
    # Minimal candidate object is not required by the fake backend beyond ID.
    candidate = object.__new__(StrategyCandidate)
    object.__setattr__(candidate, "candidate_id", "cand_real")
    object.__setattr__(candidate, "complexity_score", 1.0)
    register_real_stress_backend("cand_real", _ScenarioBackend())
    counters = BudgetCounters()
    budget = SearchBudget(max_stress_evaluations=1, max_stress_scenarios_per_candidate=3)
    tester = StressTester(budget=budget, counters=counters)
    results = tester.run(
        candidate,
        base_fitness=0.1,
        scenarios=("base_costs", "costs_2x", "wider_spread"),
    )
    assert len(results) == 3
    assert counters.stress == 1
    assert counters.stress_scenarios == 3
    assert all(r.details["is_full_event_wfo"] for r in results)
''',
        encoding="utf-8",
    )
    print(f"created {rel}")


def main() -> None:
    patch_search_budget()
    patch_fitness()
    patch_stress()
    patch_event_backend()
    patch_search_controller()
    patch_jobs()
    patch_alpha_results()
    add_tests()
    print("Alpha Miner budget-drain hardening patch applied.")


if __name__ == "__main__":
    main()
