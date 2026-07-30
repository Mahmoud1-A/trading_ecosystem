"""Export example real Stress / ParameterRobustness artifacts for reporting."""

from __future__ import annotations

import json
from pathlib import Path

from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.fitness import RobustFitness
from discovery.generator import CandidateGenerator
from discovery.parameter_robustness import ParameterRobustness
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.stress import StressTester
from discovery.stress_backend import make_stress_backend_factory
from discovery.event_wfo_backend import _synthetic_bars

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "research_integrity"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    bars = _synthetic_bars(900, seed=3)
    backend = EventDrivenDiscoveryBackend(bars=bars, require_real_bars=False)
    cand = CandidateGenerator().seed_template_mean_reversion(seed=3)
    fitness = RobustFitness(min_total_oos_trades=1, min_oos_trades_per_fold=0, max_oos_drawdown=1.0)

    factory = make_stress_backend_factory(
        backend, research_eligible=True, synthetic_stress_forbidden=True
    )
    tester = StressTester(
        budget=SearchBudget(max_stress_evaluations=4),
        counters=BudgetCounters(),
        fitness_model=fitness,
        backend_factory=factory,
        research_eligible=True,
        synthetic_stress_forbidden=True,
    )
    stress = tester.run(
        cand,
        base_fitness=1.0,
        scenarios=("base_costs", "costs_2x", "wider_spread", "delayed_execution"),
    )
    (OUT / "example_real_stress.json").write_text(
        json.dumps([r.as_dict() for r in stress], indent=2, default=str), encoding="utf-8"
    )

    rob = ParameterRobustness(
        backend=backend,
        fitness_model=fitness,
        research_eligible=True,
        synthetic_robustness_forbidden=True,
        relative_steps=(-0.1, 0.0, 0.1),
    )
    rows = rob.probe(cand)
    (OUT / "example_real_parameter_robustness.json").write_text(
        json.dumps([r.as_dict() for r in rows], indent=2, default=str), encoding="utf-8"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
