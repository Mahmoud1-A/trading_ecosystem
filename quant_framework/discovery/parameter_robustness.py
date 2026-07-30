"""Parameter robustness — reject narrow optima and cliffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.evaluator import BacktestBackend, SyntheticOOSBackend
from discovery.expression_tree import ExprNode, NodeKind, parameter_node
from discovery.fitness import RobustFitness
from discovery.types import CreationMethod


@dataclass(frozen=True)
class RobustnessResult:
    parameter: str
    base_value: float
    neighborhood: tuple[float, ...]
    fitness_curve: tuple[float, ...]
    plateau_score: float
    cliff_detected: bool
    spike_detected: bool
    accepted: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "base_value": self.base_value,
            "neighborhood": list(self.neighborhood),
            "fitness_curve": list(self.fitness_curve),
            "plateau_score": self.plateau_score,
            "cliff_detected": self.cliff_detected,
            "spike_detected": self.spike_detected,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def _replace_param(tree: ExprNode, name: str, value: float) -> ExprNode:
    kids = tuple(_replace_param(c, name, value) for c in tree.children)
    if tree.kind is NodeKind.PARAMETER and tree.name == name:
        return parameter_node(name, value)
    return ExprNode(tree.kind, tree.value_type, tree.name, kids, dict(tree.meta))


@dataclass
class ParameterRobustness:
    relative_steps: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    absolute_eps: float = 0.05
    cliff_drop: float = 0.5  # relative drop from peak
    spike_ratio: float = 2.0  # peak vs median neighbors
    backend: BacktestBackend = field(default_factory=SyntheticOOSBackend)
    fitness_model: RobustFitness = field(default_factory=RobustFitness)

    def probe(self, candidate: StrategyCandidate) -> list[RobustnessResult]:
        results: list[RobustnessResult] = []
        for name, base in candidate.parameters.items():
            neighborhood: list[float] = []
            curve: list[float] = []
            for step in self.relative_steps:
                if abs(base) < 1e-12:
                    val = base + step * self.absolute_eps * 10
                else:
                    val = base * (1.0 + step)
                neighborhood.append(val)
                perturbed = self._with_param(candidate, name, val)
                folds, _ = self.backend.evaluate(perturbed)
                fit = self.fitness_model.score(folds, complexity=perturbed.complexity_score)
                curve.append(fit.fitness)

            arr = np.asarray(curve, dtype=float)
            peak = float(np.max(arr))
            median = float(np.median(arr))
            center_idx = list(self.relative_steps).index(0.0)
            center = float(arr[center_idx])
            # Cliff: neighbors much worse than center
            neighbors = np.delete(arr, center_idx)
            cliff = bool(np.any(neighbors < center - abs(center) * self.cliff_drop - 1e-9))
            # Spike: center far above median of neighborhood
            spike = bool(center > median * self.spike_ratio + 1e-9 and center > peak * 0.95)
            plateau = 1.0 / (1.0 + float(np.std(arr)))
            accepted = not cliff and not spike and plateau >= 0.2
            reason = ""
            if cliff:
                reason = "performance_cliff"
            elif spike:
                reason = "isolated_spike"
            elif not accepted:
                reason = "unstable_plateau"
            results.append(
                RobustnessResult(
                    parameter=name,
                    base_value=base,
                    neighborhood=tuple(neighborhood),
                    fitness_curve=tuple(float(x) for x in curve),
                    plateau_score=plateau,
                    cliff_detected=cliff,
                    spike_detected=spike,
                    accepted=accepted,
                    reason=reason,
                )
            )
        return results

    def _with_param(self, candidate: StrategyCandidate, name: str, value: float) -> StrategyCandidate:
        entry = _replace_param(candidate.entry_tree, name, value)
        exit_tree = (
            _replace_param(candidate.exit_tree, name, value) if candidate.exit_tree else None
        )
        return build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            stop=candidate.stop,
            target=candidate.target,
            sizing=candidate.sizing,
            regime_gates=candidate.regime_gates,
            strategy_family=candidate.strategy_family,
            creation_method=CreationMethod.MUTATION,
            generation=candidate.generation,
            parent_ids=(candidate.candidate_id,),
            grammar_version=candidate.grammar_version,
            feature_set_version=candidate.feature_set_version,
            cost_model_version=candidate.cost_model_version,
            asset_universe=candidate.asset_universe,
            random_seed=candidate.random_seed + hash(name) % 997,
        )

    def instability_score(self, results: list[RobustnessResult]) -> float:
        if not results:
            return 0.0
        penalties = []
        for r in results:
            p = 0.0
            if r.cliff_detected:
                p += 1.0
            if r.spike_detected:
                p += 1.0
            p += max(0.0, 1.0 - r.plateau_score)
            penalties.append(p)
        return float(np.mean(penalties))
