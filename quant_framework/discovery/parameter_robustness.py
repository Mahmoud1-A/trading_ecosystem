"""Parameter robustness — real event-driven WFO for each perturbed candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.evaluator import BacktestBackend, SyntheticOOSBackend
from discovery.expression_tree import ExprNode, NodeKind, parameter_node
from discovery.fitness import RobustFitness
from discovery.stable_hash import stable_seed
from discovery.types import CreationMethod

SYNTHETIC_ROBUSTNESS_FORBIDDEN = "SYNTHETIC_ROBUSTNESS_FORBIDDEN"


@dataclass(frozen=True)
class RobustnessResult:
    parameter: str
    base_value: float
    tested_value: float
    candidate_id: str
    neighborhood: tuple[float, ...]
    fitness_curve: tuple[float, ...]
    fold_metrics: tuple[dict[str, Any], ...]
    total_oos_trades: int
    median_expectancy: float
    median_profit_factor: float
    worst_drawdown: float
    fitness: float
    plateau_score: float
    cliff_detected: bool
    spike_detected: bool
    accepted: bool
    reason: str = ""
    backend_kind: str = ""
    points: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "base_value": self.base_value,
            "tested_value": self.tested_value,
            "candidate_id": self.candidate_id,
            "neighborhood": list(self.neighborhood),
            "fitness_curve": list(self.fitness_curve),
            "fold_metrics": list(self.fold_metrics),
            "total_oos_trades": self.total_oos_trades,
            "median_expectancy": self.median_expectancy,
            "median_profit_factor": self.median_profit_factor,
            "worst_drawdown": self.worst_drawdown,
            "fitness": self.fitness,
            "plateau_score": self.plateau_score,
            "cliff_detected": self.cliff_detected,
            "spike_detected": self.spike_detected,
            "accepted": self.accepted,
            "reason": self.reason,
            "backend_kind": self.backend_kind,
            "points": list(self.points),
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
    cliff_drop: float = 0.5
    spike_ratio: float = 2.0
    backend: BacktestBackend = field(default_factory=SyntheticOOSBackend)
    fitness_model: RobustFitness = field(default_factory=RobustFitness)
    research_eligible: bool = False
    synthetic_robustness_forbidden: bool = False

    def probe(self, candidate: StrategyCandidate) -> list[RobustnessResult]:
        forbid = bool(self.synthetic_robustness_forbidden or self.research_eligible)
        if forbid and isinstance(self.backend, SyntheticOOSBackend):
            raise RuntimeError(
                f"{SYNTHETIC_ROBUSTNESS_FORBIDDEN}: research-eligible robustness "
                "cannot use SyntheticOOSBackend"
            )
        if forbid and getattr(self.backend, "backend_kind", "") == "synthetic_oos_probe":
            raise RuntimeError(SYNTHETIC_ROBUSTNESS_FORBIDDEN)

        backend_kind = str(getattr(self.backend, "backend_kind", type(self.backend).__name__))
        results: list[RobustnessResult] = []
        for name, base in candidate.parameters.items():
            neighborhood: list[float] = []
            curve: list[float] = []
            points: list[dict[str, Any]] = []
            all_folds: list[dict[str, Any]] = []
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
                fold_dicts = [f.as_dict() for f in folds]
                all_folds.extend(fold_dicts)
                points.append(
                    {
                        "parameter": name,
                        "base_value": float(base),
                        "tested_value": float(val),
                        "candidate_id": perturbed.candidate_id,
                        "fold_metrics": fold_dicts,
                        "total_oos_trades": int(sum(f.n_trades for f in folds)),
                        "median_expectancy": float(np.median([f.expectancy for f in folds]))
                        if folds
                        else 0.0,
                        "median_profit_factor": float(np.median([f.profit_factor for f in folds]))
                        if folds
                        else 0.0,
                        "worst_drawdown": float(min((f.max_drawdown for f in folds), default=0.0)),
                        "fitness": float(fit.fitness),
                        "backend_kind": backend_kind,
                    }
                )

            arr = np.asarray(curve, dtype=float)
            peak = float(np.max(arr)) if len(arr) else 0.0
            median = float(np.median(arr)) if len(arr) else 0.0
            center_idx = list(self.relative_steps).index(0.0)
            center = float(arr[center_idx]) if len(arr) else 0.0
            neighbors = np.delete(arr, center_idx) if len(arr) else arr
            cliff = bool(np.any(neighbors < center - abs(center) * self.cliff_drop - 1e-9))
            spike = bool(center > median * self.spike_ratio + 1e-9 and center > peak * 0.95)
            plateau = 1.0 / (1.0 + float(np.std(arr))) if len(arr) else 0.0
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
                    base_value=float(base),
                    tested_value=float(base),
                    candidate_id=candidate.candidate_id,
                    neighborhood=tuple(float(x) for x in neighborhood),
                    fitness_curve=tuple(float(x) for x in curve),
                    fold_metrics=tuple(all_folds),
                    total_oos_trades=int(sum(int(p["total_oos_trades"]) for p in points)),
                    median_expectancy=float(np.median([p["median_expectancy"] for p in points])),
                    median_profit_factor=float(
                        np.median([p["median_profit_factor"] for p in points])
                    ),
                    worst_drawdown=float(min(p["worst_drawdown"] for p in points)),
                    fitness=float(center),
                    plateau_score=plateau,
                    cliff_detected=cliff,
                    spike_detected=spike,
                    accepted=accepted,
                    reason=reason,
                    backend_kind=backend_kind,
                    points=tuple(points),
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
            random_seed=stable_seed(candidate.random_seed, name),
            family_provenance={
                **dict(candidate.family_provenance or {}),
                "robustness_param": name,
                "robustness_value": float(value),
            },
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
