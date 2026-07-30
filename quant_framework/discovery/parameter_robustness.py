"""Parameter robustness — real event-driven WFO for each perturbed candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.evaluator import BacktestBackend, SyntheticOOSBackend
from discovery.expression_tree import ExprNode, NodeKind, parameter_node
from discovery.feature_domains import (
    INVALID_FEATURE_THRESHOLD_DOMAIN,
    collect_candidate_threshold_violations,
)
from discovery.fitness import RobustFitness
from discovery.grammar import Grammar
from discovery.stable_hash import stable_seed
from discovery.types import CreationMethod

SYNTHETIC_ROBUSTNESS_FORBIDDEN = "SYNTHETIC_ROBUSTNESS_FORBIDDEN"
ROBUSTNESS_PARAMETER_NOT_BOUND = "ROBUSTNESS_PARAMETER_NOT_BOUND"


class ParameterNotBoundError(RuntimeError):
    """Raised when a parameter name is not present in any executable tree."""


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
    bound_tree_paths: tuple[str, ...] = ()

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
            "bound_tree_paths": list(self.bound_tree_paths),
        }


def _replace_param_with_paths(
    tree: ExprNode, name: str, value: float, *, path: str
) -> tuple[ExprNode, list[str]]:
    """Replace every PARAMETER node named ``name`` anywhere in ``tree``.

    Returns ``(new_tree, changed_paths)`` where ``changed_paths`` records the
    dotted location of every replaced node (e.g. ``"stop.0"``), so callers can
    prove a perturbation actually touched the executable tree.
    """
    if tree.kind is NodeKind.PARAMETER and tree.name == name:
        return parameter_node(name, value), [path]
    if not tree.children:
        return tree, []
    new_children: list[ExprNode] = []
    changed: list[str] = []
    any_changed = False
    for i, child in enumerate(tree.children):
        new_child, child_changed = _replace_param_with_paths(
            child, name, value, path=f"{path}.{i}"
        )
        new_children.append(new_child)
        if child_changed:
            changed.extend(child_changed)
            any_changed = True
    if not any_changed:
        return tree, []
    return (
        ExprNode(tree.kind, tree.value_type, tree.name, tuple(new_children), dict(tree.meta)),
        changed,
    )


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
    grammar: Grammar = field(default_factory=Grammar)

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
            # Verify the parameter is bound to at least one executable tree
            # before running any backend evaluation for it.
            try:
                self._with_param(candidate, name, base)
            except ParameterNotBoundError:
                results.append(
                    RobustnessResult(
                        parameter=name,
                        base_value=float(base),
                        tested_value=float(base),
                        candidate_id=candidate.candidate_id,
                        neighborhood=(),
                        fitness_curve=(),
                        fold_metrics=(),
                        total_oos_trades=0,
                        median_expectancy=0.0,
                        median_profit_factor=0.0,
                        worst_drawdown=0.0,
                        fitness=0.0,
                        plateau_score=0.0,
                        cliff_detected=False,
                        spike_detected=False,
                        accepted=False,
                        reason=ROBUSTNESS_PARAMETER_NOT_BOUND,
                        backend_kind=backend_kind,
                        points=(),
                        bound_tree_paths=(),
                    )
                )
                continue

            neighborhood: list[float] = []
            curve: list[float] = []
            points: list[dict[str, Any]] = []
            all_folds: list[dict[str, Any]] = []
            all_changed_paths: list[str] = []
            for step in self.relative_steps:
                if abs(base) < 1e-12:
                    val = base + step * self.absolute_eps * 10
                else:
                    val = base * (1.0 + step)
                neighborhood.append(val)
                perturbed, changed_paths = self._with_param(candidate, name, val)
                all_changed_paths.extend(changed_paths)

                # A perturbed value can walk a threshold outside its feature's
                # semantic domain (e.g. an ATR multiplier or z-score constant
                # pushed past its valid range). Robustness perturbation is a
                # path that generation-time checks never see, so it must be
                # validated here too — before any real backend evaluation.
                violations = collect_candidate_threshold_violations(perturbed, self.grammar)
                if violations:
                    curve.append(0.0)
                    points.append(
                        {
                            "parameter": name,
                            "base_value": float(base),
                            "tested_value": float(val),
                            "candidate_id": perturbed.candidate_id,
                            "fold_metrics": [],
                            "total_oos_trades": 0,
                            "median_expectancy": 0.0,
                            "median_profit_factor": 0.0,
                            "worst_drawdown": 0.0,
                            "fitness": 0.0,
                            "backend_kind": backend_kind,
                            "changed_tree_paths": list(changed_paths),
                            "rejection_reason": INVALID_FEATURE_THRESHOLD_DOMAIN,
                            "threshold_domain_violations": [v.as_dict() for v in violations],
                        }
                    )
                    continue

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
                        "changed_tree_paths": list(changed_paths),
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
                    bound_tree_paths=tuple(sorted(set(all_changed_paths))),
                )
            )
        return results

    def _with_param(
        self, candidate: StrategyCandidate, name: str, value: float
    ) -> tuple[StrategyCandidate, tuple[str, ...]]:
        """Replace ``name`` in every executable tree that references it.

        Covers ``entry_tree``, ``exit_tree``, ``stop``, ``target``, ``sizing``,
        and every ``regime_gate``. Raises :class:`ParameterNotBoundError` (surfaced
        by callers as ``ROBUSTNESS_PARAMETER_NOT_BOUND``) if the parameter is not
        found in any of them.
        """
        changed_paths: list[str] = []

        def _replace(tree: ExprNode | None, label: str) -> ExprNode | None:
            if tree is None:
                return None
            new_tree, paths = _replace_param_with_paths(tree, name, value, path=label)
            changed_paths.extend(paths)
            return new_tree

        entry = _replace(candidate.entry_tree, "entry_tree")
        exit_tree = _replace(candidate.exit_tree, "exit_tree")
        stop = _replace(candidate.stop, "stop")
        target = _replace(candidate.target, "target")
        sizing = _replace(candidate.sizing, "sizing")
        regime_gates = tuple(
            _replace(gate, f"regime_gate[{i}]") for i, gate in enumerate(candidate.regime_gates)
        )

        if not changed_paths:
            raise ParameterNotBoundError(
                f"{ROBUSTNESS_PARAMETER_NOT_BOUND}: parameter {name!r} is not bound to any "
                "executable tree (entry_tree/exit_tree/stop/target/sizing/regime_gate)"
            )

        new_candidate = build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            stop=stop,
            target=target,
            sizing=sizing,
            regime_gates=regime_gates,
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
                "robustness_changed_tree_paths": list(changed_paths),
            },
        )
        return new_candidate, tuple(changed_paths)

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
