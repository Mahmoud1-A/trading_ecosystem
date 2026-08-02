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
    validate_tree_threshold_domains,
)
from discovery.fitness import RobustFitness
from discovery.grammar import Grammar
from discovery.prechecks import structural_precheck
from discovery.stable_hash import stable_seed
from discovery.typecheck import InvalidDslTypeError, check_ast_types
from discovery.types import CreationMethod

SYNTHETIC_ROBUSTNESS_FORBIDDEN = "SYNTHETIC_ROBUSTNESS_FORBIDDEN"
ROBUSTNESS_PARAMETER_NOT_BOUND = "ROBUSTNESS_PARAMETER_NOT_BOUND"
REAL_ROBUSTNESS_BACKEND_REQUIRED = "REAL_ROBUSTNESS_BACKEND_REQUIRED"
ROBUSTNESS_SIGNAL_SOURCE_INVALID = "ROBUSTNESS_SIGNAL_SOURCE_INVALID"
ROBUSTNESS_BACKEND_KIND_INVALID = "ROBUSTNESS_BACKEND_KIND_INVALID"
ROBUSTNESS_WFO_INCOMPLETE = "ROBUSTNESS_WFO_INCOMPLETE"
ROBUSTNESS_INTEGRITY_FAILED = "ROBUSTNESS_INTEGRITY_FAILED"
ROBUSTNESS_POINT_NOT_APPLICABLE = "ROBUSTNESS_POINT_NOT_APPLICABLE"
ROBUSTNESS_POINT_EVALUATED = "ROBUSTNESS_POINT_EVALUATED"
ROBUSTNESS_POINT_INTEGRITY_FAILED = "ROBUSTNESS_POINT_INTEGRITY_FAILED"
ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD = "ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD"
ROBUSTNESS_BUDGET_EXHAUSTED = "ROBUSTNESS_BUDGET_EXHAUSTED"

PARAMETER_ROBUST = "PARAMETER_ROBUST"
PARAMETER_UNSTABLE = "PARAMETER_UNSTABLE"
PARAMETER_NOT_BOUND = "PARAMETER_NOT_BOUND"
PARAMETER_INSUFFICIENT_NEIGHBORHOOD = "PARAMETER_INSUFFICIENT_NEIGHBORHOOD"
PARAMETER_INTEGRITY_FAILED = "PARAMETER_INTEGRITY_FAILED"
PARAMETER_BUDGET_NOT_REACHED = "PARAMETER_BUDGET_NOT_REACHED"

_REQUIRED_SIGNAL_SOURCE = "candidate_dsl_trees"
_REQUIRED_EVENT_BACKEND_KIND = "event_driven_wfo"


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
    final_parameter_decision: str = ""
    valid_evaluated_steps: tuple[float, ...] = ()
    not_applicable_steps: tuple[float, ...] = ()
    integrity_failed_steps: tuple[float, ...] = ()
    backend_call_count: int = 0
    expectancy_curve: tuple[float, ...] = ()
    pf_curve: tuple[float, ...] = ()
    drawdown_curve: tuple[float, ...] = ()
    trade_count_curve: tuple[float, ...] = ()
    minimum_valid_neighborhood_proof: dict[str, Any] = field(default_factory=dict)

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
            "final_parameter_decision": self.final_parameter_decision,
            "valid_evaluated_steps": list(self.valid_evaluated_steps),
            "not_applicable_steps": list(self.not_applicable_steps),
            "integrity_failed_steps": list(self.integrity_failed_steps),
            "backend_call_count": self.backend_call_count,
            "expectancy_curve": list(self.expectancy_curve),
            "pf_curve": list(self.pf_curve),
            "drawdown_curve": list(self.drawdown_curve),
            "trade_count_curve": list(self.trade_count_curve),
            "minimum_valid_neighborhood_proof": dict(self.minimum_valid_neighborhood_proof),
        }


@dataclass(frozen=True)
class ParameterRobustnessSummary:
    """Parameter-level robustness decision artifact (Phase 3B.2)."""

    parameter: str
    base_value: float
    bound_executable_paths: tuple[str, ...]
    planned_steps: tuple[float, ...]
    valid_evaluated_steps: tuple[float, ...]
    not_applicable_steps: tuple[float, ...]
    integrity_failed_steps: tuple[float, ...]
    backend_call_count: int
    fitness_curve: tuple[float, ...]
    expectancy_curve: tuple[float, ...]
    pf_curve: tuple[float, ...]
    drawdown_curve: tuple[float, ...]
    trade_count_curve: tuple[float, ...]
    plateau_score: float
    cliff_detected: bool
    isolated_spike_detected: bool
    minimum_valid_neighborhood_proof: dict[str, Any]
    final_parameter_decision: str
    final_reason: str
    points: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "base_value": self.base_value,
            "bound_executable_paths": list(self.bound_executable_paths),
            "planned_steps": list(self.planned_steps),
            "valid_evaluated_steps": list(self.valid_evaluated_steps),
            "not_applicable_steps": list(self.not_applicable_steps),
            "integrity_failed_steps": list(self.integrity_failed_steps),
            "backend_call_count": self.backend_call_count,
            "fitness_curve": list(self.fitness_curve),
            "expectancy_curve": list(self.expectancy_curve),
            "pf_curve": list(self.pf_curve),
            "drawdown_curve": list(self.drawdown_curve),
            "trade_count_curve": list(self.trade_count_curve),
            "plateau_score": self.plateau_score,
            "cliff_detected": self.cliff_detected,
            "isolated_spike_detected": self.isolated_spike_detected,
            "minimum_valid_neighborhood_proof": dict(self.minimum_valid_neighborhood_proof),
            "final_parameter_decision": self.final_parameter_decision,
            "final_reason": self.final_reason,
            "points": list(self.points),
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


def _extract_train_proof(train_metrics: Any, backend: Any) -> dict[str, Any]:
    """Read integrity evidence from returned train_metrics (never hardcoded)."""
    train = dict(train_metrics) if isinstance(train_metrics, dict) else {}
    arts = {}
    last = getattr(backend, "last_run_artifacts", None)
    if isinstance(last, dict):
        arts = last
    signal_source = str(train.get("signal_source") or arts.get("signal_source") or "")
    backend_kind = str(
        train.get("backend_kind")
        or arts.get("backend_kind")
        or getattr(backend, "backend_kind", type(backend).__name__)
    )
    is_full = bool(
        train.get(
            "is_full_event_wfo",
            arts.get("is_full_event_wfo", getattr(backend, "is_full_event_wfo", False)),
        )
    )
    completed = int(
        train.get("wfo_completed_folds")
        or arts.get("wfo_completed_folds")
        or 0
    )
    return {
        "signal_source": signal_source,
        "backend_kind": backend_kind,
        "is_full_event_wfo": is_full,
        "completed_fold_count": completed,
        "orders_count": int(train.get("orders_count") or arts.get("orders_count") or 0),
        "fills_count": int(train.get("fills_count") or arts.get("fills_count") or 0),
        "trades_count": int(train.get("trades_count") or arts.get("trades_count") or 0),
        "artifact_candidate_id": str(
            train.get("candidate_id") or arts.get("candidate_id") or ""
        ),
    }


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
    expected_backend_kind: str = _REQUIRED_EVENT_BACKEND_KIND
    min_valid_neighborhood_points: int = 3
    allow_one_sided_neighborhood: bool = False
    last_probe_accounting: dict[str, Any] = field(default_factory=dict)
    last_parameter_summaries: list[ParameterRobustnessSummary] = field(default_factory=list)

    def probe(
        self,
        candidate: StrategyCandidate,
        *,
        parameter_names: tuple[str, ...] | list[str] | None = None,
        family_spec: Any | None = None,
        counters: Any | None = None,
        max_evaluations: int | None = None,
        require_integrity: bool | None = None,
    ) -> list[RobustnessResult]:
        forbid = bool(self.synthetic_robustness_forbidden or self.research_eligible)
        integrity_required = (
            bool(require_integrity)
            if require_integrity is not None
            else bool(forbid)
        )
        if forbid and isinstance(self.backend, SyntheticOOSBackend):
            raise RuntimeError(
                f"{SYNTHETIC_ROBUSTNESS_FORBIDDEN}: research-eligible robustness "
                "cannot use SyntheticOOSBackend"
            )
        if forbid and getattr(self.backend, "backend_kind", "") == "synthetic_oos_probe":
            raise RuntimeError(SYNTHETIC_ROBUSTNESS_FORBIDDEN)
        if forbid:
            kind = str(getattr(self.backend, "backend_kind", ""))
            # The existing integrity contract accepts either the concrete real
            # backend or an injected backend declaring the exact expected kind.
            # Avoid importing the local data stack when that declaration already
            # satisfies the gate; import only for the concrete-class fallback.
            if kind != self.expected_backend_kind:
                from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

                if not isinstance(self.backend, EventDrivenDiscoveryBackend):
                    raise RuntimeError(REAL_ROBUSTNESS_BACKEND_REQUIRED)

        backend_kind = str(getattr(self.backend, "backend_kind", type(self.backend).__name__))
        names = (
            list(parameter_names)
            if parameter_names is not None
            else sorted(candidate.parameters.keys())
        )
        results: list[RobustnessResult] = []
        summaries: list[ParameterRobustnessSummary] = []
        total_backend_calls = 0
        stop_reason: str | None = None

        for name in names:
            base = float(candidate.parameters.get(name, 0.0)) if name in candidate.parameters else None
            if name not in candidate.parameters:
                # Still attempt bind check for explicit names.
                base = 0.0
            try:
                _, base_paths = self._with_param(candidate, name, float(base))
            except ParameterNotBoundError:
                unbound = self._unbound_result(candidate, name, float(base or 0.0), backend_kind)
                results.append(unbound)
                summaries.append(self._summary_from_result(unbound))
                continue

            planned_steps = tuple(float(s) for s in self.relative_steps)
            points: list[dict[str, Any]] = []
            curve: list[float] = []
            exp_curve: list[float] = []
            pf_curve: list[float] = []
            dd_curve: list[float] = []
            trade_curve: list[float] = []
            valid_steps: list[float] = []
            na_steps: list[float] = []
            integ_fail_steps: list[float] = []
            neighborhood_vals: list[float] = []
            all_folds: list[dict[str, Any]] = []
            all_changed_paths: list[str] = list(base_paths)
            param_backend_calls = 0
            budget_truncated = False
            integrity_hard_fail = False

            for step in planned_steps:
                if abs(base) < 1e-12:
                    val = float(base) + float(step) * self.absolute_eps * 10
                else:
                    val = float(base) * (1.0 + float(step))
                neighborhood_vals.append(val)

                if max_evaluations is not None and counters is not None:
                    consumed = int(getattr(counters, "robustness", 0))
                    if consumed >= int(max_evaluations):
                        budget_truncated = True
                        stop_reason = ROBUSTNESS_BUDGET_EXHAUSTED
                        break

                try:
                    perturbed, changed_paths = self._with_param(candidate, name, val)
                except ParameterNotBoundError:
                    points.append(
                        self._point_record(
                            original_candidate_id=candidate.candidate_id,
                            perturbed_candidate_id=candidate.candidate_id,
                            parameter=name,
                            base_value=float(base),
                            requested_value=float(val),
                            actual_bound_value=None,
                            relative_step=float(step),
                            changed_tree_paths=(),
                            point_status=ROBUSTNESS_POINT_NOT_APPLICABLE,
                            rejection_reason=ROBUSTNESS_PARAMETER_NOT_BOUND,
                            integrity_ok=False,
                            backend_kind=backend_kind,
                        )
                    )
                    na_steps.append(float(step))
                    continue

                all_changed_paths.extend(changed_paths)
                actual_bound = float(perturbed.parameters.get(name, val))
                pre_reason = self._pre_eval_rejection(
                    perturbed,
                    family_spec=family_spec,
                    requested_value=float(val),
                    parameter=name,
                    changed_paths=changed_paths,
                )
                if pre_reason is not None:
                    points.append(
                        self._point_record(
                            original_candidate_id=candidate.candidate_id,
                            perturbed_candidate_id=perturbed.candidate_id,
                            parameter=name,
                            base_value=float(base),
                            requested_value=float(val),
                            actual_bound_value=actual_bound,
                            relative_step=float(step),
                            changed_tree_paths=changed_paths,
                            point_status=ROBUSTNESS_POINT_NOT_APPLICABLE,
                            rejection_reason=pre_reason,
                            integrity_ok=True,
                            backend_kind=backend_kind,
                        )
                    )
                    na_steps.append(float(step))
                    continue

                # Consume exactly one robustness budget unit per backend.evaluate.
                if counters is not None:
                    counters.robustness = int(getattr(counters, "robustness", 0)) + 1
                folds, train_metrics = self.backend.evaluate(perturbed)
                param_backend_calls += 1
                total_backend_calls += 1
                proof = _extract_train_proof(train_metrics, self.backend)
                fit = self.fitness_model.score(folds, complexity=perturbed.complexity_score)
                fold_dicts = [f.as_dict() for f in folds]
                all_folds.extend(fold_dicts)
                n_trades = int(sum(f.n_trades for f in folds))
                med_exp = (
                    float(np.median([f.expectancy for f in folds])) if folds else 0.0
                )
                med_pf = (
                    float(np.median([f.profit_factor for f in folds])) if folds else 0.0
                )
                worst_dd = float(min((f.max_drawdown for f in folds), default=0.0))

                integrity_reasons: list[str] = []
                if integrity_required:
                    if proof["signal_source"] != _REQUIRED_SIGNAL_SOURCE:
                        integrity_reasons.append(ROBUSTNESS_SIGNAL_SOURCE_INVALID)
                    if proof["backend_kind"] != self.expected_backend_kind:
                        integrity_reasons.append(ROBUSTNESS_BACKEND_KIND_INVALID)
                    if not proof["is_full_event_wfo"]:
                        integrity_reasons.append(ROBUSTNESS_WFO_INCOMPLETE)
                    if int(proof["completed_fold_count"]) <= 0:
                        integrity_reasons.append(ROBUSTNESS_WFO_INCOMPLETE)
                    art_cid = proof["artifact_candidate_id"]
                    if art_cid and art_cid != perturbed.candidate_id:
                        integrity_reasons.append(ROBUSTNESS_INTEGRITY_FAILED)
                    if not changed_paths:
                        integrity_reasons.append(ROBUSTNESS_PARAMETER_NOT_BOUND)
                    else:
                        # Requested value must appear on a changed path's parameter.
                        if abs(actual_bound - float(val)) > 1e-9:
                            integrity_reasons.append(ROBUSTNESS_INTEGRITY_FAILED)

                if integrity_reasons:
                    integrity_hard_fail = True
                    integ_fail_steps.append(float(step))
                    points.append(
                        self._point_record(
                            original_candidate_id=candidate.candidate_id,
                            perturbed_candidate_id=perturbed.candidate_id,
                            parameter=name,
                            base_value=float(base),
                            requested_value=float(val),
                            actual_bound_value=actual_bound,
                            relative_step=float(step),
                            changed_tree_paths=changed_paths,
                            signal_source=proof["signal_source"],
                            backend_kind=proof["backend_kind"],
                            is_full_event_wfo=proof["is_full_event_wfo"],
                            completed_fold_count=proof["completed_fold_count"],
                            orders_count=proof["orders_count"],
                            fills_count=proof["fills_count"],
                            trades_count=proof["trades_count"],
                            fold_metrics=fold_dicts,
                            total_oos_trades=n_trades,
                            median_expectancy=med_exp,
                            median_profit_factor=med_pf,
                            worst_drawdown=worst_dd,
                            fitness=float(fit.fitness),
                            point_status=ROBUSTNESS_POINT_INTEGRITY_FAILED,
                            rejection_reason=integrity_reasons[0],
                            integrity_ok=False,
                        )
                    )
                    continue

                valid_steps.append(float(step))
                curve.append(float(fit.fitness))
                exp_curve.append(med_exp)
                pf_curve.append(med_pf)
                dd_curve.append(worst_dd)
                trade_curve.append(float(n_trades))
                points.append(
                    self._point_record(
                        original_candidate_id=candidate.candidate_id,
                        perturbed_candidate_id=perturbed.candidate_id,
                        parameter=name,
                        base_value=float(base),
                        requested_value=float(val),
                        actual_bound_value=actual_bound,
                        relative_step=float(step),
                        changed_tree_paths=changed_paths,
                        signal_source=proof["signal_source"],
                        backend_kind=proof["backend_kind"],
                        is_full_event_wfo=proof["is_full_event_wfo"],
                        completed_fold_count=proof["completed_fold_count"],
                        orders_count=proof["orders_count"],
                        fills_count=proof["fills_count"],
                        trades_count=proof["trades_count"],
                        fold_metrics=fold_dicts,
                        total_oos_trades=n_trades,
                        median_expectancy=med_exp,
                        median_profit_factor=med_pf,
                        worst_drawdown=worst_dd,
                        fitness=float(fit.fitness),
                        point_status=ROBUSTNESS_POINT_EVALUATED,
                        rejection_reason="",
                        integrity_ok=True,
                    )
                )

            neighborhood_proof = self._neighborhood_proof(
                planned_steps=planned_steps,
                valid_steps=tuple(valid_steps),
                na_steps=tuple(na_steps),
            )
            decision, reason, accepted, plateau, cliff, spike, center = self._decide_parameter(
                base=float(base),
                planned_steps=planned_steps,
                valid_steps=valid_steps,
                curve=curve,
                integrity_hard_fail=integrity_hard_fail,
                budget_truncated=budget_truncated,
                neighborhood_proof=neighborhood_proof,
            )
            result = RobustnessResult(
                parameter=name,
                base_value=float(base),
                tested_value=float(base),
                candidate_id=candidate.candidate_id,
                neighborhood=tuple(float(x) for x in neighborhood_vals),
                fitness_curve=tuple(float(x) for x in curve),
                fold_metrics=tuple(all_folds),
                total_oos_trades=int(
                    sum(int(p.get("total_oos_trades") or 0) for p in points if p.get("integrity_ok"))
                ),
                median_expectancy=float(np.median(exp_curve)) if exp_curve else 0.0,
                median_profit_factor=float(np.median(pf_curve)) if pf_curve else 0.0,
                worst_drawdown=float(min(dd_curve)) if dd_curve else 0.0,
                fitness=float(center),
                plateau_score=plateau,
                cliff_detected=cliff,
                spike_detected=spike,
                accepted=accepted,
                reason=reason,
                backend_kind=backend_kind,
                points=tuple(points),
                bound_tree_paths=tuple(sorted(set(all_changed_paths))),
                final_parameter_decision=decision,
                valid_evaluated_steps=tuple(valid_steps),
                not_applicable_steps=tuple(na_steps),
                integrity_failed_steps=tuple(integ_fail_steps),
                backend_call_count=param_backend_calls,
                expectancy_curve=tuple(exp_curve),
                pf_curve=tuple(pf_curve),
                drawdown_curve=tuple(dd_curve),
                trade_count_curve=tuple(trade_curve),
                minimum_valid_neighborhood_proof=neighborhood_proof,
            )
            results.append(result)
            summaries.append(self._summary_from_result(result))
            if budget_truncated:
                break

        self.last_parameter_summaries = summaries
        self.last_probe_accounting = {
            "robustness_backend_calls": total_backend_calls,
            "robustness_counter_delta": (
                int(getattr(counters, "robustness", 0)) if counters is not None else total_backend_calls
            ),
            "max_robustness_evaluations": max_evaluations,
            "stop_reason": stop_reason,
            "parameter_count": len(results),
        }
        # When counters started mid-probe, delta for this probe is backend calls.
        if counters is not None:
            # Accounting for this probe alone uses backend call count.
            self.last_probe_accounting["robustness_counter_delta"] = total_backend_calls
        return results

    def _pre_eval_rejection(
        self,
        perturbed: StrategyCandidate,
        *,
        family_spec: Any | None,
        requested_value: float,
        parameter: str,
        changed_paths: tuple[str, ...],
    ) -> str | None:
        _ = requested_value
        _ = parameter
        _ = changed_paths
        trees: list[tuple[str, ExprNode | None]] = [
            ("entry_tree", perturbed.entry_tree),
            ("exit_tree", perturbed.exit_tree),
            ("stop", perturbed.stop),
            ("target", perturbed.target),
            ("sizing", perturbed.sizing),
            *[(f"regime_gate[{i}]", g) for i, g in enumerate(perturbed.regime_gates)],
        ]
        for label, tree in trees:
            if tree is None:
                continue
            try:
                check_ast_types(tree, path=label, candidate_id=perturbed.candidate_id)
            except InvalidDslTypeError as exc:
                return f"INVALID_DSL_TYPE:{exc.message}"

        pre = structural_precheck(
            entry=perturbed.entry_tree,
            exit=perturbed.exit_tree,
            regime_gates=perturbed.regime_gates,
            grammar=self.grammar,
            seed=int(perturbed.random_seed or 0),
        )
        if not pre.accepted:
            return f"structural_precheck:{pre.reason}"

        violations = collect_candidate_threshold_violations(perturbed, self.grammar)
        if violations:
            return INVALID_FEATURE_THRESHOLD_DOMAIN
        for label, tree in trees:
            if tree is None:
                continue
            domain_reason = validate_tree_threshold_domains(tree, self.grammar)
            if domain_reason is not None:
                return domain_reason

        if family_spec is not None:
            from discovery.multi_family_campaign import (
                FAMILY_DIRECTION_INCOHERENT,
                MultiFamilyCampaign,
                validate_family_direction_coherence,
            )

            grammar = (
                family_spec.to_grammar()
                if hasattr(family_spec, "to_grammar")
                else self.grammar
            )
            if not MultiFamilyCampaign.candidate_within_family_grammar(perturbed, grammar):
                return "FAMILY_GRAMMAR_VIOLATION"
            coherence = validate_family_direction_coherence(perturbed, family_spec)
            if not coherence.coherent:
                return coherence.rejection_reason or FAMILY_DIRECTION_INCOHERENT
        return None

    def _neighborhood_proof(
        self,
        *,
        planned_steps: tuple[float, ...],
        valid_steps: tuple[float, ...],
        na_steps: tuple[float, ...],
    ) -> dict[str, Any]:
        has_center = 0.0 in valid_steps
        lower_planned = [s for s in planned_steps if s < 0.0]
        upper_planned = [s for s in planned_steps if s > 0.0]
        lower_valid = [s for s in valid_steps if s < 0.0]
        upper_valid = [s for s in valid_steps if s > 0.0]
        lower_all_na = bool(lower_planned) and all(s in na_steps for s in lower_planned)
        upper_all_na = bool(upper_planned) and all(s in na_steps for s in upper_planned)
        one_sided_ok = False
        if self.allow_one_sided_neighborhood:
            if lower_all_na and upper_valid and has_center:
                one_sided_ok = True
            if upper_all_na and lower_valid and has_center:
                one_sided_ok = True
        enough = len(valid_steps) >= int(self.min_valid_neighborhood_points)
        sides_ok = (bool(lower_valid) and bool(upper_valid)) or one_sided_ok
        # If domain has no lower planned steps, only require upper (+ center).
        if not lower_planned and upper_valid and has_center:
            sides_ok = True
        if not upper_planned and lower_valid and has_center:
            sides_ok = True
        sufficient = bool(has_center and enough and sides_ok)
        return {
            "has_center": has_center,
            "lower_valid_count": len(lower_valid),
            "upper_valid_count": len(upper_valid),
            "valid_point_count": len(valid_steps),
            "min_required_points": int(self.min_valid_neighborhood_points),
            "one_sided_allowed": bool(self.allow_one_sided_neighborhood),
            "one_sided_applied": one_sided_ok,
            "sufficient": sufficient,
        }

    def _decide_parameter(
        self,
        *,
        base: float,
        planned_steps: tuple[float, ...],
        valid_steps: list[float],
        curve: list[float],
        integrity_hard_fail: bool,
        budget_truncated: bool,
        neighborhood_proof: dict[str, Any],
    ) -> tuple[str, str, bool, float, bool, bool, float]:
        _ = base
        _ = planned_steps
        if integrity_hard_fail:
            return PARAMETER_INTEGRITY_FAILED, ROBUSTNESS_INTEGRITY_FAILED, False, 0.0, False, False, 0.0
        if budget_truncated and not neighborhood_proof.get("sufficient"):
            return PARAMETER_BUDGET_NOT_REACHED, ROBUSTNESS_BUDGET_EXHAUSTED, False, 0.0, False, False, 0.0
        if not neighborhood_proof.get("sufficient"):
            return (
                PARAMETER_INSUFFICIENT_NEIGHBORHOOD,
                ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD,
                False,
                0.0,
                False,
                False,
                0.0,
            )

        arr = np.asarray(curve, dtype=float)
        peak = float(np.max(arr)) if len(arr) else 0.0
        median = float(np.median(arr)) if len(arr) else 0.0
        # Center fitness: prefer the evaluated 0.0 step.
        if 0.0 in valid_steps:
            center_idx = valid_steps.index(0.0)
            center = float(arr[center_idx]) if len(arr) else 0.0
            neighbors = np.delete(arr, center_idx) if len(arr) else arr
        else:
            center = float(arr[len(arr) // 2]) if len(arr) else 0.0
            neighbors = arr
        cliff = bool(np.any(neighbors < center - abs(center) * self.cliff_drop - 1e-9)) if len(neighbors) else False
        spike = bool(center > median * self.spike_ratio + 1e-9 and center > peak * 0.95)
        plateau = 1.0 / (1.0 + float(np.std(arr))) if len(arr) else 0.0
        accepted = not cliff and not spike and plateau >= 0.2
        if cliff:
            return PARAMETER_UNSTABLE, "performance_cliff", False, plateau, True, spike, center
        if spike:
            return PARAMETER_UNSTABLE, "isolated_spike", False, plateau, cliff, True, center
        if not accepted:
            return PARAMETER_UNSTABLE, "unstable_plateau", False, plateau, cliff, spike, center
        return PARAMETER_ROBUST, "parameter_plateau_ok", True, plateau, False, False, center

    @staticmethod
    def _point_record(
        *,
        original_candidate_id: str,
        perturbed_candidate_id: str,
        parameter: str,
        base_value: float,
        requested_value: float,
        actual_bound_value: float | None,
        relative_step: float,
        changed_tree_paths: tuple[str, ...] | list[str],
        point_status: str,
        rejection_reason: str,
        integrity_ok: bool,
        backend_kind: str = "",
        signal_source: str = "",
        is_full_event_wfo: bool = False,
        completed_fold_count: int = 0,
        orders_count: int = 0,
        fills_count: int = 0,
        trades_count: int = 0,
        fold_metrics: list[dict[str, Any]] | None = None,
        total_oos_trades: int = 0,
        median_expectancy: float = 0.0,
        median_profit_factor: float = 0.0,
        worst_drawdown: float = 0.0,
        fitness: float | None = None,
    ) -> dict[str, Any]:
        return {
            "original_candidate_id": original_candidate_id,
            "perturbed_candidate_id": perturbed_candidate_id,
            "parameter": parameter,
            "base_value": float(base_value),
            "requested_value": float(requested_value),
            "actual_bound_value": (
                None if actual_bound_value is None else float(actual_bound_value)
            ),
            "relative_step": float(relative_step),
            "changed_tree_paths": list(changed_tree_paths),
            "signal_source": signal_source,
            "backend_kind": backend_kind,
            "is_full_event_wfo": bool(is_full_event_wfo),
            "completed_fold_count": int(completed_fold_count),
            "orders_count": int(orders_count),
            "fills_count": int(fills_count),
            "trades_count": int(trades_count),
            "fold_metrics": list(fold_metrics or []),
            "total_oos_trades": int(total_oos_trades),
            "median_expectancy": float(median_expectancy),
            "median_profit_factor": float(median_profit_factor),
            "worst_drawdown": float(worst_drawdown),
            "fitness": fitness,
            "point_status": point_status,
            "rejection_reason": rejection_reason,
            "integrity_ok": bool(integrity_ok),
            # Backward-compatible aliases used by older callers/tests.
            "tested_value": float(requested_value),
            "candidate_id": perturbed_candidate_id,
        }

    def _unbound_result(
        self, candidate: StrategyCandidate, name: str, base: float, backend_kind: str
    ) -> RobustnessResult:
        return RobustnessResult(
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
            final_parameter_decision=PARAMETER_NOT_BOUND,
            valid_evaluated_steps=(),
            not_applicable_steps=(),
            integrity_failed_steps=(),
            backend_call_count=0,
            minimum_valid_neighborhood_proof={"sufficient": False, "reason": "unbound"},
        )

    @staticmethod
    def _summary_from_result(result: RobustnessResult) -> ParameterRobustnessSummary:
        return ParameterRobustnessSummary(
            parameter=result.parameter,
            base_value=result.base_value,
            bound_executable_paths=result.bound_tree_paths,
            planned_steps=tuple(
                float(p.get("relative_step"))
                for p in result.points
                if "relative_step" in p
            )
            or tuple(result.neighborhood),
            valid_evaluated_steps=result.valid_evaluated_steps,
            not_applicable_steps=result.not_applicable_steps,
            integrity_failed_steps=result.integrity_failed_steps,
            backend_call_count=result.backend_call_count,
            fitness_curve=result.fitness_curve,
            expectancy_curve=result.expectancy_curve,
            pf_curve=result.pf_curve,
            drawdown_curve=result.drawdown_curve,
            trade_count_curve=result.trade_count_curve,
            plateau_score=result.plateau_score,
            cliff_detected=result.cliff_detected,
            isolated_spike_detected=result.spike_detected,
            minimum_valid_neighborhood_proof=dict(result.minimum_valid_neighborhood_proof),
            final_parameter_decision=result.final_parameter_decision
            or (PARAMETER_ROBUST if result.accepted else PARAMETER_UNSTABLE),
            final_reason=result.reason,
            points=result.points,
        )

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
