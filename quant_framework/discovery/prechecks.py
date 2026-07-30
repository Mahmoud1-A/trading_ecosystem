"""Limit checks, constant-signal rejection, and cheap structural prechecks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discovery.expression_tree import DSLValidationError, ExprNode, eval_scalar
from discovery.grammar import Grammar, GrammarLimits
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.types import NodeKind, ValueType


@dataclass(frozen=True)
class PrecheckResult:
    accepted: bool
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"accepted": self.accepted, "reason": self.reason}


def validate_limits(tree: ExprNode, limits: GrammarLimits) -> None:
    if tree.depth > limits.max_tree_depth:
        raise DSLValidationError(
            f"tree depth {tree.depth} exceeds max_tree_depth={limits.max_tree_depth}"
        )
    if tree.n_nodes > limits.max_nodes:
        raise DSLValidationError(
            f"node count {tree.n_nodes} exceeds max_nodes={limits.max_nodes}"
        )
    features = tree.feature_ids()
    if len(features) > limits.max_distinct_features:
        raise DSLValidationError(
            f"distinct features {len(features)} exceeds max={limits.max_distinct_features}"
        )
    params = tree.parameter_names()
    if len(params) > limits.max_free_parameters:
        raise DSLValidationError(
            f"parameters {len(params)} exceeds max={limits.max_free_parameters}"
        )
    for node in tree.walk():
        if node.kind is NodeKind.OPERATOR:
            spec = OPERATOR_REGISTRY[OperatorId(node.name)]
            if spec.lookback_child is not None:
                child = node.children[spec.lookback_child]
                if child.kind in {NodeKind.CONSTANT, NodeKind.PARAMETER}:
                    lb = float(child.meta.get("value", child.meta.get("default", 0)))
                    if lb > limits.max_rolling_lookback or lb < 1:
                        raise DSLValidationError(
                            f"lookback {lb} outside [1, {limits.max_rolling_lookback}]"
                        )


def count_ops(tree: ExprNode, op: OperatorId) -> int:
    return sum(1 for n in tree.walk() if n.kind is NodeKind.OPERATOR and n.name == op.value)


def validate_strategy_structure(
    *,
    entry: ExprNode,
    exit: ExprNode | None,
    regime_gates: tuple[ExprNode, ...],
    grammar: Grammar,
) -> None:
    limits = grammar.limits
    for tree in (entry, exit, *regime_gates):
        if tree is not None:
            validate_limits(tree, limits)
    if entry.value_type not in {ValueType.BOOLEAN, ValueType.ORDER_INTENT}:
        raise DSLValidationError("entry tree must produce BOOLEAN or ORDER_INTENT")
    if count_ops(entry, OperatorId.ENTRY_LONG) + count_ops(entry, OperatorId.ENTRY_SHORT) > limits.max_entry_conditions:
        raise DSLValidationError("too many entry conditions")
    if exit is not None:
        exits = (
            count_ops(exit, OperatorId.EXIT_SIGNAL)
            + count_ops(exit, OperatorId.TIME_EXIT)
            + count_ops(exit, OperatorId.SESSION_EXIT)
        )
        if exits > limits.max_exit_conditions:
            raise DSLValidationError("too many exit conditions")
    if len(regime_gates) > limits.max_regime_gates:
        raise DSLValidationError("too many regime gates")
    cross = 0
    for fid in entry.feature_ids():
        try:
            if grammar.leaf_for(fid).is_cross_asset:
                cross += 1
        except KeyError as exc:
            raise DSLValidationError(f"entry references unknown feature {fid!r}") from exc
    if cross > limits.max_cross_asset_inputs:
        raise DSLValidationError("too many cross-asset inputs")


def is_constant_boolean(tree: ExprNode) -> bool | None:
    """Return True/False if tree is constant boolean; None if not constant."""
    # Pure constant / param-free boolean probe
    try:
        # If any FEATURE node exists, not constant
        if any(n.kind is NodeKind.FEATURE for n in tree.walk()):
            return None
        val = eval_scalar(tree)
        if tree.value_type is ValueType.BOOLEAN or tree.value_type is ValueType.ORDER_INTENT:
            return bool(val)
        return None
    except Exception:
        return None


def probe_always_same(
    tree: ExprNode,
    *,
    feature_names: tuple[str, ...],
    n_probes: int = 32,
    seed: int = 0,
) -> bool | None:
    """
    Return True if all probes are truthy, False if all falsy, None if mixed/non-boolean.
    """
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_probes):
        bindings = {f: float(rng.normal()) for f in feature_names}
        # also bind parameters to defaults via eval_scalar
        values.append(eval_scalar(tree, bindings=bindings))
    arr = np.asarray(values)
    if not np.isfinite(arr).all():
        raise DSLValidationError("expression produced non-finite probe outputs")
    truth = arr > 0.5
    if truth.all():
        return True
    if (~truth).all():
        return False
    return None


def structural_precheck(
    *,
    entry: ExprNode,
    exit: ExprNode | None,
    regime_gates: tuple[ExprNode, ...],
    grammar: Grammar,
    seed: int = 0,
) -> PrecheckResult:
    try:
        validate_strategy_structure(
            entry=entry, exit=exit, regime_gates=regime_gates, grammar=grammar
        )
    except DSLValidationError as exc:
        return PrecheckResult(False, str(exc))

    const = is_constant_boolean(entry)
    if const is True:
        return PrecheckResult(False, "always-true entry signal")
    if const is False:
        return PrecheckResult(False, "always-false entry signal")

    probe = probe_always_same(entry, feature_names=entry.feature_ids() or ("_none",), seed=seed)
    if probe is True:
        return PrecheckResult(False, "entry always true on probe data")
    if probe is False:
        return PrecheckResult(False, "entry always false on probe data")

    # Contradictory long+short without gate
    if count_ops(entry, OperatorId.ENTRY_LONG) and count_ops(entry, OperatorId.ENTRY_SHORT):
        if OperatorId.AND.value not in entry.operator_ids() and OperatorId.OR.value in entry.operator_ids():
            # Heuristic: simultaneous unconditional long/short OR is contradictory
            return PrecheckResult(False, "contradictory entry long/short")

    # Protected division presence is fine; evaluate a divide-by-near-zero probe
    for node in entry.walk():
        if node.kind is NodeKind.OPERATOR and node.name == OperatorId.DIVIDE_PROTECTED.value:
            val = eval_scalar(
                node,
                bindings={n.name: 1.0 for n in node.walk() if n.kind is NodeKind.FEATURE},
            )
            if not np.isfinite(val):
                return PrecheckResult(False, "protected division produced non-finite value")

    return PrecheckResult(True, "ok")
