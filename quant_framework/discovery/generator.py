"""Deterministic random generation of typed strategy candidates."""

from __future__ import annotations

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.expression_tree import (
    DSLValidationError,
    ExprNode,
    constant_node,
    feature_node,
    op_node,
    parameter_node,
)
from discovery.grammar import GRAMMAR_VERSION, Grammar
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.prechecks import structural_precheck
from discovery.repair import clamp_lookbacks, repair_entry
from discovery.typecheck import check_ast_types, check_strategy_trees, parse_dsl_validation_error
from discovery.types import CreationMethod, NUMERIC_TYPES, ValueType


class CandidateGenerator:
    def __init__(
        self,
        grammar: Grammar | None = None,
        *,
        feature_set_version: str = "feature_set_v1_phase6b",
        cost_model_version: str = "cost_v1",
        strategy_family: str = "dsl_generated",
    ) -> None:
        self.grammar = grammar or Grammar()
        self.feature_set_version = feature_set_version
        self.cost_model_version = cost_model_version
        self.strategy_family = strategy_family

    def _rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _features_of_type(self, value_type: ValueType) -> list:
        return [l for l in self.grammar.feature_leaves if l.value_type is value_type]

    def _random_feature(self, rng: np.random.Generator, *, value_type: ValueType | None = None) -> ExprNode:
        leaves = list(self.grammar.feature_leaves)
        if value_type is not None:
            typed = self._features_of_type(value_type)
            if not typed:
                raise ValueError(f"no feature leaves of type {value_type}")
            leaves = typed
        leaf = leaves[int(rng.integers(0, len(leaves)))]
        return feature_node(leaf.feature_id, leaf.value_type)

    def _random_const(self, rng: np.random.Generator) -> ExprNode:
        return constant_node(float(rng.normal(0, 1)))

    def _random_param(self, rng: np.random.Generator, idx: int) -> ExprNode:
        return parameter_node(f"p{idx}", float(rng.uniform(-2, 2)))

    def _boolean_leaf(self, rng: np.random.Generator, param_counter: list[int]) -> ExprNode:
        numeric_leaves = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
        if not numeric_leaves:
            numeric_leaves = list(self.grammar.feature_leaves)
        leaf = numeric_leaves[int(rng.integers(0, len(numeric_leaves)))]
        left = feature_node(leaf.feature_id, leaf.value_type)
        right = (
            self._random_const(rng)
            if rng.random() < 0.7
            else self._random_param(rng, param_counter[0])
        )
        if right.kind.value == "PARAMETER":
            param_counter[0] += 1
        cmp_ops = (OperatorId.LESS_THAN, OperatorId.GREATER_THAN, OperatorId.LESS_EQUAL)
        return op_node(cmp_ops[int(rng.integers(0, len(cmp_ops)))], left, right)

    def _leaf(
        self,
        rng: np.random.Generator,
        *,
        target_type: ValueType,
        param_counter: list[int],
    ) -> ExprNode:
        """Return a node whose value_type exactly matches target_type."""
        if target_type is ValueType.BOOLEAN:
            return self._boolean_leaf(rng, param_counter)
        if target_type is ValueType.ORDER_INTENT:
            cond = self._leaf(rng, target_type=ValueType.BOOLEAN, param_counter=param_counter)
            return op_node(OperatorId.ENTRY_LONG, cond)
        if target_type is ValueType.REGIME:
            typed = self._features_of_type(ValueType.REGIME)
            if typed:
                return self._random_feature(rng, value_type=ValueType.REGIME)
            return constant_node(float(int(rng.integers(0, 3))), ValueType.REGIME)
        if target_type is ValueType.SCALAR:
            if self._features_of_type(ValueType.SCALAR) and rng.random() < 0.4:
                return self._random_feature(rng, value_type=ValueType.SCALAR)
            if rng.random() < 0.5:
                return self._random_const(rng)
            node = self._random_param(rng, param_counter[0])
            param_counter[0] += 1
            return node
        typed = self._features_of_type(target_type)
        if typed:
            return self._random_feature(rng, value_type=target_type)
        numeric_leaves = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
        def _num_feat() -> ExprNode:
            leaf = numeric_leaves[int(rng.integers(0, len(numeric_leaves)))]
            return feature_node(leaf.feature_id, leaf.value_type)

        if target_type is ValueType.RATIO:
            return op_node(OperatorId.DIVIDE_PROTECTED, _num_feat(), self._random_const(rng))
        if target_type is ValueType.VOLATILITY:
            return op_node(OperatorId.ROLLING_STD, _num_feat(), constant_node(20.0))
        if target_type is ValueType.RETURN:
            return op_node(OperatorId.PERCENT_CHANGE, _num_feat())
        if target_type is ValueType.RANK:
            return op_node(OperatorId.ROLLING_RANK, _num_feat(), constant_node(20.0))
        if target_type is ValueType.ZSCORE:
            return op_node(OperatorId.ROLLING_ZSCORE, _num_feat(), constant_node(20.0))
        raise DSLValidationError(f"cannot synthesize leaf for type {target_type.value}")

    def _child_for_slot(
        self,
        rng: np.random.Generator,
        *,
        allowed: frozenset[ValueType],
        max_depth: int,
        param_counter: list[int],
        lookback: bool,
    ) -> ExprNode:
        if lookback:
            lb = int(rng.integers(2, min(self.grammar.limits.max_rolling_lookback, 40) + 1))
            return constant_node(float(lb))
        ordered = sorted(
            allowed,
            key=lambda t: (
                0 if t in {ValueType.BOOLEAN, ValueType.REGIME, ValueType.SCALAR} else 1,
                t.value,
            ),
        )
        last_err: Exception | None = None
        for _ in range(max(3, len(ordered) * 2)):
            child_type = ordered[int(rng.integers(0, len(ordered)))]
            try:
                if max_depth <= 1 or rng.random() < 0.35:
                    child = self._leaf(rng, target_type=child_type, param_counter=param_counter)
                else:
                    child = self._grow(
                        rng,
                        target_type=child_type,
                        max_depth=max_depth - 1,
                        param_counter=param_counter,
                    )
                if child.value_type in allowed:
                    return child
                for alt in ordered:
                    fixed = self._leaf(rng, target_type=alt, param_counter=param_counter)
                    if fixed.value_type in allowed:
                        return fixed
            except DSLValidationError as exc:
                last_err = exc
                continue
        if last_err is not None:
            raise last_err
        raise DSLValidationError(
            f"could not build child for allowed types {[t.value for t in allowed]}"
        )

    def _grow(
        self,
        rng: np.random.Generator,
        *,
        target_type: ValueType,
        max_depth: int,
        param_counter: list[int],
    ) -> ExprNode:
        if max_depth <= 1 or rng.random() < 0.35:
            return self._leaf(rng, target_type=target_type, param_counter=param_counter)

        candidates = [
            oid
            for oid, spec in OPERATOR_REGISTRY.items()
            if spec.output_type is target_type and oid not in {OperatorId.ENTRY_SHORT}
        ]
        if not candidates:
            return self._leaf(rng, target_type=target_type, param_counter=param_counter)

        order = list(candidates)
        rng.shuffle(order)
        for op in order[: min(8, len(order))]:
            spec = OPERATOR_REGISTRY[op]
            try:
                children = [
                    self._child_for_slot(
                        rng,
                        allowed=allowed,
                        max_depth=max_depth,
                        param_counter=param_counter,
                        lookback=spec.lookback_child == i,
                    )
                    for i, allowed in enumerate(spec.input_types)
                ]
                node = op_node(op, *children)
                check_ast_types(node, path="grow", operation="generate")
                return node
            except DSLValidationError:
                continue
        return self._leaf(rng, target_type=target_type, param_counter=param_counter)

    def generate_entry(self, seed: int) -> ExprNode:
        rng = self._rng(seed)
        counter = [0]
        raw = self._grow(
            rng,
            target_type=ValueType.BOOLEAN,
            max_depth=min(4, self.grammar.limits.max_tree_depth),
            param_counter=counter,
        )
        entry = repair_entry(clamp_lookbacks(raw, self.grammar), self.grammar)
        check_ast_types(entry, path="entry", operation="generate")
        return entry

    def generate_exit(self, seed: int) -> ExprNode:
        rng = self._rng(seed + 17)
        numeric_leaves = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
        leaf = numeric_leaves[int(rng.integers(0, len(numeric_leaves)))]
        feature = feature_node(leaf.feature_id, leaf.value_type)
        thr = self._random_const(rng)
        cond = op_node(OperatorId.GREATER_THAN, feature, thr)
        exit_tree = op_node(OperatorId.EXIT_SIGNAL, cond)
        check_ast_types(exit_tree, path="exit", operation="generate")
        return exit_tree

    def generate(
        self,
        seed: int,
        *,
        with_exit: bool = True,
        with_stop: bool = True,
        max_attempts: int = 32,
    ) -> StrategyCandidate:
        last_reason = "unknown"
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            s = seed + attempt * 997
            try:
                entry = self.generate_entry(s)
                exit_tree = self.generate_exit(s) if with_exit else None
                stop = None
                target = None
                if with_stop:
                    atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
                    stop = op_node(OperatorId.ATR_STOP, atr, constant_node(1.5))
                    target = op_node(OperatorId.ATR_TARGET, atr, constant_node(2.0))
                check_strategy_trees(entry, exit_tree, stop, target, operation="generate")
                check = structural_precheck(
                    entry=entry, exit=exit_tree, regime_gates=(), grammar=self.grammar, seed=s
                )
                if not check.accepted:
                    last_reason = check.reason
                    continue
                cand = build_candidate(
                    entry_tree=entry,
                    exit_tree=exit_tree,
                    stop=stop,
                    target=target,
                    strategy_family=self.strategy_family,
                    creation_method=CreationMethod.RANDOM,
                    grammar_version=self.grammar.version,
                    feature_set_version=self.feature_set_version,
                    cost_model_version=self.cost_model_version,
                    random_seed=s,
                )
                check_strategy_trees(
                    cand.entry_tree,
                    cand.exit_tree,
                    cand.stop,
                    cand.target,
                    *cand.regime_gates,
                    operation="generate",
                    candidate_id=cand.candidate_id,
                )
                return cand
            except DSLValidationError as exc:
                last_exc = parse_dsl_validation_error(exc, operation="generate")
                last_reason = str(last_exc)
                continue
        if last_exc is not None:
            raise parse_dsl_validation_error(last_exc, operation="generate")
        raise RuntimeError(f"Failed to generate valid candidate: {last_reason}")

    def seed_template_mean_reversion(self, seed: int = 0) -> StrategyCandidate:
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        entry_cond = op_node(OperatorId.LESS_THAN, z, parameter_node("z_entry", -2.0))
        entry = op_node(OperatorId.ENTRY_LONG, entry_cond)
        exit_cond = op_node(OperatorId.GREATER_THAN, z, parameter_node("z_exit", -0.25))
        exit_tree = op_node(OperatorId.EXIT_SIGNAL, exit_cond)
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, constant_node(1.5))
        cand = build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            stop=stop,
            strategy_family="mean_reversion_template",
            creation_method=CreationMethod.SEED_TEMPLATE,
            grammar_version=GRAMMAR_VERSION,
            feature_set_version=self.feature_set_version,
            cost_model_version=self.cost_model_version,
            random_seed=seed,
        )
        check_strategy_trees(
            cand.entry_tree,
            cand.exit_tree,
            cand.stop,
            operation="seed_template",
            candidate_id=cand.candidate_id,
        )
        return cand
