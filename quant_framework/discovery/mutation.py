"""Typed mutation operators for strategy candidates."""

from __future__ import annotations

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.expression_tree import DSLValidationError, ExprNode, constant_node, feature_node, op_node
from discovery.grammar import Grammar
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.prechecks import structural_precheck
from discovery.repair import clamp_lookbacks, repair_entry
from discovery.typecheck import check_ast_types, check_strategy_trees, parse_dsl_validation_error
from discovery.types import CreationMethod, NodeKind, ValueType


class Mutator:
    def __init__(self, grammar: Grammar | None = None) -> None:
        self.grammar = grammar or Grammar()

    def _nodes(self, tree: ExprNode) -> list[ExprNode]:
        return list(tree.walk())

    def _replace(self, root: ExprNode, target_id: int, new_node: ExprNode) -> ExprNode:
        """Replace node at walk-index; require identical value_type to preserve typing."""
        counter = [0]

        def _walk(node: ExprNode) -> ExprNode:
            idx = counter[0]
            counter[0] += 1
            if idx == target_id:
                if new_node.value_type is not node.value_type:
                    return node
                return new_node
            kids = tuple(_walk(c) for c in node.children)
            return ExprNode(node.kind, node.value_type, node.name, kids, dict(node.meta))

        return _walk(root)

    def mutate_tree(self, tree: ExprNode, rng: np.random.Generator) -> ExprNode:
        nodes = self._nodes(tree)
        idx = int(rng.integers(0, len(nodes)))
        node = nodes[idx]
        choice = int(rng.integers(0, 5))

        if choice == 0 and node.kind is NodeKind.FEATURE:
            same = [l for l in self.grammar.feature_leaves if l.value_type is node.value_type]
            if not same:
                return tree
            leaf = same[int(rng.integers(0, len(same)))]
            return self._replace(tree, idx, feature_node(leaf.feature_id, leaf.value_type))

        if choice == 1 and node.kind is NodeKind.CONSTANT:
            val = float(node.meta.get("value", 0.0)) * float(rng.uniform(0.5, 1.5))
            val += float(rng.normal(0, 0.1))
            return self._replace(tree, idx, constant_node(val, node.value_type))

        if choice == 2 and node.kind is NodeKind.OPERATOR:
            spec = OPERATOR_REGISTRY[OperatorId(node.name)]
            alts = [
                oid
                for oid, s in OPERATOR_REGISTRY.items()
                if s.arity == spec.arity
                and s.output_type is spec.output_type
                and s.input_types == spec.input_types
            ]
            if alts:
                new_op = alts[int(rng.integers(0, len(alts)))]
                return self._replace(tree, idx, op_node(new_op, *node.children))

        if choice == 3:
            for i, n in enumerate(nodes):
                if n.kind is not NodeKind.OPERATOR:
                    continue
                spec = OPERATOR_REGISTRY[OperatorId(n.name)]
                if spec.lookback_child is None:
                    continue
                kids = list(n.children)
                lb = int(rng.integers(2, self.grammar.limits.max_rolling_lookback + 1))
                kids[spec.lookback_child] = constant_node(float(lb))
                return self._replace(tree, i, op_node(OperatorId(n.name), *kids))

        # Simplify only BOOLEAN / ORDER_INTENT sites so parent typing stays valid
        if node.value_type is ValueType.BOOLEAN:
            from discovery.types import NUMERIC_TYPES

            numeric = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
            leaf = numeric[int(rng.integers(0, len(numeric)))] if numeric else self.grammar.feature_leaves[0]
            simplified = op_node(
                OperatorId.LESS_THAN,
                feature_node(leaf.feature_id, leaf.value_type),
                constant_node(float(rng.normal())),
            )
            return self._replace(tree, idx, simplified)
        if node.value_type is ValueType.ORDER_INTENT:
            from discovery.types import NUMERIC_TYPES

            numeric = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
            leaf = numeric[int(rng.integers(0, len(numeric)))] if numeric else self.grammar.feature_leaves[0]
            cond = op_node(
                OperatorId.LESS_THAN,
                feature_node(leaf.feature_id, leaf.value_type),
                constant_node(float(rng.normal())),
            )
            return self._replace(tree, idx, op_node(OperatorId.ENTRY_LONG, cond))
        return tree

    def mutate(self, parent: StrategyCandidate, *, seed: int) -> StrategyCandidate:
        rng = np.random.default_rng(seed)
        try:
            entry = repair_entry(
                clamp_lookbacks(self.mutate_tree(parent.entry_tree, rng), self.grammar),
                self.grammar,
            )
            check_ast_types(
                entry,
                path="mutation/entry",
                operation="mutation",
                parent_ids=(parent.candidate_id,),
            )
        except DSLValidationError:
            entry = parent.entry_tree

        exit_tree = parent.exit_tree
        if exit_tree is not None and rng.random() < 0.5:
            try:
                exit_tree = clamp_lookbacks(self.mutate_tree(exit_tree, rng), self.grammar)
                check_ast_types(
                    exit_tree,
                    path="mutation/exit",
                    operation="mutation",
                    parent_ids=(parent.candidate_id,),
                )
            except DSLValidationError:
                exit_tree = parent.exit_tree

        check = structural_precheck(
            entry=entry,
            exit=exit_tree,
            regime_gates=parent.regime_gates,
            grammar=self.grammar,
            seed=seed,
        )
        if not check.accepted:
            entry = parent.entry_tree
            exit_tree = parent.exit_tree

        cand = build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            stop=parent.stop,
            target=parent.target,
            sizing=parent.sizing,
            regime_gates=parent.regime_gates,
            strategy_family=parent.strategy_family,
            creation_method=CreationMethod.MUTATION,
            generation=parent.generation + 1,
            parent_ids=(parent.candidate_id,),
            grammar_version=parent.grammar_version,
            feature_set_version=parent.feature_set_version,
            cost_model_version=parent.cost_model_version,
            asset_universe=parent.asset_universe,
            random_seed=seed,
        )
        try:
            check_strategy_trees(
                cand.entry_tree,
                cand.exit_tree,
                cand.stop,
                cand.target,
                *cand.regime_gates,
                operation="mutation",
                candidate_id=cand.candidate_id,
                parent_ids=(parent.candidate_id,),
            )
        except DSLValidationError as exc:
            raise parse_dsl_validation_error(
                exc,
                operation="mutation",
                candidate_id=cand.candidate_id,
                parent_ids=(parent.candidate_id,),
            ) from exc
        return cand
