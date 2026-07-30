"""Type-preserving crossover for strategy expression trees."""

from __future__ import annotations

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.expression_tree import DSLValidationError, ExprNode
from discovery.grammar import Grammar
from discovery.prechecks import structural_precheck
from discovery.repair import clamp_lookbacks, repair_entry
from discovery.typecheck import check_strategy_trees, parse_dsl_validation_error
from discovery.types import CreationMethod, ValueType


class CrossoverError(ValueError):
    pass


class Crossover:
    def __init__(self, grammar: Grammar | None = None) -> None:
        self.grammar = grammar or Grammar()

    def _typed_sites(self, tree: ExprNode, value_type: ValueType) -> list[int]:
        return [i for i, n in enumerate(tree.walk()) if n.value_type is value_type]

    def _swap(self, root: ExprNode, index: int, replacement: ExprNode) -> ExprNode:
        counter = [0]

        def _walk(node: ExprNode) -> ExprNode:
            idx = counter[0]
            counter[0] += 1
            if idx == index:
                if node.value_type is not replacement.value_type:
                    raise CrossoverError("type mismatch during crossover")
                return replacement
            kids = tuple(_walk(c) for c in node.children)
            try:
                return ExprNode(node.kind, node.value_type, node.name, kids, dict(node.meta))
            except DSLValidationError as exc:
                raise CrossoverError(str(exc)) from exc

        return _walk(root)

    def crossover_trees(
        self,
        a: ExprNode,
        b: ExprNode,
        rng: np.random.Generator,
    ) -> tuple[ExprNode, ExprNode]:
        types_a = {n.value_type for n in a.walk()}
        types_b = {n.value_type for n in b.walk()}
        shared = sorted(types_a & types_b, key=lambda t: t.value)
        if not shared:
            raise CrossoverError("no shared value types for crossover")

        # Try several typed sites — parent slots may reject even same ValueType
        # when operator input unions differ across trees.
        for _ in range(16):
            vt = shared[int(rng.integers(0, len(shared)))]
            sites_a = self._typed_sites(a, vt)
            sites_b = self._typed_sites(b, vt)
            if not sites_a or not sites_b:
                continue
            ia = sites_a[int(rng.integers(0, len(sites_a)))]
            ib = sites_b[int(rng.integers(0, len(sites_b)))]
            node_a = list(a.walk())[ia]
            node_b = list(b.walk())[ib]
            try:
                child_a = self._swap(a, ia, node_b)
                child_b = self._swap(b, ib, node_a)
                return child_a, child_b
            except CrossoverError:
                continue
        raise CrossoverError("no compatible crossover sites")

    def crossover(
        self,
        parent_a: StrategyCandidate,
        parent_b: StrategyCandidate,
        *,
        seed: int,
    ) -> tuple[StrategyCandidate, StrategyCandidate]:
        rng = np.random.default_rng(seed)
        try:
            e1, e2 = self.crossover_trees(parent_a.entry_tree, parent_b.entry_tree, rng)
            e1 = repair_entry(clamp_lookbacks(e1, self.grammar), self.grammar)
            e2 = repair_entry(clamp_lookbacks(e2, self.grammar), self.grammar)
        except (CrossoverError, DSLValidationError):
            e1, e2 = parent_a.entry_tree, parent_b.entry_tree

        def _build(entry: ExprNode, parents: tuple[str, ...], s: int) -> StrategyCandidate:
            check = structural_precheck(
                entry=entry, exit=parent_a.exit_tree, regime_gates=(), grammar=self.grammar, seed=s
            )
            if not check.accepted:
                entry = parent_a.entry_tree
            cand = build_candidate(
                entry_tree=entry,
                exit_tree=parent_a.exit_tree,
                stop=parent_a.stop,
                target=parent_a.target,
                strategy_family=parent_a.strategy_family,
                creation_method=CreationMethod.CROSSOVER,
                generation=max(parent_a.generation, parent_b.generation) + 1,
                parent_ids=parents,
                grammar_version=parent_a.grammar_version,
                feature_set_version=parent_a.feature_set_version,
                cost_model_version=parent_a.cost_model_version,
                asset_universe=parent_a.asset_universe,
                random_seed=s,
                family_provenance=dict(parent_a.family_provenance or {}),
            )
            try:
                check_strategy_trees(
                    cand.entry_tree,
                    cand.exit_tree,
                    cand.stop,
                    cand.target,
                    operation="crossover",
                    candidate_id=cand.candidate_id,
                    parent_ids=parents,
                )
            except DSLValidationError as exc:
                raise parse_dsl_validation_error(
                    exc,
                    operation="crossover",
                    candidate_id=cand.candidate_id,
                    parent_ids=parents,
                ) from exc
            return cand

        return (
            _build(e1, (parent_a.candidate_id, parent_b.candidate_id), seed),
            _build(e2, (parent_b.candidate_id, parent_a.candidate_id), seed + 1),
        )
