"""Complexity scoring for DSL candidates."""

from __future__ import annotations

from discovery.expression_tree import ExprNode
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.types import NodeKind


def complexity_score(*trees: ExprNode | None) -> float:
    score = 0.0
    for tree in trees:
        if tree is None:
            continue
        for node in tree.walk():
            if node.kind is NodeKind.OPERATOR:
                spec = OPERATOR_REGISTRY[OperatorId(node.name)]
                score += float(spec.complexity_cost)
            elif node.kind is NodeKind.FEATURE:
                score += 1.0
            elif node.kind is NodeKind.PARAMETER:
                score += 0.5
            else:
                score += 0.25
        score += 0.1 * tree.depth
    return float(score)
