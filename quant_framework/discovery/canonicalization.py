"""Canonical serialization of expression trees for stable candidate IDs."""

from __future__ import annotations

import json
from typing import Any

from discovery.expression_tree import ExprNode
from discovery.operators import OperatorId
from discovery.types import NodeKind


# Operators that are commutative — sort children for stable identity
_COMMUTATIVE = frozenset(
    {
        OperatorId.ADD.value,
        OperatorId.MULTIPLY_BOUNDED.value,
        OperatorId.MIN.value,
        OperatorId.MAX.value,
        OperatorId.AND.value,
        OperatorId.OR.value,
    }
)


def canonicalize_node(node: ExprNode) -> ExprNode:
    from discovery.typecheck import check_ast_types

    children = tuple(canonicalize_node(c) for c in node.children)
    if node.kind is NodeKind.OPERATOR and node.name in _COMMUTATIVE and len(children) == 2:
        left, right = children
        if _key(left) > _key(right):
            children = (right, left)
    meta = {k: node.meta[k] for k in sorted(node.meta)}
    out = ExprNode(
        kind=node.kind,
        value_type=node.value_type,
        name=node.name,
        children=children,
        meta=meta,
    )
    check_ast_types(out, path="canonicalize", operation="normalization")
    return out


def _key(node: ExprNode) -> str:
    return json.dumps(node.as_dict(), sort_keys=True, separators=(",", ":"))


def canonical_dict(node: ExprNode | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return canonicalize_node(node).as_dict()


def canonical_payload(
    *,
    entry: ExprNode | None,
    exit: ExprNode | None,
    stop: ExprNode | None,
    target: ExprNode | None,
    regime_gates: tuple[ExprNode, ...],
    sizing: ExprNode | None,
    strategy_family: str,
    grammar_version: str,
    feature_set_version: str,
    parameters: dict[str, float],
) -> dict[str, Any]:
    return {
        "strategy_family": strategy_family,
        "grammar_version": grammar_version,
        "feature_set_version": feature_set_version,
        "entry": canonical_dict(entry),
        "exit": canonical_dict(exit),
        "stop": canonical_dict(stop),
        "target": canonical_dict(target),
        "regime_gates": [canonical_dict(g) for g in regime_gates],
        "sizing": canonical_dict(sizing),
        "parameters": {k: parameters[k] for k in sorted(parameters)},
    }
