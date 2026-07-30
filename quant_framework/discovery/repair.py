"""Repair invalid or over-limit trees into valid typed DSL expressions."""

from __future__ import annotations

from discovery.expression_tree import DSLValidationError, ExprNode, constant_node, feature_node, op_node
from discovery.grammar import Grammar
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.typecheck import check_ast_types
from discovery.types import NodeKind, ValueType


def _leaf_matching(allowed: frozenset[ValueType], grammar: Grammar) -> ExprNode:
    """Build a minimal node whose type is in ``allowed``."""
    from discovery.types import NUMERIC_TYPES

    for vt in sorted(allowed, key=lambda t: t.value):
        typed = [l for l in grammar.feature_leaves if l.value_type is vt]
        if typed:
            leaf = typed[0]
            return feature_node(leaf.feature_id, leaf.value_type)
        if vt is ValueType.SCALAR:
            return constant_node(0.0)
        if vt is ValueType.REGIME:
            typed_r = [l for l in grammar.feature_leaves if l.value_type is ValueType.REGIME]
            if typed_r:
                leaf = typed_r[0]
                return feature_node(leaf.feature_id, leaf.value_type)
            return constant_node(0.0, ValueType.REGIME)
        if vt is ValueType.BOOLEAN:
            numeric = [l for l in grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
            leaf = numeric[0] if numeric else grammar.feature_leaves[0]
            return op_node(
                OperatorId.LESS_THAN,
                feature_node(leaf.feature_id, leaf.value_type),
                constant_node(0.0),
            )
    numeric = [l for l in grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
    leaf = numeric[0] if numeric else grammar.feature_leaves[0]
    return feature_node(leaf.feature_id, leaf.value_type)


def truncate_to_limits(tree: ExprNode, grammar: Grammar) -> ExprNode:
    """Reduce depth/nodes by replacing deep subtrees with type-compatible leaves."""
    limits = grammar.limits

    def _trim(node: ExprNode, depth: int) -> ExprNode:
        if depth >= limits.max_tree_depth or node.n_nodes > limits.max_nodes:
            if node.kind is NodeKind.FEATURE:
                return node
            return _leaf_matching(frozenset({node.value_type}), grammar)
        if not node.children:
            return node
        if node.kind is NodeKind.OPERATOR:
            spec = OPERATOR_REGISTRY[OperatorId(node.name)]
            kids_list: list[ExprNode] = []
            for i, c in enumerate(node.children):
                trimmed = _trim(c, depth + 1)
                allowed = spec.input_types[i]
                if trimmed.value_type not in allowed:
                    trimmed = _leaf_matching(allowed, grammar)
                kids_list.append(trimmed)
            kids = tuple(kids_list)
            try:
                return ExprNode(
                    kind=node.kind,
                    value_type=node.value_type,
                    name=node.name,
                    children=kids,
                    meta=dict(node.meta),
                )
            except DSLValidationError:
                return _leaf_matching(frozenset({node.value_type}), grammar)
        kids = tuple(_trim(c, depth + 1) for c in node.children)
        return ExprNode(
            kind=node.kind,
            value_type=node.value_type,
            name=node.name,
            children=kids,
            meta=dict(node.meta),
        )

    trimmed = _trim(tree, 1)
    if trimmed.n_nodes > limits.max_nodes or trimmed.depth > limits.max_tree_depth:
        leaf = grammar.feature_leaves[0]
        return op_node(
            OperatorId.LESS_THAN,
            feature_node(leaf.feature_id, leaf.value_type),
            constant_node(-2.0),
        )
    return trimmed


def repair_entry(tree: ExprNode, grammar: Grammar) -> ExprNode:
    """Ensure entry is BOOLEAN or ENTRY_* ORDER_INTENT and within limits."""
    fixed = truncate_to_limits(tree, grammar)
    check_ast_types(fixed, path="repair_entry/pre", operation="normalize")
    if fixed.value_type is ValueType.ORDER_INTENT:
        return fixed
    if fixed.value_type is ValueType.BOOLEAN:
        out = op_node(OperatorId.ENTRY_LONG, fixed)
        check_ast_types(out, path="repair_entry", operation="normalize")
        return out
    boolish = op_node(OperatorId.LESS_THAN, fixed, constant_node(0.0))
    out = op_node(OperatorId.ENTRY_LONG, boolish)
    check_ast_types(out, path="repair_entry", operation="normalize")
    return out


def clamp_lookbacks(tree: ExprNode, grammar: Grammar) -> ExprNode:
    max_lb = grammar.limits.max_rolling_lookback

    def _fix(node: ExprNode) -> ExprNode:
        kids = tuple(_fix(c) for c in node.children)
        meta = dict(node.meta)
        if node.kind is NodeKind.OPERATOR:
            spec = OPERATOR_REGISTRY[OperatorId(node.name)]
            if spec.lookback_child is not None:
                child = kids[spec.lookback_child]
                if child.kind in {NodeKind.CONSTANT, NodeKind.PARAMETER}:
                    key = "value" if child.kind is NodeKind.CONSTANT else "default"
                    raw = float(child.meta.get(key, 1))
                    lb = int(max(1, min(max_lb, round(raw))))
                    kids_list = list(kids)
                    kids_list[spec.lookback_child] = constant_node(float(lb))
                    kids = tuple(kids_list)
        return ExprNode(node.kind, node.value_type, node.name, kids, meta)

    out = _fix(tree)
    check_ast_types(out, path="clamp_lookbacks", operation="normalize")
    return out
