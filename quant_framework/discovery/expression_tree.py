"""Typed expression trees — the only executable representation of strategy logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from discovery.operators import FORBIDDEN_BUILTINS, OPERATOR_REGISTRY, OperatorId, OperatorSpec
from discovery.types import NodeKind, ValueType


class DSLValidationError(ValueError):
    """Raised when a tree violates type, arity, or safety rules."""


@dataclass(frozen=True)
class ExprNode:
    """Immutable expression node. Never holds callable Python code."""

    kind: NodeKind
    value_type: ValueType
    # FEATURE: feature_id; CONSTANT/PARAMETER: numeric value or name; OPERATOR: OperatorId value
    name: str
    children: tuple[ExprNode, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name in FORBIDDEN_BUILTINS or any(b in self.name for b in ("__", "os.", "sys.")):
            raise DSLValidationError(f"Forbidden identifier in DSL node: {self.name!r}")
        if self.kind is NodeKind.OPERATOR:
            try:
                op = OperatorId(self.name)
            except ValueError as exc:
                raise DSLValidationError(f"Unknown operator {self.name!r}") from exc
            spec = OPERATOR_REGISTRY[op]
            if len(self.children) != spec.arity:
                raise DSLValidationError(
                    f"{op.value} expects arity {spec.arity}, got {len(self.children)}"
                )
            for i, child in enumerate(self.children):
                allowed = spec.input_types[i]
                if child.value_type not in allowed:
                    raise DSLValidationError(
                        f"{op.value} child[{i}] type {child.value_type.value} "
                        f"not in {[t.value for t in allowed]}"
                    )
            if self.value_type is not spec.output_type:
                raise DSLValidationError(
                    f"{op.value} output must be {spec.output_type.value}, got {self.value_type.value}"
                )

    @property
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth for c in self.children)

    @property
    def n_nodes(self) -> int:
        return 1 + sum(c.n_nodes for c in self.children)

    def walk(self) -> Iterator[ExprNode]:
        yield self
        for c in self.children:
            yield from c.walk()

    def feature_ids(self) -> tuple[str, ...]:
        return tuple(sorted({n.name for n in self.walk() if n.kind is NodeKind.FEATURE}))

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted({n.name for n in self.walk() if n.kind is NodeKind.PARAMETER}))

    def operator_ids(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.walk() if n.kind is NodeKind.OPERATOR)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value_type": self.value_type.value,
            "name": self.name,
            "children": [c.as_dict() for c in self.children],
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any], *, _validate_root: bool = True) -> ExprNode:
        from discovery.typecheck import check_ast_types

        node = ExprNode(
            kind=NodeKind(payload["kind"]),
            value_type=ValueType(payload["value_type"]),
            name=str(payload["name"]),
            children=tuple(
                ExprNode.from_dict(c, _validate_root=False)
                for c in payload.get("children", ())
            ),
            meta=dict(payload.get("meta") or {}),
        )
        if _validate_root:
            check_ast_types(node, path="deserialize", operation="deserialization")
        return node


def feature_node(feature_id: str, value_type: ValueType) -> ExprNode:
    return ExprNode(NodeKind.FEATURE, value_type, feature_id)


def constant_node(value: float, value_type: ValueType = ValueType.SCALAR) -> ExprNode:
    if not np.isfinite(value):
        raise DSLValidationError("Constant must be finite")
    return ExprNode(NodeKind.CONSTANT, value_type, f"{value:.10g}", meta={"value": float(value)})


def parameter_node(name: str, default: float, value_type: ValueType = ValueType.SCALAR) -> ExprNode:
    if name in FORBIDDEN_BUILTINS:
        raise DSLValidationError(f"Forbidden parameter name {name!r}")
    if not np.isfinite(default):
        raise DSLValidationError("Parameter default must be finite")
    return ExprNode(NodeKind.PARAMETER, value_type, name, meta={"default": float(default)})


def op_node(op: OperatorId, *children: ExprNode) -> ExprNode:
    spec: OperatorSpec = OPERATOR_REGISTRY[op]
    return ExprNode(NodeKind.OPERATOR, spec.output_type, op.value, children=tuple(children))


def protected_divide(numer: float, denom: float, *, eps: float = 1e-8) -> float:
    """Finite protected division used by the DSL evaluator — never raises."""
    if not np.isfinite(numer) or not np.isfinite(denom):
        return 0.0
    if abs(denom) < eps:
        return 0.0
    out = numer / denom
    if not np.isfinite(out):
        return 0.0
    return float(out)


def eval_scalar(node: ExprNode, *, bindings: dict[str, float] | None = None) -> float:
    """
    Evaluate a pure scalar tree (no series ops) for prechecks.

    Series operators return 0.0 placeholders here; full bar evaluation is Phase 6D.
    """
    bindings = bindings or {}
    if node.kind is NodeKind.CONSTANT:
        return float(node.meta["value"])
    if node.kind is NodeKind.PARAMETER:
        return float(bindings.get(node.name, node.meta.get("default", 0.0)))
    if node.kind is NodeKind.FEATURE:
        return float(bindings.get(node.name, 0.0))
    op = OperatorId(node.name)
    kids = [eval_scalar(c, bindings=bindings) for c in node.children]
    if op is OperatorId.ADD:
        return kids[0] + kids[1]
    if op is OperatorId.SUBTRACT:
        return kids[0] - kids[1]
    if op is OperatorId.MULTIPLY_BOUNDED:
        return float(np.clip(kids[0] * kids[1], -1e6, 1e6))
    if op is OperatorId.DIVIDE_PROTECTED:
        return protected_divide(kids[0], kids[1])
    if op is OperatorId.ABS:
        return abs(kids[0])
    if op is OperatorId.NEGATE:
        return -kids[0]
    if op is OperatorId.MIN:
        return min(kids[0], kids[1])
    if op is OperatorId.MAX:
        return max(kids[0], kids[1])
    if op is OperatorId.CLIP:
        return float(np.clip(kids[0], kids[1], kids[2]))
    if op is OperatorId.GREATER_THAN:
        return 1.0 if kids[0] > kids[1] else 0.0
    if op is OperatorId.LESS_THAN:
        return 1.0 if kids[0] < kids[1] else 0.0
    if op is OperatorId.GREATER_EQUAL:
        return 1.0 if kids[0] >= kids[1] else 0.0
    if op is OperatorId.LESS_EQUAL:
        return 1.0 if kids[0] <= kids[1] else 0.0
    if op is OperatorId.BETWEEN:
        return 1.0 if kids[1] <= kids[0] <= kids[2] else 0.0
    if op is OperatorId.AND:
        return 1.0 if kids[0] and kids[1] else 0.0
    if op is OperatorId.OR:
        return 1.0 if kids[0] or kids[1] else 0.0
    if op is OperatorId.NOT:
        return 0.0 if kids[0] else 1.0
    # Series / strategy wrappers — scalar probe returns child or 0
    if kids:
        return float(kids[0])
    return 0.0
