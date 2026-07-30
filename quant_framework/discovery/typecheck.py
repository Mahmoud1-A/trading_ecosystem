"""Recursive typed-AST validation for the Strategy DSL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from discovery.expression_tree import DSLValidationError, ExprNode
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.types import NodeKind, ValueType


INVALID_DSL_TYPE = "INVALID_DSL_TYPE"


@dataclass
class TypeViolation:
    path: str
    operator: str | None
    expected_types: list[str]
    actual_type: str
    child_index: int | None = None
    node_name: str = ""
    node_kind: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InvalidDslTypeError(DSLValidationError):
    """Typed AST violation — reject candidate, do not abort the campaign."""

    def __init__(
        self,
        message: str,
        *,
        violations: list[TypeViolation] | None = None,
        operation: str = "",
        candidate_id: str | None = None,
        parent_ids: tuple[str, ...] = (),
    ) -> None:
        self.message = message
        self.violations = list(violations or [])
        self.operation = operation
        self.candidate_id = candidate_id
        self.parent_ids = tuple(parent_ids)
        first = self.violations[0] if self.violations else None
        self.node_path = first.path if first else ""
        self.operator = first.operator if first else None
        self.expected_types = list(first.expected_types) if first else []
        self.actual_type = first.actual_type if first else ""
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": INVALID_DSL_TYPE,
            "message": self.message,
            "operation": self.operation,
            "candidate_id": self.candidate_id,
            "parent_ids": list(self.parent_ids),
            "node_path": self.node_path,
            "operator": self.operator,
            "expected_types": list(self.expected_types),
            "actual_type": self.actual_type,
            "violations": [v.as_dict() for v in self.violations],
        }


def check_ast_types(
    node: ExprNode,
    *,
    path: str = "root",
    operation: str = "",
    candidate_id: str | None = None,
    parent_ids: tuple[str, ...] = (),
) -> None:
    """
    Recursively verify operator arity/output and child input-type contracts.

    Raises :class:`InvalidDslTypeError` on the first structural type failure set
    (all violations in the tree are collected first).
    """
    violations: list[TypeViolation] = []

    def _walk(n: ExprNode, p: str) -> None:
        if n.kind is NodeKind.OPERATOR:
            try:
                op = OperatorId(n.name)
            except ValueError:
                violations.append(
                    TypeViolation(
                        path=p,
                        operator=n.name,
                        expected_types=[],
                        actual_type=n.value_type.value,
                        node_name=n.name,
                        node_kind=n.kind.value,
                    )
                )
                return
            spec = OPERATOR_REGISTRY[op]
            if len(n.children) != spec.arity:
                violations.append(
                    TypeViolation(
                        path=p,
                        operator=op.value,
                        expected_types=[f"arity={spec.arity}"],
                        actual_type=f"arity={len(n.children)}",
                        node_name=n.name,
                        node_kind=n.kind.value,
                    )
                )
            if n.value_type is not spec.output_type:
                violations.append(
                    TypeViolation(
                        path=p,
                        operator=op.value,
                        expected_types=[spec.output_type.value],
                        actual_type=n.value_type.value,
                        node_name=n.name,
                        node_kind=n.kind.value,
                    )
                )
            for i, child in enumerate(n.children):
                allowed = spec.input_types[i]
                child_path = f"{p}/{op.value}[{i}]"
                if child.value_type not in allowed:
                    violations.append(
                        TypeViolation(
                            path=child_path,
                            operator=op.value,
                            expected_types=sorted(t.value for t in allowed),
                            actual_type=child.value_type.value,
                            child_index=i,
                            node_name=child.name,
                            node_kind=child.kind.value,
                        )
                    )
                _walk(child, child_path)
        else:
            for i, child in enumerate(n.children):
                _walk(child, f"{p}/{n.name}[{i}]")

    _walk(node, path)
    if violations:
        first = violations[0]
        msg = (
            f"{INVALID_DSL_TYPE}: {first.operator or node.name} at {first.path} "
            f"got {first.actual_type}, expected {first.expected_types}"
        )
        raise InvalidDslTypeError(
            msg,
            violations=violations,
            operation=operation,
            candidate_id=candidate_id,
            parent_ids=parent_ids,
        )


def check_strategy_trees(
    *trees: ExprNode | None,
    operation: str = "",
    candidate_id: str | None = None,
    parent_ids: tuple[str, ...] = (),
) -> None:
    for i, tree in enumerate(trees):
        if tree is None:
            continue
        check_ast_types(
            tree,
            path=f"tree[{i}]",
            operation=operation,
            candidate_id=candidate_id,
            parent_ids=parent_ids,
        )


def parse_dsl_validation_error(
    exc: Exception,
    *,
    operation: str = "",
    candidate_id: str | None = None,
    parent_ids: tuple[str, ...] = (),
) -> InvalidDslTypeError:
    """Normalize bare DSLValidationError into InvalidDslTypeError."""
    if isinstance(exc, InvalidDslTypeError):
        if operation and not exc.operation:
            exc.operation = operation
        if candidate_id and not exc.candidate_id:
            exc.candidate_id = candidate_id
        if parent_ids and not exc.parent_ids:
            exc.parent_ids = tuple(parent_ids)
        return exc
    message = str(exc)
    # REGIME_GATE child[1] type SCALAR not in ['REGIME', 'BOOLEAN']
    operator = None
    actual = ""
    expected: list[str] = []
    child_index = None
    if " child[" in message and " type " in message:
        try:
            operator = message.split(" child[", 1)[0].strip()
            rest = message.split(" child[", 1)[1]
            child_index = int(rest.split("]", 1)[0])
            actual = rest.split(" type ", 1)[1].split(" ", 1)[0]
            if " not in " in message:
                raw = message.split(" not in ", 1)[1].strip()
                expected = [x.strip(" []'\"") for x in raw.strip("[]").split(",") if x.strip()]
        except Exception:  # noqa: BLE001
            pass
    violation = TypeViolation(
        path=f"parsed/{operator or 'unknown'}[{child_index}]",
        operator=operator,
        expected_types=expected,
        actual_type=actual,
        child_index=child_index,
    )
    return InvalidDslTypeError(
        f"{INVALID_DSL_TYPE}: {message}",
        violations=[violation],
        operation=operation,
        candidate_id=candidate_id,
        parent_ids=parent_ids,
    )
