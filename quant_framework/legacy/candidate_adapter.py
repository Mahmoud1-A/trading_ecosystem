"""Adapt legacy records into typed DSL candidates (LEGACY_IMPORT)."""

from __future__ import annotations

from discovery.candidate import StrategyCandidate
from discovery.expression_tree import ExprNode, constant_node, feature_node, op_node, parameter_node
from discovery.grammar import GRAMMAR_VERSION
from discovery.lineage import legacy_import_candidate
from discovery.operators import OperatorId
from discovery.types import CreationMethod, ValueType
from legacy.records import LegacyStrategyRecord


FORBIDDEN_INHERITED_FIELDS = frozenset(
    {
        "book_member",
        "vault_eligible",
        "paper_status",
        "legacy_fitness",
        "promotion_status",
        "fitness_score",
        "book_membership",
    }
)


def _default_entry_from_params(params: dict[str, float]) -> ExprNode:
    z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
    z_entry = float(params.get("z_entry", -2.0))
    cond = op_node(OperatorId.LESS_THAN, z, parameter_node("z_entry", z_entry))
    return op_node(OperatorId.ENTRY_LONG, cond)


def _default_exit_from_params(params: dict[str, float]) -> ExprNode:
    z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
    z_exit = float(params.get("z_exit", -0.25))
    cond = op_node(OperatorId.GREATER_THAN, z, parameter_node("z_exit", z_exit))
    return op_node(OperatorId.EXIT_SIGNAL, cond)


def adapt_legacy_record(
    record: LegacyStrategyRecord,
    *,
    feature_set_version: str = "feature_set_v1_phase6b",
    grammar_version: str = GRAMMAR_VERSION,
    random_seed: int = 0,
) -> StrategyCandidate:
    """
    Convert a legacy record into a new-system candidate.

    Never copies Book / Vault / paper / fitness / promotion status.
    """
    if record.expression is not None:
        entry = ExprNode.from_dict(record.expression)
        exit_tree = None
    else:
        entry = _default_entry_from_params(record.parameters)
        exit_tree = _default_exit_from_params(record.parameters)

    candidate = legacy_import_candidate(
        entry_tree=entry,
        exit_tree=exit_tree,
        legacy_id=record.legacy_id,
        grammar_version=grammar_version,
        feature_set_version=feature_set_version,
        random_seed=random_seed,
        strategy_family=record.strategy_family,
    )
    assert candidate.creation_method is CreationMethod.LEGACY_IMPORT
    # Prove inherited status fields are absent from the live candidate payload
    payload = candidate.as_dict()
    for field in FORBIDDEN_INHERITED_FIELDS:
        assert field not in payload
    return candidate
