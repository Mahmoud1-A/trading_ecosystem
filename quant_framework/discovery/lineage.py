"""Discovery lineage helpers — material changes create new lineage IDs."""

from __future__ import annotations

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.expression_tree import ExprNode
from discovery.types import CreationMethod
from registry.hashing import sha256_json
from validation.candidate_lineage import CandidateLineage


def lineage_from_candidate(candidate: StrategyCandidate) -> CandidateLineage:
    return CandidateLineage.create(
        strategy_family=candidate.strategy_family,
        parameters=dict(candidate.parameters),
        config_snapshot={
            "grammar_version": candidate.grammar_version,
            "feature_set_version": candidate.feature_set_version,
            "candidate_id": candidate.candidate_id,
            "entry": candidate.entry_tree.as_dict(),
        },
        code_hash=candidate.grammar_version,
        data_hash=candidate.feature_set_version,
        cost_model_version=candidate.cost_model_version,
        candidate_id=candidate.candidate_id,
    )


def legacy_import_candidate(
    *,
    entry_tree: ExprNode,
    exit_tree: ExprNode | None,
    legacy_id: str,
    grammar_version: str,
    feature_set_version: str,
    random_seed: int,
    strategy_family: str = "legacy_import",
) -> StrategyCandidate:
    """
    Import a legacy strategy as LEGACY_IMPORT.

    Always receives a new candidate_id and lineage_id — never inherits Book /
    Vault / promotion status from the legacy system.
    """
    lineage_id = "lin_legacy_" + sha256_json(
        {
            "legacy_id": legacy_id,
            "creation_method": CreationMethod.LEGACY_IMPORT.value,
            "grammar_version": grammar_version,
            "feature_set_version": feature_set_version,
            "entry": entry_tree.as_dict(),
        }
    )[:20]
    return build_candidate(
        entry_tree=entry_tree,
        exit_tree=exit_tree,
        strategy_family=strategy_family,
        creation_method=CreationMethod.LEGACY_IMPORT,
        generation=0,
        parent_ids=(f"legacy:{legacy_id}",),
        grammar_version=grammar_version,
        feature_set_version=feature_set_version,
        random_seed=random_seed,
        lineage_id=lineage_id,
    )
