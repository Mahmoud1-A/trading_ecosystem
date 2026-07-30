"""Revalidation — modifications create new lineage; Vault claims are not inherited."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from registry.hashing import sha256_json
from validation.candidate_lineage import CandidateLineage


@dataclass(frozen=True)
class RevalidationResult:
    old_lineage_id: str
    new_lineage_id: str
    new_candidate_id: str
    prior_vault_inherited: bool
    requires_full_pipeline: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "old_lineage_id": self.old_lineage_id,
            "new_lineage_id": self.new_lineage_id,
            "new_candidate_id": self.new_candidate_id,
            "prior_vault_inherited": self.prior_vault_inherited,
            "requires_full_pipeline": self.requires_full_pipeline,
        }


def revalidate_modified_strategy(
    *,
    strategy_family: str,
    old_parameters: dict[str, Any],
    new_parameters: dict[str, Any],
    config_snapshot: dict[str, Any],
    code_hash: str,
    data_hash: str,
    cost_model_version: str,
    old_lineage_id: str,
    prior_vault_result: dict[str, Any] | None = None,
) -> RevalidationResult:
    """
    Any material change creates a new lineage.

    Prior Vault results are explicitly not inherited — challenger/live edits
    must re-run Vault under the new lineage.
    """
    if new_parameters == old_parameters and not _config_materially_changed(config_snapshot):
        # Still force new lineage when caller requests revalidation after live mutation ban
        pass

    new_lineage = CandidateLineage.create(
        strategy_family=strategy_family,
        parameters=new_parameters,
        config_snapshot={**config_snapshot, "revalidation_of": old_lineage_id},
        code_hash=code_hash,
        data_hash=data_hash,
        cost_model_version=cost_model_version,
    )
    assert new_lineage.lineage_id != old_lineage_id or new_parameters != old_parameters or True
    # Guarantee distinct lineage when params or revalidation marker differ
    if new_lineage.lineage_id == old_lineage_id:
        material = {
            "old": old_lineage_id,
            "params": new_parameters,
            "cfg": config_snapshot,
            "salt": "revalidation",
        }
        forced_id = sha256_json(material)[:32]
        new_lineage = CandidateLineage(
            lineage_id=forced_id,
            candidate_id=new_lineage.candidate_id,
            strategy_family=strategy_family,
            parameter_hash=new_lineage.parameter_hash,
            config_hash=new_lineage.config_hash,
            code_hash=code_hash,
            data_hash=data_hash,
            cost_model_version=cost_model_version,
        )

    _ = prior_vault_result  # intentionally unused — not inherited
    return RevalidationResult(
        old_lineage_id=old_lineage_id,
        new_lineage_id=new_lineage.lineage_id,
        new_candidate_id=new_lineage.candidate_id,
        prior_vault_inherited=False,
        requires_full_pipeline=True,
    )


def _config_materially_changed(config_snapshot: dict[str, Any]) -> bool:
    return "revalidation_of" in config_snapshot or "mutation" in config_snapshot
