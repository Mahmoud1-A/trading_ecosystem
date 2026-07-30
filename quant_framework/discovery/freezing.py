"""Freeze candidates before Vault submission — immutable snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from discovery.candidate import StrategyCandidate
from registry.hashing import sha256_json
from validation.candidate_lineage import CandidateLineage


@dataclass(frozen=True)
class FrozenCandidate:
    """Immutable handoff artifact required before Vault access."""

    frozen_id: str
    candidate_id: str
    lineage_id: str
    strategy_family: str
    parameters: dict[str, float]
    config_hash: str
    feature_set_version: str
    cost_model_version: str
    grammar_version: str
    code_hash: str
    data_hash: str
    execution_assumptions: dict[str, Any]
    expression_snapshot: dict[str, Any]
    oos_fitness: float
    ranking_source: str
    discovery_run_id: str
    frozen_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "frozen_id": self.frozen_id,
            "candidate_id": self.candidate_id,
            "lineage_id": self.lineage_id,
            "strategy_family": self.strategy_family,
            "parameters": dict(self.parameters),
            "config_hash": self.config_hash,
            "feature_set_version": self.feature_set_version,
            "cost_model_version": self.cost_model_version,
            "grammar_version": self.grammar_version,
            "code_hash": self.code_hash,
            "data_hash": self.data_hash,
            "execution_assumptions": dict(self.execution_assumptions),
            "expression_snapshot": dict(self.expression_snapshot),
            "oos_fitness": self.oos_fitness,
            "ranking_source": self.ranking_source,
            "discovery_run_id": self.discovery_run_id,
            "frozen_at": self.frozen_at,
        }

    def to_lineage(self) -> CandidateLineage:
        return CandidateLineage(
            lineage_id=self.lineage_id,
            candidate_id=self.candidate_id,
            strategy_family=self.strategy_family,
            parameter_hash=sha256_json(self.parameters),
            config_hash=self.config_hash,
            code_hash=self.code_hash,
            data_hash=self.data_hash,
            cost_model_version=self.cost_model_version,
        )


class FreezeError(ValueError):
    pass


def freeze_candidate(
    candidate: StrategyCandidate,
    *,
    oos_fitness: float,
    ranking_source: str,
    discovery_run_id: str,
    code_hash: str,
    data_hash: str,
    execution_assumptions: dict[str, Any] | None = None,
) -> FrozenCandidate:
    if ranking_source != "validation_oos":
        raise FreezeError("cannot freeze candidate ranked on training metrics")
    assumptions = execution_assumptions or {"pipeline": "event_driven_wfo"}
    config_hash = sha256_json(
        {
            "grammar_version": candidate.grammar_version,
            "feature_set_version": candidate.feature_set_version,
            "cost_model_version": candidate.cost_model_version,
            "execution_assumptions": assumptions,
        }
    )
    payload = {
        "candidate_id": candidate.candidate_id,
        "lineage_id": candidate.lineage_id,
        "config_hash": config_hash,
        "oos_fitness": oos_fitness,
        "discovery_run_id": discovery_run_id,
    }
    frozen_id = "frozen_" + sha256_json(payload)[:24]
    return FrozenCandidate(
        frozen_id=frozen_id,
        candidate_id=candidate.candidate_id,
        lineage_id=candidate.lineage_id,
        strategy_family=candidate.strategy_family,
        parameters=dict(candidate.parameters),
        config_hash=config_hash,
        feature_set_version=candidate.feature_set_version,
        cost_model_version=candidate.cost_model_version,
        grammar_version=candidate.grammar_version,
        code_hash=code_hash,
        data_hash=data_hash,
        execution_assumptions=assumptions,
        expression_snapshot=candidate.entry_tree.as_dict(),
        oos_fitness=oos_fitness,
        ranking_source=ranking_source,
        discovery_run_id=discovery_run_id,
    )
