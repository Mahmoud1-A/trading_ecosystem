"""Candidate lineage — any material change creates a new lineage_id."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class CandidateLineage:
    """
    Immutable identity for a candidate under evaluation.

    Any change to parameters, features, strategy logic, risk rules,
    portfolio construction, or cost model produces a new lineage_id.
    """

    lineage_id: str
    candidate_id: str
    strategy_family: str
    parameter_hash: str
    config_hash: str
    code_hash: str
    data_hash: str
    cost_model_version: str
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    @staticmethod
    def create(
        *,
        strategy_family: str,
        parameters: dict[str, Any],
        config_snapshot: dict[str, Any],
        code_hash: str,
        data_hash: str,
        cost_model_version: str,
        candidate_id: str | None = None,
    ) -> CandidateLineage:
        param_hash = _stable_hash(parameters)
        cfg_hash = _stable_hash(config_snapshot)
        lineage_material = {
            "strategy_family": strategy_family,
            "parameter_hash": param_hash,
            "config_hash": cfg_hash,
            "code_hash": code_hash,
            "data_hash": data_hash,
            "cost_model_version": cost_model_version,
        }
        lineage_id = _stable_hash(lineage_material)[:32]
        return CandidateLineage(
            lineage_id=lineage_id,
            candidate_id=candidate_id or str(uuid4()),
            strategy_family=strategy_family,
            parameter_hash=param_hash,
            config_hash=cfg_hash,
            code_hash=code_hash,
            data_hash=data_hash,
            cost_model_version=cost_model_version,
        )

    def with_parameter_change(self, new_parameters: dict[str, Any]) -> CandidateLineage:
        """Return a new lineage for modified parameters (cannot reuse vault as untouched)."""
        return CandidateLineage.create(
            strategy_family=self.strategy_family,
            parameters=new_parameters,
            config_snapshot={"config_hash": self.config_hash},
            code_hash=self.code_hash,
            data_hash=self.data_hash,
            cost_model_version=self.cost_model_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "candidate_id": self.candidate_id,
            "strategy_family": self.strategy_family,
            "parameter_hash": self.parameter_hash,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "data_hash": self.data_hash,
            "cost_model_version": self.cost_model_version,
            "created_at": self.created_at,
        }
