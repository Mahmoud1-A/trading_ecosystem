"""Immutable portfolio versioning and freeze lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from registry.hashing import sha256_json


class PortfolioPipelineStage(str, Enum):
    PORTFOLIO_BUILT = "PORTFOLIO_BUILT"
    STABILITY_PASSED = "STABILITY_PASSED"
    FROZEN = "FROZEN"
    VAULT_ELIGIBLE = "VAULT_ELIGIBLE"
    VAULT_TESTED = "VAULT_TESTED"
    VAULT_PASSED = "VAULT_PASSED"
    VAULT_FAILED = "VAULT_FAILED"
    PAPER_ELIGIBLE = "PAPER_ELIGIBLE"


class PortfolioVersionError(ValueError):
    pass


@dataclass(frozen=True)
class PortfolioVersion:
    portfolio_id: str
    portfolio_version: str
    member_candidate_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    weights: dict[str, float]
    construction_method: str
    constraints: dict[str, Any]
    dataset_versions: dict[str, str]
    feature_set_version: str
    cost_model_version: str
    risk_profile: dict[str, Any]
    wfo_results: dict[str, Any]
    stress_results: dict[str, Any]
    frozen: bool
    stage: PortfolioPipelineStage
    creation_timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_version": self.portfolio_version,
            "member_candidate_ids": list(self.member_candidate_ids),
            "lineage_ids": list(self.lineage_ids),
            "weights": dict(self.weights),
            "construction_method": self.construction_method,
            "constraints": dict(self.constraints),
            "dataset_versions": dict(self.dataset_versions),
            "feature_set_version": self.feature_set_version,
            "cost_model_version": self.cost_model_version,
            "risk_profile": dict(self.risk_profile),
            "wfo_results": dict(self.wfo_results),
            "stress_results": dict(self.stress_results),
            "frozen": self.frozen,
            "stage": self.stage.value,
            "creation_timestamp": self.creation_timestamp,
        }

    def with_stage(self, stage: PortfolioPipelineStage) -> PortfolioVersion:
        if self.frozen and stage not in {
            PortfolioPipelineStage.FROZEN,
            PortfolioPipelineStage.VAULT_ELIGIBLE,
            PortfolioPipelineStage.VAULT_TESTED,
            PortfolioPipelineStage.VAULT_PASSED,
            PortfolioPipelineStage.VAULT_FAILED,
            PortfolioPipelineStage.PAPER_ELIGIBLE,
        }:
            raise PortfolioVersionError("cannot mutate frozen portfolio contents")
        # Stage transitions create a new version object without altering frozen payload
        payload = self.as_dict()
        payload["stage"] = stage
        payload["frozen"] = self.frozen or stage is PortfolioPipelineStage.FROZEN
        # Keep same portfolio_version identity when only stage advances after freeze
        return PortfolioVersion(
            portfolio_id=self.portfolio_id,
            portfolio_version=self.portfolio_version,
            member_candidate_ids=self.member_candidate_ids,
            lineage_ids=self.lineage_ids,
            weights=dict(self.weights),
            construction_method=self.construction_method,
            constraints=dict(self.constraints),
            dataset_versions=dict(self.dataset_versions),
            feature_set_version=self.feature_set_version,
            cost_model_version=self.cost_model_version,
            risk_profile=dict(self.risk_profile),
            wfo_results=dict(self.wfo_results),
            stress_results=dict(self.stress_results),
            frozen=payload["frozen"],
            stage=stage,
            creation_timestamp=self.creation_timestamp,
        )


def build_portfolio_version(
    *,
    weights: dict[str, float],
    lineage_ids: dict[str, str],
    construction_method: str,
    constraints: dict[str, Any],
    feature_set_version: str,
    cost_model_version: str,
    dataset_versions: dict[str, str] | None = None,
    risk_profile: dict[str, Any] | None = None,
    wfo_results: dict[str, Any] | None = None,
    stress_results: dict[str, Any] | None = None,
    stage: PortfolioPipelineStage = PortfolioPipelineStage.PORTFOLIO_BUILT,
) -> PortfolioVersion:
    member_ids = tuple(sorted(weights.keys()))
    payload = {
        "members": list(member_ids),
        "weights": {k: weights[k] for k in member_ids},
        "method": construction_method,
        "constraints": constraints,
        "feature_set_version": feature_set_version,
        "cost_model_version": cost_model_version,
    }
    portfolio_id = "port_" + sha256_json(payload)[:20]
    version = "pv_" + sha256_json({**payload, "stage": stage.value})[:16]
    return PortfolioVersion(
        portfolio_id=portfolio_id,
        portfolio_version=version,
        member_candidate_ids=member_ids,
        lineage_ids=tuple(lineage_ids[cid] for cid in member_ids),
        weights={cid: weights[cid] for cid in member_ids},
        construction_method=construction_method,
        constraints=constraints,
        dataset_versions=dataset_versions or {},
        feature_set_version=feature_set_version,
        cost_model_version=cost_model_version,
        risk_profile=risk_profile or {},
        wfo_results=wfo_results or {},
        stress_results=stress_results or {},
        frozen=False,
        stage=stage,
    )


def freeze_portfolio(version: PortfolioVersion) -> PortfolioVersion:
    if version.frozen:
        raise PortfolioVersionError("portfolio already frozen")
    frozen = version.with_stage(PortfolioPipelineStage.FROZEN)
    # Re-hash content into immutable frozen version id
    payload = frozen.as_dict()
    payload["frozen"] = True
    new_ver = "pv_frozen_" + sha256_json(payload)[:16]
    return PortfolioVersion(
        portfolio_id=frozen.portfolio_id,
        portfolio_version=new_ver,
        member_candidate_ids=frozen.member_candidate_ids,
        lineage_ids=frozen.lineage_ids,
        weights=dict(frozen.weights),
        construction_method=frozen.construction_method,
        constraints=dict(frozen.constraints),
        dataset_versions=dict(frozen.dataset_versions),
        feature_set_version=frozen.feature_set_version,
        cost_model_version=frozen.cost_model_version,
        risk_profile=dict(frozen.risk_profile),
        wfo_results=dict(frozen.wfo_results),
        stress_results=dict(frozen.stress_results),
        frozen=True,
        stage=PortfolioPipelineStage.FROZEN,
        creation_timestamp=frozen.creation_timestamp,
    )


def mutate_frozen_weights(version: PortfolioVersion, new_weights: dict[str, float]) -> None:
    if version.frozen:
        raise PortfolioVersionError("portfolio versions are immutable after freezing")
    raise PortfolioVersionError("use build_portfolio_version for new weights")
