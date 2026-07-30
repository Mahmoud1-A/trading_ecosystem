"""Portfolio promotion and Vault eligibility — frozen portfolios only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from portfolio.versioning import PortfolioPipelineStage, PortfolioVersion, PortfolioVersionError
from validation.vault import ValidationVault, VaultAccessError


class PortfolioVaultError(PermissionError):
    pass


@dataclass
class PortfolioPromoter:
    vault: ValidationVault | None = None

    def mark_vault_eligible(self, version: PortfolioVersion) -> PortfolioVersion:
        if not version.frozen:
            raise PortfolioVaultError("only frozen portfolios can access Vault")
        return version.with_stage(PortfolioPipelineStage.VAULT_ELIGIBLE)

    def submit_to_vault(
        self,
        version: PortfolioVersion,
        *,
        evaluate_fn: Callable[[pd.DataFrame], dict[str, Any]],
    ) -> tuple[PortfolioVersion, dict[str, Any]]:
        if not version.frozen:
            raise PortfolioVaultError("only frozen portfolios can access Vault")
        if self.vault is None:
            raise PortfolioVaultError("no vault configured")
        if version.stage not in {
            PortfolioPipelineStage.FROZEN,
            PortfolioPipelineStage.VAULT_ELIGIBLE,
        }:
            # Allow re-entry only via new portfolio version / lineage
            if version.stage in {
                PortfolioPipelineStage.VAULT_TESTED,
                PortfolioPipelineStage.VAULT_PASSED,
                PortfolioPipelineStage.VAULT_FAILED,
            }:
                raise PortfolioVaultError("vault already tested for this portfolio version")

        from validation.candidate_lineage import CandidateLineage

        lineage = CandidateLineage.create(
            strategy_family="portfolio",
            parameters=dict(version.weights),
            config_snapshot={
                "portfolio_id": version.portfolio_id,
                "portfolio_version": version.portfolio_version,
                "feature_set_version": version.feature_set_version,
            },
            code_hash=version.portfolio_version,
            data_hash=version.dataset_versions.get("data", "unknown"),
            cost_model_version=version.cost_model_version,
            candidate_id=version.portfolio_id,
        )
        try:
            result = self.vault.evaluate_frozen_candidate(
                lineage,
                evaluate_fn=evaluate_fn,
                for_optimization=False,
                for_ranking=False,
            )
        except VaultAccessError as exc:
            failed = version.with_stage(PortfolioPipelineStage.VAULT_FAILED)
            raise PortfolioVaultError(str(exc)) from exc

        tested = version.with_stage(PortfolioPipelineStage.VAULT_TESTED)
        passed = tested.with_stage(PortfolioPipelineStage.VAULT_PASSED)
        return passed, result
