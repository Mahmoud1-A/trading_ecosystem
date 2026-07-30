"""Narrow Vault interface for the Miner — no raw Vault data exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from discovery.freezing import FreezeError, FrozenCandidate
from validation.vault import ValidationVault, VaultAccessError


class MinerVaultError(PermissionError):
    pass


@dataclass
class VaultSubmissionRequest:
    """Only frozen identifiers may be submitted — never raw Vault frames."""

    frozen: FrozenCandidate
    portfolio_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frozen_id": self.frozen.frozen_id,
            "candidate_id": self.frozen.candidate_id,
            "lineage_id": self.frozen.lineage_id,
            "config_hash": self.frozen.config_hash,
            "feature_set_version": self.frozen.feature_set_version,
            "cost_model_version": self.frozen.cost_model_version,
            "execution_assumptions": dict(self.frozen.execution_assumptions),
            "portfolio_id": self.portfolio_id,
        }


@dataclass
class MinerVaultGateway:
    """
    One-shot Vault access for frozen candidates only.

    The Miner never receives Vault timestamps, features, returns, or metrics
    except through the opaque evaluate callback result after freeze.
    """

    vault: ValidationVault
    _submitted_frozen_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self._submitted_frozen_ids is None:
            self._submitted_frozen_ids = set()

    def submit(
        self,
        request: VaultSubmissionRequest,
        *,
        evaluate_fn: Callable[[pd.DataFrame], dict[str, Any]],
    ) -> dict[str, Any]:
        frozen = request.frozen
        if not isinstance(frozen, FrozenCandidate):
            raise FreezeError("Vault accepts only FrozenCandidate")
        assert self._submitted_frozen_ids is not None
        if frozen.frozen_id in self._submitted_frozen_ids:
            raise MinerVaultError(f"frozen_id {frozen.frozen_id} already submitted")

        lineage = frozen.to_lineage()
        try:
            result = self.vault.evaluate_frozen_candidate(
                lineage,
                evaluate_fn=evaluate_fn,
                for_optimization=False,
                for_ranking=False,
                for_discovery_dashboard=False,
            )
        except VaultAccessError as exc:
            raise MinerVaultError(str(exc)) from exc

        self._submitted_frozen_ids.add(frozen.frozen_id)
        return {
            "request": request.as_dict(),
            "vault_result": result,
        }

    def peek_vault_forbidden(self) -> None:
        """Guard used by tests — Miner code paths must not call this."""
        raise MinerVaultError("Miner must not access Vault data directly")
