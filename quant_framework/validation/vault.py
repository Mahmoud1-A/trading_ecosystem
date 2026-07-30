"""Immutable final Validation Vault — one-shot access per lineage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from validation.candidate_lineage import CandidateLineage

logger = logging.getLogger(__name__)


class VaultAccessError(PermissionError):
    """Raised when vault rules are violated (reuse, optimization access, etc.)."""


@dataclass
class VaultAccessRecord:
    vault_version: str
    candidate_id: str
    lineage_id: str
    evaluation_timestamp: str
    result: dict[str, Any]
    access_count: int


@dataclass
class ValidationVault:
    """
    Immutable hold-out vault.

    Rules:
    - Never used in optimization or candidate ranking.
    - Only evaluated after a candidate is frozen.
    - A lineage may access a specific vault version only once.
    - Parameter/logic changes create a new lineage (cannot reuse as untouched).
    """

    vault_version: str
    data: pd.DataFrame
    data_hash: str
    store_path: Path | None = None
    _access_log: dict[str, VaultAccessRecord] = field(default_factory=dict, repr=False)
    _frozen: bool = True

    def __post_init__(self) -> None:
        if self.store_path is not None:
            self.store_path = Path(self.store_path)
            self.store_path.mkdir(parents=True, exist_ok=True)
            self._load_access_log()

    def _key(self, lineage: CandidateLineage) -> str:
        return f"{self.vault_version}:{lineage.lineage_id}"

    def _access_log_path(self) -> Path | None:
        if self.store_path is None:
            return None
        return self.store_path / f"vault_{self.vault_version}_access.json"

    def _load_access_log(self) -> None:
        path = self._access_log_path()
        if path is None or not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        for k, v in raw.items():
            self._access_log[k] = VaultAccessRecord(**v)

    def _persist_access_log(self) -> None:
        path = self._access_log_path()
        if path is None:
            return
        payload = {
            k: {
                "vault_version": r.vault_version,
                "candidate_id": r.candidate_id,
                "lineage_id": r.lineage_id,
                "evaluation_timestamp": r.evaluation_timestamp,
                "result": r.result,
                "access_count": r.access_count,
            }
            for k, r in self._access_log.items()
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def has_been_accessed(self, lineage: CandidateLineage) -> bool:
        return self._key(lineage) in self._access_log

    def access_count(self, lineage: CandidateLineage) -> int:
        rec = self._access_log.get(self._key(lineage))
        return 0 if rec is None else rec.access_count

    def evaluate_frozen_candidate(
        self,
        lineage: CandidateLineage,
        *,
        evaluate_fn: Callable[[pd.DataFrame], dict[str, Any]],
        for_optimization: bool = False,
        for_ranking: bool = False,
        for_discovery_dashboard: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate a frozen candidate once on vault data.

        Blocks optimization / ranking / discovery dashboard misuse and second access.
        """
        if for_optimization:
            raise VaultAccessError("Vault data must never be used in optimization")
        if for_ranking:
            raise VaultAccessError("Vault data must never appear in candidate ranking")
        if for_discovery_dashboard:
            raise VaultAccessError("Vault data must never appear in discovery dashboards")

        key = self._key(lineage)
        if key in self._access_log:
            raise VaultAccessError(
                f"Lineage {lineage.lineage_id} already accessed vault {self.vault_version} "
                f"(access_count={self._access_log[key].access_count}). "
                "Modified candidates must create a new lineage."
            )

        logger.info(
            "Vault one-shot access vault=%s lineage=%s candidate=%s",
            self.vault_version,
            lineage.lineage_id,
            lineage.candidate_id,
        )
        result = evaluate_fn(self.data.copy())
        record = VaultAccessRecord(
            vault_version=self.vault_version,
            candidate_id=lineage.candidate_id,
            lineage_id=lineage.lineage_id,
            evaluation_timestamp=datetime.now(tz=timezone.utc).isoformat(),
            result=result,
            access_count=1,
        )
        self._access_log[key] = record
        self._persist_access_log()
        return {
            "vault_version": self.vault_version,
            "candidate_id": lineage.candidate_id,
            "lineage_id": lineage.lineage_id,
            "evaluation_timestamp": record.evaluation_timestamp,
            "result": result,
            "access_count": 1,
        }
