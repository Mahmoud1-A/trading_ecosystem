"""Experiment registry — persist every trial including rejected candidates."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from metrics.dsr import DSRResult, compute_deflated_sharpe
from metrics.pbo import PBOResult, compute_pbo
from metrics.trial_population import TrialPopulation, TrialSelection, build_trial_population
from registry.hashing import git_commit_hash, sha256_json

logger = logging.getLogger(__name__)


class TrialStatus(str, Enum):
    """Outcome of a trial. FAILED trials still consumed a test."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class TrialRecord:
    trial_id: str
    candidate_id: str
    lineage_id: str
    strategy_family: str
    parameters: dict[str, Any]
    config_snapshot: dict[str, Any]
    system_version: str
    git_commit_hash: str
    data_hash: str
    random_seed: int
    train_window: dict[str, str] | None
    validation_window: dict[str, str] | None
    vault_version: str | None
    gross_metrics: dict[str, Any]
    net_metrics: dict[str, Any]
    ranking_score: float | None
    rejection_reason: str | None
    trade_log_path: str | None
    equity_curve_path: str | None
    execution_assumptions: dict[str, Any]
    cost_model_version: str
    code_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    trial_status: str = TrialStatus.COMPLETED.value
    failure_reason: str | None = None
    fold_records: list[dict[str, Any]] = field(default_factory=list)
    ranking_source: str = "validation_oos"

    @property
    def failed(self) -> bool:
        return self.trial_status == TrialStatus.FAILED.value

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """
    Append-only trial ledger.

    Rejected candidates are never hidden — required for DSR/PBO.
    """

    def __init__(self, store_path: Path) -> None:
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.store_path / "trial_ledger.jsonl"
        self._trials: list[TrialRecord] = []
        self._load()

    def _load(self) -> None:
        if not self.ledger_path.exists():
            return
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            self._trials.append(TrialRecord(**json.loads(line)))

    def record(self, trial: TrialRecord) -> TrialRecord:
        self._trials.append(trial)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trial.as_dict(), default=str) + "\n")
        logger.info(
            "Registry trial=%s lineage=%s rejected=%s score=%s",
            trial.trial_id,
            trial.lineage_id,
            trial.rejection_reason,
            trial.ranking_score,
        )
        return trial

    def create_trial(
        self,
        *,
        candidate_id: str,
        lineage_id: str,
        strategy_family: str,
        parameters: dict[str, Any],
        config_snapshot: dict[str, Any],
        system_version: str,
        data_hash: str,
        random_seed: int,
        cost_model_version: str,
        code_hash: str,
        gross_metrics: dict[str, Any] | None = None,
        net_metrics: dict[str, Any] | None = None,
        ranking_score: float | None = None,
        rejection_reason: str | None = None,
        train_window: dict[str, str] | None = None,
        validation_window: dict[str, str] | None = None,
        vault_version: str | None = None,
        trade_log_path: str | None = None,
        equity_curve_path: str | None = None,
        execution_assumptions: dict[str, Any] | None = None,
        cwd: Path | None = None,
        trial_status: TrialStatus | str = TrialStatus.COMPLETED,
        failure_reason: str | None = None,
        fold_records: list[dict[str, Any]] | None = None,
        ranking_source: str = "validation_oos",
        trial_id: str | None = None,
    ) -> TrialRecord:
        trial = TrialRecord(
            trial_id=trial_id or str(uuid4()),
            candidate_id=candidate_id,
            lineage_id=lineage_id,
            strategy_family=strategy_family,
            parameters=parameters,
            config_snapshot=config_snapshot,
            system_version=system_version,
            git_commit_hash=git_commit_hash(cwd),
            data_hash=data_hash,
            random_seed=random_seed,
            train_window=train_window,
            validation_window=validation_window,
            vault_version=vault_version,
            gross_metrics=gross_metrics or {},
            net_metrics=net_metrics or {},
            ranking_score=ranking_score,
            rejection_reason=rejection_reason,
            trade_log_path=trade_log_path,
            equity_curve_path=equity_curve_path,
            execution_assumptions=execution_assumptions or {},
            cost_model_version=cost_model_version,
            code_hash=code_hash,
            trial_status=(
                trial_status.value if isinstance(trial_status, TrialStatus) else str(trial_status)
            ),
            failure_reason=failure_reason,
            fold_records=list(fold_records or []),
            ranking_source=ranking_source,
        )
        return self.record(trial)

    def all_trials(self) -> list[TrialRecord]:
        return list(self._trials)

    def rejected_trials(self) -> list[TrialRecord]:
        return [t for t in self._trials if t.rejection_reason]

    def accepted_trials(self) -> list[TrialRecord]:
        return [t for t in self._trials if not t.rejection_reason]

    def failed_trials(self) -> list[TrialRecord]:
        return [t for t in self._trials if t.failed]

    def trial_history_for_dsr(self) -> list[float]:
        """Full net ranking scores including rejects — required for DSR/PBO."""
        scores: list[float] = []
        for t in self._trials:
            if t.ranking_score is not None:
                scores.append(float(t.ranking_score))
            elif "sharpe" in t.net_metrics:
                scores.append(float(t.net_metrics["sharpe"]))
        return scores

    def trial_population(
        self,
        *,
        correlation_matrix: Any | None = None,
    ) -> TrialPopulation:
        """
        Full trial population for DSR/PBO — never a Top-N shortlist.

        Scored trials contribute their ranking score (rejected candidates included);
        failed trials have no score but still count toward ``total_trials`` so the
        multiple-testing adjustment reflects the real search intensity.
        """
        scores: list[float] = []
        failed = 0
        for t in self._trials:
            if t.failed:
                failed += 1
                continue
            if t.ranking_score is not None:
                scores.append(float(t.ranking_score))
            elif "sharpe" in t.net_metrics:
                scores.append(float(t.net_metrics["sharpe"]))
        return build_trial_population(
            scores,
            total_trials=len(self._trials),
            rejected_trials=len(self.rejected_trials()),
            failed_trials=failed,
            selection=TrialSelection.ALL_TRIALS,
            correlation_matrix=correlation_matrix,
        )

    def evaluate_overfitting(
        self,
        *,
        observed_sharpe: float,
        n_observations: int,
        performance_matrix: Any | None = None,
        n_splits: int = 8,
        correlation_matrix: Any | None = None,
    ) -> dict[str, Any]:
        """
        Registry-level DSR + PBO evaluation over the complete trial population.

        Deterministic: the same ledger yields byte-identical results. Both statistics
        carry their own status, so an unmet requirement surfaces as
        INSUFFICIENT_DATA rather than a misleadingly precise number.
        """
        population = self.trial_population(correlation_matrix=correlation_matrix)
        dsr: DSRResult = compute_deflated_sharpe(
            observed_sharpe, population, n_observations=n_observations
        )
        pbo: PBOResult | None = None
        if performance_matrix is not None:
            pbo = compute_pbo(performance_matrix, population=population, n_splits=n_splits)
        return {
            "trial_population": population.as_dict(),
            "dsr": dsr.as_dict(),
            "pbo": pbo.as_dict() if pbo is not None else None,
            "registry_snapshot_hash": self.snapshot_hash(),
        }

    def snapshot_hash(self) -> str:
        return sha256_json([t.as_dict() for t in self._trials])
