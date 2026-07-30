"""Run types, states, and persistent run records (Phase 12)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from registry.hashing import sha256_json


class RunType(str, Enum):
    RESEARCH_DEMO = "RESEARCH_DEMO"
    ALPHA_MINER = "ALPHA_MINER"
    WFO_ONLY = "WFO_ONLY"
    PORTFOLIO_BUILD = "PORTFOLIO_BUILD"
    VAULT_EVALUATION = "VAULT_EVALUATION"
    SHADOW_RUNTIME = "SHADOW_RUNTIME"
    PAPER_RUNTIME = "PAPER_RUNTIME"


# Explicitly blocked — dashboard must never activate these
FORBIDDEN_LIVE_MODES = frozenset(
    {
        "LIVE",
        "CANARY_LIVE",
        "SMALL_LIVE",
        "LIMITED_PRODUCTION",
        "PRODUCTION",
    }
)


class RunState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL_STATES = frozenset(
    {
        RunState.CANCELLED,
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.INTERRUPTED,
    }
)


@dataclass
class RunRecord:
    run_id: str
    run_type: RunType
    state: RunState
    system_version: str
    config_snapshot: dict[str, Any]
    config_hash: str
    dataset_ids: list[str] = field(default_factory=list)
    feature_set_version: str = "feature_set_v1_phase6b"
    grammar_version: str = "strategy_dsl_v1"
    cost_model_version: str = "cost_v1"
    random_seed: int = 42
    process_id: int | None = None
    worker_id: str | None = None
    created_at: str = ""
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float = 0.0
    progress_pct: float = 0.0
    current_stage: str = ""
    current_message: str = ""
    generated_count: int = 0
    evaluated_count: int = 0
    rejected_count: int = 0
    qualified_count: int = 0
    finalist_count: int = 0
    portfolio_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    artifact_dir: str = ""
    terminal_reason: str | None = None
    # Result honesty
    software_success: bool | None = None
    statistical_validation: str | None = None
    profitability_result: str | None = None
    total_return_pct: float | None = None
    vault_result: str | None = None
    paper_eligible: bool | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    artifact_manifest: list[str] = field(default_factory=list)
    cancel_requested: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_type"] = self.run_type.value
        d["state"] = self.state.value
        return d

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> RunRecord:
        return RunRecord(
            run_id=raw["run_id"],
            run_type=RunType(raw["run_type"]),
            state=RunState(raw["state"]),
            system_version=raw.get("system_version", ""),
            config_snapshot=dict(raw.get("config_snapshot") or {}),
            config_hash=raw.get("config_hash", ""),
            dataset_ids=list(raw.get("dataset_ids") or []),
            feature_set_version=raw.get("feature_set_version", ""),
            grammar_version=raw.get("grammar_version", ""),
            cost_model_version=raw.get("cost_model_version", ""),
            random_seed=int(raw.get("random_seed", 42)),
            process_id=raw.get("process_id"),
            worker_id=raw.get("worker_id"),
            created_at=raw.get("created_at", ""),
            queued_at=raw.get("queued_at"),
            started_at=raw.get("started_at"),
            completed_at=raw.get("completed_at"),
            elapsed_seconds=float(raw.get("elapsed_seconds", 0.0)),
            progress_pct=float(raw.get("progress_pct", 0.0)),
            current_stage=raw.get("current_stage", ""),
            current_message=raw.get("current_message", ""),
            generated_count=int(raw.get("generated_count", 0)),
            evaluated_count=int(raw.get("evaluated_count", 0)),
            rejected_count=int(raw.get("rejected_count", 0)),
            qualified_count=int(raw.get("qualified_count", 0)),
            finalist_count=int(raw.get("finalist_count", 0)),
            portfolio_count=int(raw.get("portfolio_count", 0)),
            warning_count=int(raw.get("warning_count", 0)),
            error_count=int(raw.get("error_count", 0)),
            artifact_dir=raw.get("artifact_dir", ""),
            terminal_reason=raw.get("terminal_reason"),
            software_success=raw.get("software_success"),
            statistical_validation=raw.get("statistical_validation"),
            profitability_result=raw.get("profitability_result"),
            total_return_pct=raw.get("total_return_pct"),
            vault_result=raw.get("vault_result"),
            paper_eligible=raw.get("paper_eligible"),
            summary=dict(raw.get("summary") or {}),
            artifact_manifest=list(raw.get("artifact_manifest") or []),
            cancel_requested=bool(raw.get("cancel_requested", False)),
        )


def new_run_id() -> str:
    return "run_" + uuid4().hex[:16]


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def build_run_record(
    *,
    run_type: RunType,
    system_version: str,
    config_snapshot: dict[str, Any],
    artifact_root: str,
    random_seed: int = 42,
    dataset_ids: list[str] | None = None,
    feature_set_version: str = "feature_set_v1_phase6b",
    grammar_version: str = "strategy_dsl_v1",
    cost_model_version: str = "cost_v1",
) -> RunRecord:
    run_id = new_run_id()
    cfg_hash = "cfg_" + sha256_json(config_snapshot)[:24]
    return RunRecord(
        run_id=run_id,
        run_type=run_type,
        state=RunState.CREATED,
        system_version=system_version,
        config_snapshot=config_snapshot,
        config_hash=cfg_hash,
        dataset_ids=list(dataset_ids or ["synthetic_demo"]),
        feature_set_version=feature_set_version,
        grammar_version=grammar_version,
        cost_model_version=cost_model_version,
        random_seed=random_seed,
        created_at=now_iso(),
        artifact_dir=str(Pathish(artifact_root) / run_id),
    )


def Pathish(p: str):
    from pathlib import Path

    return Path(p)
