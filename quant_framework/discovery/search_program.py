"""Persistent Alpha Miner search programs spanning multiple run sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from registry.hashing import sha256_json


class SearchMode(str, Enum):
    NEW_SEARCH = "NEW_SEARCH"
    RESUME_SEARCH = "RESUME_SEARCH"
    EXTEND_BUDGET = "EXTEND_BUDGET"
    REEVALUATE_FROZEN_CANDIDATES = "REEVALUATE_FROZEN_CANDIDATES"


class PipelinePhase(str, Enum):
    """Ordered resume phases for a Multi-Family evolutionary campaign."""

    GENERATION_0 = "GENERATION_0"
    EVOLUTION = "EVOLUTION"
    PENDING_EVAL = "PENDING_EVAL"
    STRESS = "STRESS"
    ROBUSTNESS = "ROBUSTNESS"
    STATISTICS = "STATISTICS"
    COMPLETE = "COMPLETE"


RESUME_PHASE_ORDER: tuple[PipelinePhase, ...] = (
    PipelinePhase.STRESS,
    PipelinePhase.ROBUSTNESS,
    PipelinePhase.STATISTICS,
    PipelinePhase.PENDING_EVAL,
    PipelinePhase.EVOLUTION,
)


INCOMPATIBLE_FINGERPRINT = "INCOMPATIBLE_SEARCH_FINGERPRINT"
MISSING_CHECKPOINT = "MISSING_SEARCH_CHECKPOINT"
INVALID_RESUME_MODE = "INVALID_RESUME_MODE"


def new_search_program_id() -> str:
    return "sp_" + uuid4().hex[:16]


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def compute_compatibility_fingerprint(
    *,
    dataset_hash: str,
    timeframe: str,
    wfo_config_hash: str,
    cost_model_version: str,
    risk_model_version: str,
    execution_engine_code_hash: str,
    multi_family_config_hash: str,
    seed: int,
    family_ids: list[str] | None = None,
) -> str:
    """Dataset/config/code fingerprint that gates true resume vs reevaluation."""
    payload = {
        "dataset_hash": str(dataset_hash),
        "timeframe": str(timeframe),
        "wfo_config_hash": str(wfo_config_hash),
        "cost_model_version": str(cost_model_version),
        "risk_model_version": str(risk_model_version),
        "execution_engine_code_hash": str(execution_engine_code_hash),
        "multi_family_config_hash": str(multi_family_config_hash),
        "seed": int(seed),
        "family_ids": list(family_ids) if family_ids else None,
    }
    return "fp_" + sha256_json(payload)[:32]


def hash_wfo_config(wfo: dict[str, Any] | None) -> str:
    return "wfo_" + sha256_json(dict(wfo or {}))[:24]


def hash_multi_family_config(mf: dict[str, Any] | None) -> str:
    """Hash structural multi-family settings (exclude session budget extensions)."""
    raw = dict(mf or {})
    # Session-only budget extensions must not change program identity.
    for k in (
        "max_runtime_seconds",
        "total_candidate_budget",
        "max_full_wfo",
        "max_evaluated_candidates",
        "max_stress_evaluations",
        "max_robustness_evaluations",
        "search_mode",
        "search_program_id",
        "source_run_id",
        "resumed_from_run_id",
        "additional_runtime_seconds",
        "additional_generated_budget",
        "additional_full_wfo_budget",
    ):
        raw.pop(k, None)
    return "mf_" + sha256_json(raw)[:24]


@dataclass
class SearchSessionRecord:
    run_id: str
    search_mode: str
    source_run_id: str | None = None
    resumed_from_run_id: str | None = None
    started_at: str = ""
    completed_at: str | None = None
    terminal_reason: str | None = None
    session_generated: int = 0
    session_evaluated: int = 0
    session_full_wfo: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> SearchSessionRecord:
        return SearchSessionRecord(
            run_id=str(raw["run_id"]),
            search_mode=str(raw.get("search_mode") or SearchMode.NEW_SEARCH.value),
            source_run_id=raw.get("source_run_id"),
            resumed_from_run_id=raw.get("resumed_from_run_id"),
            started_at=str(raw.get("started_at") or ""),
            completed_at=raw.get("completed_at"),
            terminal_reason=raw.get("terminal_reason"),
            session_generated=int(raw.get("session_generated") or 0),
            session_evaluated=int(raw.get("session_evaluated") or 0),
            session_full_wfo=int(raw.get("session_full_wfo") or 0),
            notes=dict(raw.get("notes") or {}),
        )


@dataclass
class SearchProgramRecord:
    search_program_id: str
    compatibility_fingerprint: str
    created_at: str
    seed: int
    family_ids: list[str] = field(default_factory=list)
    sessions: list[SearchSessionRecord] = field(default_factory=list)
    cumulative_generated: int = 0
    cumulative_evaluated: int = 0
    cumulative_full_wfo: int = 0
    latest_checkpoint_path: str | None = None
    latest_run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "search_program_id": self.search_program_id,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "created_at": self.created_at,
            "seed": self.seed,
            "family_ids": list(self.family_ids),
            "sessions": [s.as_dict() for s in self.sessions],
            "cumulative_generated": self.cumulative_generated,
            "cumulative_evaluated": self.cumulative_evaluated,
            "cumulative_full_wfo": self.cumulative_full_wfo,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "latest_run_id": self.latest_run_id,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> SearchProgramRecord:
        return SearchProgramRecord(
            search_program_id=str(raw["search_program_id"]),
            compatibility_fingerprint=str(raw["compatibility_fingerprint"]),
            created_at=str(raw.get("created_at") or ""),
            seed=int(raw.get("seed") or 42),
            family_ids=list(raw.get("family_ids") or []),
            sessions=[
                SearchSessionRecord.from_dict(s) for s in (raw.get("sessions") or [])
            ],
            cumulative_generated=int(raw.get("cumulative_generated") or 0),
            cumulative_evaluated=int(raw.get("cumulative_evaluated") or 0),
            cumulative_full_wfo=int(raw.get("cumulative_full_wfo") or 0),
            latest_checkpoint_path=raw.get("latest_checkpoint_path"),
            latest_run_id=raw.get("latest_run_id"),
            metadata=dict(raw.get("metadata") or {}),
        )


class SearchProgramStore:
    """Filesystem store for search programs under ``{root}/search_programs/{id}/``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def program_dir(self, search_program_id: str) -> Path:
        return self.root / search_program_id

    def program_path(self, search_program_id: str) -> Path:
        return self.program_dir(search_program_id) / "program.json"

    def checkpoint_path(self, search_program_id: str) -> Path:
        return self.program_dir(search_program_id) / "checkpoint.json"

    def evaluation_cache_dir(self, search_program_id: str) -> Path:
        return self.program_dir(search_program_id) / "evaluation_cache"

    def create(
        self,
        *,
        compatibility_fingerprint: str,
        seed: int,
        family_ids: list[str] | None = None,
        search_program_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchProgramRecord:
        pid = search_program_id or new_search_program_id()
        rec = SearchProgramRecord(
            search_program_id=pid,
            compatibility_fingerprint=compatibility_fingerprint,
            created_at=now_iso(),
            seed=int(seed),
            family_ids=list(family_ids or []),
            metadata=dict(metadata or {}),
        )
        self.save(rec)
        self.evaluation_cache_dir(pid).mkdir(parents=True, exist_ok=True)
        return rec

    def save(self, rec: SearchProgramRecord) -> None:
        d = self.program_dir(rec.search_program_id)
        d.mkdir(parents=True, exist_ok=True)
        path = self.program_path(rec.search_program_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec.as_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def get(self, search_program_id: str) -> SearchProgramRecord | None:
        path = self.program_path(search_program_id)
        if not path.is_file():
            return None
        return SearchProgramRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def append_session(self, search_program_id: str, session: SearchSessionRecord) -> SearchProgramRecord:
        rec = self.get(search_program_id)
        if rec is None:
            raise KeyError(search_program_id)
        rec.sessions.append(session)
        rec.latest_run_id = session.run_id
        self.save(rec)
        return rec

    def update_cumulative(
        self,
        search_program_id: str,
        *,
        generated: int | None = None,
        evaluated: int | None = None,
        full_wfo: int | None = None,
        checkpoint_path: str | None = None,
    ) -> SearchProgramRecord:
        rec = self.get(search_program_id)
        if rec is None:
            raise KeyError(search_program_id)
        if generated is not None:
            rec.cumulative_generated = int(generated)
        if evaluated is not None:
            rec.cumulative_evaluated = int(evaluated)
        if full_wfo is not None:
            rec.cumulative_full_wfo = int(full_wfo)
        if checkpoint_path is not None:
            rec.latest_checkpoint_path = checkpoint_path
        self.save(rec)
        return rec

    def delete(self, search_program_id: str) -> None:
        """Remove a program directory (used to roll back failed atomic bootstrap)."""
        import shutil

        d = self.program_dir(search_program_id)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def assert_fingerprint_compatible(
    stored: str,
    incoming: str,
    *,
    mode: SearchMode,
) -> None:
    """True resume requires an identical fingerprint; reevaluation may diverge."""
    if mode is SearchMode.REEVALUATE_FROZEN_CANDIDATES:
        return
    if stored != incoming:
        raise ValueError(
            f"{INCOMPATIBLE_FINGERPRINT}: stored={stored!r} incoming={incoming!r} "
            f"mode={mode.value}"
        )
