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

# Strict semantic components that gate true resume (order is diagnostic display order).
# ``execution_semantic_hash`` is an allowlisted content hash of evaluation-critical
# modules. ``repository_git_sha`` is provenance-only and never gates resume alone.
FINGERPRINT_COMPONENT_KEYS: tuple[str, ...] = (
    "dataset_hash",
    "timeframe",
    "wfo_config_hash",
    "cost_model_version",
    "risk_model_version",
    "execution_semantic_hash",
    "multi_family_config_hash",
    "seed",
    "family_ids",
)

# Subset persisted on checkpoints / eval-cache wiring (plus optional provenance).
ENGINE_FINGERPRINT_KEYS: tuple[str, ...] = (
    "dataset_hash",
    "timeframe",
    "wfo_config_hash",
    "cost_model_version",
    "risk_model_version",
    "execution_semantic_hash",
)

PROVENANCE_COMPONENT_KEYS: tuple[str, ...] = (
    "repository_git_sha",
)

FINGERPRINT_REASON_CODES: dict[str, str] = {
    "dataset_hash": "DATASET_HASH_CHANGED",
    "timeframe": "TIMEFRAME_CHANGED",
    "wfo_config_hash": "WFO_CONFIG_CHANGED",
    "cost_model_version": "COST_MODEL_CHANGED",
    "risk_model_version": "RISK_MODEL_CHANGED",
    "execution_semantic_hash": "EXECUTION_SEMANTICS_CHANGED",
    # Legacy alias retained for diagnostics of pre-migration programs.
    "execution_engine_code_hash": "CODE_HASH_CHANGED",
    "multi_family_config_hash": "MULTI_FAMILY_STRUCTURE_CHANGED",
    "seed": "SEED_CHANGED",
    "family_ids": "FAMILY_IDS_CHANGED",
}

CONTROL_PLANE_CODE_CHANGED = "CONTROL_PLANE_CODE_CHANGED"
EXECUTION_SEMANTICS_CHANGED = "EXECUTION_SEMANTICS_CHANGED"

# Structural research semantics only — session/runtime/dashboard metadata excluded.
MULTI_FAMILY_STRUCTURAL_KEYS: tuple[str, ...] = (
    "requested_family_count",
    "min_candidates_per_family",
    "initial_candidates_per_family",
    "adaptive_reallocation",
    "family_ids",
    "family_local_evolution",
    "evolution_generations",
    "stagnation_generations",
    "minimum_improvement",
    "allow_cross_family_crossover",
    "population_size",
    "min_oos_trades",
    "min_oos_trades_per_fold",
    "max_oos_drawdown",
    "max_stress_scenarios_per_candidate",
    "min_stress_pass_rate",
    "stress_scenarios",
    "fail_closed_unsupported_stress",
    "max_parameters_per_candidate",
    "max_points_per_parameter",
    "allow_one_sided_neighborhood",
    "min_valid_neighborhood_points",
    "min_dsr",
    "max_pbo",
    "pbo_n_splits",
    "behavioral_similarity_threshold",
    "min_oos_observations_for_dsr",
)


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
    multi_family_config_hash: str,
    seed: int,
    family_ids: list[str] | None = None,
    execution_semantic_hash: str | None = None,
    execution_engine_code_hash: str | None = None,
) -> str:
    """Dataset/config/code fingerprint that gates true resume vs reevaluation.

    ``execution_semantic_hash`` is preferred. ``execution_engine_code_hash`` is
    accepted as a legacy alias (older callers / polluted aggregates).
    """
    semantic = (
        str(execution_semantic_hash)
        if execution_semantic_hash not in (None, "")
        else str(execution_engine_code_hash or "")
    )
    payload = {
        "dataset_hash": str(dataset_hash),
        "timeframe": str(timeframe),
        "wfo_config_hash": str(wfo_config_hash),
        "cost_model_version": str(cost_model_version),
        "risk_model_version": str(risk_model_version),
        "execution_semantic_hash": semantic,
        "multi_family_config_hash": str(multi_family_config_hash),
        "seed": int(seed),
        "family_ids": list(family_ids) if family_ids else None,
    }
    return "fp_" + sha256_json(payload)[:32]


def hash_wfo_config(wfo: dict[str, Any] | None) -> str:
    return "wfo_" + sha256_json(dict(wfo or {}))[:24]


def _canonical_scalar(value: Any) -> Any:
    """Normalize JSON-ish values so equivalent configs hash identically."""
    if isinstance(value, tuple):
        return [_canonical_scalar(v) for v in value]
    if isinstance(value, list):
        return [_canonical_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical_scalar(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, float):
        # Avoid -0.0 vs 0.0 and tiny float noise in thresholds.
        if value == 0.0:
            return 0.0
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return value


def canonical_multi_family_payload(mf: dict[str, Any] | None) -> dict[str, Any]:
    """Extract structural multi-family research semantics for fingerprinting.

    Session budgets, resume metadata, and dashboard-only fields are excluded so
    EXTEND_BUDGET / Continue Search clones of an enriched config_snapshot keep
    the same identity as the original NEW_SEARCH.

    Missing structural keys are filled with FamilyCampaignConfig defaults, and
    ``initial_candidates_per_family`` is normalized to the effective gen-0 size
    so sparse UI configs match checkpoint ``as_dict()`` payloads.
    """
    raw = dict(mf or {})

    def _get_int(key: str, default: int) -> int:
        if raw.get(key) is None:
            return int(default)
        return int(raw[key])

    def _get_float(key: str, default: float) -> float:
        if raw.get(key) is None:
            return float(default)
        return float(raw[key])

    def _get_bool(key: str, default: bool) -> bool:
        if key not in raw or raw.get(key) is None:
            return bool(default)
        return bool(raw[key])

    min_per_family = _get_int("min_candidates_per_family", 10)
    if raw.get("initial_candidates_per_family") is not None:
        initial_per_family = max(1, int(raw["initial_candidates_per_family"]))
    else:
        initial_per_family = max(1, min_per_family)

    family_ids = raw.get("family_ids")
    stress = raw.get("stress_scenarios")

    payload: dict[str, Any] = {
        "requested_family_count": _get_int(
            "requested_family_count", int(raw.get("family_count") or 6)
        ),
        "min_candidates_per_family": min_per_family,
        "initial_candidates_per_family": initial_per_family,
        "adaptive_reallocation": _get_bool("adaptive_reallocation", True),
        "family_ids": sorted(str(x) for x in family_ids) if family_ids else None,
        "family_local_evolution": _get_bool("family_local_evolution", False),
        "evolution_generations": _get_int("evolution_generations", 2),
        "stagnation_generations": _get_int("stagnation_generations", 99),
        "minimum_improvement": _canonical_scalar(_get_float("minimum_improvement", 1e-4)),
        "allow_cross_family_crossover": _get_bool("allow_cross_family_crossover", False),
        "population_size": _get_int("population_size", 2),
        "min_oos_trades": _get_int("min_oos_trades", 1),
        "min_oos_trades_per_fold": _get_int("min_oos_trades_per_fold", 1),
        "max_oos_drawdown": _canonical_scalar(
            _get_float("max_oos_drawdown", float(raw.get("max_drawdown_limit") or 0.20))
        ),
        "max_stress_scenarios_per_candidate": _get_int(
            "max_stress_scenarios_per_candidate", 10
        ),
        "min_stress_pass_rate": _canonical_scalar(_get_float("min_stress_pass_rate", 0.5)),
        "stress_scenarios": [str(x) for x in stress] if stress is not None else None,
        "fail_closed_unsupported_stress": _get_bool(
            "fail_closed_unsupported_stress", True
        ),
        "max_parameters_per_candidate": _get_int("max_parameters_per_candidate", 2),
        "max_points_per_parameter": _get_int("max_points_per_parameter", 5),
        "allow_one_sided_neighborhood": _get_bool("allow_one_sided_neighborhood", False),
        "min_valid_neighborhood_points": _get_int("min_valid_neighborhood_points", 3),
        "min_dsr": _canonical_scalar(_get_float("min_dsr", 0.95)),
        "max_pbo": _canonical_scalar(_get_float("max_pbo", 0.50)),
        "pbo_n_splits": _get_int("pbo_n_splits", 4),
        "behavioral_similarity_threshold": _canonical_scalar(
            _get_float("behavioral_similarity_threshold", 0.85)
        ),
        "min_oos_observations_for_dsr": _get_int("min_oos_observations_for_dsr", 20),
    }
    return payload


def hash_multi_family_config(mf: dict[str, Any] | None) -> str:
    """Hash structural multi-family settings via an explicit semantic allowlist."""
    return "mf_" + sha256_json(canonical_multi_family_payload(mf))[:24]


def build_fingerprint_components(
    *,
    dataset_hash: str,
    timeframe: str,
    wfo_config_hash: str,
    cost_model_version: str,
    risk_model_version: str,
    multi_family_config_hash: str,
    seed: int,
    family_ids: list[str] | None = None,
    execution_semantic_hash: str | None = None,
    execution_engine_code_hash: str | None = None,
    repository_git_sha: str | None = None,
) -> dict[str, Any]:
    """Full strict-component map used for diagnostics and identity checks.

    ``repository_git_sha`` is stored for audit/provenance and is not a resume gate.
    """
    semantic = (
        str(execution_semantic_hash)
        if execution_semantic_hash not in (None, "")
        else str(execution_engine_code_hash or "")
    )
    out: dict[str, Any] = {
        "dataset_hash": str(dataset_hash),
        "timeframe": str(timeframe),
        "wfo_config_hash": str(wfo_config_hash),
        "cost_model_version": str(cost_model_version),
        "risk_model_version": str(risk_model_version),
        "execution_semantic_hash": semantic,
        "multi_family_config_hash": str(multi_family_config_hash),
        "seed": int(seed),
        "family_ids": sorted(str(x) for x in family_ids) if family_ids else None,
    }
    if repository_git_sha not in (None, ""):
        out["repository_git_sha"] = str(repository_git_sha)
    return out


def engine_fingerprint_subset(components: dict[str, Any]) -> dict[str, str]:
    """Checkpoint / campaign subset including provenance when present.

    Also mirrors ``execution_semantic_hash`` into ``execution_engine_code_hash``
    so older campaign/eval-cache call sites keep working without touching
    evaluation-critical modules.
    """
    keys = list(ENGINE_FINGERPRINT_KEYS) + list(PROVENANCE_COMPONENT_KEYS)
    out = {
        k: str(components[k])
        for k in keys
        if components.get(k) not in (None, "")
    }
    if "execution_semantic_hash" in out:
        out["execution_engine_code_hash"] = out["execution_semantic_hash"]
    return out


@dataclass
class FingerprintDiff:
    """Component-level compatibility diagnostics for resume gating."""

    stored_fingerprint: str
    incoming_fingerprint: str
    stored_components: dict[str, Any]
    incoming_components: dict[str, Any]
    changed_components: list[str]
    changed_reason_codes: list[str]
    stored_multi_family_canonical: dict[str, Any] = field(default_factory=dict)
    incoming_multi_family_canonical: dict[str, Any] = field(default_factory=dict)
    stored_git_commit_sha: str | None = None
    incoming_git_commit_sha: str | None = None
    compatible: bool = False
    migrated_aggregate: bool = False
    control_plane_code_changed: bool = False
    audit_reason_codes: list[str] = field(default_factory=list)
    legacy_code_hash_migrated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stored_fingerprint": self.stored_fingerprint,
            "incoming_fingerprint": self.incoming_fingerprint,
            "stored_components": dict(self.stored_components),
            "incoming_components": dict(self.incoming_components),
            "changed_components": list(self.changed_components),
            "changed_reason_codes": list(self.changed_reason_codes),
            "stored_multi_family_canonical": dict(self.stored_multi_family_canonical),
            "incoming_multi_family_canonical": dict(self.incoming_multi_family_canonical),
            "stored_git_commit_sha": self.stored_git_commit_sha,
            "incoming_git_commit_sha": self.incoming_git_commit_sha,
            "stored_repository_git_sha": self.stored_git_commit_sha,
            "incoming_repository_git_sha": self.incoming_git_commit_sha,
            "stored_execution_semantic_hash": self.stored_components.get(
                "execution_semantic_hash"
            ),
            "incoming_execution_semantic_hash": self.incoming_components.get(
                "execution_semantic_hash"
            ),
            "stored_dataset_hash": self.stored_components.get("dataset_hash"),
            "incoming_dataset_hash": self.incoming_components.get("dataset_hash"),
            "stored_wfo_hash": self.stored_components.get("wfo_config_hash"),
            "incoming_wfo_hash": self.incoming_components.get("wfo_config_hash"),
            "compatible": self.compatible,
            "migrated_aggregate": self.migrated_aggregate,
            "control_plane_code_changed": self.control_plane_code_changed,
            "audit_reason_codes": list(self.audit_reason_codes),
            "legacy_code_hash_migrated": self.legacy_code_hash_migrated,
        }


class IncompatibleSearchFingerprint(ValueError):
    """Raised when strict resume components diverge."""

    def __init__(self, diff: FingerprintDiff) -> None:
        self.diff = diff
        reasons = ",".join(diff.changed_reason_codes) or INCOMPATIBLE_FINGERPRINT
        super().__init__(
            f"{INCOMPATIBLE_FINGERPRINT}: {reasons} "
            f"stored={diff.stored_fingerprint!r} incoming={diff.incoming_fingerprint!r}"
        )


def _component_semantic_value(components: dict[str, Any]) -> str | None:
    """Prefer execution_semantic_hash; fall back to legacy execution_engine_code_hash."""
    sem = components.get("execution_semantic_hash")
    if sem not in (None, ""):
        return str(sem)
    legacy = components.get("execution_engine_code_hash")
    if legacy not in (None, ""):
        return str(legacy)
    return None


def _component_repo_sha(components: dict[str, Any]) -> str | None:
    repo = components.get("repository_git_sha")
    if repo not in (None, ""):
        return str(repo)
    legacy = components.get("execution_engine_code_hash")
    from discovery.execution_semantic_hash import looks_like_git_sha

    if looks_like_git_sha(legacy):
        return str(legacy)
    return None


def compare_fingerprint_components(
    *,
    stored_fingerprint: str,
    incoming_fingerprint: str,
    stored_components: dict[str, Any],
    incoming_components: dict[str, Any],
    stored_multi_family_canonical: dict[str, Any] | None = None,
    incoming_multi_family_canonical: dict[str, Any] | None = None,
    legacy_code_hash_migrated: bool = False,
) -> FingerprintDiff:
    """Compare strict semantic components; aggregates alone are not authoritative.

    ``repository_git_sha`` may differ without blocking resume when
    ``execution_semantic_hash`` matches; that case records
    ``CONTROL_PLANE_CODE_CHANGED`` as a non-blocking audit reason.
    """
    changed: list[str] = []
    for key in FINGERPRINT_COMPONENT_KEYS:
        if key == "execution_semantic_hash":
            left = _component_semantic_value(stored_components)
            right = _component_semantic_value(incoming_components)
        else:
            left = stored_components.get(key)
            right = incoming_components.get(key)
        if key == "family_ids":
            left = sorted(str(x) for x in left) if left else None
            right = sorted(str(x) for x in right) if right else None
        elif key == "seed":
            left = int(left) if left is not None else None
            right = int(right) if right is not None else None
        else:
            left = None if left is None else str(left)
            right = None if right is None else str(right)
        if left != right:
            changed.append(key)
    reasons = [FINGERPRINT_REASON_CODES[k] for k in changed if k in FINGERPRINT_REASON_CODES]
    compatible = not changed
    migrated = compatible and stored_fingerprint != incoming_fingerprint

    stored_repo = _component_repo_sha(stored_components)
    incoming_repo = _component_repo_sha(incoming_components)
    audit: list[str] = []
    control_plane_changed = False
    if (
        compatible
        and stored_repo
        and incoming_repo
        and stored_repo != incoming_repo
    ):
        control_plane_changed = True
        audit.append(CONTROL_PLANE_CODE_CHANGED)

    return FingerprintDiff(
        stored_fingerprint=str(stored_fingerprint),
        incoming_fingerprint=str(incoming_fingerprint),
        stored_components=dict(stored_components),
        incoming_components=dict(incoming_components),
        changed_components=changed,
        changed_reason_codes=reasons,
        stored_multi_family_canonical=dict(stored_multi_family_canonical or {}),
        incoming_multi_family_canonical=dict(incoming_multi_family_canonical or {}),
        stored_git_commit_sha=stored_repo,
        incoming_git_commit_sha=incoming_repo,
        compatible=compatible,
        migrated_aggregate=migrated,
        control_plane_code_changed=control_plane_changed,
        audit_reason_codes=audit,
        legacy_code_hash_migrated=bool(legacy_code_hash_migrated),
    )


def persist_fingerprint_diff(artifact_dir: Path, diff: FingerprintDiff) -> Path:
    """Write fingerprint_diff.json under a run artifact directory."""
    art = Path(artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    path = art / "fingerprint_diff.json"
    path.write_text(json.dumps(diff.as_dict(), indent=2, default=str), encoding="utf-8")
    return path


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
    stored_components: dict[str, Any] | None = None,
    incoming_components: dict[str, Any] | None = None,
    stored_multi_family_canonical: dict[str, Any] | None = None,
    incoming_multi_family_canonical: dict[str, Any] | None = None,
    artifact_dir: Path | None = None,
    legacy_code_hash_migrated: bool = False,
) -> FingerprintDiff | None:
    """True resume requires identical strict semantic components.

    Aggregate fingerprint equality is preferred, but when an older program was
    hashed with a broader multi-family payload (including runtime/dashboard
    metadata), component-level comparison using the canonical structural
    allowlist is authoritative. Reevaluation may diverge.

    Repository git SHA may change while ``execution_semantic_hash`` stays
    identical; that yields a compatible diff with ``CONTROL_PLANE_CODE_CHANGED``
    audit metadata rather than a hard reject.
    """
    if mode is SearchMode.REEVALUATE_FROZEN_CANDIDATES:
        return None
    if stored == incoming and not legacy_code_hash_migrated:
        # Still surface control-plane code drift when components are provided.
        if stored_components is not None and incoming_components is not None:
            diff = compare_fingerprint_components(
                stored_fingerprint=stored,
                incoming_fingerprint=incoming,
                stored_components=stored_components,
                incoming_components=incoming_components,
                stored_multi_family_canonical=stored_multi_family_canonical,
                incoming_multi_family_canonical=incoming_multi_family_canonical,
                legacy_code_hash_migrated=False,
            )
            if diff.control_plane_code_changed:
                return diff
        return None

    if stored_components is not None and incoming_components is not None:
        diff = compare_fingerprint_components(
            stored_fingerprint=stored,
            incoming_fingerprint=incoming,
            stored_components=stored_components,
            incoming_components=incoming_components,
            stored_multi_family_canonical=stored_multi_family_canonical,
            incoming_multi_family_canonical=incoming_multi_family_canonical,
            legacy_code_hash_migrated=legacy_code_hash_migrated,
        )
        if diff.compatible:
            # Aggregate drift only (e.g. pre-runtime metadata polluted the old hash)
            # and/or legacy whole-repo SHA → semantic-hash schema migration.
            if legacy_code_hash_migrated:
                diff.migrated_aggregate = True
            return diff
        if artifact_dir is not None:
            persist_fingerprint_diff(artifact_dir, diff)
        raise IncompatibleSearchFingerprint(diff)

    # No component diagnostics available — fall back to opaque aggregate check.
    diff = FingerprintDiff(
        stored_fingerprint=stored,
        incoming_fingerprint=incoming,
        stored_components=dict(stored_components or {}),
        incoming_components=dict(incoming_components or {}),
        changed_components=["compatibility_fingerprint"],
        changed_reason_codes=[INCOMPATIBLE_FINGERPRINT],
        stored_multi_family_canonical=dict(stored_multi_family_canonical or {}),
        incoming_multi_family_canonical=dict(incoming_multi_family_canonical or {}),
        compatible=False,
    )
    if artifact_dir is not None:
        persist_fingerprint_diff(artifact_dir, diff)
    raise IncompatibleSearchFingerprint(diff)
