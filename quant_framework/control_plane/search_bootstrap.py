"""Bootstrap a search checkpoint from a prior Alpha Miner run's artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from discovery.candidate import StrategyCandidate, collect_parameters
from discovery.evaluation_cache import EvaluationCache
from discovery.expression_tree import ExprNode
from discovery.legacy_parameters import (
    LEGACY_PARAMETER_VALUE_AMBIGUOUS,
    LegacyParameterError,
    finite_float,
    normalize_legacy_candidate_parameters,
    sanitize_fitness_components,
    value_shape,
)
from discovery.search_checkpoint import (
    CHECKPOINT_VERSION,
    FamilyCheckpointState,
    SearchCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from discovery.search_program import (
    CONTROL_PLANE_CODE_CHANGED,
    MISSING_CHECKPOINT,
    PipelinePhase,
    SearchMode,
    SearchProgramStore,
    assert_fingerprint_compatible,
    build_fingerprint_components,
    canonical_multi_family_payload,
    engine_fingerprint_subset,
    hash_multi_family_config,
    new_search_program_id,
    now_iso,
)
from discovery.execution_semantic_hash import (
    normalize_legacy_code_hash_component,
    resolve_repo_root,
)
from discovery.search_resume import (
    mark_score_qualified_pending,
    rebuild_candidates,
    rebuild_evaluation_records,
)
from discovery.types import CreationMethod

LEGACY_CANDIDATE_SPEC_INCOMPLETE = "LEGACY_CANDIDATE_SPEC_INCOMPLETE"


class LegacyBootstrapError(RuntimeError):
    """Fail-closed migration error during legacy search bootstrap."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        candidate_id: str | None = None,
        parameter: str | None = None,
        phase: str = "legacy_bootstrap",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.candidate_id = candidate_id
        self.parameter = parameter
        self.phase = phase
        self.details = dict(details or {})


@dataclass
class LegacyBootstrapReport:
    source_run_id: str | None = None
    search_program_id: str | None = None
    trials_scanned: int = 0
    candidates_reconstructed: int = 0
    rejected_history_only_preserved: int = 0
    list_valued_parameter_fields: list[dict[str, Any]] = field(default_factory=list)
    values_recovered_from_dsl: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_candidate_ids: list[str] = field(default_factory=list)
    incomplete_candidate_ids: list[str] = field(default_factory=list)
    migration_records: list[dict[str, Any]] = field(default_factory=list)
    pending_score_qualified_restored: list[str] = field(default_factory=list)
    reconstruction_unavailable_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_run_id": self.source_run_id,
            "search_program_id": self.search_program_id,
            "trials_scanned": self.trials_scanned,
            "candidates_reconstructed": self.candidates_reconstructed,
            "rejected_history_only_preserved": self.rejected_history_only_preserved,
            "list_valued_parameter_fields": list(self.list_valued_parameter_fields),
            "values_recovered_from_dsl": list(self.values_recovered_from_dsl),
            "ambiguous_candidate_ids": list(self.ambiguous_candidate_ids),
            "incomplete_candidate_ids": list(self.incomplete_candidate_ids),
            "migration_records": list(self.migration_records),
            "pending_score_qualified_restored": list(self.pending_score_qualified_restored),
            "reconstruction_unavailable_ids": list(self.reconstruction_unavailable_ids),
        }


@dataclass
class PreparedSearchSession:
    search_program_id: str
    resume_checkpoint: SearchCheckpoint | None
    checkpoint_path: Path
    evaluation_cache: EvaluationCache
    created_new_program: bool
    bootstrapped_from_source: bool
    bootstrap_report: LegacyBootstrapReport | None = None
    fingerprint_migrated: bool = False
    control_plane_code_changed: bool = False
    legacy_code_hash_migrated: bool = False
    migration_info: dict[str, Any] = field(default_factory=dict)


def _load_source_multi_family(artifacts_root: Path, source_run_id: str | None) -> dict[str, Any]:
    if not source_run_id:
        return {}
    src = Path(artifacts_root) / str(source_run_id)
    for name in ("run_record.json", "run_summary.json", "multi_family_campaign.json"):
        path = src / name
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if name == "multi_family_campaign.json":
            cfg = raw.get("config") or (raw.get("budget_allocation") or {}).get("config")
            if isinstance(cfg, dict):
                return dict(cfg)
            continue
        snap = raw.get("config_snapshot") or {}
        mf = snap.get("multi_family") or raw.get("multi_family")
        if isinstance(mf, dict):
            return dict(mf)
    return {}


def _stored_fingerprint_context(
    *,
    program: Any,
    checkpoint: SearchCheckpoint | None,
    artifacts_root: Path,
    source_run_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild stored strict components using canonical multi-family semantics."""
    stored_mf: dict[str, Any] = {}
    if checkpoint is not None and checkpoint.config:
        stored_mf = dict(checkpoint.config)
    source_mf = _load_source_multi_family(artifacts_root, source_run_id)
    if not stored_mf:
        stored_mf = dict(source_mf)

    ckpt_comps = dict(checkpoint.fingerprint_components) if checkpoint is not None else {}
    meta_comps: dict[str, Any] = {}
    if isinstance(getattr(program, "metadata", None), dict):
        meta_comps = dict(program.metadata.get("fingerprint_components") or {})
    source_comps = dict(source_mf.get("fingerprint_components") or {})

    def _pick(key: str) -> str:
        for src in (ckpt_comps, meta_comps, source_comps):
            if src.get(key) not in (None, ""):
                return str(src[key])
        return ""

    mf_hash = hash_multi_family_config(stored_mf) if stored_mf else str(
        ckpt_comps.get("multi_family_config_hash")
        or meta_comps.get("multi_family_config_hash")
        or source_comps.get("multi_family_config_hash")
        or ""
    )
    family_ids = list(program.family_ids) if program.family_ids else None
    if family_ids is None and stored_mf.get("family_ids"):
        family_ids = list(stored_mf.get("family_ids") or [])

    stored_components = build_fingerprint_components(
        dataset_hash=_pick("dataset_hash"),
        timeframe=_pick("timeframe"),
        wfo_config_hash=_pick("wfo_config_hash"),
        cost_model_version=_pick("cost_model_version"),
        risk_model_version=_pick("risk_model_version"),
        execution_semantic_hash=_pick("execution_semantic_hash"),
        execution_engine_code_hash=_pick("execution_engine_code_hash"),
        repository_git_sha=_pick("repository_git_sha"),
        multi_family_config_hash=mf_hash,
        seed=int(program.seed),
        family_ids=family_ids,
    )
    # Ensure legacy whole-repo SHA remains visible for schema migration even when
    # build_fingerprint_components promoted it into execution_semantic_hash.
    legacy = _pick("execution_engine_code_hash")
    if legacy:
        stored_components["execution_engine_code_hash"] = legacy
    return stored_components, canonical_multi_family_payload(stored_mf)


def _tree_from_raw(raw: Any) -> ExprNode | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"tree payload must be dict, got {type(raw).__name__}")
    return ExprNode.from_dict(raw)


def _authoritative_candidate_payload(trial: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer a full persisted candidate payload over entry-only legacy snapshots."""
    snap = dict(trial.get("config_snapshot") or {})
    for key in ("candidate", "strategy_candidate", "candidate_payload"):
        raw = snap.get(key)
        if isinstance(raw, dict) and (raw.get("entry_tree") or raw.get("expression_tree")):
            payload = dict(raw)
            payload.setdefault("candidate_id", trial.get("candidate_id"))
            payload.setdefault("lineage_id", trial.get("lineage_id") or trial.get("candidate_id"))
            if "parameters" not in payload and trial.get("parameters") is not None:
                payload["parameters"] = dict(trial.get("parameters") or {})
            return payload

    entry_raw = snap.get("entry_tree") or snap.get("expression_tree")
    if not isinstance(entry_raw, dict):
        return None

    explicit_trees = any(
        k in snap for k in ("exit_tree", "stop", "target", "sizing", "regime_gates")
    )
    if not explicit_trees:
        return None

    creation = snap.get("creation_method") or CreationMethod.RANDOM.value
    return {
        "candidate_id": str(trial["candidate_id"]),
        "lineage_id": str(trial.get("lineage_id") or trial["candidate_id"]),
        "generation": int(snap.get("generation") or 0),
        "parent_ids": list(snap.get("parent_ids") or ()),
        "creation_method": creation,
        "strategy_family": str(trial.get("strategy_family") or "dsl_generated"),
        "expression_tree": snap.get("expression_tree") or entry_raw,
        "entry_tree": entry_raw,
        "exit_tree": snap.get("exit_tree"),
        "stop": snap.get("stop"),
        "target": snap.get("target"),
        "sizing": snap.get("sizing"),
        "regime_gates": list(snap.get("regime_gates") or []),
        "feature_ids": list(snap.get("feature_ids") or []),
        "parameters": dict(trial.get("parameters") or {}),
        "complexity_score": float(snap.get("complexity") or snap.get("complexity_score") or 0.0),
        "grammar_version": str(snap.get("grammar_version") or ""),
        "feature_set_version": str(snap.get("feature_set_version") or ""),
        "cost_model_version": str(trial.get("cost_model_version") or "cost_v1"),
        "asset_universe": list(snap.get("asset_universe") or ("ES",)),
        "random_seed": int(trial.get("random_seed") or snap.get("random_seed") or 0),
        "creation_timestamp": str(snap.get("creation_timestamp") or ""),
        "family_provenance": dict(snap.get("family_provenance") or {}),
    }


def _candidate_complete_for_stress(cand: StrategyCandidate) -> bool:
    """Executable Stress requires the same tree set that produced ``parameters``."""
    tree_params = collect_parameters(
        cand.entry_tree,
        cand.exit_tree,
        cand.stop,
        cand.target,
        cand.sizing,
        *cand.regime_gates,
    )
    orphan = sorted(set(cand.parameters) - set(tree_params))
    return not orphan


def _candidate_from_trial(
    trial: dict[str, Any],
    *,
    report: LegacyBootstrapReport,
    require_complete: bool,
) -> StrategyCandidate | None:
    """Rebuild a candidate from a legacy trial.

    Returns ``None`` when reconstruction is unavailable (caller may preserve the
    trial in the multiple-testing population without an executable spec).
    """
    cid = str(trial.get("candidate_id") or "")
    snap = dict(trial.get("config_snapshot") or {})
    payload = _authoritative_candidate_payload(trial)

    try:
        if payload is not None:
            entry = _tree_from_raw(payload.get("entry_tree") or payload.get("expression_tree"))
            if entry is None:
                raise ValueError("missing entry_tree")
            exit_tree = _tree_from_raw(payload.get("exit_tree"))
            stop = _tree_from_raw(payload.get("stop"))
            target = _tree_from_raw(payload.get("target"))
            sizing = _tree_from_raw(payload.get("sizing"))
            regime_gates = tuple(
                ExprNode.from_dict(g) for g in (payload.get("regime_gates") or [])
            )
            creation = payload.get("creation_method") or CreationMethod.RANDOM.value
            try:
                method = (
                    creation
                    if isinstance(creation, CreationMethod)
                    else CreationMethod(str(creation))
                )
            except ValueError:
                method = CreationMethod.RANDOM
            params, recovered = normalize_legacy_candidate_parameters(
                (entry, exit_tree, stop, target, sizing, *regime_gates),
                dict(trial.get("parameters") or payload.get("parameters") or {}),
                candidate_id=cid,
            )
            for item in recovered:
                report.list_valued_parameter_fields.append(item)
                if item.get("recovered_from_dsl") is not None:
                    report.values_recovered_from_dsl.append(item)
            feature_ids = tuple(str(f) for f in (payload.get("feature_ids") or ()))
            if not feature_ids:
                from discovery.candidate import collect_features

                feature_ids = collect_features(
                    entry, exit_tree, stop, target, sizing, *regime_gates
                )
            cand = StrategyCandidate(
                candidate_id=str(payload.get("candidate_id") or cid),
                lineage_id=str(payload.get("lineage_id") or cid),
                generation=int(payload.get("generation") or 0),
                parent_ids=tuple(str(p) for p in (payload.get("parent_ids") or ())),
                creation_method=method,
                strategy_family=str(
                    payload.get("strategy_family")
                    or trial.get("strategy_family")
                    or "dsl_generated"
                ),
                expression_tree=_tree_from_raw(payload.get("expression_tree")) or entry,
                entry_tree=entry,
                exit_tree=exit_tree,
                stop=stop,
                target=target,
                sizing=sizing,
                regime_gates=regime_gates,
                feature_ids=feature_ids,
                parameters=params,
                complexity_score=float(payload.get("complexity_score") or 0.0),
                grammar_version=str(payload.get("grammar_version") or ""),
                feature_set_version=str(payload.get("feature_set_version") or ""),
                cost_model_version=str(
                    payload.get("cost_model_version")
                    or trial.get("cost_model_version")
                    or "cost_v1"
                ),
                asset_universe=tuple(
                    str(a) for a in (payload.get("asset_universe") or ("ES",))
                ),
                random_seed=int(payload.get("random_seed") or trial.get("random_seed") or 0),
                creation_timestamp=str(payload.get("creation_timestamp") or ""),
                family_provenance=dict(payload.get("family_provenance") or {}),
            )
        else:
            entry_raw = snap.get("expression_tree") or snap.get("entry_tree")
            if not isinstance(entry_raw, dict):
                report.migration_records.append(
                    {
                        "candidate_id": cid,
                        "status": "reconstruction_unavailable",
                        "reason": "missing_entry_tree",
                    }
                )
                return None
            entry = ExprNode.from_dict(entry_raw)
            creation = snap.get("creation_method") or CreationMethod.RANDOM.value
            try:
                method = CreationMethod(str(creation))
            except ValueError:
                method = CreationMethod.RANDOM
            params, recovered = normalize_legacy_candidate_parameters(
                entry,
                trial.get("parameters"),
                candidate_id=cid,
            )
            for item in recovered:
                report.list_valued_parameter_fields.append(item)
                if item.get("recovered_from_dsl") is not None:
                    report.values_recovered_from_dsl.append(item)
            # Entry-only legacy snapshot — do not invent exit/stop/target.
            cand = StrategyCandidate(
                candidate_id=cid,
                lineage_id=str(trial.get("lineage_id") or cid),
                generation=int(snap.get("generation") or 0),
                parent_ids=tuple(str(p) for p in (snap.get("parent_ids") or ())),
                creation_method=method,
                strategy_family=str(trial.get("strategy_family") or "dsl_generated"),
                expression_tree=entry,
                entry_tree=entry,
                exit_tree=None,
                stop=None,
                target=None,
                sizing=None,
                regime_gates=(),
                feature_ids=tuple(str(f) for f in (snap.get("feature_ids") or entry.feature_ids())),
                parameters=params,
                complexity_score=float(snap.get("complexity") or 0.0),
                grammar_version=str(snap.get("grammar_version") or ""),
                feature_set_version=str(snap.get("feature_set_version") or ""),
                cost_model_version=str(trial.get("cost_model_version") or "cost_v1"),
                asset_universe=("ES",),
                random_seed=int(trial.get("random_seed") or 0),
                family_provenance=dict(snap.get("family_provenance") or {}),
            )
    except LegacyParameterError as exc:
        wrapped = LegacyBootstrapError(
            exc.code,
            str(exc).split(": ", 1)[-1],
            candidate_id=exc.candidate_id or cid or None,
            parameter=exc.parameter,
            phase="normalize_legacy_candidate_parameters",
            details=exc.details,
        )
        if require_complete:
            report.ambiguous_candidate_ids.append(cid)
            raise wrapped from exc
        report.ambiguous_candidate_ids.append(cid)
        report.reconstruction_unavailable_ids.append(cid)
        report.migration_records.append(
            {
                "candidate_id": cid,
                "status": "reconstruction_unavailable",
                "reason": wrapped.code,
                "parameter": wrapped.parameter,
                "details": wrapped.details,
            }
        )
        return None
    except LegacyBootstrapError as exc:
        if require_complete:
            if exc.code == LEGACY_PARAMETER_VALUE_AMBIGUOUS:
                report.ambiguous_candidate_ids.append(cid)
            else:
                report.incomplete_candidate_ids.append(cid)
            raise
        report.ambiguous_candidate_ids.append(cid)
        report.reconstruction_unavailable_ids.append(cid)
        report.migration_records.append(
            {
                "candidate_id": cid,
                "status": "reconstruction_unavailable",
                "reason": exc.code,
                "parameter": exc.parameter,
                "details": exc.details,
            }
        )
        return None
    except Exception as exc:  # noqa: BLE001
        if require_complete:
            raise LegacyBootstrapError(
                LEGACY_CANDIDATE_SPEC_INCOMPLETE,
                f"failed to reconstruct candidate: {exc}",
                candidate_id=cid or None,
                phase="candidate_from_trial",
                details={"error": str(exc), "error_type": type(exc).__name__},
            ) from exc
        report.reconstruction_unavailable_ids.append(cid)
        report.migration_records.append(
            {
                "candidate_id": cid,
                "status": "reconstruction_unavailable",
                "reason": type(exc).__name__,
                "error": str(exc),
            }
        )
        return None

    if require_complete and not _candidate_complete_for_stress(cand):
        report.incomplete_candidate_ids.append(cid)
        raise LegacyBootstrapError(
            LEGACY_CANDIDATE_SPEC_INCOMPLETE,
            (
                "SCORE_QUALIFIED candidate missing authoritative executable trees; "
                "refusing to run Stress on a modified strategy"
            ),
            candidate_id=cid or None,
            phase="candidate_from_trial",
            details={
                "has_exit_tree": cand.exit_tree is not None,
                "has_stop": cand.stop is not None,
                "has_target": cand.target is not None,
                "parameter_keys": sorted(cand.parameters),
                "tree_parameter_keys": sorted(
                    collect_parameters(
                        cand.entry_tree,
                        cand.exit_tree,
                        cand.stop,
                        cand.target,
                        cand.sizing,
                        *cand.regime_gates,
                    )
                ),
            },
        )

    if not require_complete and not _candidate_complete_for_stress(cand):
        # History-only: keep eval record / population membership without executable spec.
        report.reconstruction_unavailable_ids.append(cid)
        report.migration_records.append(
            {
                "candidate_id": cid,
                "status": "reconstruction_unavailable",
                "reason": "entry_only_or_incomplete_spec",
                "has_exit_tree": cand.exit_tree is not None,
                "has_stop": cand.stop is not None,
                "has_target": cand.target is not None,
            }
        )
        return None

    return cand


def _eval_record_from_trial(trial: dict[str, Any]) -> dict[str, Any]:
    rejected = trial.get("rejection_reason")
    ranking = trial.get("ranking_score")
    snap = dict(trial.get("config_snapshot") or {})
    outcome = "REJECTED" if rejected else "REGISTERED"
    if trial.get("trial_status") == "FAILED":
        outcome = "EVAL_FAILED"
    fitness = None
    if ranking is not None:
        net = dict(trial.get("net_metrics") or {})
        fitness = {
            "fitness": float(ranking),
            "ranking_source": trial.get("ranking_source") or "validation_oos",
            "components": sanitize_fitness_components(net),
            "fold_scores": [],
            "rejected": bool(rejected),
            "rejection_reason": rejected,
        }
    folds = []
    for fr in trial.get("fold_records") or []:
        if not isinstance(fr, dict):
            continue
        folds.append(
            {
                "fold_id": int(fr.get("fold_id") or 0),
                "expectancy": float(fr.get("expectancy") or 0.0),
                "sharpe": float(fr.get("sharpe") or 0.0),
                "profit_factor": float(fr.get("profit_factor") or 0.0),
                "calmar": float(fr.get("calmar") or 0.0),
                "max_drawdown": float(fr.get("max_drawdown") or 0.0),
                "n_trades": int(fr.get("n_trades") or 0),
            }
        )
    is_full = bool(snap.get("is_full_event_wfo"))
    signal_source = str(snap.get("signal_source") or "")
    if is_full and not signal_source:
        signal_source = "candidate_dsl_trees"
    train_metrics = dict(trial.get("gross_metrics") or {})
    train_metrics.setdefault("signal_source", signal_source or train_metrics.get("signal_source"))
    train_metrics.setdefault("is_full_event_wfo", is_full)
    train_metrics.setdefault(
        "wfo_completed_folds",
        int(snap.get("wfo_completed_folds") or len(folds) or (3 if is_full else 0)),
    )
    return {
        "outcome": outcome,
        "candidate_id": str(trial["candidate_id"]),
        "lineage_id": str(trial.get("lineage_id") or ""),
        "trial_id": trial.get("trial_id"),
        "fitness": fitness,
        "rejection_reason": rejected,
        "oos_folds": folds,
        "train_metrics": train_metrics,
        "runtime_seconds": 0.0,
        "memory_mb": 0.0,
        "stress_results": {},
        "robustness_results": {},
        "behavioral_cluster": None,
        "meta": {
            "is_full_event_wfo": is_full,
            "baseline_wfo_artifacts": {
                "signal_source": signal_source or "candidate_dsl_trees",
                "is_full_event_wfo": is_full,
                "wfo_completed_folds": int(train_metrics.get("wfo_completed_folds") or 0),
                "backend_kind": str(snap.get("backend_kind") or "event_driven_wfo"),
            },
            "bootstrap_from_trial_ledger": True,
            "net_metrics_non_scalar": {
                str(k): value_shape(v)
                for k, v in (trial.get("net_metrics") or {}).items()
                if finite_float(v) is None
            },
        },
    }


def _score_qualified_ids_from_campaign(campaign: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for ev in campaign.get("candidate_status_history") or []:
        if str(ev.get("new_status") or "") == "SCORE_QUALIFIED":
            cid = str(ev.get("candidate_id") or "")
            if cid:
                ids.add(cid)
    for gen in campaign.get("generation_records") or []:
        for cid in gen.get("score_qualified_candidate_ids") or []:
            ids.add(str(cid))
    return ids


def _pending_score_qualified_ids(
    campaign: dict[str, Any],
    trials: Iterable[dict[str, Any]],
) -> set[str]:
    """IDs that must be executable for Stress after legacy bootstrap."""
    sq = _score_qualified_ids_from_campaign(campaign)
    stressed = {
        str(s.get("candidate_id") or "")
        for s in (campaign.get("candidate_stress_summaries") or [])
        if s.get("final_decision") in {"STRESS_PASSED", "STRESS_FAILED"}
    }
    pending = {cid for cid in sq if cid and cid not in stressed}
    if pending:
        return pending
    # Legacy runs may lack SCORE_QUALIFIED status events; infer from unrejeced ranks.
    inferred: set[str] = set()
    for trial in trials:
        if trial.get("rejection_reason"):
            continue
        if trial.get("ranking_score") is None:
            continue
        inferred.add(str(trial["candidate_id"]))
    return inferred


def bootstrap_checkpoint_from_run(
    artifact_dir: Path,
    *,
    search_program_id: str,
    compatibility_fingerprint: str,
    session_run_id: str,
    search_mode: str,
    fingerprint_components: dict[str, str] | None = None,
    source_run_id: str | None = None,
    resumed_from_run_id: str | None = None,
) -> tuple[SearchCheckpoint | None, LegacyBootstrapReport]:
    """Build a resume checkpoint from ``multi_family_campaign.json`` + trial ledger.

    Used when a RUNTIME_EXHAUSTED run predates durable checkpoints but still has
    SCORE_QUALIFIED candidates awaiting Stress.
    """
    art = Path(artifact_dir)
    report = LegacyBootstrapReport(
        source_run_id=source_run_id or art.name,
        search_program_id=search_program_id,
    )
    campaign_path = art / "multi_family_campaign.json"
    ledger_path = art / "registry" / "trial_ledger.jsonl"
    if not campaign_path.is_file():
        return None, report
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trials.append(json.loads(line))
    report.trials_scanned = len(trials)

    pending_sq = _pending_score_qualified_ids(campaign, trials)
    candidates: dict[str, dict[str, Any]] = {}
    evaluation_records: dict[str, dict[str, Any]] = {}
    trial_population_ids: list[str] = []

    for trial in trials:
        cid = str(trial.get("candidate_id") or "")
        if not cid:
            continue
        trial_population_ids.append(cid)
        require_complete = cid in pending_sq
        cand = _candidate_from_trial(
            trial, report=report, require_complete=require_complete
        )
        evaluation_records[cid] = _eval_record_from_trial(trial)
        if cand is None:
            if trial.get("rejection_reason") or cid not in pending_sq:
                report.rejected_history_only_preserved += 1
            continue
        candidates[cand.candidate_id] = cand.as_dict()
        report.candidates_reconstructed += 1

    family_states: dict[str, FamilyCheckpointState] = {}
    for st in campaign.get("family_stats") or []:
        fid = str(st.get("family_id") or "")
        if not fid:
            continue
        evaluated_ids: list[str] = []
        for trial in trials:
            tcid = str(trial.get("candidate_id") or "")
            if not tcid or tcid not in evaluation_records:
                continue
            cand_raw = candidates.get(tcid) or {}
            prov = dict(
                cand_raw.get("family_provenance")
                or (trial.get("config_snapshot") or {}).get("family_provenance")
                or {}
            )
            fam = str(
                prov.get("family_id")
                or cand_raw.get("strategy_family")
                or trial.get("strategy_family")
                or ""
            )
            if fam == fid:
                evaluated_ids.append(tcid)
        family_states[fid] = FamilyCheckpointState(
            family_id=fid,
            generated_count=int(st.get("generated") or 0),
            generation_0_count=int(st.get("generation_0_generated") or st.get("generated") or 0),
            descendant_count=int(st.get("descendants_generated") or 0),
            mutation_count=int(st.get("mutation_children") or 0),
            crossover_count=int(st.get("crossover_children") or 0),
            full_wfo_count=int(st.get("full_wfo") or 0),
            highest_generation=int(st.get("highest_generation_reached") or 0),
            structural_parents_found=int(st.get("structural_parents_found") or 0),
            family_stop_reason=st.get("family_stop_reason")
            or campaign.get("aggregated_stop_reason"),
            family_gen_cap=int(st.get("allocation_generated") or st.get("generated") or 0),
            family_wfo_cap=int(st.get("allocation_wfo") or st.get("full_wfo") or 0),
            generation_0_complete=True,
            evaluated_ids=sorted(set(evaluated_ids)),
        )

    gates: dict[str, str] = {}
    for ev in campaign.get("candidate_status_history") or []:
        cid = str(ev.get("candidate_id") or "")
        new_status = str(ev.get("new_status") or "")
        if cid and new_status:
            gates[cid] = new_status

    alloc = dict(campaign.get("budget_allocation") or {})
    family_stats = list(campaign.get("family_stats") or [])
    generated_sum = sum(int(st.get("generated") or 0) for st in family_stats)
    evaluated_sum = sum(int(st.get("evaluated") or 0) for st in family_stats)
    full_wfo_sum = sum(int(st.get("full_wfo") or 0) for st in family_stats)
    ckpt = SearchCheckpoint(
        version=CHECKPOINT_VERSION,
        search_program_id=search_program_id,
        compatibility_fingerprint=compatibility_fingerprint,
        session_run_id=session_run_id,
        source_run_id=source_run_id,
        resumed_from_run_id=resumed_from_run_id,
        search_mode=search_mode,
        pipeline_phase=PipelinePhase.STRESS.value,
        updated_at=now_iso(),
        seed=int((campaign.get("config") or {}).get("seed") or 42),
        config=dict(alloc.get("config") or campaign.get("config") or {}),
        family_specs=list(campaign.get("families") or []),
        candidates=candidates,
        candidate_gates=gates,
        evaluation_records=evaluation_records,
        family_states=family_states,
        generation_records=list(campaign.get("generation_records") or []),
        status_history=list(campaign.get("candidate_status_history") or []),
        campaign_generated=int(
            alloc.get("campaign_generated") or generated_sum or len(evaluation_records)
        ),
        campaign_evaluated=int(
            alloc.get("campaign_evaluated") or evaluated_sum or len(evaluation_records)
        ),
        campaign_full_wfo=int(alloc.get("campaign_full_wfo") or full_wfo_sum or 0),
        stop_reason=campaign.get("aggregated_stop_reason"),
        stress_summaries=list(campaign.get("candidate_stress_summaries") or []),
        robustness_summaries=list(campaign.get("candidate_robustness_summaries") or []),
        statistics_summaries=list(campaign.get("candidate_statistics_summaries") or []),
        clusters=list(campaign.get("clusters") or []),
        behavioral_signatures=list(campaign.get("behavioral_signatures") or []),
        shortlist_rejects=list(campaign.get("shortlist_rejects") or []),
        research_shortlist=list(campaign.get("research_shortlist") or []),
        population_stats=dict(campaign.get("population_stats") or {}),
        trial_population_candidate_ids=sorted(set(trial_population_ids)),
        fingerprint_components=dict(fingerprint_components or {}),
        budget_caps={
            "total_candidate_budget": alloc.get("total_candidate_budget"),
            "max_full_wfo": alloc.get("max_full_wfo"),
        },
    )
    mark_score_qualified_pending(ckpt)
    # Ensure every pending SCORE_QUALIFIED has an executable reconstructed candidate.
    missing_pending = [cid for cid in ckpt.pending_stress_ids if cid not in ckpt.candidates]
    if missing_pending:
        report.incomplete_candidate_ids.extend(missing_pending)
        raise LegacyBootstrapError(
            LEGACY_CANDIDATE_SPEC_INCOMPLETE,
            (
                "pending SCORE_QUALIFIED candidates lack executable specs: "
                + ", ".join(missing_pending[:8])
            ),
            candidate_id=missing_pending[0],
            phase="validate_pending_score_qualified",
            details={"missing_pending_ids": missing_pending},
        )
    report.pending_score_qualified_restored = list(ckpt.pending_stress_ids)
    if ckpt.pending_stress_ids:
        ckpt.pipeline_phase = PipelinePhase.STRESS.value

    # Validate checkpoint can be rehydrated before publish.
    rebuild_candidates(ckpt)
    rebuild_evaluation_records(ckpt)
    return ckpt, report


def prepare_search_program_session(
    *,
    search_mode: str,
    program_id: str | None,
    program_store: SearchProgramStore,
    compatibility_fingerprint: str,
    seed: int,
    family_ids: list[str] | None,
    run_id: str,
    source_run_id: str | None,
    resumed_from_run_id: str | None,
    artifacts_root: Path,
    fingerprint_components: dict[str, Any] | None = None,
    multi_family_config: dict[str, Any] | None = None,
    run_artifact_dir: Path | None = None,
) -> PreparedSearchSession:
    """Create or resume a search program, bootstrapping legacy runs when needed.

    NEW_SEARCH always creates a fresh program with no checkpoint.
    RESUME / EXTEND / REEVALUATE never proceed with a null checkpoint: when the
    program has no checkpoint yet and ``source_run_id`` points at prior artifacts,
    ``bootstrap_checkpoint_from_run`` is used and the result is persisted atomically.
    """
    mode = SearchMode(str(search_mode or SearchMode.NEW_SEARCH.value).upper())
    created_new = False
    bootstrapped = False
    fingerprint_migrated = False
    control_plane_code_changed = False
    legacy_code_hash_migrated = False
    migration_info: dict[str, Any] = {}
    resume_ckpt: SearchCheckpoint | None = None
    bootstrap_report: LegacyBootstrapReport | None = None
    incoming_components = dict(fingerprint_components or {})
    incoming_mf_canonical = canonical_multi_family_payload(multi_family_config)

    if mode is SearchMode.NEW_SEARCH:
        pid = program_id or new_search_program_id()
        if program_store.get(pid) is None:
            program_store.create(
                compatibility_fingerprint=compatibility_fingerprint,
                seed=int(seed),
                family_ids=list(family_ids) if family_ids else None,
                search_program_id=pid,
                metadata={
                    "created_by_run_id": run_id,
                    "fingerprint_components": dict(incoming_components),
                    "repository_git_sha": incoming_components.get("repository_git_sha"),
                    "execution_semantic_hash": incoming_components.get(
                        "execution_semantic_hash"
                    ),
                },
            )
            created_new = True
        ckpt_path = program_store.checkpoint_path(pid)
        cache = EvaluationCache(program_store.evaluation_cache_dir(pid))
        return PreparedSearchSession(
            search_program_id=pid,
            resume_checkpoint=None,
            checkpoint_path=ckpt_path,
            evaluation_cache=cache,
            created_new_program=created_new,
            bootstrapped_from_source=False,
            bootstrap_report=None,
            fingerprint_migrated=False,
        )

    # RESUME_SEARCH / EXTEND_BUDGET / REEVALUATE_FROZEN_CANDIDATES
    deferred_create = False
    if not program_id:
        pid = new_search_program_id()
        deferred_create = True
    else:
        pid = str(program_id)
        existing = program_store.get(pid)
        if existing is None:
            deferred_create = True
        else:
            ckpt_path_early = program_store.checkpoint_path(pid)
            stored_ckpt = load_checkpoint(ckpt_path_early)
            stored_components, stored_mf_canonical = _stored_fingerprint_context(
                program=existing,
                checkpoint=stored_ckpt,
                artifacts_root=artifacts_root,
                source_run_id=source_run_id,
            )
            if (
                "multi_family_config_hash" not in incoming_components
                and multi_family_config is not None
            ):
                incoming_components["multi_family_config_hash"] = hash_multi_family_config(
                    multi_family_config
                )
            if "seed" not in incoming_components:
                incoming_components["seed"] = int(seed)
            if "family_ids" not in incoming_components:
                incoming_components["family_ids"] = (
                    sorted(str(x) for x in family_ids) if family_ids else None
                )

            repo_root = resolve_repo_root(Path(__file__).resolve().parents[2])
            current_sem = str(incoming_components.get("execution_semantic_hash") or "")
            normalized_stored, legacy_info = normalize_legacy_code_hash_component(
                stored_components,
                repo_root=repo_root,
                current_execution_semantic_hash=current_sem or None,
            )
            migration_info = dict(legacy_info)
            legacy_code_hash_migrated = bool(legacy_info.get("legacy_schema"))

            migrate_diff = assert_fingerprint_compatible(
                existing.compatibility_fingerprint,
                compatibility_fingerprint,
                mode=mode,
                stored_components=normalized_stored,
                incoming_components=incoming_components,
                stored_multi_family_canonical=stored_mf_canonical,
                incoming_multi_family_canonical=incoming_mf_canonical,
                artifact_dir=run_artifact_dir,
                legacy_code_hash_migrated=legacy_code_hash_migrated,
            )
            if migrate_diff is not None and (
                migrate_diff.migrated_aggregate
                or migrate_diff.control_plane_code_changed
                or legacy_code_hash_migrated
            ):
                # Atomic program metadata migration — never touch counters/queues/cache.
                original_fp = existing.compatibility_fingerprint
                original_comps = dict(
                    (existing.metadata or {}).get("fingerprint_components")
                    or stored_components
                )
                existing.compatibility_fingerprint = compatibility_fingerprint
                meta = dict(existing.metadata or {})
                history = list(meta.get("fingerprint_migration_history") or [])
                history.append(
                    {
                        "migrated_at": now_iso(),
                        "from_fingerprint": original_fp,
                        "to_fingerprint": compatibility_fingerprint,
                        "from_components": original_comps,
                        "to_components": dict(incoming_components),
                        "legacy_code_hash_migrated": legacy_code_hash_migrated,
                        "legacy_info": dict(legacy_info),
                        "control_plane_code_changed": bool(
                            migrate_diff.control_plane_code_changed
                        ),
                        "audit_reason_codes": list(migrate_diff.audit_reason_codes),
                        "run_id": run_id,
                    }
                )
                meta["fingerprint_migration_history"] = history
                meta["fingerprint_components"] = dict(incoming_components)
                meta["fingerprint_migrated_at"] = now_iso()
                meta["fingerprint_migrated_from"] = original_fp
                meta["repository_git_sha"] = incoming_components.get(
                    "repository_git_sha"
                )
                meta["execution_semantic_hash"] = incoming_components.get(
                    "execution_semantic_hash"
                )
                if migrate_diff.control_plane_code_changed:
                    control_plane_code_changed = True
                    audits = list(meta.get("audit_events") or [])
                    audits.append(
                        {
                            "reason": CONTROL_PLANE_CODE_CHANGED,
                            "stored_repository_git_sha": migrate_diff.stored_git_commit_sha,
                            "incoming_repository_git_sha": migrate_diff.incoming_git_commit_sha,
                            "execution_semantic_hash": incoming_components.get(
                                "execution_semantic_hash"
                            ),
                            "at": now_iso(),
                            "run_id": run_id,
                        }
                    )
                    meta["audit_events"] = audits
                existing.metadata = meta
                program_store.save(existing)

                # Update checkpoint fingerprint components in place (identity only).
                if stored_ckpt is not None:
                    stored_ckpt.compatibility_fingerprint = compatibility_fingerprint
                    stored_ckpt.fingerprint_components = engine_fingerprint_subset(
                        incoming_components
                    )
                    save_checkpoint(ckpt_path_early, stored_ckpt)

                fingerprint_migrated = True
                migration_info["migrated"] = True
                migration_info["from_fingerprint"] = original_fp
                migration_info["to_fingerprint"] = compatibility_fingerprint

    ckpt_path = program_store.checkpoint_path(pid)
    # Do not mkdir the program dir until bootstrap publishes (atomic rollback).
    cache: EvaluationCache | None = None
    resume_ckpt = None if deferred_create else load_checkpoint(ckpt_path)
    if resume_ckpt is not None:
        cache = EvaluationCache(program_store.evaluation_cache_dir(pid))

    if resume_ckpt is not None:
        try:
            for cid in list(resume_ckpt.pending_stress_ids):
                raw = resume_ckpt.candidates.get(cid)
                if raw is None:
                    raise LegacyBootstrapError(
                        LEGACY_CANDIDATE_SPEC_INCOMPLETE,
                        "pending SCORE_QUALIFIED missing from checkpoint candidates",
                        candidate_id=cid,
                        phase="validate_existing_checkpoint",
                    )
                cand = StrategyCandidate.from_dict(raw)
                if not _candidate_complete_for_stress(cand):
                    raise LegacyBootstrapError(
                        LEGACY_CANDIDATE_SPEC_INCOMPLETE,
                        (
                            "pending SCORE_QUALIFIED checkpoint candidate is incomplete; "
                            "refusing Stress on a modified strategy"
                        ),
                        candidate_id=cid,
                        phase="validate_existing_checkpoint",
                    )
            rebuild_evaluation_records(resume_ckpt)
        except LegacyBootstrapError:
            if not source_run_id:
                raise
            # Retry path: drop the broken checkpoint and rebuild from source artifacts.
            resume_ckpt = None
            if ckpt_path.is_file():
                ckpt_path.unlink()

    if resume_ckpt is None and source_run_id:
        src = Path(artifacts_root) / str(source_run_id)
        tmp_root = Path(tempfile.mkdtemp(prefix=f"legacy_bootstrap_{pid}_"))
        published = False
        try:
            tmp_ckpt_path = tmp_root / "checkpoint.json"
            resume_ckpt, bootstrap_report = bootstrap_checkpoint_from_run(
                src,
                search_program_id=pid,
                compatibility_fingerprint=compatibility_fingerprint,
                session_run_id=run_id,
                search_mode=mode.value,
                fingerprint_components=engine_fingerprint_subset(incoming_components),
                source_run_id=str(source_run_id),
                resumed_from_run_id=str(resumed_from_run_id or source_run_id),
            )
            if resume_ckpt is None:
                raise RuntimeError(
                    f"{MISSING_CHECKPOINT}: program={pid!r} mode={mode.value} "
                    f"source_run_id={source_run_id!r}"
                )
            save_checkpoint(tmp_ckpt_path, resume_ckpt)
            # Re-load + validate from the temporary location before publish.
            validated = load_checkpoint(tmp_ckpt_path)
            assert validated is not None
            rebuild_candidates(validated)
            rebuild_evaluation_records(validated)

            if deferred_create:
                program_store.create(
                    compatibility_fingerprint=compatibility_fingerprint,
                    seed=int(seed),
                    family_ids=list(family_ids) if family_ids else None,
                    search_program_id=pid,
                    metadata={
                        "created_by_run_id": run_id,
                        "legacy_bootstrap": True,
                        "source_run_id": source_run_id,
                        "fingerprint_components": dict(incoming_components),
                    },
                )
                created_new = True

            program_store.evaluation_cache_dir(pid).mkdir(parents=True, exist_ok=True)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_ckpt_path, ckpt_path)
            report_path = program_store.program_dir(pid) / "legacy_bootstrap_report.json"
            if bootstrap_report is not None:
                bootstrap_report.search_program_id = pid
                report_path.write_text(
                    json.dumps(bootstrap_report.as_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
                try:
                    (src / "legacy_bootstrap_report.json").write_text(
                        json.dumps(bootstrap_report.as_dict(), indent=2, default=str),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
            program_store.update_cumulative(
                pid,
                generated=resume_ckpt.campaign_generated,
                evaluated=resume_ckpt.campaign_evaluated,
                full_wfo=resume_ckpt.campaign_full_wfo,
                checkpoint_path=str(ckpt_path),
            )
            bootstrapped = True
            published = True
            resume_ckpt = validated
            cache = EvaluationCache(program_store.evaluation_cache_dir(pid))
        except Exception:
            if deferred_create and not published:
                program_store.delete(pid)
            raise
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if resume_ckpt is None:
        if deferred_create:
            program_store.delete(pid)
        raise RuntimeError(
            f"{MISSING_CHECKPOINT}: program={pid!r} mode={mode.value} "
            f"source_run_id={source_run_id!r}"
        )

    if cache is None:
        cache = EvaluationCache(program_store.evaluation_cache_dir(pid))

    return PreparedSearchSession(
        search_program_id=pid,
        resume_checkpoint=resume_ckpt,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
        created_new_program=created_new,
        bootstrapped_from_source=bootstrapped,
        bootstrap_report=bootstrap_report,
        fingerprint_migrated=fingerprint_migrated,
        control_plane_code_changed=control_plane_code_changed,
        legacy_code_hash_migrated=legacy_code_hash_migrated,
        migration_info=migration_info,
    )
