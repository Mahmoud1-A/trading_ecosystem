"""Bootstrap a search checkpoint from a prior Alpha Miner run's artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from discovery.candidate import StrategyCandidate
from discovery.evaluation_cache import EvaluationCache
from discovery.expression_tree import ExprNode
from discovery.search_checkpoint import (
    CHECKPOINT_VERSION,
    FamilyCheckpointState,
    SearchCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from discovery.search_program import (
    MISSING_CHECKPOINT,
    PipelinePhase,
    SearchMode,
    SearchProgramStore,
    assert_fingerprint_compatible,
    new_search_program_id,
    now_iso,
)
from discovery.search_resume import mark_score_qualified_pending
from discovery.types import CreationMethod


@dataclass
class PreparedSearchSession:
    search_program_id: str
    resume_checkpoint: SearchCheckpoint | None
    checkpoint_path: Path
    evaluation_cache: EvaluationCache
    created_new_program: bool
    bootstrapped_from_source: bool


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
    fingerprint_components: dict[str, str] | None = None,
) -> PreparedSearchSession:
    """Create or resume a search program, bootstrapping legacy runs when needed.

    NEW_SEARCH always creates a fresh program with no checkpoint.
    RESUME / EXTEND / REEVALUATE never proceed with a null checkpoint: when the
    program has no checkpoint yet and ``source_run_id`` points at prior artifacts,
    ``bootstrap_checkpoint_from_run`` is used and the result is persisted.
    """
    mode = SearchMode(str(search_mode or SearchMode.NEW_SEARCH.value).upper())
    created_new = False
    bootstrapped = False
    resume_ckpt: SearchCheckpoint | None = None

    if mode is SearchMode.NEW_SEARCH:
        pid = program_id or new_search_program_id()
        if program_store.get(pid) is None:
            program_store.create(
                compatibility_fingerprint=compatibility_fingerprint,
                seed=int(seed),
                family_ids=list(family_ids) if family_ids else None,
                search_program_id=pid,
                metadata={"created_by_run_id": run_id},
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
        )

    # RESUME_SEARCH / EXTEND_BUDGET / REEVALUATE_FROZEN_CANDIDATES
    if not program_id:
        # Legacy RUNTIME_EXHAUSTED run: assign a program id and create the record.
        pid = new_search_program_id()
        program_store.create(
            compatibility_fingerprint=compatibility_fingerprint,
            seed=int(seed),
            family_ids=list(family_ids) if family_ids else None,
            search_program_id=pid,
            metadata={
                "created_by_run_id": run_id,
                "legacy_bootstrap": True,
                "source_run_id": source_run_id,
            },
        )
        created_new = True
    else:
        pid = str(program_id)
        existing = program_store.get(pid)
        if existing is None:
            # Program id was assigned by Continue Search but record not written yet.
            program_store.create(
                compatibility_fingerprint=compatibility_fingerprint,
                seed=int(seed),
                family_ids=list(family_ids) if family_ids else None,
                search_program_id=pid,
                metadata={
                    "created_by_run_id": run_id,
                    "legacy_bootstrap": True,
                    "source_run_id": source_run_id,
                },
            )
            created_new = True
        else:
            assert_fingerprint_compatible(
                existing.compatibility_fingerprint,
                compatibility_fingerprint,
                mode=mode,
            )

    ckpt_path = program_store.checkpoint_path(pid)
    cache = EvaluationCache(program_store.evaluation_cache_dir(pid))
    resume_ckpt = load_checkpoint(ckpt_path)

    if resume_ckpt is None and source_run_id:
        src = Path(artifacts_root) / str(source_run_id)
        resume_ckpt = bootstrap_checkpoint_from_run(
            src,
            search_program_id=pid,
            compatibility_fingerprint=compatibility_fingerprint,
            session_run_id=run_id,
            search_mode=mode.value,
            fingerprint_components=fingerprint_components,
            source_run_id=str(source_run_id),
            resumed_from_run_id=str(resumed_from_run_id or source_run_id),
        )
        if resume_ckpt is not None:
            save_checkpoint(ckpt_path, resume_ckpt)
            bootstrapped = True
            program_store.update_cumulative(
                pid,
                generated=resume_ckpt.campaign_generated,
                evaluated=resume_ckpt.campaign_evaluated,
                full_wfo=resume_ckpt.campaign_full_wfo,
                checkpoint_path=str(ckpt_path),
            )

    if resume_ckpt is None:
        raise RuntimeError(
            f"{MISSING_CHECKPOINT}: program={pid!r} mode={mode.value} "
            f"source_run_id={source_run_id!r}"
        )

    return PreparedSearchSession(
        search_program_id=pid,
        resume_checkpoint=resume_ckpt,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
        created_new_program=created_new,
        bootstrapped_from_source=bootstrapped,
    )


def _candidate_from_trial(trial: dict[str, Any]) -> StrategyCandidate | None:
    snap = dict(trial.get("config_snapshot") or {})
    entry_raw = snap.get("expression_tree") or snap.get("entry_tree")
    if not isinstance(entry_raw, dict):
        return None
    try:
        entry = ExprNode.from_dict(entry_raw)
    except Exception:  # noqa: BLE001
        return None
    creation = snap.get("creation_method") or CreationMethod.RANDOM.value
    try:
        method = CreationMethod(str(creation))
    except ValueError:
        method = CreationMethod.RANDOM
    return StrategyCandidate(
        candidate_id=str(trial["candidate_id"]),
        lineage_id=str(trial.get("lineage_id") or trial["candidate_id"]),
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
        feature_ids=tuple(str(f) for f in (snap.get("feature_ids") or ())),
        parameters={str(k): float(v) for k, v in (trial.get("parameters") or {}).items()},
        complexity_score=float(snap.get("complexity") or 0.0),
        grammar_version=str(snap.get("grammar_version") or ""),
        feature_set_version=str(snap.get("feature_set_version") or ""),
        cost_model_version=str(trial.get("cost_model_version") or "cost_v1"),
        asset_universe=("ES",),
        random_seed=int(trial.get("random_seed") or 0),
        family_provenance=dict(snap.get("family_provenance") or {}),
    )


def _eval_record_from_trial(trial: dict[str, Any]) -> dict[str, Any]:
    rejected = trial.get("rejection_reason")
    ranking = trial.get("ranking_score")
    snap = dict(trial.get("config_snapshot") or {})
    outcome = "REJECTED" if rejected else "REGISTERED"
    if trial.get("trial_status") == "FAILED":
        outcome = "EVAL_FAILED"
    fitness = None
    if ranking is not None:
        fitness = {
            "fitness": float(ranking),
            "ranking_source": trial.get("ranking_source") or "validation_oos",
            "components": dict(trial.get("net_metrics") or {}),
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
        },
    }


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
) -> SearchCheckpoint | None:
    """Build a resume checkpoint from ``multi_family_campaign.json`` + trial ledger.

    Used when a RUNTIME_EXHAUSTED run predates durable checkpoints but still has
    SCORE_QUALIFIED candidates awaiting Stress.
    """
    art = Path(artifact_dir)
    campaign_path = art / "multi_family_campaign.json"
    ledger_path = art / "registry" / "trial_ledger.jsonl"
    if not campaign_path.is_file():
        return None
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trials.append(json.loads(line))

    candidates: dict[str, dict[str, Any]] = {}
    evaluation_records: dict[str, dict[str, Any]] = {}
    for trial in trials:
        cand = _candidate_from_trial(trial)
        if cand is None:
            continue
        candidates[cand.candidate_id] = cand.as_dict()
        evaluation_records[cand.candidate_id] = _eval_record_from_trial(trial)

    family_states: dict[str, FamilyCheckpointState] = {}
    for st in campaign.get("family_stats") or []:
        fid = str(st.get("family_id") or "")
        if not fid:
            continue
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
            evaluated_ids=[
                cid
                for cid, rec in evaluation_records.items()
                if (candidates.get(cid) or {}).get("strategy_family") == fid
                or ((candidates.get(cid) or {}).get("family_provenance") or {}).get("family_id")
                == fid
            ],
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
            alloc.get("campaign_generated") or generated_sum or len(candidates)
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
        trial_population_candidate_ids=sorted(evaluation_records.keys()),
        fingerprint_components=dict(fingerprint_components or {}),
        budget_caps={
            "total_candidate_budget": alloc.get("total_candidate_budget"),
            "max_full_wfo": alloc.get("max_full_wfo"),
        },
    )
    mark_score_qualified_pending(ckpt)
    # If score-qualified awaiting stress, stay in STRESS phase (resume priority).
    if ckpt.pending_stress_ids:
        ckpt.pipeline_phase = PipelinePhase.STRESS.value
    return ckpt
